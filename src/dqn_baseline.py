"""Modern learned distributed baseline: DQN-routing (neural Q-routing).

Q-routing (Boyan & Littman 1994) ties MAPPO on delivery at the training scale
but its TABLE cannot transfer or scale (see qscale-transfer). The natural modern
learned baseline is the same update rule with a shared-weight neural function
approximator over the env's candidate features — the deep analogue of
Q-routing, i.e. what a practitioner would deploy today instead of the table:

  Q_theta(candidate features of (node, neighbor), dst embedding)
    ~= min path cost to dst through that neighbor

Update rule mirrors QRoutingPolicy.observe_transition exactly (fitted-Q /
value-regression form): after each env step, for every agent decision,
    target = immediate(edge delay, neighbor queue) + min_k Q_target(features of
              neighbor's candidates, dst)
and the online net regresses stored (features, target) pairs from a replay
buffer (Huber). A target network syncs periodically. Execution is fully
distributed: one shared net reading only local candidate features + dst.

This file is deliberately separate from mappo_evaluation.py, which is hashed by
code_fingerprint — editing it would invalidate checkpoint reuse.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper

MAX_NODE_ID = 200  # dst embedding table covers every constellation size tested


class QHead(nn.Module):
    def __init__(self, feat_dim: int = 26, hidden: int = 128, dst_emb: int = 0):
        super().__init__()
        self.dst_emb = dst_emb
        if dst_emb > 0:
            self.dst = nn.Embedding(MAX_NODE_ID + 1, dst_emb)
            in_dim = feat_dim + dst_emb
        else:
            # one-hot dst (n<=MAX_NODE_ID): strictly covers the tabular
            # Q(node,dst,neighbor) index -- no dst representation to learn
            self.dst = nn.Identity()
            in_dim = feat_dim + MAX_NODE_ID + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, dst_ids: torch.Tensor) -> torch.Tensor:
        # feats (B, feat_dim), dst_ids (B,) -> Q values (B,)
        if self.dst_emb > 0:
            d = self.dst(dst_ids)
        else:
            d = torch.nn.functional.one_hot(dst_ids, MAX_NODE_ID + 1).float()
        h = torch.cat([feats, d], dim=-1)
        return self.net(h).squeeze(-1)


class DqnRoutingPolicy:
    """Per-node DQN routing baseline (shared weights, CTDE-free, distributed)."""

    def __init__(self, seed: int = 0, feat_dim: int = 26,
                 epsilon: float = 0.3, eps_min: float = 0.05, eps_decay_episodes: int = 100,
                 lr: float = 1e-3, gamma: float = 1.0, batch: int = 64,
                 replay_cap: int = 100_000, target_sync: int = 250,
                 device: str = "cpu"):
        self.feat_dim = feat_dim
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay_episodes = eps_decay_episodes
        self.gamma = gamma
        self.batch = batch
        self.target_sync = target_sync
        self.device = device
        self.rng = random.Random(seed)
        self.net = QHead(feat_dim).to(device)
        self.target_net = QHead(feat_dim).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay: deque = deque(maxlen=replay_cap)
        self._updates = 0
        self._episode = 0
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None
        self.pending: list = []
        self.training = True

    # ---- env plumbing (mirrors QRoutingPolicy) -------------------------------
    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper

    def on_episode_start(self) -> None:
        self._episode += 1
        if self.training and self._episode <= self.eps_decay_episodes:
            span = self.eps_decay_episodes
            self.epsilon = 0.3 - (0.3 - self.eps_min) * (self._episode / span)

    def _q_all(self, feats: np.ndarray, dst: int) -> np.ndarray:
        """Q for a stack of candidate features (k, feat_dim) at one dst."""
        with torch.no_grad():
            f = torch.from_numpy(feats).float().to(self.device)
            d = torch.full((f.shape[0],), dst, dtype=torch.long, device=self.device)
            return self.net(f, d).cpu().numpy()

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        self.pending = []
        for external_index, sat in enumerate(self.wrapper.external_to_internal):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            feats = np.asarray(obs["candidate_features"], dtype=np.float32)[feasible - 1]
            qvals = self._q_all(feats, packet.dst)
            if self.training and self.rng.random() < self.epsilon:
                pick = self.rng.randrange(len(feasible))
            else:
                # selection mirrors tabular Q-routing: learned Q + immediate
                # (delay + neighbor queue + rho) of each candidate
                immediate = feats[:, 2] + feats[:, 1] + feats[:, 4]
                pick = int(np.argmin(qvals + immediate))
            action = int(feasible[pick])
            neighbor = obs["neighbor_ids"][action - 1]
            actions[external_index] = action
            self.pending.append((sat, packet.dst, neighbor, feats[pick].copy(),
                                 int(packet.packet_id)))
        return actions

    def observe_transition(self, info: Dict) -> None:
        if not self.training or self.wrapper is None:
            return
        env = self.wrapper.env
        # fitted-Q targets for every decision made this step: immediate edge
        # cost + neighbor queue (exactly QRoutingPolicy's immediate) + min over
        # the forwarded packet's candidate features AT THE NEIGHBOR under the
        # target net. Crucially the bootstrap features are recomputed for THIS
        # packet (dst/hops/visited context) at the neighbor via the env's own
        # feature code -- using the neighbor's current HOL features (a
        # different packet) corrupts the Bellman target and training collapses
        # to random-walk routing.
        for sat, dst, neighbor, feat, pid in self.pending:
            if neighbor <= 0 or neighbor - 1 >= len(self.wrapper._obs):
                continue
            packet = env.packets.get(pid)
            # terminal anchor: forwarding to dst ENDS the episode for this
            # packet (cost 0 thereafter). The tabular version gets this for
            # free -- q[dst][dst][*] is never updated, frozen at init, which
            # anchors the whole Bellman system; without it the neural
            # bootstrap is circular (no base case) and never converges.
            boot = 0.0
            cand_vs = []
            if neighbor != dst and packet is not None:
                cand_vs = [v for v in
                           self.wrapper._obs[neighbor - 1]["neighbor_ids"] if v > 0]
            if cand_vs:
                nfeats = np.asarray(
                    [env._candidate_features(packet, neighbor, v) for v in cand_vs],
                    dtype=np.float32)
                with torch.no_grad():
                    f = torch.from_numpy(nfeats).float().to(self.device)
                    d = torch.full((f.shape[0],), dst, dtype=torch.long,
                                   device=self.device)
                    boot = float(self.target_net(f, d).min())
            edge = env.graph.get((sat, neighbor))
            immediate = (edge.delay_ms / env.cfg.env.d_ref_ms) if edge else 2.0
            immediate += len(env.queues[neighbor]) / max(1, env.cfg.max_queue_packets)
            # ONLINE TD per decision — the neural analogue of tabular Q-routing's
            # alpha-blended update. A large replay + slow target sync (classic
            # fitted-Q) fails here: early random-bootstrap targets flood the
            # buffer and the Bellman system never converges (delivery ~0.2,
            # random-walk level); per-decision updates toward a target net that
            # tracks closely converge like the table does.
            self._td_update(feat, dst, immediate + self.gamma * boot)
        self.pending = []

    def _td_update(self, feat: np.ndarray, dst: int, target: float) -> None:
        f = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        d = torch.tensor([dst], dtype=torch.long, device=self.device)
        t = torch.tensor([target], dtype=torch.float32, device=self.device)
        q = self.net(f, d)
        loss = nn.functional.smooth_l1_loss(q, t)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self._updates += 1
        if self._updates % self.target_sync == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def _learn(self) -> None:
        if len(self.replay) < self.batch:
            return
        batch = self.rng.sample(self.replay, self.batch)
        feats = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
        dsts = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        targets = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        q = self.net(feats, dsts)
        loss = nn.functional.smooth_l1_loss(q, targets)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self._updates += 1
        if self._updates % self.target_sync == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def save(self, path) -> None:
        import torch as _t
        _t.save({"net": self.net.state_dict(), "feat_dim": self.feat_dim,
                 "episode": self._episode}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DqnRoutingPolicy":
        import torch as _t
        ck = _t.load(path, map_location=device, weights_only=False)
        pol = cls(seed=0, feat_dim=int(ck["feat_dim"]), device=device)
        pol.net.load_state_dict(ck["net"])
        pol.target_net.load_state_dict(ck["net"])
        pol.freeze()
        return pol

    def freeze(self) -> None:
        self.training = False
        self.epsilon = 0.0
