"""Synchronous satellite-level environment for shared-parameter MAPPO.

Every satellite is a logical agent.  All agents observe one frozen slot
snapshot, choose simultaneously, and the environment resolves transmissions in
batch.  This file does not implement MAPPO; it fixes the environment semantics
that must exist before a multi-agent training result is meaningful.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from leo_marl_env import EnvConfig, LeoRoutingEnv, LinkState, SCENARIOS


NO_OP = 0


@dataclass
class MultiAgentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    episode_slots: int = 30
    max_queue_packets: int = 45
    exogenous_packets_per_slot: int = 0
    initial_packets: int = 8
    link_capacity_packets: int = 1
    shared_reward: bool = True
    seed: int = 11


@dataclass
class RoutedPacket:
    packet_id: int
    src: int
    dst: int
    owner: int
    created_slot: int
    hop_count: int = 0
    last_next_hop: Optional[int] = None
    visited: Tuple[int, ...] = ()


class SynchronousLeoMultiAgentEnv:
    """FIFO/HOL routing with simultaneous satellite actions."""

    def __init__(self, cfg: Optional[MultiAgentConfig] = None):
        self.cfg = cfg or MultiAgentConfig()
        self.rng = random.Random(self.cfg.seed)
        self.base = LeoRoutingEnv(self.cfg.env)
        self.n_agents = self.cfg.env.n_sats
        self.max_degree = self.base.max_degree
        self.candidate_feature_dim = self.base.candidate_feature_dim
        self.action_size = self.max_degree + 1
        self.slot = 1
        self.next_packet_id = 1
        self.queues: Dict[int, Deque[int]] = {
            i: deque() for i in range(1, self.n_agents + 1)
        }
        self.packets: Dict[int, RoutedPacket] = {}
        self.generated: set[int] = set()
        self.delivered: set[int] = set()
        self.dropped: set[int] = set()
        self.drop_reasons: Dict[int, str] = {}
        self.graph: Dict[Tuple[int, int], LinkState] = {}
        self.used_rate = [
            [0.0 for _ in range(self.n_agents + 1)]
            for _ in range(self.n_agents + 1)
        ]
        self.last_transition: Dict = {}
        self.trace: List[Dict] = []

    @classmethod
    def from_scenario(cls, name: str, seed: int = 11):
        env_cfg = EnvConfig(seed=seed, scenario=SCENARIOS[name])
        return cls(MultiAgentConfig(env=env_cfg, seed=seed))

    def reset(
        self,
        seed: Optional[int] = None,
        initial_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Tuple[List[Dict], Dict]:
        if seed is not None:
            self.cfg.seed = seed
        self.rng = random.Random(self.cfg.seed)
        self.base = LeoRoutingEnv(self.cfg.env)
        self.base.rng = random.Random(self.cfg.seed)
        self.slot = 1
        self.next_packet_id = 1
        self.queues = {i: deque() for i in range(1, self.n_agents + 1)}
        self.packets = {}
        self.generated = set()
        self.delivered = set()
        self.dropped = set()
        self.drop_reasons = {}
        self.used_rate = [
            [0.0 for _ in range(self.n_agents + 1)]
            for _ in range(self.n_agents + 1)
        ]
        self.last_transition = {}
        self.trace = []

        if initial_pairs is None:
            initial_pairs = self._sample_initial_pairs(self.cfg.initial_packets)
        for src, dst in initial_pairs:
            self._create_packet(src, dst)
        self._refresh_graph()
        observations = self.observe()
        return observations, {
            "slot": self.slot,
            "active_agents": self._active_agents(),
            "global_state": self.global_state(),
        }

    def observe(self) -> List[Dict]:
        self._refresh_graph()
        return [self._agent_observation(i) for i in range(1, self.n_agents + 1)]

    def step(
        self,
        actions: Sequence[int],
        agent_order: Optional[Sequence[int]] = None,
    ) -> Tuple[List[Dict], List[float], bool, bool, Dict]:
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")
        self._refresh_graph()
        frozen_obs = self.observe()
        frozen_graph = self.graph
        order = list(agent_order or range(1, self.n_agents + 1))
        if sorted(order) != list(range(1, self.n_agents + 1)):
            raise ValueError("agent_order must be a permutation of satellite IDs")

        rewards = [0.0 for _ in range(self.n_agents)]
        proposals: List[Dict] = []
        mask_ledger: Dict[int, Dict] = {}
        immediate_drops: List[Tuple[int, int, str]] = []

        for sat in order:
            obs = frozen_obs[sat - 1]
            action = int(actions[sat - 1])
            mask_ledger[sat] = {
                "packet_id": obs["hol_packet_id"],
                "mask": list(obs["action_mask"]),
                "reasons": list(obs["mask_reasons"]),
            }
            if obs["hol_packet_id"] is None:
                if action != NO_OP:
                    rewards[sat - 1] -= self.cfg.env.w_invalid
                continue

            feasible_count = sum(obs["action_mask"][1:])
            if feasible_count == 0:
                immediate_drops.append((sat, obs["hol_packet_id"], "no_route"))
                rewards[sat - 1] -= self.cfg.env.w_invalid
                continue
            if action <= 0 or action >= self.action_size or not obs["action_mask"][action]:
                immediate_drops.append((sat, obs["hol_packet_id"], "invalid_action"))
                rewards[sat - 1] -= self.cfg.env.w_invalid
                continue

            next_hop = obs["neighbor_ids"][action - 1]
            packet = self.packets[obs["hol_packet_id"]]
            edge = frozen_graph[(sat, next_hop)]
            reward = self._forward_reward(packet, sat, next_hop, edge)
            proposals.append(
                {
                    "sat": sat,
                    "packet_id": packet.packet_id,
                    "next_hop": next_hop,
                    "reward": reward,
                }
            )

        # Invalid/no-route outcomes are applied before accepted transmissions,
        # but only after every action has been read from the frozen snapshot.
        for sat, packet_id, reason in sorted(immediate_drops, key=lambda x: x[1]):
            self._remove_hol(sat, packet_id)
            self._drop_packet(packet_id, reason)

        accepted, blocked = self._resolve_link_capacity(proposals)
        transmitted_ids: List[int] = []
        link_counts: Dict[str, int] = defaultdict(int)
        incoming: Dict[int, List[Dict]] = defaultdict(list)

        for item in accepted:
            sat = item["sat"]
            packet_id = item["packet_id"]
            next_hop = item["next_hop"]
            self._remove_hol(sat, packet_id)
            incoming[next_hop].append(item)
            transmitted_ids.append(packet_id)
            link_counts[f"{sat}->{next_hop}"] += 1
            rewards[sat - 1] += item["reward"]
            self.used_rate[sat][next_hop] += self.cfg.env.packet_demand_mbps

        for item in blocked:
            rewards[item["sat"] - 1] -= self.cfg.env.w_load

        # Queue admission is deterministic by packet ID, not by satellite loop order.
        for next_hop in sorted(incoming):
            for item in sorted(incoming[next_hop], key=lambda x: x["packet_id"]):
                packet = self.packets[item["packet_id"]]
                packet.hop_count += 1
                packet.last_next_hop = next_hop
                packet.owner = next_hop
                packet.visited = packet.visited + (next_hop,)
                if next_hop == packet.dst:
                    self.delivered.add(packet.packet_id)
                    rewards[item["sat"] - 1] += self.cfg.env.w_deliver
                elif packet.hop_count >= self.cfg.env.max_local_hops:
                    self._drop_packet(packet.packet_id, "ttl_exceeded")
                    rewards[item["sat"] - 1] -= self.cfg.env.w_loop
                elif len(self.queues[next_hop]) >= self.cfg.max_queue_packets:
                    self._drop_packet(packet.packet_id, "queue_overflow")
                    rewards[item["sat"] - 1] -= self.cfg.env.w_invalid
                else:
                    self.queues[next_hop].append(packet.packet_id)

        if self.cfg.exogenous_packets_per_slot > 0:
            for src, dst in self._sample_initial_pairs(
                self.cfg.exogenous_packets_per_slot
            ):
                if len(self.queues[src]) < self.cfg.max_queue_packets:
                    self._create_packet(src, dst)

        concurrent_actions = len(accepted)
        global_reward = sum(rewards) / max(1, concurrent_actions)
        if self.cfg.shared_reward:
            rewards = [global_reward if obs["hol_packet_id"] is not None else 0.0 for obs in frozen_obs]

        transition = {
            "slot": self.slot,
            "actions": [int(a) for a in actions],
            "accepted": sorted(transmitted_ids),
            "blocked": sorted(x["packet_id"] for x in blocked),
            "link_counts": dict(sorted(link_counts.items())),
            "mask_ledger": mask_ledger,
            "concurrent_non_noop": concurrent_actions,
            "global_reward": global_reward,
        }
        self.last_transition = transition
        self.trace.append(transition)
        self._decay_load()
        self.slot += 1
        self.validate_invariants()

        terminated = len(self._backlog_ids()) == 0 and self.cfg.exogenous_packets_per_slot == 0
        truncated = self.slot > self.cfg.episode_slots
        next_obs = self.observe()
        info = {
            **transition,
            "delivered": len(self.delivered),
            "dropped": len(self.dropped),
            "backlog": len(self._backlog_ids()),
            "trace_hash": self.trace_hash(),
            "global_state": self.global_state(),
        }
        return next_obs, rewards, terminated, truncated, info

    def global_state(self) -> Dict:
        self._refresh_graph()
        node_features = []
        hol_features = []
        for sat in range(1, self.n_agents + 1):
            u, w = self.base.orbital_geodetic_coord(sat, self.slot)
            node_features.append(
                [len(self.queues[sat]) / max(1, self.cfg.max_queue_packets), u, w]
            )
            if self.queues[sat]:
                packet = self.packets[self.queues[sat][0]]
                du, dw = self.base.orbital_geodetic_coord(packet.dst, self.slot)
                hol_features.append(
                    [
                        1.0,
                        du,
                        dw,
                        packet.hop_count / max(1, self.cfg.env.max_local_hops),
                        len(set(packet.visited)) / max(1, self.n_agents),
                    ]
                )
            else:
                hol_features.append([0.0] * 5)
        edge_features = [
            [
                u / self.n_agents,
                v / self.n_agents,
                edge.delay_ms / self.cfg.env.d_ref_ms,
                edge.rho,
                edge.reliability,
                min(1.0, edge.t_rem / self.cfg.env.t_safe),
            ]
            for (u, v), edge in sorted(self.graph.items())
        ]
        return {
            "slot_phase": (self.slot % self.cfg.env.max_steps) / max(1, self.cfg.env.max_steps),
            "node_features": node_features,
            "hol_features": hol_features,
            "edge_features": edge_features,
        }

    def validate_invariants(self) -> None:
        backlog = self._backlog_ids()
        if len(backlog) != sum(len(q) for q in self.queues.values()):
            raise AssertionError("single ownership violated: duplicate packet in queues")
        terminal = self.delivered | self.dropped
        if self.delivered & self.dropped:
            raise AssertionError("delivered and dropped sets overlap")
        if backlog & terminal:
            raise AssertionError("terminal packet still has queue ownership")
        if self.generated != backlog | terminal:
            raise AssertionError("packet identity conservation failed")
        accepted = self.last_transition.get("accepted", [])
        if len(accepted) != len(set(accepted)):
            raise AssertionError("packet transmitted more than once in one slot")
        for count in self.last_transition.get("link_counts", {}).values():
            if count > self.cfg.link_capacity_packets:
                raise AssertionError("directed link capacity exceeded")

    def trace_hash(self) -> str:
        payload = json.dumps(self.trace, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def state_digest(self) -> str:
        state = {
            "slot": self.slot,
            "queues": {str(k): list(v) for k, v in sorted(self.queues.items())},
            "delivered": sorted(self.delivered),
            "dropped": sorted(self.dropped),
            "packets": {
                str(pid): {
                    "owner": p.owner,
                    "hop": p.hop_count,
                    "last": p.last_next_hop,
                    "visited": list(p.visited),
                }
                for pid, p in sorted(self.packets.items())
            },
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _agent_observation(self, sat: int) -> Dict:
        queue = self.queues[sat]
        if not queue:
            return {
                "agent_id": sat,
                "hol_packet_id": None,
                "neighbor_ids": [0] * self.max_degree,
                "candidate_features": [[0.0] * self.candidate_feature_dim for _ in range(self.max_degree)],
                "action_mask": [True] + [False] * self.max_degree,
                "mask_reasons": ["inactive_no_op"] + ["inactive"] * self.max_degree,
            }

        packet = self.packets[queue[0]]
        neighbors = self.base._neighbors(sat)[: self.max_degree]
        padded_ids = neighbors + [0] * (self.max_degree - len(neighbors))
        features: List[List[float]] = []
        candidate_mask: List[bool] = []
        reasons: List[str] = []
        for neighbor in padded_ids:
            if neighbor == 0:
                features.append([0.0] * self.candidate_feature_dim)
                candidate_mask.append(False)
                reasons.append("padding")
                continue
            features.append(self._candidate_features(packet, sat, neighbor))
            reason = self._mask_reason(packet, sat, neighbor)
            candidate_mask.append(reason == "feasible")
            reasons.append(reason)
        return {
            "agent_id": sat,
            "hol_packet_id": packet.packet_id,
            "neighbor_ids": padded_ids,
            "candidate_features": features,
            "action_mask": [False] + candidate_mask,
            "mask_reasons": ["active_no_op_forbidden"] + reasons,
            "remaining_hop_ratio": max(
                0.0,
                (self.cfg.env.max_local_hops - packet.hop_count)
                / max(1, self.cfg.env.max_local_hops),
            ),
        }

    def _candidate_features(self, packet: RoutedPacket, u: int, v: int) -> List[float]:
        edge = self.graph[(u, v)]
        vu, vw = self.base.orbital_geodetic_coord(v, self.slot)
        uu, uw = self.base.orbital_geodetic_coord(u, self.slot)
        du, dw = self.base.orbital_geodetic_coord(packet.dst, self.slot)
        return [
            len(self.queues[u]) / max(1, self.cfg.max_queue_packets),
            len(self.queues[v]) / max(1, self.cfg.max_queue_packets),
            edge.delay_ms / self.cfg.env.d_ref_ms,
            self._remaining_bandwidth(u, v) / self.cfg.env.capacity_mbps,
            edge.rho,
            edge.reliability,
            min(1.0, edge.t_rem / self.cfg.env.t_safe),
            0.0,
            self.base._progress_value(u, v, packet.dst),
            vu,
            vw,
            uu,
            uw,
            du,
            dw,
            max(0.0, (self.cfg.env.max_local_hops - packet.hop_count) / max(1, self.cfg.env.max_local_hops)),
            packet.hop_count / max(1, self.cfg.env.max_local_hops),
            float(packet.last_next_hop is not None and v != packet.last_next_hop),
            len(set(packet.visited)) / max(1, self.n_agents),
            (self.slot % self.cfg.env.max_steps) / max(1, self.cfg.env.max_steps),
        ]

    def _mask_reason(self, packet: RoutedPacket, u: int, v: int) -> str:
        if v in packet.visited:
            return "visited"
        edge = self.graph.get((u, v))
        if edge is None or not edge.available:
            return "unavailable"
        if self._remaining_bandwidth(u, v) < self.cfg.env.b_min_mbps:
            return "bandwidth"
        if len(self.queues[v]) >= self.cfg.max_queue_packets:
            return "queue_full"
        if edge.reliability < self.cfg.env.r_min:
            return "reliability"
        if edge.t_rem < self.cfg.env.t_safe:
            return "lifetime"
        return "feasible"

    def _resolve_link_capacity(self, proposals: Iterable[Dict]):
        by_link: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        for item in proposals:
            by_link[(item["sat"], item["next_hop"])].append(item)
        accepted: List[Dict] = []
        blocked: List[Dict] = []
        for link in sorted(by_link):
            items = sorted(by_link[link], key=lambda x: x["packet_id"])
            accepted.extend(items[: self.cfg.link_capacity_packets])
            blocked.extend(items[self.cfg.link_capacity_packets :])
        return accepted, blocked

    def _forward_reward(self, packet: RoutedPacket, u: int, v: int, edge: LinkState) -> float:
        return (
            -self.cfg.env.w_delay * edge.delay_ms / self.cfg.env.d_ref_ms
            -self.cfg.env.w_queue * len(self.queues[v]) / max(1, self.cfg.max_queue_packets)
            -self.cfg.env.w_load * edge.rho
            -self.cfg.env.w_risk * (1.0 - edge.reliability)
            -self.cfg.env.w_lifetime * self.cfg.env.t_safe / max(self.cfg.env.t_safe, edge.t_rem)
            +self.cfg.env.w_progress * self.base._progress_value(u, v, packet.dst)
            -self.cfg.env.w_switch * float(packet.last_next_hop is not None and v != packet.last_next_hop)
        )

    def _refresh_graph(self) -> None:
        self.base.time_slot = self.slot
        self.base.used_rate = self.used_rate
        self.base.queues = [0.0] + [
            float(len(self.queues[i])) for i in range(1, self.n_agents + 1)
        ]
        self.graph = self.base._build_topology(self.slot)
        self.base.graph = self.graph

    def _decay_load(self) -> None:
        for u in range(1, self.n_agents + 1):
            for v in range(1, self.n_agents + 1):
                self.used_rate[u][v] *= self.cfg.env.load_decay

    def _remaining_bandwidth(self, u: int, v: int) -> float:
        edge = self.graph[(u, v)]
        return edge.capacity_mbps - edge.used_rate_mbps

    def _remove_hol(self, sat: int, packet_id: int) -> None:
        if not self.queues[sat] or self.queues[sat][0] != packet_id:
            raise AssertionError("HOL ownership changed during frozen-slot resolution")
        self.queues[sat].popleft()

    def _drop_packet(self, packet_id: int, reason: str) -> None:
        self.dropped.add(packet_id)
        self.drop_reasons[packet_id] = reason

    def _create_packet(self, src: int, dst: int) -> int:
        if src == dst:
            raise ValueError("packet source and destination must differ")
        packet_id = self.next_packet_id
        self.next_packet_id += 1
        self.packets[packet_id] = RoutedPacket(
            packet_id=packet_id,
            src=src,
            dst=dst,
            owner=src,
            created_slot=self.slot,
            visited=(src,),
        )
        self.generated.add(packet_id)
        self.queues[src].append(packet_id)
        return packet_id

    def _sample_initial_pairs(self, count: int) -> List[Tuple[int, int]]:
        sources = list(range(1, self.n_agents + 1))
        self.rng.shuffle(sources)
        pairs = []
        for src in sources[: min(count, self.n_agents)]:
            dst = self.rng.randint(1, self.n_agents)
            while dst == src:
                dst = self.rng.randint(1, self.n_agents)
            pairs.append((src, dst))
        return pairs

    def _active_agents(self) -> List[int]:
        return [sat for sat, queue in self.queues.items() if queue]

    def _backlog_ids(self) -> set[int]:
        ids: List[int] = []
        for queue in self.queues.values():
            ids.extend(queue)
        return set(ids)


def first_feasible_actions(observations: Sequence[Dict]) -> List[int]:
    actions = []
    for obs in observations:
        valid = [i for i, allowed in enumerate(obs["action_mask"]) if allowed]
        actions.append(valid[0] if valid else NO_OP)
    return actions


if __name__ == "__main__":
    env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
    obs, _ = env.reset(seed=11)
    for _ in range(5):
        obs, rewards, terminated, truncated, info = env.step(
            first_feasible_actions(obs)
        )
        print(
            {
                "slot": info["slot"],
                "concurrent": info["concurrent_non_noop"],
                "delivered": info["delivered"],
                "dropped": info["dropped"],
                "backlog": info["backlog"],
                "reward": round(sum(rewards), 4),
            }
        )
        if terminated or truncated:
            break
