"""Closed-loop ns-3 policy server: the policy decides INSIDE the ns-3 event loop.

Replay mode (ns3_leo_validation.cc) precomputes full source-routed paths in the
slot-based env and replays them — it validates PATHS, not the policy acting on
ns-3's own state. This server implements the closed loop:

    slot boundary -> ns-3 reports RAW per-node state (queue lengths, HOL packet
    fields, per-link transmissions) over TCP -> this server reconstructs the
    env's 26-dim candidate features by calling the REAL env methods
    (leo_multiagent_env._candidate_features / _mask_reason) on an env instance
    whose dynamic state (queues / used_rate / previous_contention / route_cache
    / slot) is overwritten with the ns-3-reported values -> the checkpoint
    actor (or Dijkstra baseline) picks next hops -> ns-3 executes the forwards
    on its real FIFO data plane -> the resulting queues feed the next slot.

Feature FORMULA parity is exact (single source of truth: the env code); feature
VALUES for queue/bandwidth/contention terms come from ns-3's actual data plane.
No fingerprint-hashed file is modified — env state is overridden per instance.

Wire protocol (line-based, both ends under our control):
  client -> HELLO <n_nodes> <action_size> <policy_name>
  server -> READY <n_nodes> <action_size>
  per slot:
    client -> SLOT <episode> <slot> <n_agents> <n_tx>
              AG <id> <queue_len> <incoming> <pkt_id> <dst> <cls> <hops>
                 <prev_node> <created> <visited_mask> <c0>..<c6>
              (pkt_id < 0 = no HOL; c0=0 hold placeholder, c1..c6 neighbor ids)
              TX <u> <v> <count>          (directed transmissions last slot)
    server -> ACT <n>
              <agent_id> <action_slot> <next_hop_id>
"""
from __future__ import annotations

import argparse
import socket
import threading
import types
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MULTIAGENT_LOADS, MultiAgentConfig
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from ns3_policy_bridge import Ns3PolicyBridge

VARIANT = "no_lifetime"


def make_wrapper(scenario: str, seed: int) -> CleanMARLLeoMultiAgentWrapper:
    ini, exo = MULTIAGENT_LOADS[scenario]
    envc = EnvConfig(scenario=SCENARIOS[scenario], seed=seed)
    cfg = MultiAgentConfig(env=envc, initial_packets=ini,
                           exogenous_packets_per_slot=exo, seed=seed,
                           variant=VARIANT)
    return CleanMARLLeoMultiAgentWrapper(cfg=cfg)


class EpisodeFeatureContext:
    """Env instance per workload seed: geometry/edge statics from the model,
    dynamic state overwritten each slot from the ns-3 report."""

    def __init__(self, scenario: str, seed: int):
        wrapper = make_wrapper(scenario, seed)
        wrapper.reset(seed=seed)
        self.env = wrapper.env
        self.n_agents = wrapper.n_agents
        self.max_degree = self.env.max_degree
        # used_rate replication: U_t = (U_{t-1} + A_{t-1}) * load_decay
        self.used_rate: Dict[int, Dict[int, float]] = {
            u: {v: 0.0 for v in range(1, self.n_agents + 1)}
            for u in range(1, self.n_agents + 1)
        }
        self.last_tx: Dict[tuple, int] = {}
        self.env.route_cache = {}

    def advance_to_slot(self, slot: int, link_tx: Dict[tuple, int]) -> None:
        """Apply last slot's accepted transmissions + decay, refresh edge
        used-rate in the env graph (mirrors env step: adds happen during the
        slot, _decay_load at slot end)."""
        cfg = self.env.cfg
        for (u, v), n in link_tx.items():
            if (u, v) in self.used_rate:
                self.used_rate[u][v] += n * cfg.env.packet_demand_mbps
        for u in self.used_rate:
            for v in self.used_rate[u]:
                self.used_rate[u][v] *= cfg.env.load_decay
        for (u, v), edge in self.env.graph.items():
            edge.used_rate_mbps = self.used_rate[u][v]
        self.env.slot = slot

    def set_state(self, queue_lens: Dict[int, int], incoming: Dict[int, int]) -> None:
        # only len() is read by the feature/mask code
        self.env.queues = {n: [0] * q for n, q in queue_lens.items()}
        self.env.previous_contention = [0] * (self.n_agents + 1)
        for node, count in incoming.items():
            self.env.previous_contention[node] = max(0, count - 1)


def make_packet_shim(dst, cls, hops, prev_node, created, visited_mask, n_agents):
    visited = {i + 1 for i in range(n_agents) if visited_mask >> i & 1}
    return types.SimpleNamespace(
        dst=dst, traffic_class=cls, hop_count=hops,
        previous_node=(prev_node if prev_node > 0 else None),
        created_slot=created, visited=visited, src=0, packet_id=-1,
    )


class DijkstraProvider:
    """Delay-weighted SPF on the env graph (static topology: table per dst)."""

    def __init__(self, contexts):
        self.contexts = contexts

    def decide(self, agents, ctx):
        import heapq
        dist_cache: Dict[int, Dict[int, float]] = {}

        def dist_to(dst: int) -> Dict[int, float]:
            if dst not in dist_cache:
                graph = ctx.env.graph
                d = {dst: 0.0}
                pq = [(0.0, dst)]
                while pq:
                    cost, node = heapq.heappop(pq)
                    if cost != d.get(node):
                        continue
                    for (src, nbr), edge in graph.items():
                        if nbr != node or not edge.available:
                            continue
                        cand = cost + edge.delay_ms
                        if cand < d.get(src, float("inf")):
                            d[src] = cand
                            heapq.heappush(pq, (cand, src))
                dist_cache[dst] = d
            return dist_cache[dst]

        decisions = []
        for ag in agents:
            if ag["pkt_id"] < 0:
                decisions.append((ag["id"], 0, 0))
                continue
            stale = dist_to(ag["dst"])
            best, best_score = 0, float("inf")
            for action, neighbor in enumerate(ag["candidates"]):
                if neighbor <= 0 or not ag["candidate_mask"][action]:
                    continue
                edge = ctx.env.graph.get((ag["sat"], neighbor))
                if edge is None:
                    continue
                score = edge.delay_ms + stale.get(neighbor, float("inf"))
                if score < best_score:
                    best, best_score = action, score
            if best == 0:
                decisions.append((ag["id"], 0, 0))       # forced hold
            else:
                decisions.append((ag["id"], best, ag["candidates"][best]))
        return decisions


class ClosedLoopServer:
    def __init__(self, checkpoint: Path, scenario: str, seeds, device="cpu"):
        self.bridge = Ns3PolicyBridge(checkpoint, device=device)
        self.scenario = scenario
        self.seeds = list(seeds)
        self.contexts: Dict[int, EpisodeFeatureContext] = {}
        for seed in self.seeds:
            self.contexts[seed] = EpisodeFeatureContext(scenario, seed)
        self.dijkstra = DijkstraProvider(self.contexts)
        self.feature_dim = self.bridge.feature_dim
        self.action_size = self.bridge.action_size

    # ---- one slot: raw report -> features via env methods -> policy -> actions
    def handle_slot(self, parts, lines, tx_lines) -> list:
        episode, slot = int(parts[1]), int(parts[2])
        ctx = self.contexts[self.seeds[episode]]
        agents_raw = []
        for line in lines:
            f = line.split()
            agents_raw.append({
                "id": int(f[1]), "queue_len": int(f[2]), "incoming": int(f[3]),
                "pkt_id": int(f[4]), "dst": int(f[5]), "cls": int(f[6]),
                "hops": int(f[7]), "prev_node": int(f[8]),
                "created": int(f[9]), "visited_mask": int(f[10]),
                "candidates": [int(x) for x in f[11:18]],
            })
        link_tx = {}
        for line in tx_lines:
            f = line.split()
            link_tx[(int(f[1]), int(f[2]))] = int(f[3])

        ctx.advance_to_slot(slot, link_tx)
        ctx.set_state({a["id"]: a["queue_len"] for a in agents_raw},
                      {a["id"]: a["incoming"] for a in agents_raw})

        v1_agents, decisions_meta = [], []
        for a in agents_raw:
            sat = a["id"]
            # action 0 = hold; actions 1..max_degree map to candidates[1..6]
            # (row 0 of candidate_features is the hold placeholder, zeros --
            # matching the wrapper's flattened action_size*feature_dim obs)
            feats = np.zeros((self.action_size, self.feature_dim), dtype=np.float32)
            cmask = [False] + [False] * (self.action_size - 1)
            if a["pkt_id"] >= 0:
                packet = make_packet_shim(a["dst"], a["cls"], a["hops"],
                                          a["prev_node"], a["created"],
                                          a["visited_mask"], ctx.n_agents)
                for action, neighbor in enumerate(a["candidates"]):
                    if neighbor <= 0:
                        continue
                    feats[action] = ctx.env._candidate_features(packet, sat, neighbor)
                    cmask[action] = (
                        ctx.env._mask_reason(packet, sat, neighbor) == "feasible")
            no_route = not any(cmask)
            action_mask = [no_route] + cmask[1:]
            v1_agents.append({
                "agent_id": sat,
                "candidate_features": feats.tolist(),
                "action_mask": action_mask,
                "candidate_next_hops": a["candidates"],
                "packet_id": a["pkt_id"],
            })
            decisions_meta.append({"sat": sat, "dst": a["dst"],
                                   "candidates": a["candidates"],
                                   "candidate_mask": cmask,
                                   "pkt_id": a["pkt_id"], "id": a["id"],
                                   "cls": a["cls"]})

        policy_name = self.current_policy
        if policy_name == "mappo":
            message = {"schema_version": 1, "type": "slot_state",
                       "episode_id": episode, "time_slot": slot,
                       "agents": v1_agents}
            reply = self.bridge.decide(message)
            out = [(d["agent_id"], d["action_slot"], d["next_hop_id"])
                   for d in reply["decisions"]]
        else:
            out = self.dijkstra.decide(decisions_meta, ctx)

        # route_cache mirrors env: updated per executed (non-hold) decision
        for (agent_id, action_slot, _nh), meta in zip(out, decisions_meta):
            if action_slot > 0 and meta["pkt_id"] >= 0:
                ctx.env.route_cache[(meta["sat"], meta["dst"], meta["cls"])] = \
                    meta["candidates"][action_slot]
        return out

    # ---- TCP serve loop (one connection; ns-3 is single-run) ----------------
    def serve_connection(self, conn: socket.socket) -> None:
        f = conn.makefile("rwb", buffering=1 << 16)

        def readline():
            return f.readline().decode().strip()

        hello = readline().split()
        assert hello[0] == "HELLO", hello
        self.current_policy = hello[3]
        f.write(f"READY {hello[1]} {hello[2]}\n".encode())
        f.flush()

        while True:
            line = readline()
            if not line or line == "BYE":
                break
            parts = line.split()
            if parts[0] != "SLOT":
                continue
            n_agents, n_tx = int(parts[3]), int(parts[4])
            ag_lines = [readline() for _ in range(n_agents)]
            tx_lines = [readline() for _ in range(n_tx)]
            try:
                decisions = self.handle_slot(parts, ag_lines, tx_lines)
            except Exception as error:          # surface, never deadlock C++
                import traceback
                traceback.print_exc()
                f.write(f"ERR {error}\n".encode())
                f.flush()
                break
            f.write(f"ACT {len(decisions)}\n".encode())
            for agent_id, action_slot, next_hop in decisions:
                f.write(f"{agent_id} {action_slot} {next_hop}\n".encode())
            f.flush()
        f.close()

    def listen(self, host="0.0.0.0", port=7341) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        self._srv = srv
        return srv.getsockname()[1]

    def serve_once(self):
        conn, _ = self._srv.accept()
        with conn:
            self.serve_connection(conn)
        self._srv.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--workload-seeds", default="21001,21002,21003,21004,21005")
    ap.add_argument("--port", type=int, default=7341)
    args = ap.parse_args()
    seeds = [int(s) for s in args.workload_seeds.split(",")]
    server = ClosedLoopServer(args.checkpoint, args.scenario, seeds)
    port = server.listen(port=args.port)
    print(f"closed-loop server listening on :{port} (policies mappo/dijkstra)", flush=True)
    server.serve_once()
    print("closed-loop server: connection finished", flush=True)


if __name__ == "__main__":
    main()
