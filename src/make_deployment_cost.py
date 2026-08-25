"""Deployment-cost table: on-board feasibility of MAPPO vs baselines.

Reviews of LEO routing work ask: can this run on a satellite? Produces a CSV +
markdown table with, per policy:
  - parameters / model memory (fp32)
  - per-slot decision latency (measured: one full-constellation forward, CPU;
    GPU when available)
  - analytic MACs per per-satellite decision (<= 7 candidates)
  - Q-routing table sizes at every constellation scale (n^3 entries) for the
    memory/transferability contrast
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from mappo_design import SharedCandidateActor
from dqn_baseline import DqnRoutingPolicy, QHead


def actor_from_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    fdim = int(ck["candidate_feature_dim"])
    a = ck["args"]
    actor = SharedCandidateActor(candidate_feature_dim=fdim,
                                 hidden_dim=int(a["actor_hidden_dim"]),
                                 num_layers=int(a["actor_num_layers"])).to(device)
    prefix = "shared_candidate_actor."
    actor.load_state_dict({k[len(prefix):]: v for k, v in ck["actor"].items()
                           if k.startswith(prefix)})
    actor.eval()
    return actor, fdim, int(a["actor_hidden_dim"]), int(a["actor_num_layers"])


def measure(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="experiments/train-main/checkpoints/medium_load/no_lifetime/seed_1024/leo_multi__medium_load__seed-1024__2026-08-12_01-33-57__EXP-004/validation_best.pt")
    ap.add_argument("--output", default="experiments/deployment-cost")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = []
    n_agents, n_cand = 24, 7

    # MAPPO actor
    actor, fdim, hidden, layers = actor_from_checkpoint(args.checkpoint, args.device)
    params = sum(p.numel() for p in actor.parameters())
    x = torch.randn(n_agents, n_cand, fdim, device=args.device)
    m = torch.ones(n_agents, n_cand, dtype=torch.bool, device=args.device)
    with torch.no_grad():
        cpu_ms = measure(lambda: actor(x, m))
    # analytic MACs for ONE satellite's decision (7 candidates, scorer MLP)
    macs = (fdim * hidden) + (layers - 1) * hidden * hidden + hidden * n_cand
    rows.append({
        "policy": "mappo_actor", "params": params,
        "model_mem_kb": params * 4 / 1024,
        "macs_per_sat_decision": macs,
        "slot_decision_ms_cpu": round(cpu_ms, 3),
        "note": f"shared scorer h={hidden} x{layers}, feat={fdim}",
    })

    # DQN baseline head (same measurement)
    dqn = DqnRoutingPolicy(seed=0)
    dparams = sum(p.numel() for p in dqn.net.parameters())
    feats = torch.randn(n_agents * n_cand, 26)
    dsts = torch.randint(1, 25, (n_agents * n_cand,))
    with torch.no_grad():
        d_ms = measure(lambda: dqn.net(feats, dsts))
    rows.append({
        "policy": "dqn_routing_head", "params": dparams,
        "model_mem_kb": dparams * 4 / 1024,
        "macs_per_sat_decision": (26 + 8) * 128 + 128 * 128 + 128,
        "slot_decision_ms_cpu": round(d_ms, 3),
        "note": "MLP 34-128-128-1 + dst embedding",
    })

    # Q-tables at each scale
    for n in (24, 66, 110, 132):
        entries = (n + 1) ** 3
        rows.append({
            "policy": f"q_routing_table_n{n}", "params": entries,
            "model_mem_kb": entries * 4 / 1024,
            "macs_per_sat_decision": 0,
            "slot_decision_ms_cpu": "",
            "note": "table lookup O(1); must be retrained per constellation",
        })

    # SPF (Dijkstra) reference: reverse-Dijkstra per (slot,dst) — measure on env
    import heapq
    edges = [(u, v, 8.0 + 4.0 * ((u - 1) // 6 != (v - 1) // 6))
             for u in range(1, 25) for v in
             (u - 1, u + 1, u + 5, u + 6, u + 7, u - 5, u - 6, u - 7)
             if 1 <= v <= 24]
    def one_dijkstra():
        dist = {1: 0.0}
        q = [(0.0, 1)]
        while q:
            d, node = heapq.heappop(q)
            if d != dist.get(node):
                continue
            for (u, v, w) in edges:
                if u == node and d + w < dist.get(v, 1e9):
                    dist[v] = d + w
                    heapq.heappush(q, (d + w, v))
        return dist
    one_dijkstra()  # warm
    t0 = time.perf_counter()
    for _ in range(200):
        one_dijkstra()
    dij_ms = (time.perf_counter() - t0) / 200 * 1000
    rows.append({
        "policy": "dijkstra_reverse_spf", "params": 0, "model_mem_kb": 0.0,
        "macs_per_sat_decision": "",
        "slot_decision_ms_cpu": round(dij_ms, 3),
        "note": "per (slot, dst) on n24; needs CURRENT global link state (oracle)",
    })

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with open(out / "deployment_cost.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{'policy':26s} {'params':>12s} {'mem KB':>10s} {'slot ms':>8s}")
    for r in rows:
        print(f"{r['policy']:26s} {r['params']:>12,d} {r['model_mem_kb']:>10.1f} "
              f"{str(r['slot_decision_ms_cpu']):>8s}")
    print(f"=> wrote {out/'deployment_cost.csv'}")


if __name__ == "__main__":
    main()
