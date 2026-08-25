"""Why does MAPPO lose hotspot_high_load by ~1pp? Mechanistic decomposition.

hotspot is the one scenario where MAPPO trails (delivery 0.290 vs Dijkstra
0.299, Q-routing 0.309). This script decomposes the difference:
  - drop-reason composition per policy (queue-full / TTL / deadline / backlog)
  - queue occupancy + path length (hop_count) + delay of DELIVERED packets
  - offered load vs capacity sanity (is the system simply saturated?)
Writes experiments/hotspot-mechanism/hotspot_mechanism.csv + console table.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    QRoutingPolicy,
    heuristic_policy,
    load_checkpoint_policy,
)

VARIANT = "no_lifetime"
CKPT = ("experiments/train-main/checkpoints/hotspot_high_load/no_lifetime/"
        "seed_1024/leo_multi__hotspot_high_load__seed-1024__2026-08-12_05-33-03__EXP-004/"
        "validation_best.pt")


def run_policy(name, policy, seeds):
    reasons = Counter()
    hops, delays, queues, gen, dele, backlog = [], [], [], 0, 0, 0
    for seed in seeds:
        wrapper = CleanMARLLeoMultiAgentWrapper(scenario="hotspot_high_load",
                                                seed=seed, variant=VARIANT)
        binder = getattr(policy, "bind", None)
        if binder:
            binder(wrapper)
        obs, _ = wrapper.reset(seed=seed)
        term = trunc = False
        q_samples = []
        while not term and not trunc:
            q_samples.extend(len(q) for q in wrapper.env.queues.values())
            act = policy(obs, wrapper.get_avail_actions())
            obs, _r, term, trunc, info = wrapper.step(act)
        env = wrapper.env
        gen += len(env.generated)
        dele += len(env.delivered)
        backlog += len(env._backlog_ids())
        for pid in env.dropped:
            reasons[env.drop_reasons.get(pid, "unknown")] += 1
        for pid in env.delivered:
            hops.append(env.packets[pid].hop_count)
            delays.append(env.delivery_slots[pid] - env.packets[pid].created_slot + 1)
        queues.append(float(np.mean(q_samples)))
        wrapper.close()
    n = len(seeds)
    return {
        "policy": name, "generated_per_ep": gen / n, "delivered_per_ep": dele / n,
        "backlog_per_ep": backlog / n,
        "drop_deadline": reasons.get("deadline_exceeded", 0) / n,
        "drop_ttl": reasons.get("ttl_exceeded", 0) / n,
        "drop_queue": (reasons.get("queue_overflow", 0)
                       + reasons.get("source_queue_overflow", 0)) / n,
        "drop_other": sum(v for k, v in reasons.items()
                          if k not in ("deadline_exceeded", "ttl_exceeded",
                                       "queue_overflow", "source_queue_overflow")) / n,
        "delivered_mean_hops": float(np.mean(hops)) if hops else 0.0,
        "delivered_mean_delay": float(np.mean(delays)) if delays else 0.0,
        "mean_queue": float(np.mean(queues)),
    }


def main():
    import glob
    ckpts = glob.glob(("experiments/train-main/checkpoints/hotspot_high_load/"
                       "no_lifetime/seed_1024/*/validation_best.pt"))
    assert ckpts, "hotspot seed_1024 checkpoint not found"
    ck = ckpts[0]
    seeds = [11001, 11002, 11003]
    rows = []
    mappo, _ = load_checkpoint_policy(ck, device="cpu")
    rows.append(run_policy("mappo", mappo, seeds))
    rows.append(run_policy("global_dijkstra", GlobalDijkstraPolicy(), seeds))
    q = QRoutingPolicy(seed=7)
    q.train_and_freeze = None
    # load the retrained no_lifetime table
    z = np.load("experiments/eval-main/q_routing_models/hotspot_high_load_seed_7.npz")
    q.q = z["q"]; q.freeze()
    rows.append(run_policy("q_routing", q, seeds))
    rows.append(run_policy("full_heuristic",
                           heuristic_policy("full_heuristic", seed=1234), seeds))

    out = Path("experiments/hotspot-mechanism")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hotspot_mechanism.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    cols = ["policy", "delivered_per_ep", "backlog_per_ep", "drop_deadline",
            "drop_ttl", "drop_queue", "delivered_mean_hops",
            "delivered_mean_delay", "mean_queue"]
    print(" | ".join(f"{c:>18s}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:>18.1f}" if isinstance(r[c], float) else f"{r[c]:>18s}"
                         for c in cols))
    print(f"=> wrote {out/'hotspot_mechanism.csv'}")


if __name__ == "__main__":
    main()
