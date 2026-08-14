"""Q-routing constellation-scale transfer vs retraining (the "why MAPPO" experiment).

After the eval-variant fix, Q-routing (retrained no_lifetime) essentially TIES
MAPPO on delivery at the training scale (n24). The remaining differentiator is
transfer: MAPPO's actor is permutation-equivariant and neighbor-relative, so an
n24 checkpoint deploys zero-shot at any constellation size; a Q-table is indexed
by concrete node ids, so it cannot. This experiment quantifies that with data:

  q_routing_transfer    n24-trained table deployed as-is at n66/n110/n132
                        (table sized for the target, trained entries copied
                        over; untrained node/dst entries stay at init -> the
                        policy degrades to immediate-cost greedy there). This
                        is the honest "just deploy the old table" attempt.
  q_routing_retrained   fresh Q-routing trained ON the target constellation
                        with the same episode budget as the n24 headline
                        (500 episodes) -- the cost MAPPO avoids entirely.
  q_routing_n24_indist  the n24 tables evaluated at n24 on the SAME workload
                        seeds as the scale experiment (reference row).

MAPPO-transfer and Dijkstra rows are NOT rerun here: run_scale_experiment.py
(now mask-parity fixed) produces them on the same workload seeds (16001+) --
join on (target_scale, workload_seed) for paired comparisons.

Mask parity: every policy runs under variant=no_lifetime (the checkpoints'
variant; the n24 tables were also trained under it).

Outputs (experiments/IEEE-QSCALE/):
  qscale_matrix.csv   per (scale, policy, seed) episode metrics
  qscale_aggregate.csv bootstrap means + 95% CI per (scale, policy, metric)
  manifest.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mappo_evaluation import QRoutingPolicy, evaluate_policy
from run_scale_experiment import (
    DEFAULT_SCALES,
    aggregate_scale,
    wrapper_factory,
    write_csv,
)

VARIANT = "no_lifetime"
SCENARIO = "medium_load"


def train_q_routing_scaled(factory, n_nodes: int, seed: int, episodes: int) -> QRoutingPolicy:
    """train_q_routing's loop, parameterized by a (scaled) wrapper factory."""
    policy = QRoutingPolicy(n_nodes=n_nodes, seed=seed)
    for episode in range(episodes):
        workload_seed = 9001 + episode % 20
        wrapper = factory(workload_seed)
        observation, _ = wrapper.reset(seed=workload_seed)
        policy.bind(wrapper)
        terminated = truncated = False
        while not terminated and not truncated:
            actions = policy(observation, wrapper.get_avail_actions())
            observation, _, terminated, truncated, info = wrapper.step(actions)
            policy.observe_transition(info)
        wrapper.close()
    policy.freeze()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path,
                    default=Path("experiments/IEEE-NOLIFE-EVALFIX/q_routing_models"))
    ap.add_argument("--output", type=Path, default=Path("experiments/IEEE-QSCALE"))
    ap.add_argument("--scales", default="n24,n66,n110,n132")
    ap.add_argument("--table-seeds", default="7,42,1024,123,456,789,2024,314")
    ap.add_argument("--retrain-seeds", default="7,42,1024")
    ap.add_argument("--q-episodes", type=int, default=500,
                    help="retraining episode budget (matches the n24 headline)")
    ap.add_argument("--workload-seed-start", type=int, default=16001)
    ap.add_argument("--workload-seeds", type=int, default=20)
    args = ap.parse_args()

    scales = [s for s in DEFAULT_SCALES if s[2] in args.scales.split(",")]
    table_seeds = [int(s) for s in args.table_seeds.split(",")]
    retrain_seeds = [int(s) for s in args.retrain_seeds.split(",")]
    wseeds = list(range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds))

    tables = {}
    for seed in table_seeds:
        path = args.tables_dir / f"{SCENARIO}_seed_{seed}.npz"
        tables[seed] = np.load(path)["q"]
    print(f"qscale: {len(tables)} n24 tables, retrain seeds {retrain_seeds} "
          f"@ {args.q_episodes} eps, scales {[s[2] for s in scales]}, {len(wseeds)} wseeds")
    args.output.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for (npl, spp, label) in scales:
        n = npl * spp
        fct = wrapper_factory(npl, spp, SCENARIO, VARIANT)
        print(f"\n=== {label} (n={n}, {npl}x{spp}) ===")

        if label == "n24":
            # in-distribution reference: the saved tables, same wseeds
            for seed in table_seeds:
                policy = QRoutingPolicy(n_nodes=n, seed=seed)
                policy.q = tables[seed].copy()
                policy.freeze()
                all_rows.extend(evaluate_policy(label, "q_routing_n24_indist", policy,
                                                seed, wseeds, wrapper_factory=fct, variant=VARIANT))
        else:
            # (a) deploy the n24 table as-is at this scale
            for seed in table_seeds:
                policy = QRoutingPolicy(n_nodes=n, seed=seed)
                policy.q[:25, :25, :25] = tables[seed]
                policy.freeze()
                all_rows.extend(evaluate_policy(label, "q_routing_transfer", policy,
                                                seed, wseeds, wrapper_factory=fct, variant=VARIANT))

            # (b) retrain fresh on this constellation, matched episode budget
            for seed in retrain_seeds:
                t0 = time.time()
                policy = train_q_routing_scaled(fct, n, seed, args.q_episodes)
                train_min = (time.time() - t0) / 60
                np.savez_compressed(args.output / f"q_retrained_{label}_seed_{seed}.npz", q=policy.q)
                all_rows.extend(evaluate_policy(label, "q_routing_retrained", policy,
                                                seed, wseeds, wrapper_factory=fct, variant=VARIANT))
                print(f"  [retrain seed={seed}] {train_min:.1f} min for {args.q_episodes} eps")

        def mean(pol, m="delivery_ratio"):
            xs = [float(getattr(r, m)) for r in all_rows
                  if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        parts = [f"  delivery: {p}={mean(p):.3f}" for p in sorted(
            {r.policy for r in all_rows if r.scenario == label})]
        print("\n".join(parts))

    write_csv(args.output / "qscale_matrix.csv", [asdict(r) for r in all_rows])
    write_csv(args.output / "qscale_aggregate.csv", aggregate_scale(all_rows, scales))
    (args.output / "manifest.json").write_text(json.dumps({
        "experiment": "q_routing_scale_transfer_vs_retrain",
        "variant": VARIANT, "scenario": SCENARIO,
        "scales": [{"label": l, "n_planes": p, "sats_per_plane": s, "n_sats": p * s}
                   for p, s, l in scales],
        "table_seeds": table_seeds, "retrain_seeds": retrain_seeds,
        "q_episodes": args.q_episodes, "workload_seeds": wseeds,
        "tables_dir": str(args.tables_dir),
        "note": "join with IEEE-SCALE-EVALFIX transfer_matrix.csv on (target_scale, workload_seed) "
                "for MAPPO-transfer / Dijkstra comparisons (same wseeds)",
    }, indent=2), encoding="utf-8")
    print(f"\n=> wrote {args.output/'qscale_matrix.csv'} + qscale_aggregate.csv")


if __name__ == "__main__":
    main()
