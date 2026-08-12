"""Traffic-load sweep: delivery vs offered load (the congestion-aware argument).

Fixes the topology (synthetic 24-sat medium_load traffic pattern) and sweeps the
offered load from light to heavily congested. At each load point it evaluates
the existing 8-seed MAPPO checkpoints + Dijkstra oracle + distributed heuristic.

Motivation: at light load shortest-path Dijkstra is optimal and a learned policy
merely matches it; the *distributed learned* approach earns its keep under
CONGESTION, where Dijkstra's paths converge onto the same links (hot-spots,
queue buildup, drops) and a load-aware policy can route around the contention.
This sweep locates the load regime where MAPPO pulls ahead of (or stays even
with) the centralized oracle, and where the heuristic collapses.

Load is parameterized by exogenous packets/slot; initial packets track at the
medium_load 2:1 ratio (initial = 2*exogenous). Reported metrics: delivery,
drop rate, throughput/slot, mean queue, P95 delay.

Outputs (experiments/IEEE-LOAD/):
  load_matrix.csv     -- per (load_point, policy, seed) episode metrics
  aggregate_load.csv  -- per (load_point, policy) bootstrap means + 95% CI

Usage:
  # quick 2-point probe (hypothesis check)
  python run_load_sweep.py --exo 6,20 --workload-seeds 3
  # full sweep
  python run_load_sweep.py --exo 2,4,6,8,10,12,16,20 --workload-seeds 15
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig, MULTIAGENT_LOADS
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    evaluate_policy,
    heuristic_policy,
    load_checkpoint_policy,
)
from run_scale_experiment import bootstrap_ci, find_source_checkpoints, write_csv

METRICS = [
    "delivery_ratio", "drop_rate", "throughput_packets_per_slot",
    "average_delay_slots", "p95_delay_slots", "mean_queue_packets",
]


def load_point(exo: int):
    """(label, initial_packets, exogenous_per_slot) at 2:1 initial:exo ratio."""
    ini = 2 * exo
    return (f"exo{exo}", ini, exo)


def wrapper_factory(scenario: str, ini: int, exo: int, variant: str):
    def make(seed: int) -> CleanMARLLeoMultiAgentWrapper:
        envc = EnvConfig(scenario=SCENARIOS[scenario], seed=seed)  # default 4x6, 30-slot
        cfg = MultiAgentConfig(
            env=envc, initial_packets=ini, exogenous_packets_per_slot=exo,
            seed=seed, variant=variant,
        )
        return CleanMARLLeoMultiAgentWrapper(cfg=cfg)
    return make


def aggregate_load(rows, labels) -> list[dict]:
    out = []
    for label in labels:
        for policy in sorted({r.policy for r in rows if r.scenario == label}):
            sel = [r for r in rows if r.scenario == label and r.policy == policy]
            if not sel:
                continue
            for m in METRICS:
                vals = [float(getattr(r, m)) for r in sel]
                mean, lo, hi = bootstrap_ci(vals)
                out.append({
                    "load_point": label, "policy": policy, "metric": m,
                    "n": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi,
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("experiments/IEEE-REPRO-CHECK"))
    ap.add_argument("--output", type=Path, default=Path("experiments/IEEE-LOAD"))
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--exo", default="2,4,6,8,10,12,16,20",
                    help="comma list of exogenous packets/slot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workload-seed-start", type=int, default=18001)
    ap.add_argument("--workload-seeds", type=int, default=15)
    args = ap.parse_args()

    exos = [int(x) for x in args.exo.split(",")]
    points = [load_point(e) for e in exos]
    scenario = args.scenario
    wseeds = list(range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds))
    sources = find_source_checkpoints(args.source_dir, scenario)
    if not sources:
        raise FileNotFoundError(f"no checkpoints under {args.source_dir}/checkpoints/{scenario}/seed_*")

    print(f"load sweep: {len(sources)} seeds x {len(points)} load points, "
          f"{len(wseeds)} workload seeds, device={args.device}")

    actors = []
    for seed, ckpt in sources:
        policy, info = load_checkpoint_policy(ckpt, device=args.device)
        variant = info.get("args", {}).get("leo_variant", "full") if isinstance(info, dict) else "full"
        actors.append((seed, policy, variant))

    all_rows = []
    labels = []
    for label, ini, exo in points:
        labels.append(label)
        print(f"\n=== load {label} (initial={ini}, exo/slot={exo}) ===")
        for seed, policy, variant in actors:
            rows = evaluate_policy(
                label, "mappo", policy, seed, wseeds,
                wrapper_factory=wrapper_factory(scenario, ini, exo, variant), variant=variant,
            )
            all_rows.extend(rows)
        fct = wrapper_factory(scenario, ini, exo, "full")
        all_rows.extend(evaluate_policy(label, "global_dijkstra", GlobalDijkstraPolicy(), -1, wseeds, wrapper_factory=fct))
        all_rows.extend(evaluate_policy(label, "full_heuristic", heuristic_policy("full_heuristic", seed=1234), -1, wseeds, wrapper_factory=fct))

        def mean(pol, m="delivery_ratio"):
            xs = [float(getattr(r, m)) for r in all_rows if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        md, dd = mean("mappo"), mean("global_dijkstra")
        print(f"  delivery: MAPPO={md:.3f}  Dijkstra={dd:.3f}  heuristic={mean('full_heuristic'):.3f}  "
              f"gap(M-D)={md-dd:+.4f}")
        mq = np.mean([float(r.mean_queue_packets) for r in all_rows if r.scenario == label and r.policy == "global_dijkstra"])
        print(f"  Dijkstra mean_queue={mq:.2f}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "load_matrix.csv", [asdict(r) for r in all_rows])
    write_csv(out / "aggregate_load.csv", aggregate_load(all_rows, labels))
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "traffic_load_sweep", "scenario": scenario,
        "load_points": [{"label": l, "initial": i, "exogenous": e} for l, i, e in points],
        "policy_seeds": [s for s, _ in sources], "workload_seeds": wseeds,
    }, indent=2), encoding="utf-8")
    print(f"\n=> wrote {out/'load_matrix.csv'} + {out/'aggregate_load.csv'}")


if __name__ == "__main__":
    main()
