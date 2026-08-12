"""Zero-shot realism transfer: synthetic-grid-trained MAPPO -> real TLE topologies.

Loads headline checkpoints (trained on the regular synthetic 4x6 grid) and
evaluates each policy seed ZERO-SHOT on frozen real-orbital-geometry
topologies (Starlink-24, OneWeb-24), plus Dijkstra/heuristic oracles. This
tests *topology generalization*: does a policy learned on a regular synthetic
grid transfer to the irregular contact schedule of a real constellation?

This is the realism analog of the constellation-scale transfer experiment
(run_scale_experiment.py): there we vary SIZE at fixed (synthetic) geometry;
here we vary GEOMETRY at fixed size (24 sats). The shared candidate actor is
neighbor-relative + permutation-equivariant, so it carries no assumption about
global constellation structure -> a synthetic-trained actor should deploy on a
real contact schedule unchanged.

Workload is held identical to the synthetic headline (medium_load: initial=12,
exogenous=6, 30-slot episodes) so the only varying factor is the link
topology -> apples-to-apples synthetic-vs-real comparison.

Outputs (experiments/IEEE-REALISM/):
  realism_matrix.csv    -- per (topology, policy, seed) episode metrics
  aggregate_realism.csv -- per (topology, policy) bootstrap means + 95% CI

Usage:
  python run_realism_transfer.py \
      --source-dir experiments/IEEE-REPRO-CHECK \
      --output experiments/IEEE-REALISM --scenario medium_load --device cpu
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cleanmarl_tle_wrapper import CleanMARLTLEWrapper
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    evaluate_policy,
    heuristic_policy,
    load_checkpoint_policy,
)
from run_scale_experiment import bootstrap_ci, find_source_checkpoints, write_csv

# (label, links_csv). None links_csv => skipped (sanity handled by caller).
DEFAULT_TOPOLOGIES = [
    ("starlink_24", "data/starlink_24_links.csv"),
    ("oneweb_24", "data/oneweb_24_links.csv"),
]

METRICS = [
    "delivery_ratio", "drop_rate", "throughput_packets_per_slot",
    "average_delay_slots", "p95_delay_slots", "mean_queue_packets",
]


def tle_wrapper_factory(links_csv: str, scenario: str, variant: str):
    def make(seed: int) -> CleanMARLTLEWrapper:
        return CleanMARLTLEWrapper(
            topology_csv=links_csv, scenario=scenario, seed=seed, variant=variant
        )
    return make


def aggregate_realism(rows, topology_labels) -> list[dict]:
    out = []
    for label in topology_labels:
        for policy in sorted({r.policy for r in rows if r.scenario == label}):
            sel = [r for r in rows if r.scenario == label and r.policy == policy]
            if not sel:
                continue
            for m in METRICS:
                vals = [float(getattr(r, m)) for r in sel]
                mean, lo, hi = bootstrap_ci(vals)
                out.append({
                    "topology": label, "policy": policy, "metric": m,
                    "n": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi,
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("experiments/IEEE-REPRO-CHECK"))
    ap.add_argument("--output", type=Path, default=Path("experiments/IEEE-REALISM"))
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workload-seed-start", type=int, default=17001)
    ap.add_argument("--workload-seeds", type=int, default=20)
    ap.add_argument("--topologies", nargs="*", default=None,
                    help="override topology list as label:csv pairs")
    args = ap.parse_args()

    if args.topologies:
        tops = [(p.split(":", 1)[0], p.split(":", 1)[1]) for p in args.topologies]
    else:
        tops = [(lbl, str(Path(csv))) for lbl, csv in DEFAULT_TOPOLOGIES]

    scenario = args.scenario
    wseeds = list(range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds))
    sources = find_source_checkpoints(args.source_dir, scenario)
    if not sources:
        raise FileNotFoundError(f"no checkpoints under {args.source_dir}/checkpoints/{scenario}/seed_*")

    print(f"realism transfer: {len(sources)} synthetic source seeds -> "
          f"{len(tops)} real topologies, {len(wseeds)} workload seeds, device={args.device}")

    actors = []
    for seed, ckpt in sources:
        policy, info = load_checkpoint_policy(ckpt, device=args.device)
        variant = info.get("args", {}).get("leo_variant", "full") if isinstance(info, dict) else "full"
        actors.append((seed, policy, variant))
        print(f"  loaded synthetic seed={seed} <- {ckpt.name} (variant={variant})")

    all_rows = []
    labels = []
    for label, links_csv in tops:
        labels.append(label)
        if not Path(links_csv).exists():
            print(f"  [skip] {label}: links csv not found ({links_csv})")
            continue
        print(f"\n=== real topology {label} ({links_csv}) ===")
        for seed, policy, variant in actors:
            rows = evaluate_policy(
                label, "mappo_realism_transfer", policy, seed, wseeds,
                wrapper_factory=tle_wrapper_factory(links_csv, scenario, variant), variant=variant,
            )
            all_rows.extend(rows)
        fct = tle_wrapper_factory(links_csv, scenario, "full")
        all_rows.extend(evaluate_policy(label, "global_dijkstra", GlobalDijkstraPolicy(), -1, wseeds, wrapper_factory=fct))
        all_rows.extend(evaluate_policy(label, "full_heuristic", heuristic_policy("full_heuristic", seed=1234), -1, wseeds, wrapper_factory=fct))

        def mean(pol, m="delivery_ratio"):
            xs = [float(getattr(r, m)) for r in all_rows if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        def mthrough(pol):
            xs = [float(r.throughput_packets_per_slot) for r in all_rows if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        print(f"  delivery: MAPPO-transfer={mean('mappo_realism_transfer'):.3f}  "
              f"Dijkstra={mean('global_dijkstra'):.3f}  heuristic={mean('full_heuristic'):.3f}")
        dt = mthrough("global_dijkstra")
        print(f"  thru/slot ratio MAPPO/Dijkstra="
              f"{mthrough('mappo_realism_transfer')/max(1e-9,dt):.3f}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "realism_matrix.csv", [asdict(r) for r in all_rows])
    write_csv(out / "aggregate_realism.csv", aggregate_realism(all_rows, labels))
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "realism_transfer_synthetic_to_TLE", "mode": "transfer",
        "source_dir": str(args.source_dir), "source_topology": "synthetic_4x6_grid",
        "topologies": [{"label": l, "links_csv": c} for l, c in tops],
        "scenario": scenario, "policy_seeds": [s for s, _ in sources],
        "workload_seeds": wseeds,
        "workload": "identical to synthetic headline (medium_load), only topology varies",
    }, indent=2), encoding="utf-8")
    print(f"\n=> wrote {out/'realism_matrix.csv'} + {out/'aggregate_realism.csv'}")


if __name__ == "__main__":
    main()
