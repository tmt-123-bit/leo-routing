"""Link-failure robustness sweep: performance vs fault rate.

Fixes medium_load traffic and sweeps the fraction of faulted links
(fault_link_ratio) from none to severe. At each fault rate it evaluates MAPPO
+ Dijkstra oracle + distributed heuristic.

Motivation: LEO topology is defined by link volatility. The env's break model is
TIME-CORRELATED (p_break = 1 - t_rem/t_safe), so the reliability observation is
*predictive* of failures -> a reliability-aware learned policy may degrade more
gracefully than reactive shortest-path Dijkstra (which only re-routes AFTER a
link drops, with global post-fault knowledge). This sweep locates the regime.

Optionally takes --source-scenario to pick which checkpoints drive MAPPO:
  fault_links  : trained WITH faults (in-distribution around 8%) -> degradation curve
  medium_load  : trained WITHOUT faults -> zero-shot fault generalization

reliability_penalty tracks fault_link_ratio at the 1.25x the fault_links preset
uses (0.08 -> 0.10), so the 0.08 point reproduces the fault_links scenario.

Outputs (experiments/IEEE-FAULT[-<tag>]/):
  fault_matrix.csv / aggregate_fault.csv

Usage:
  python run_fault_sweep.py --source-scenario fault_links --fault-rates 0,0.04,0.08,0.12,0.16,0.20
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from leo_marl_env import EnvConfig, ScenarioConfig
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


def scenario_with_fault(fault_ratio: float) -> ScenarioConfig:
    return ScenarioConfig(
        name=f"fault{fault_ratio:.2f}",
        background_packets_per_slot=6, hotspot_packets_per_slot=12, hotspot_ratio=0.65,
        fault_link_ratio=fault_ratio,
        reliability_penalty=round(fault_ratio * 1.25, 4),
    )


def wrapper_factory(fault_ratio: float, ini: int, exo: int, variant: str):
    def make(seed: int) -> CleanMARLLeoMultiAgentWrapper:
        envc = EnvConfig(scenario=scenario_with_fault(fault_ratio), seed=seed)
        cfg = MultiAgentConfig(env=envc, initial_packets=ini, exogenous_packets_per_slot=exo,
                               seed=seed, variant=variant)
        return CleanMARLLeoMultiAgentWrapper(cfg=cfg)
    return make


def aggregate_fault(rows, labels) -> list[dict]:
    out = []
    for label in labels:
        for policy in sorted({r.policy for r in rows if r.scenario == label}):
            sel = [r for r in rows if r.scenario == label and r.policy == policy]
            if not sel:
                continue
            for m in METRICS:
                vals = [float(getattr(r, m)) for r in sel]
                mean, lo, hi = bootstrap_ci(vals)
                out.append({"fault_point": label, "policy": policy, "metric": m,
                            "n": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("experiments/IEEE-REPRO-CHECK"))
    ap.add_argument("--source-scenario", default="fault_links",
                    help="which trained checkpoints to load (fault_links / medium_load / ...)")
    ap.add_argument("--output", type=Path, default=Path("experiments/IEEE-FAULT"))
    ap.add_argument("--traffic-scenario", default="medium_load",
                    help="traffic pattern for offered load (default medium_load)")
    ap.add_argument("--fault-rates", default="0,0.04,0.08,0.12,0.16,0.20")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workload-seed-start", type=int, default=19001)
    ap.add_argument("--workload-seeds", type=int, default=15)
    args = ap.parse_args()

    rates = [float(x) for x in args.fault_rates.split(",")]
    ini, exo = MULTIAGENT_LOADS[args.traffic_scenario]
    wseeds = list(range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds))
    sources = find_source_checkpoints(args.source_dir, args.source_scenario)
    if not sources:
        raise FileNotFoundError(f"no checkpoints under {args.source_dir}/checkpoints/{args.source_scenario}/seed_*")

    print(f"fault sweep: {len(sources)} seeds (source={args.source_scenario}) x {len(rates)} fault rates, "
          f"traffic={args.traffic_scenario} (ini={ini},exo={exo}), device={args.device}")

    actors = []
    for seed, ckpt in sources:
        policy, info = load_checkpoint_policy(ckpt, device=args.device)
        variant = info.get("args", {}).get("leo_variant", "full") if isinstance(info, dict) else "full"
        actors.append((seed, policy, variant))

    all_rows, labels = [], []
    for fr in rates:
        label = f"fr{fr:.2f}"
        labels.append(label)
        print(f"\n=== fault_rate={fr:.2f} (reliability_penalty={fr*1.25:.3f}) ===")
        for seed, policy, variant in actors:
            rows = evaluate_policy(label, "mappo", policy, seed, wseeds,
                                   wrapper_factory=wrapper_factory(fr, ini, exo, variant), variant=variant)
            all_rows.extend(rows)
        # mask parity: oracles/heuristics must run under the SAME env variant
        # as the transferred checkpoints — mask-following baselines are
        # constrained by the action mask, so a variant mismatch silently
        # handicaps one side (see ns3_trace_extractor factory note).
        fct = wrapper_factory(fr, ini, exo, actors[0][2])
        all_rows.extend(evaluate_policy(label, "global_dijkstra", GlobalDijkstraPolicy(), -1, wseeds, wrapper_factory=fct))
        all_rows.extend(evaluate_policy(label, "full_heuristic", heuristic_policy("full_heuristic", seed=1234), -1, wseeds, wrapper_factory=fct))

        def mean(pol, m="delivery_ratio"):
            xs = [float(getattr(r, m)) for r in all_rows if r.scenario == label and r.policy == pol]
            return float(np.mean(xs)) if xs else float("nan")
        md, dd = mean("mappo"), mean("global_dijkstra")
        print(f"  delivery: MAPPO={md:.3f}  Dijkstra={dd:.3f}  heuristic={mean('full_heuristic'):.3f}  gap(M-D)={md-dd:+.4f}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fault_matrix.csv", [asdict(r) for r in all_rows])
    write_csv(out / "aggregate_fault.csv", aggregate_fault(all_rows, labels))
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "link_fault_robustness_sweep", "source_scenario": args.source_scenario,
        "traffic_scenario": args.traffic_scenario, "fault_rates": rates,
        "policy_seeds": [s for s, _ in sources], "workload_seeds": wseeds,
    }, indent=2), encoding="utf-8")
    print(f"\n=> wrote {out/'fault_matrix.csv'} + {out/'aggregate_fault.csv'}")


if __name__ == "__main__":
    main()
