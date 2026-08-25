"""Train + evaluate the DQN-routing baseline (the modern learned baseline).

Per scenario: train DqnRoutingPolicy for --episodes episodes (budget-matched to
Q-routing's headline 500), evaluate on the SAME held-out workload seeds as the
EVALFIX headline (11001+, 50 eps) under variant=no_lifetime (mask parity), and
write per-scenario episode metrics. Launch one process per scenario for
parallelism; --merge at the end combines everything and emits aggregate +
paired-vs-MAPPO statistics (joining eval-main episode_metrics.csv).

Usage:
  python src/run_dqn_baseline.py --scenario medium_load --seeds 7,42,1024 &
  ... (one per scenario) ...
  python src/run_dqn_baseline.py --merge
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from dqn_baseline import DqnRoutingPolicy
from mappo_evaluation import EpisodeMetrics, evaluate_policy

VARIANT = "no_lifetime"
SCENARIOS = ["low_load", "medium_load", "hotspot_high_load", "frequent_break", "fault_links"]


def train_dqn(scenario: str, seed: int, episodes: int) -> DqnRoutingPolicy:
    policy = DqnRoutingPolicy(seed=seed)
    for episode in range(episodes):
        workload_seed = 9001 + episode % 20
        wrapper = CleanMARLLeoMultiAgentWrapper(
            scenario=scenario, seed=workload_seed, variant=VARIANT)
        policy.on_episode_start()
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


def run_scenario(args, scenario: str) -> None:
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        ck = ckpt_dir / f"dqn__{scenario}__seed{seed}.pt"
        if ck.exists() and not args.force:
            policy = DqnRoutingPolicy.load(ck, device=args.device)
            print(f"[{scenario}] seed={seed}: loaded {ck.name}", flush=True)
        else:
            t0 = time.time()
            policy = train_dqn(scenario, seed, args.episodes)
            policy.save(ck)
            print(f"[{scenario}] seed={seed}: trained {args.episodes} eps in "
                  f"{(time.time()-t0)/60:.1f} min -> {ck.name}", flush=True)
        rows.extend(evaluate_policy(
            scenario, "dqn_routing", policy, seed,
            range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds),
            variant=VARIANT))
    path = out / f"episode_metrics_{scenario}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in rows])
    d = np.mean([r.delivery_ratio for r in rows])
    print(f"[{scenario}] wrote {path.name}: dqn delivery={d:.3f} "
          f"({len(rows)} eps, {len(args.seeds.split(','))} seeds)", flush=True)


def merge(args) -> None:
    """Combine per-scenario CSVs; emit aggregate + paired tests vs MAPPO/others."""
    from scipy import stats as st

    out = args.output
    all_rows = []
    for scenario in SCENARIOS:
        p = out / f"episode_metrics_{scenario}.csv"
        if not p.exists():
            print(f"[merge] MISSING {p.name}", flush=True)
            continue
        with open(p, encoding="utf-8-sig") as f:
            all_rows.extend(list(csv.DictReader(f)))
    with open(out / "episode_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    # aggregate (bootstrap CI) per scenario/policy/metric
    evalfix = list(csv.DictReader(
        open("experiments/eval-main/episode_metrics.csv", encoding="utf-8-sig")))
    METRICS = ["delivery_ratio", "drop_rate", "throughput_packets_per_slot",
               "average_delay_slots", "p95_delay_slots", "mean_queue_packets"]
    agg = []
    for scenario in SCENARIOS:
        for policy in ["dqn_routing", "mappo", "global_dijkstra", "q_routing"]:
            src = all_rows if policy == "dqn_routing" else evalfix
            sel = [r for r in src if r["scenario"] == scenario and r["policy"] == policy]
            if not sel:
                continue
            for m in METRICS:
                vals = np.array([float(r[m]) for r in sel])
                rng = np.random.default_rng(1234)
                bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(5000)]
                agg.append({"scenario": scenario, "policy": policy, "metric": m,
                            "n": len(vals), "mean": float(vals.mean()),
                            "ci95_low": float(np.percentile(bs, 2.5)),
                            "ci95_high": float(np.percentile(bs, 97.5))})
    with open(out / "aggregate_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)

    # paired (by workload seed) dqn vs each reference
    paired = []
    for scenario in SCENARIOS:
        for ref in ["mappo", "global_dijkstra", "q_routing"]:
            dqn = {}
            refm = {}
            for r in all_rows:
                if r["scenario"] == scenario and r["policy"] == "dqn_routing":
                    dqn.setdefault(r["workload_seed"], []).append(float(r["delivery_ratio"]))
            for r in evalfix:
                if r["scenario"] == scenario and r["policy"] == ref:
                    refm.setdefault(r["workload_seed"], []).append(float(r["delivery_ratio"]))
            common = sorted(set(dqn) & set(refm))
            if not common:
                continue
            a = np.array([np.mean(dqn[k]) for k in common])
            b = np.array([np.mean(refm[k]) for k in common])
            try:
                p = float(st.wilcoxon(a - b).pvalue)
            except ValueError:
                p = 1.0
            paired.append({"scenario": scenario, "comparison": f"dqn_vs_{ref}",
                           "n_workloads": len(common),
                           "dqn_mean": float(a.mean()), "ref_mean": float(b.mean()),
                           "mean_difference": float((a - b).mean()),
                           "wilcoxon_p": p})
    with open(out / "paired_tests.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader(); w.writerows(paired)
    for row in paired:
        print(f"  {row['scenario']:18s} {row['comparison']:24s} "
              f"diff={row['mean_difference']*100:+.1f}pp p={row['wilcoxon_p']:.2e}", flush=True)
    print(f"[merge] wrote episode_metrics/aggregate/paired to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="single scenario to run (omit + --merge)")
    ap.add_argument("--seeds", default="7,42,1024")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--eval-seed-start", type=int, default=11001)
    ap.add_argument("--eval-seeds", type=int, default=50)
    ap.add_argument("--output", type=Path, default=Path("experiments/dqn-baseline"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.merge:
        merge(args)
    elif args.scenario:
        run_scenario(args, args.scenario)
    else:
        ap.error("give --scenario <name> or --merge")


if __name__ == "__main__":
    main()
