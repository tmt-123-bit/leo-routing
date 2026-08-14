"""Reward-weight sensitivity sweep (the standard 'is it robust to your reward weights?' panel).

One-at-a-time (OAT) perturbation of the three most consequential reward weights on
medium_load / no_lifetime (the operating regime + the adopted variant):
  w_deliver (2.0) -- primary delivery incentive   -> {1.0 (0.5x), 4.0 (2x)}
  w_load    (1.0) -- congestion / load-balance     -> {0.5 (0.5x), 2.0 (2x)}  [the Dijkstra-beating mechanism]
  w_switch  (0.2) -- routing stability penalty     -> {0.1 (0.5x), 1.0 (5x)}
plus a baseline (no override) re-trained at the same budget for an internally-valid
comparison. Each config is trained 3 seeds x 50k on GPU, then evaluated on standard
medium_load (reward-INDEPENDENT metrics: delivery, throughput, P95 delay, queue,
imbalance) vs the Dijkstra oracle.

Reward overrides are injected via the LEO_REWARD_OVERRIDES env var (read by
CleanMARLLeoMultiAgentWrapper._apply_reward_overrides) -- no cleanmarl trainer edit.
The var is CLEARED during evaluation so every config is scored in the identical
standard env.

A robust policy should stay within a tight band of the baseline across all
perturbations; the scientifically interesting case is w_load (if the load-balance /
P95 advantage degrades at w_load=0.5, that localizes the mechanism).

Outputs (experiments/IEEE-REWARD-SENSITIVITY/):
  sensitivity_matrix.csv     -- per (config, policy, seed, wl_seed) episode metrics
  aggregate_sensitivity.csv  -- per (config, policy, metric) bootstrap mean + 95% CI
  summary.csv                -- per config delivery/throughput/p95 + delta-vs-baseline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig, MULTIAGENT_LOADS
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from mappo_evaluation import (
    EpisodeMetrics,
    GlobalDijkstraPolicy,
    evaluate_policy,
    load_checkpoint_policy,
)
from run_scale_experiment import bootstrap_ci, write_csv

# (name, overrides). Defaults: w_deliver=2.0, w_load=1.0, w_switch=0.2.
CONFIGS = [
    ("baseline",   {}),
    ("deliver_lo", {"w_deliver": 1.0}),
    ("deliver_hi", {"w_deliver": 4.0}),
    ("load_lo",    {"w_load": 0.5}),
    ("load_hi",    {"w_load": 2.0}),
    ("switch_lo",  {"w_switch": 0.1}),
    ("switch_hi",  {"w_switch": 1.0}),
]
METRICS = [
    "delivery_ratio", "drop_rate", "throughput_packets_per_slot",
    "average_delay_slots", "p95_delay_slots", "mean_queue_packets",
    "global_load_imbalance",
]


def train_one(args, config_name: str, overrides: dict, seed: int) -> Path:
    run_root = args.output / config_name / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    # skip if already trained (validation_best.pt present)
    existing = sorted(run_root.glob("*/validation_best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        return existing[0]

    env = os.environ.copy()
    env["LEO_REWARD_OVERRIDES"] = json.dumps(overrides)  # "" / "{}" = no-op
    cmd = [
        sys.executable, str(args.cleanmarl / "cleanmarl" / "mappo.py"),
        "--env-type", "leo_multi", "--env-name", args.scenario,
        "--leo-project-path", str(args.project), "--leo-variant", "no_lifetime",
        "--seed", str(seed),
        "--batch-size", "4", "--total-timesteps", str(args.timesteps),
        "--epochs", "3", "--num-minibatches", "4",
        "--eval-steps", "40", "--num-eval-ep", "50", "--save-every-steps", "5000",
        "--checkpoint-dir", str(run_root), "--run-tag", "REW-SENS",
        "--train-seed-start", "9001", "--train-seed-count", "200",
        "--validation-seed-start", "10001", "--device", args.device,
    ]
    print(f"  [train] {config_name} seed={seed} overrides={overrides}", flush=True)
    log = run_root / "trainer_stdout.log"
    completed = subprocess.run(cmd, cwd=args.cleanmarl, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", check=False)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"training failed: {config_name}/seed_{seed}; see {log}")
    cks = sorted(run_root.glob("*/validation_best.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not cks:
        raise RuntimeError(f"no checkpoint produced: {config_name}/seed_{seed}; see {log}")
    return cks[0]


def find_checkpoint(args, config_name: str, seed: int):
    root = args.output / config_name / f"seed_{seed}"
    cks = sorted(root.glob("*/validation_best.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[0] if cks else None


def medium_wrapper_factory(variant: str):
    ini, exo = MULTIAGENT_LOADS["medium_load"]
    def make(seed: int) -> CleanMARLLeoMultiAgentWrapper:
        envc = EnvConfig(scenario=SCENARIOS["medium_load"], seed=seed)
        cfg = MultiAgentConfig(env=envc, initial_packets=ini, exogenous_packets_per_slot=exo,
                               seed=seed, variant=variant)
        return CleanMARLLeoMultiAgentWrapper(cfg=cfg)
    return make


def aggregate(rows, config_names) -> list[dict]:
    out = []
    for cfg in config_names:
        for policy in sorted({r.policy for r in rows if r.scenario == cfg}):
            sel = [r for r in rows if r.scenario == cfg and r.policy == policy]
            for m in METRICS:
                vals = [float(getattr(r, m)) for r in sel]
                if vals:
                    mean, lo, hi = bootstrap_ci(vals)
                    out.append({"config": cfg, "policy": policy, "metric": m,
                                "n": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    ap.add_argument("--project", type=Path, default=Path("F:/leo-routing-preliminary-matlab"))
    ap.add_argument("--output", type=Path, default=Path("experiments/IEEE-REWARD-SENSITIVITY"))
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--workload-seed-start", type=int, default=21001)
    ap.add_argument("--workload-seeds", type=int, default=20)
    ap.add_argument("--train-only", action="store_true", help="train then exit (eval separately)")
    args = ap.parse_args()

    # Resolve to ABSOLUTE paths: the training subprocess runs with cwd=cleanmarl,
    # so a relative --checkpoint-dir would land under cleanmarl/, not the project.
    args.output = args.output.resolve()
    args.project = args.project.resolve()
    args.cleanmarl = args.cleanmarl.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    wseeds = list(range(args.workload_seed_start, args.workload_seed_start + args.workload_seeds))

    # ---- phase 1: train (2-way parallel) ----
    jobs = [(name, ov, sd) for name, ov in CONFIGS for sd in seeds]
    print(f"=== TRAIN {len(jobs)} jobs ({len(CONFIGS)} configs x {len(seeds)} seeds), "
          f"{args.timesteps} steps, parallel={args.max_parallel}, device={args.device} ===", flush=True)
    pending = list(jobs)
    running = []  # (popen, name, seed, log_path)
    done = 0
    while pending or running:
        while pending and len(running) < args.max_parallel:
            name, ov, sd = pending.pop(0)
            run_root = args.output / name / f"seed_{sd}"
            run_root.mkdir(parents=True, exist_ok=True)
            if find_checkpoint(args, name, sd):
                print(f"  [skip] {name} seed={sd} (exists)", flush=True)
                done += 1
                continue
            env = os.environ.copy()
            env["LEO_REWARD_OVERRIDES"] = json.dumps(ov)
            cmd = [
                sys.executable, str(args.cleanmarl / "cleanmarl" / "mappo.py"),
                "--env-type", "leo_multi", "--env-name", args.scenario,
                "--leo-project-path", str(args.project), "--leo-variant", "no_lifetime",
                "--seed", str(sd), "--batch-size", "4",
                "--total-timesteps", str(args.timesteps), "--epochs", "3", "--num-minibatches", "4",
                "--eval-steps", "40", "--num-eval-ep", "50", "--save-every-steps", "5000",
                "--checkpoint-dir", str(run_root), "--run-tag", "REW-SENS",
                "--train-seed-start", "9001", "--train-seed-count", "200",
                "--validation-seed-start", "10001", "--device", args.device,
            ]
            log = open(run_root / "trainer_stdout.log", "w", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=args.cleanmarl, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, text=True)
            running.append((p, name, sd, log))
            print(f"  [start] {name} seed={sd} pid={p.pid} overrides={ov}", flush=True)
        # poll
        still = []
        for p, name, sd, log in running:
            rc = p.poll()
            if rc is None:
                still.append((p, name, sd, log))
            else:
                log.close()
                done += 1
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  [done {done}/{len(jobs)}] {name} seed={sd} {status}", flush=True)
        running = still
        if running and pending:
            import time; time.sleep(5)

    if args.train_only:
        print("train-only complete", flush=True); return

    # ---- phase 2: evaluate (env var CLEARED -> standard env) ----
    os.environ.pop("LEO_REWARD_OVERRIDES", None)
    config_names = [c for c, _ in CONFIGS]
    print(f"\n=== EVAL {len(config_names)} configs x {len(seeds)} seeds x {len(wseeds)} wl seeds ===", flush=True)
    all_rows: list[EpisodeMetrics] = []
    for name in config_names:
        actors = []
        for sd in seeds:
            ck = find_checkpoint(args, name, sd)
            if ck is None:
                print(f"  [warn] no checkpoint for {name}/seed_{sd}", flush=True); continue
            policy, info = load_checkpoint_policy(ck, device="cpu")
            actors.append((sd, policy))
        if not actors:
            continue
        fct = medium_wrapper_factory("no_lifetime")
        for sd, policy in actors:
            all_rows.extend(evaluate_policy(name, "mappo", policy, sd, wseeds,
                                            wrapper_factory=fct, variant="no_lifetime"))
        # mask parity: the oracle must run under the SAME variant as the
        # no_lifetime checkpoints (mask-following baselines are constrained by
        # the action mask; a mismatch silently handicaps them).
        all_rows.extend(evaluate_policy(name, "global_dijkstra", GlobalDijkstraPolicy(), -1, wseeds,
                                        wrapper_factory=medium_wrapper_factory("no_lifetime")))
        md = np.mean([float(r.delivery_ratio) for r in all_rows if r.scenario == name and r.policy == "mappo"])
        dd = np.mean([float(r.delivery_ratio) for r in all_rows if r.scenario == name and r.policy == "global_dijkstra"])
        print(f"  {name:<12} delivery MAPPO={md:.3f} Dij={dd:.3f} gap={md-dd:+.3f}", flush=True)

    write_csv(args.output / "sensitivity_matrix.csv", [asdict(r) for r in all_rows])
    write_csv(args.output / "aggregate_sensitivity.csv", aggregate(all_rows, config_names))

    # ---- phase 3: summary (delta vs baseline) ----
    agg = aggregate(all_rows, config_names)
    cell = {(r["config"], r["policy"], r["metric"]): r for r in agg}
    summ = []
    base_m = cell.get(("baseline", "mappo", "delivery_ratio"), {}).get("mean")
    for name in config_names:
        row = {"config": name}
        for m in ["delivery_ratio", "throughput_packets_per_slot", "p95_delay_slots",
                  "mean_queue_packets", "global_load_imbalance"]:
            c = cell.get((name, "mappo", m))
            if c:
                row[f"mappo_{m}"] = round(c["mean"], 4)
        if base_m is not None and (name, "mappo", "delivery_ratio") in cell:
            row["delivery_delta_vs_baseline"] = round(
                cell[(name, "mappo", "delivery_ratio")]["mean"] - base_m, 4)
        summ.append(row)
    write_csv(args.output / "summary.csv", summ)
    print(f"\n=> wrote sensitivity_matrix.csv + aggregate_sensitivity.csv + summary.csv")


if __name__ == "__main__":
    main()
