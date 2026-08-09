"""EXP-004: train, select, and test the revised satellite-level MAPPO."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable

import numpy as np
from scipy import stats

from mappo_evaluation import (
    EpisodeMetrics,
    GlobalDijkstraPolicy,
    evaluate_policy,
    heuristic_policy,
    load_checkpoint_policy,
    train_q_routing,
)


ALL_SCENARIOS = [
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links",
]
POLICY_SEEDS = [7, 42, 1024]
METRICS = [
    "delivery_ratio",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "routing_switches",
    "episode_reward",
    "global_delay_cost",
    "global_queue_cost",
    "global_load_imbalance",
    "global_switch_cost",
    "global_throughput_reward",
    "global_control_overhead_ratio",
    "global_drop_cost",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", type=Path, default=Path("experiments/EXP-004"))
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def mode_config(mode: str) -> Dict:
    if mode == "full":
        return {
            "scenarios": ALL_SCENARIOS,
            "timesteps": 50000,
            "validation_episodes": 50,
            "test_episodes": 50,
            "eval_every_rollouts": 40,
            "save_every_steps": 5000,
            "batch_size": 4,
            "q_routing_train_episodes": 500,
        }
    return {
        "scenarios": ["medium_load", "frequent_break"],
        "timesteps": 300,
        "validation_episodes": 20,
        "test_episodes": 20,
        "eval_every_rollouts": 2,
        "save_every_steps": 150,
        "batch_size": 2,
        "q_routing_train_episodes": 100,
    }


def find_selected_checkpoint(run_root: Path) -> Path | None:
    manifests = sorted(
        run_root.glob("*/run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = manifest.get("validation_best_checkpoint")
        if selected:
            selected_path = Path(selected)
            if not selected_path.is_absolute():
                selected_path = manifest_path.parent / selected_path.name
            if selected_path.exists():
                return selected_path
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint(args) -> Dict[str, str]:
    paths = {
        "trainer": args.cleanmarl / "cleanmarl" / "mappo.py",
        "environment": args.project / "leo_multiagent_env.py",
        "wrapper": args.project / "cleanmarl_leo_multiagent_wrapper.py",
        "design": args.project / "mappo_design.py",
        "evaluation": args.project / "mappo_evaluation.py",
    }
    return {name: sha256(path.resolve()) for name, path in paths.items()}


def train_one(
    args,
    config: Dict,
    scenario: str,
    policy_seed: int,
    variant: str = "full",
    trainer_overrides: list[str] | None = None,
) -> Path:
    if variant == "full":
        run_root = args.output.resolve() / "checkpoints" / scenario / f"seed_{policy_seed}"
    else:
        run_root = (
            args.output.resolve()
            / "checkpoints"
            / scenario
            / variant
            / f"seed_{policy_seed}"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "code": code_fingerprint(args),
        "scenario": scenario,
        "policy_seed": policy_seed,
        "variant": variant,
        "config": config,
        "trainer_overrides": trainer_overrides or [],
    }
    fingerprint_path = run_root / "code_fingerprint.json"
    existing = find_selected_checkpoint(run_root)
    fingerprint_matches = (
        fingerprint_path.exists()
        and json.loads(fingerprint_path.read_text(encoding="utf-8")) == fingerprint
    )
    if args.skip_training:
        if existing is None or not fingerprint_matches:
            raise FileNotFoundError(f"no selected checkpoint under {run_root}")
        return existing
    if existing is not None and fingerprint_matches:
        return existing

    command = [
        sys.executable,
        str(args.cleanmarl / "cleanmarl" / "mappo.py"),
        "--env-type", "leo_multi",
        "--env-name", scenario,
        "--leo-project-path", str(args.project),
        "--leo-variant", variant if variant != "no_ppo_protection" else "full",
        "--seed", str(policy_seed),
        "--batch-size", str(config["batch_size"]),
        "--total-timesteps", str(config["timesteps"]),
        "--epochs", "3",
        "--num-minibatches", "4",
        "--eval-steps", str(config["eval_every_rollouts"]),
        "--num-eval-ep", str(config["validation_episodes"]),
        "--save-every-steps", str(config["save_every_steps"]),
        "--checkpoint-dir", str(run_root),
        "--run-tag", "EXP-004",
        "--train-seed-start", "9001",
        "--train-seed-count", "20",
        "--validation-seed-start", "10001",
        "--device", args.device,
    ]
    command.extend(trainer_overrides or [])
    completed = subprocess.run(
        command,
        cwd=args.cleanmarl,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    (run_root / "trainer_stdout.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"training failed for {scenario}/seed={policy_seed}; "
            f"see {run_root / 'trainer_stdout.log'}"
        )
    selected = find_selected_checkpoint(run_root)
    if selected is None:
        raise RuntimeError(f"training produced no validation-best checkpoint: {run_root}")
    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )
    return selected


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[EpisodeMetrics]) -> list[Dict]:
    result = []
    keys = sorted({(row.scenario, row.policy) for row in rows})
    for group_index, (scenario, policy) in enumerate(keys):
        selected = [row for row in rows if row.scenario == scenario and row.policy == policy]
        for metric_index, metric in enumerate(METRICS):
            values = np.asarray([getattr(row, metric) for row in selected], dtype=float)
            rng = np.random.default_rng(4000 + 100 * group_index + metric_index)
            policy_seeds = sorted({row.policy_seed for row in selected})
            workload_seeds = sorted({row.workload_seed for row in selected})
            lookup = {
                (row.policy_seed, row.workload_seed): getattr(row, metric)
                for row in selected
            }
            bootstrap_means = []
            for _ in range(5000):
                sampled_policies = rng.choice(
                    policy_seeds, size=len(policy_seeds), replace=True
                )
                sampled_workloads = rng.choice(
                    workload_seeds, size=len(workload_seeds), replace=True
                )
                sample = [
                    lookup[(int(policy_seed), int(workload_seed))]
                    for policy_seed in sampled_policies
                    for workload_seed in sampled_workloads
                ]
                bootstrap_means.append(float(np.mean(sample)))
            ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
            result.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "metric": metric,
                    "n": len(values),
                    "policy_seed_count": len(policy_seeds),
                    "workload_seed_count": len(workload_seeds),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "ci_method": "hierarchical_bootstrap_5000",
                }
            )
    return result


def paired_tests(rows: list[EpisodeMetrics]) -> list[Dict]:
    results = []
    for scenario in sorted({row.scenario for row in rows}):
        for baseline in [
            "delay_only",
            "full_heuristic",
            "global_dijkstra",
            "q_routing",
        ]:
            for metric in METRICS:
                mappo_by_workload = {}
                baseline_by_workload = {}
                for row in rows:
                    if row.scenario != scenario:
                        continue
                    if row.policy == "mappo":
                        mappo_by_workload.setdefault(row.workload_seed, []).append(
                            getattr(row, metric)
                        )
                    elif row.policy == baseline:
                        baseline_by_workload.setdefault(row.workload_seed, []).append(
                            getattr(row, metric)
                        )
                common = sorted(set(mappo_by_workload) & set(baseline_by_workload))
                learned = np.asarray(
                    [np.mean(mappo_by_workload[seed]) for seed in common]
                )
                reference = np.asarray(
                    [np.mean(baseline_by_workload[seed]) for seed in common]
                )
                differences = learned - reference
                try:
                    test = stats.wilcoxon(differences)
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                results.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_workloads": len(common),
                        "mappo_mean": float(learned.mean()),
                        "baseline_mean": float(reference.mean()),
                        "mean_difference": float(differences.mean()),
                        "median_difference": float(np.median(differences)),
                        "paired_cohen_dz": float(
                            differences.mean() / differences.std(ddof=1)
                        ) if len(differences) > 1 and differences.std(ddof=1) > 0 else 0.0,
                        "wilcoxon_statistic": statistic,
                        "raw_p_value": p_value,
                    }
                )
    if results:
        ordered = sorted(range(len(results)), key=lambda i: results[i]["raw_p_value"])
        adjusted = [1.0] * len(results)
        total = len(results)
        running = 1.0
        for rank in range(total, 0, -1):
            index = ordered[rank - 1]
            candidate = results[index]["raw_p_value"] * total / rank
            running = min(running, candidate)
            adjusted[index] = min(1.0, running)
        for row, value in zip(results, adjusted):
            row["benjamini_hochberg_p"] = value
    return results


def main():
    args = parse_args()
    config = mode_config(args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    workload_seeds = range(11001, 11001 + config["test_episodes"])
    all_rows: list[EpisodeMetrics] = []
    checkpoints = {}

    for scenario in config["scenarios"]:
        for policy_seed in POLICY_SEEDS:
            checkpoint = train_one(args, config, scenario, policy_seed)
            checkpoints[f"{scenario}/seed_{policy_seed}"] = str(checkpoint)
            policy, _ = load_checkpoint_policy(checkpoint, device=args.device)
            all_rows.extend(
                evaluate_policy(
                    scenario,
                    "mappo",
                    policy,
                    policy_seed,
                    workload_seeds,
                )
            )
        for baseline in ["delay_only", "full_heuristic", "random"]:
            all_rows.extend(
                evaluate_policy(
                    scenario,
                    baseline,
                    heuristic_policy(baseline, seed=1234),
                    -1,
                    workload_seeds,
                )
            )
        all_rows.extend(
            evaluate_policy(
                scenario,
                "global_dijkstra",
                GlobalDijkstraPolicy(),
                -1,
                workload_seeds,
            )
        )
        q_model_dir = args.output / "q_routing_models"
        q_model_dir.mkdir(parents=True, exist_ok=True)
        for policy_seed in POLICY_SEEDS:
            q_policy = train_q_routing(
                scenario,
                seed=policy_seed,
                episodes=config["q_routing_train_episodes"],
            )
            np.savez_compressed(
                q_model_dir / f"{scenario}_seed_{policy_seed}.npz",
                q=q_policy.q,
            )
            all_rows.extend(
                evaluate_policy(
                    scenario,
                    "q_routing",
                    q_policy,
                    policy_seed,
                    workload_seeds,
                )
            )

    episode_dicts = [asdict(row) for row in all_rows]
    write_csv(args.output / "episode_metrics.csv", episode_dicts)
    write_csv(args.output / "aggregate_metrics.csv", aggregate_rows(all_rows))
    write_csv(args.output / "paired_tests.csv", paired_tests(all_rows))
    (args.output / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "EXP-004",
                "mode": args.mode,
                "config": config,
                "policy_seeds": POLICY_SEEDS,
                "train_workload_seeds": [9001, 9020],
                "validation_workload_seeds": [
                    10001,
                    10000 + config["validation_episodes"],
                ],
                "test_workload_seeds": [
                    11001,
                    11000 + config["test_episodes"],
                ],
                "checkpoints": checkpoints,
                "code_fingerprint": code_fingerprint(args),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"EXP-004 complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
