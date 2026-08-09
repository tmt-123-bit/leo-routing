"""Controlled one-factor-at-a-time ablations for the revised LEO MAPPO."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import stats

from mappo_evaluation import evaluate_policy, load_checkpoint_policy
from run_exp004_mappo import POLICY_SEEDS, train_one


VARIANTS = [
    "full",
    "no_queue",
    "no_lifetime",
    "no_credit",
    "no_packet_context",
    "flat_critic",
    "no_ppo_protection",
]
SCENARIOS = ["medium_load", "frequent_break"]
PRIMARY_METRICS = [
    "delivery_ratio",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "routing_switches",
    "global_control_overhead_ratio",
]


def mode_config(mode: str):
    if mode == "full":
        return {
            "timesteps": 5000,
            "validation_episodes": 50,
            "test_episodes": 50,
            "eval_every_rollouts": 10,
            "save_every_steps": 1000,
            "batch_size": 4,
        }
    return {
        "timesteps": 600,
        "validation_episodes": 20,
        "test_episodes": 20,
        "eval_every_rollouts": 4,
        "save_every_steps": 300,
        "batch_size": 2,
    }


def overrides_for(variant: str) -> list[str]:
    if variant != "no_ppo_protection":
        return []
    return [
        "--clip-gradients", "0",
        "--target-kl", "0",
        "--no-normalize-advantage",
    ]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_effects(rows, scenarios, variants):
    output = []
    for scenario in scenarios:
        for variant in variants:
            if variant == "full":
                continue
            for metric in PRIMARY_METRICS:
                full = {}
                ablated = {}
                for row in rows:
                    if row.scenario != scenario:
                        continue
                    key = (row.policy_seed, row.workload_seed)
                    if row.policy == "mappo_full":
                        full[key] = getattr(row, metric)
                    elif row.policy == f"mappo_{variant}":
                        ablated[key] = getattr(row, metric)
                common = sorted(set(full) & set(ablated))
                differences = np.asarray(
                    [ablated[key] - full[key] for key in common], dtype=float
                )
                try:
                    result = stats.wilcoxon(differences)
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                output.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "metric": metric,
                        "paired_samples": len(common),
                        "full_mean": float(np.mean([full[key] for key in common])),
                        "ablated_mean": float(np.mean([ablated[key] for key in common])),
                        "ablated_minus_full": float(differences.mean()),
                        "median_difference": float(np.median(differences)),
                        "wilcoxon_statistic": statistic,
                        "raw_p_value": p_value,
                    }
                )
    ordered = sorted(range(len(output)), key=lambda index: output[index]["raw_p_value"])
    adjusted = [1.0] * len(output)
    running = 1.0
    total = len(output)
    for rank in range(total, 0, -1):
        index = ordered[rank - 1]
        running = min(running, output[index]["raw_p_value"] * total / rank)
        adjusted[index] = min(1.0, running)
    for row, value in zip(output, adjusted):
        row["benjamini_hochberg_p"] = value
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", type=Path, default=Path("experiments/ABLATION"))
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    if "full" not in args.variants:
        parser.error("--variants must include full as the paired reference")
    config = mode_config(args.mode)
    train_args = SimpleNamespace(
        output=args.output,
        cleanmarl=args.cleanmarl,
        project=args.project,
        skip_training=args.skip_training,
        device=args.device,
    )
    workload_seeds = range(13001, 13001 + config["test_episodes"])
    rows = []
    checkpoints = {}
    jobs = [
        (scenario, variant, policy_seed)
        for scenario in args.scenarios
        for variant in args.variants
        for policy_seed in POLICY_SEEDS
    ]

    def train_job(job):
        scenario, variant, policy_seed = job
        return train_one(
            train_args,
            config,
            scenario,
            policy_seed,
            variant=variant,
            trainer_overrides=overrides_for(variant),
        )

    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        future_to_job = {executor.submit(train_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            scenario, variant, policy_seed = future_to_job[future]
            key = f"{scenario}/{variant}/seed_{policy_seed}"
            checkpoints[key] = str(future.result())
            print(f"trained {key}", flush=True)

    for scenario, variant, policy_seed in jobs:
        checkpoint = Path(checkpoints[f"{scenario}/{variant}/seed_{policy_seed}"])
        policy, _ = load_checkpoint_policy(checkpoint, device=args.device)
        rows.extend(
            evaluate_policy(
                scenario,
                f"mappo_{variant}",
                policy,
                policy_seed,
                workload_seeds,
                variant=variant,
            )
        )
    dictionaries = [asdict(row) for row in rows]
    write_csv(args.output / "episode_metrics.csv", dictionaries)
    write_csv(
        args.output / "paired_ablation_effects.csv",
        paired_effects(rows, args.scenarios, args.variants),
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "controlled_ablation",
                "mode": args.mode,
                "one_factor_at_a_time": True,
                "variants": args.variants,
                "scenarios": args.scenarios,
                "policy_seeds": POLICY_SEEDS,
                "train_workload_seeds": [9001, 9020],
                "validation_seed_start": 10001,
                "test_seed_range": [13001, 13000 + config["test_episodes"]],
                "config": config,
                "max_parallel": args.max_parallel,
                "checkpoints": checkpoints,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Ablation experiment complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
