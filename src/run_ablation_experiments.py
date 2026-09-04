"""Planned component contrasts and an adjacent lifetime-mechanism ladder."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from hierarchical_statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    add_multiplicity_corrections,
    pair_crossed_matrices,
    paired_crossed_summary,
    statistical_analysis_manifest,
)
from mappo_evaluation import evaluate_policy, load_checkpoint_policy
from run_exp004_mappo import (
    POLICY_SEEDS,
    mode_config as headline_mode_config,
    train_one,
)
from variant_definitions import (
    PLANNED_CONTRASTS,
    PlannedContrast,
    canonical_variant_name,
    planned_contrasts_manifest,
)


VARIANTS = [
    "proposed",
    "no_queue",
    "no_credit",
    "no_packet_context",
    "flat_critic",
    "no_ppo_protection",
    "with_lifetime_feature",
    "with_lifetime_reward",
    "with_hard_lifetime_mask",
]
SCENARIOS = [
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links",
]
# The historical 13001..13050 panel was inspected during the legacy 5k
# ablation, so it is ineligible for confirmatory evaluation.
TEST_WORKLOAD_SEEDS = tuple(range(31001, 31051))
DEVELOPMENT_WORKLOAD_SEEDS = tuple(range(12001, 12051))
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
        # Keep every training and validation-selection setting identical to the
        # headline 50k experiment.  The former "full" ablation preset was 5k.
        return dict(headline_mode_config("full"))
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


def legacy_contrasts(variants) -> tuple[PlannedContrast, ...]:
    """Build explicit references when re-auditing the historical 5k table."""

    return tuple(
        PlannedContrast(
            name=f"legacy_{variant}_vs_full",
            family="legacy_post_hoc",
            reference="full",
            treatment=variant,
            changed_flags=(),
            component_kind="legacy_confounded",
        )
        for variant in variants
        if variant != "full"
    )


def paired_effects(rows, scenarios, contrasts=PLANNED_CONTRASTS):
    output = []
    for scenario_index, scenario in enumerate(scenarios):
        for contrast_index, contrast in enumerate(contrasts):
            reference = [
                row
                for row in rows
                if row.scenario == scenario
                and row.policy == f"mappo_{contrast.reference}"
            ]
            treatment = [
                row
                for row in rows
                if row.scenario == scenario
                and row.policy == f"mappo_{contrast.treatment}"
            ]
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                paired = pair_crossed_matrices(treatment, reference, metric)
                bootstrap_rng_seed = (
                    18000
                    + 1000 * scenario_index
                    + 100 * contrast_index
                    + metric_index
                )
                summary = paired_crossed_summary(
                    paired,
                    rng_seed=bootstrap_rng_seed,
                    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                )
                output.append(
                    {
                        "scenario": scenario,
                        "contrast": contrast.name,
                        "contrast_family": contrast.family,
                        "reference_variant": contrast.reference,
                        "treatment_variant": contrast.treatment,
                        "changed_flags": ";".join(contrast.changed_flags),
                        "component_kind": contrast.component_kind,
                        "variant": contrast.treatment,
                        "metric": metric,
                        "paired_policy_seeds": summary["policy_seed_count"],
                        "paired_workloads": summary["workload_seed_count"],
                        "paired_episode_cells": summary["episode_pair_count"],
                        "bootstrap_rng_seed": summary["bootstrap_rng_seed"],
                        "policy_seed_ids_json": json.dumps(
                            summary["policy_seed_ids"],
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ),
                        "policy_seed_mean_differences_json": json.dumps(
                            summary["policy_seed_mean_differences"],
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ),
                        "policy_seed_pairing": summary["policy_seed_pairing"],
                        "reference_mean": summary["reference_mean"],
                        "treatment_mean": summary["treatment_mean"],
                        "treatment_minus_reference": summary["mean_difference"],
                        "median_difference": summary[
                            "median_policy_seed_mean_difference"
                        ],
                        "policy_seed_mean_difference_std": summary[
                            "policy_seed_mean_difference_std"
                        ],
                        "difference_ci95_low": summary["difference_ci95_low"],
                        "difference_ci95_high": summary["difference_ci95_high"],
                        "bootstrap_standard_error": summary[
                            "bootstrap_standard_error"
                        ],
                        "difference_ci_method": summary["difference_ci_method"],
                        "ci_resamples": summary["ci_resamples"],
                        "paired_cohen_dz": summary["paired_cohen_dz"],
                        "paired_hedges_gz": summary["paired_hedges_gz"],
                        "rank_biserial_correlation": summary[
                            "rank_biserial_correlation"
                        ],
                        "positive_policy_seed_fraction": summary[
                            "positive_policy_seed_fraction"
                        ],
                        "sign_flip_statistic": summary["sign_flip_statistic"],
                        "sign_flip_permutations": summary[
                            "sign_flip_permutations"
                        ],
                        "sign_flip_tail_tie_permutations": summary[
                            "sign_flip_tail_tie_permutations"
                        ],
                        "sign_flip_zero_difference_count": summary[
                            "sign_flip_zero_difference_count"
                        ],
                        "sign_flip_tail_ties_included": summary[
                            "sign_flip_tail_ties_included"
                        ],
                        "raw_p_value": summary["raw_p_value"],
                        "p_value_method": summary["p_value_method"],
                        "seed_level_wilcoxon_statistic": summary[
                            "seed_level_wilcoxon_statistic"
                        ],
                        "seed_level_wilcoxon_p": summary[
                            "seed_level_wilcoxon_p"
                        ],
                        "seed_level_wilcoxon_requested_method": summary[
                            "seed_level_wilcoxon_requested_method"
                        ],
                        "seed_level_wilcoxon_zero_method": summary[
                            "seed_level_wilcoxon_zero_method"
                        ],
                        "seed_level_wilcoxon_zero_count": summary[
                            "seed_level_wilcoxon_zero_count"
                        ],
                        "seed_level_wilcoxon_absolute_tie_group_count": summary[
                            "seed_level_wilcoxon_absolute_tie_group_count"
                        ],
                        "seed_level_wilcoxon_absolute_tied_value_count": summary[
                            "seed_level_wilcoxon_absolute_tied_value_count"
                        ],
                        "seed_level_wilcoxon_method": summary[
                            "seed_level_wilcoxon_method"
                        ],
                    }
                )
    add_multiplicity_corrections(
        output,
        family_keys=("metric",),
        holm_key="confirmatory_holm_within_metric_p",
        benjamini_hochberg_key="within_metric_bh_sensitivity_p",
        family_size_key="within_metric_family_size",
    )
    add_multiplicity_corrections(
        output,
        holm_key="global_holm_sensitivity_p",
        benjamini_hochberg_key="global_bh_exploratory_p",
        family_size_key="global_family_size",
    )
    for row in output:
        if row["contrast_family"] == "legacy_post_hoc":
            row["multiplicity_role"] = "legacy_exploratory"
        else:
            row["multiplicity_role"] = (
                "primary_confirmatory_family"
                if row["metric"] == "delivery_ratio"
                else "secondary_metric_family"
            )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/archive/ablation-dev"),
    )
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args(argv)
    if args.mode == "full":
        parser.error(
            "--mode full is reserved for ablation_matrix_runner.py, which "
            "enforces the frozen 540-job protocol and test lock"
        )
    if "proposed" not in args.variants:
        parser.error("--variants must include proposed as the canonical reference")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = mode_config(args.mode)
    train_args = SimpleNamespace(
        output=args.output,
        cleanmarl=args.cleanmarl,
        project=args.project,
        skip_training=args.skip_training,
        device=args.device,
    )
    workload_seeds = DEVELOPMENT_WORKLOAD_SEEDS[: config["test_episodes"]]
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
                variant=canonical_variant_name(variant),
            )
        )
    dictionaries = [asdict(row) for row in rows]
    write_csv(args.output / "episode_metrics.csv", dictionaries)
    write_csv(
        args.output / "paired_ablation_effects.csv",
        paired_effects(
            rows,
            args.scenarios,
            tuple(
                contrast
                for contrast in PLANNED_CONTRASTS
                if contrast.reference in args.variants
                and contrast.treatment in args.variants
            ),
        ),
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": "controlled_ablation",
                "inferential_status": "exploratory_development_only",
                "formal_full_runner": "ablation_matrix_runner.py",
                "mode": args.mode,
                "one_factor_at_a_time": False,
                "design": "controlled_components_plus_adjacent_lifetime_ladder",
                "variants": args.variants,
                "planned_contrasts": planned_contrasts_manifest(),
                "scenarios": args.scenarios,
                "policy_seeds": POLICY_SEEDS,
                "train_workload_seeds": [9001, 9200],
                "validation_seed_start": 10001,
                "development_evaluation_seed_range": [
                    DEVELOPMENT_WORKLOAD_SEEDS[0],
                    DEVELOPMENT_WORKLOAD_SEEDS[config["test_episodes"] - 1],
                ],
                "formal_test_seeds_used": False,
                "config": config,
                "max_parallel": args.max_parallel,
                "checkpoints": checkpoints,
                "statistical_analysis": statistical_analysis_manifest(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Exploratory ablation run complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
