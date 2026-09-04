"""Build an auditable summary from completed results without retraining.

The corrected legacy 50k evaluation is the default evidence package. It is
reported with policy-seed-level inference and explicit retrospective labels.
The paused 9 x 5 x 12 study remains an optional future validation and is never
allowed to hide or replace the completed legacy results unless all frozen gates
pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCENARIOS = (
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links",
)
POLICY_SEEDS = (
    2009387241,
    688652842,
    1824192069,
    1446495455,
    140273932,
    1016147766,
    885581732,
    247192783,
    1367579272,
    1996870291,
    646895287,
    489261598,
)
CONFIGURATIONS = (
    "proposed",
    "no_queue",
    "no_credit",
    "no_packet_context",
    "flat_critic",
    "no_ppo_protection",
    "with_lifetime_feature",
    "with_lifetime_reward",
    "with_hard_lifetime_mask",
)
TEST_WORKLOAD_SEEDS = tuple(range(31001, 31051))
ENDPOINTS = (
    "delivery_ratio",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "routing_switches",
    "global_control_overhead_ratio",
)
PLANNED_CONTRASTS = (
    "remove_queue_mechanism_package",
    "remove_centered_local_credit",
    "remove_packet_context",
    "replace_graph_critic_with_flat_critic",
    "remove_ppo_protection_package",
    "add_lifetime_feature",
    "add_lifetime_reward",
    "add_hard_lifetime_mask",
)
CONTRAST_VARIANTS = {
    "remove_queue_mechanism_package": ("proposed", "no_queue"),
    "remove_centered_local_credit": ("proposed", "no_credit"),
    "remove_packet_context": ("proposed", "no_packet_context"),
    "replace_graph_critic_with_flat_critic": ("proposed", "flat_critic"),
    "remove_ppo_protection_package": ("proposed", "no_ppo_protection"),
    "add_lifetime_feature": ("proposed", "with_lifetime_feature"),
    "add_lifetime_reward": ("with_lifetime_feature", "with_lifetime_reward"),
    "add_hard_lifetime_mask": (
        "with_lifetime_reward",
        "with_hard_lifetime_mask",
    ),
}
TRAINING_SOURCE_FILES = {
    "training_entry": "run_exp004_mappo.py",
    "base_environment": "leo_marl_env.py",
    "environment": "leo_multiagent_env.py",
    "wrapper": "cleanmarl_leo_multiagent_wrapper.py",
    "variant_definitions": "variant_definitions.py",
    "design": "mappo_design.py",
    "evaluation": "mappo_evaluation.py",
    "trainer_snapshot": "cleanmarl_mappo_leo.py",
}
ANALYSIS_SOURCE_FILES = {
    "ablation_analysis": "run_ablation_experiments.py",
    "statistics": "hierarchical_statistics.py",
}
ALL_REPOSITORY_SOURCE_FILES = {
    "runner": "ablation_matrix_runner.py",
    **TRAINING_SOURCE_FILES,
    **ANALYSIS_SOURCE_FILES,
}

EXPECTED_TRAINING_JOBS = 540
EXPECTED_TEST_ROWS = 27_000
EXPECTED_EFFECT_ROWS = len(SCENARIOS) * len(PLANNED_CONTRASTS) * len(ENDPOINTS)
MATRIX_NAME = "controlled_ablation_50k_v2"

REQUIRED_EFFECT_COLUMNS = {
    "scenario",
    "contrast",
    "reference_variant",
    "treatment_variant",
    "metric",
    "paired_policy_seeds",
    "paired_workloads",
    "paired_episode_cells",
    "treatment_minus_reference",
    "difference_ci95_low",
    "difference_ci95_high",
    "difference_ci_method",
    "ci_resamples",
    "raw_p_value",
    "p_value_method",
    "sign_flip_permutations",
    "confirmatory_holm_within_metric_p",
    "within_metric_family_size",
}

REQUIRED_HEADLINE_COLUMNS = {
    "scenario",
    "baseline",
    "metric",
    "paired_policy_seeds",
    "paired_workloads",
    "paired_episode_cells",
    "mappo_mean",
    "baseline_mean",
    "mean_difference",
    "difference_ci95_low",
    "difference_ci95_high",
    "difference_ci_method",
    "ci_resamples",
    "positive_policy_seed_fraction",
    "raw_p_value",
    "p_value_method",
    "sign_flip_permutations",
}

LEGACY_BASELINES = (
    "delay_only",
    "full_heuristic",
    "global_dijkstra",
    "ospf_ecmp",
    "q_routing",
)
LEGACY_METRICS = (
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
)
LEGACY_ABLATION_CONTRASTS = (
    "legacy_flat_critic_vs_full",
    "legacy_no_credit_vs_full",
    "legacy_no_lifetime_vs_full",
    "legacy_no_packet_context_vs_full",
    "legacy_no_ppo_protection_vs_full",
    "legacy_no_queue_vs_full",
)
LEGACY_POLICY_SEEDS = (7, 42, 1024, 123, 456, 789, 2024, 314)


@dataclass(frozen=True)
class ConfirmatoryAblation:
    complete: bool
    root: Path
    training_completed: int | None
    frozen_training_jobs: int | None
    evaluation_completed: int | None
    validated_rows: int | None
    effects: tuple[dict[str, str], ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ReanalysisArtifact:
    complete: bool
    name: str
    source: Path
    root: Path
    source_rows: int | None
    output_rows: int | None
    rows: tuple[dict[str, str], ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class LegacyProvenance:
    complete: bool
    issues: tuple[str, ...]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def source_hashes(
    project_root: Path, source_files: Mapping[str, str]
) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for name, relative_path in source_files.items():
        path = project_root / "src" / relative_path
        if not path.is_file():
            return None
        result[name] = sha256_file(path)
    return result


def planned_contrast_names(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        return ()
    return tuple(item.get("name") for item in value)


def nested_int(value: Mapping[str, Any] | None, *keys: str) -> int | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    try:
        return int(current)
    except (TypeError, ValueError):
        return None


def finite_float(row: Mapping[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {column!r} value") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {column!r} value")
    return value


def _valid_frozen_checkpoint(checkpoint: Any) -> bool:
    if not isinstance(checkpoint, Mapping):
        return False
    hash_fields = (
        "training_fingerprint",
        "checkpoint_sha256",
        "run_manifest_sha256",
        "run_config_sha256",
        "training_metrics_sha256",
        "trainer_log_sha256",
        "training_code_fingerprint_sha256",
    )
    if any(not is_sha256(checkpoint.get(field)) for field in hash_fields):
        return False
    try:
        actual_steps = int(checkpoint["actual_environment_steps"])
        selected_step = int(checkpoint["selected_checkpoint_step"])
        earliest_best_step = int(checkpoint["earliest_best_validation_step"])
        score = [float(value) for value in checkpoint["selected_validation_score"]]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        50_000 <= actual_steps <= 50_119
        and 0 < selected_step == earliest_best_step <= actual_steps
        and len(score) == 4
        and all(math.isfinite(value) for value in score)
    )


def inspect_confirmatory_ablation(project_root: Path) -> ConfirmatoryAblation:
    """Validate every publication gate before exposing confirmatory effects."""

    root = project_root / "experiments" / "ablation-50k-v2"
    spec = load_json(root / "matrix_spec.json")
    historical_seed_audit_path = root / "historical_workload_seed_audit.json"
    historical_seed_audit = load_json(historical_seed_audit_path)
    training_freeze = load_json(root / "training_freeze_manifest.json")
    audit = load_json(root / "matrix_audit.json")
    merge = load_json(root / "episode_metrics_manifest.json")
    statistics = load_json(root / "statistical_analysis_manifest.json")
    effect_path = root / "paired_ablation_effects.csv"
    effects = load_csv(effect_path)
    issues: list[str] = []
    frozen_training_jobs: int | None = None
    training_freeze_hash: str | None = None
    historical_seed_audit_hash: str | None = None
    historical_seed_audit_content_hash: str | None = None

    spec_hash: str | None = None
    if spec is None:
        issues.append("matrix_spec.json is missing or invalid")
    else:
        declared_spec_hash = spec.get("spec_sha256")
        unhashed_spec = dict(spec)
        unhashed_spec.pop("spec_sha256", None)
        if not isinstance(declared_spec_hash, str):
            issues.append("matrix spec lacks spec_sha256")
        elif sha256_json(unhashed_spec) != declared_spec_hash:
            issues.append("matrix spec self-hash is invalid")
        else:
            spec_hash = declared_spec_hash

        frozen_fields = {
            "matrix_name": MATRIX_NAME,
            "scenarios": list(SCENARIOS),
            "variants": list(CONFIGURATIONS),
            "policy_seeds": list(POLICY_SEEDS),
            "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
            "expected_training_jobs": EXPECTED_TRAINING_JOBS,
            "expected_evaluation_rows": EXPECTED_TEST_ROWS,
            "full_budget_steps": 50_000,
        }
        for key, expected in frozen_fields.items():
            if spec.get(key) != expected:
                issues.append(f"matrix spec {key} differs from the frozen protocol")
        if planned_contrast_names(spec.get("planned_contrasts")) != PLANNED_CONTRASTS:
            issues.append("matrix spec contrasts differ from the frozen protocol")

        jobs = spec.get("jobs")
        observed_jobs: set[tuple[str, str, int]] = set()
        if isinstance(jobs, list):
            try:
                observed_jobs = {
                    (
                        str(job["scenario"]),
                        str(job["variant"]),
                        int(job["policy_seed"]),
                    )
                    for job in jobs
                    if isinstance(job, Mapping)
                }
            except (KeyError, TypeError, ValueError):
                observed_jobs = set()
        expected_jobs = {
            (scenario, configuration, policy_seed)
            for scenario in SCENARIOS
            for configuration in CONFIGURATIONS
            for policy_seed in POLICY_SEEDS
        }
        if (
            not isinstance(jobs, list)
            or len(jobs) != EXPECTED_TRAINING_JOBS
            or observed_jobs != expected_jobs
        ):
            issues.append("matrix spec jobs are not the unique frozen 9 x 5 x 12 grid")

        config = spec.get("config")
        required_config = {
            "timesteps": 50_000,
            "validation_episodes": 50,
            "test_episodes": 50,
            "eval_every_rollouts": 40,
            "save_every_steps": 5_000,
            "batch_size": 4,
        }
        if not isinstance(config, Mapping):
            issues.append("matrix spec lacks the full-budget runtime config")
        else:
            for key, expected in required_config.items():
                if config.get(key) != expected:
                    issues.append(
                        f"matrix runtime config {key} is {config.get(key)!r}, not {expected}"
                    )

    if historical_seed_audit is None:
        issues.append("historical_workload_seed_audit.json is missing or invalid")
    else:
        historical_content = historical_seed_audit.get("content")
        if historical_seed_audit.get("status") != (
            "passed_no_selected_workload_seed_overlap"
        ):
            issues.append("historical seed audit did not pass")
        if not isinstance(historical_content, Mapping):
            issues.append("historical seed audit has no content object")
        else:
            declared_historical_content_hash = historical_seed_audit.get(
                "content_sha256"
            )
            if (
                not is_sha256(declared_historical_content_hash)
                or declared_historical_content_hash
                != sha256_json(historical_content)
            ):
                issues.append("historical seed audit content SHA-256 is invalid")
            else:
                historical_seed_audit_content_hash = (
                    declared_historical_content_hash
                )
            if historical_content.get("selected_policy_seeds") != list(
                POLICY_SEEDS
            ):
                issues.append("historical seed audit targets the wrong policy seeds")
            if historical_content.get("selected_workload_seeds") != list(
                TEST_WORKLOAD_SEEDS
            ):
                issues.append("historical seed audit targets the wrong test workloads")
            if any(
                historical_content.get(field) != []
                for field in (
                    "selected_prior_overlap",
                    "selected_workload_seed_prior_overlap",
                    "selected_policy_seed_prior_overlap",
                )
            ):
                issues.append("historical seed audit reports selected-seed overlap")
            if any(
                not isinstance(historical_content.get(field), list)
                for field in (
                    "scanned_csv_files",
                    "historical_workload_seed_ranges",
                    "historical_policy_seeds",
                )
            ):
                issues.append("historical seed audit has malformed scan evidence")
        if historical_seed_audit_path.is_file():
            historical_seed_audit_hash = sha256_file(historical_seed_audit_path)

    if training_freeze is None:
        issues.append("training_freeze_manifest.json is missing or invalid")
    else:
        freeze_content = training_freeze.get("content")
        if training_freeze.get("status") != "frozen_before_test_evaluation":
            issues.append("training freeze is not active")
        if not isinstance(freeze_content, Mapping):
            issues.append("training freeze has no content object")
        else:
            declared_freeze_hash = training_freeze.get("content_sha256")
            if (
                not is_sha256(declared_freeze_hash)
                or declared_freeze_hash != sha256_json(freeze_content)
            ):
                issues.append("training freeze content SHA-256 is invalid")
            else:
                training_freeze_hash = declared_freeze_hash
            if freeze_content.get("matrix_name") != MATRIX_NAME:
                issues.append("training freeze has the wrong matrix identity")
            if freeze_content.get("protocol_spec_sha256") != spec_hash:
                issues.append("training freeze does not bind to the matrix spec")
            if freeze_content.get("expected_training_jobs") != EXPECTED_TRAINING_JOBS:
                issues.append("training freeze does not cover 540 jobs")
            if freeze_content.get(
                "historical_workload_seed_audit_sha256"
            ) != historical_seed_audit_hash or freeze_content.get(
                "historical_workload_seed_audit_content_sha256"
            ) != historical_seed_audit_content_hash:
                issues.append("training freeze does not bind to the historical seed audit")
            training_code = freeze_content.get("training_code_sha256")
            if (
                not isinstance(training_code, Mapping)
                or not training_code
                or any(not is_sha256(digest) for digest in training_code.values())
            ):
                issues.append("training freeze has invalid training-code hashes")
            else:
                current_training_code = source_hashes(
                    project_root, TRAINING_SOURCE_FILES
                )
                if current_training_code is None:
                    issues.append("current training source set is incomplete")
                else:
                    current_training_code["trainer"] = current_training_code[
                        "trainer_snapshot"
                    ]
                    if dict(training_code) != current_training_code:
                        issues.append(
                            "current training sources differ from the frozen code hashes"
                        )
            protocol_path = project_root / "docs" / "ABLATION_50K_PROTOCOL.md"
            protocol_digest = (
                sha256_file(protocol_path) if protocol_path.is_file() else None
            )
            current_all_code = source_hashes(
                project_root, ALL_REPOSITORY_SOURCE_FILES
            )
            if current_all_code is None:
                issues.append("current protocol source set is incomplete")
            elif protocol_digest is None:
                issues.append("frozen protocol document is missing")
            else:
                current_all_code["trainer"] = current_all_code["trainer_snapshot"]
                current_all_code["protocol_document"] = protocol_digest
                if freeze_content.get("all_code_sha256") != current_all_code:
                    issues.append(
                        "current protocol sources differ from the complete frozen "
                        "code hashes"
                    )
            if protocol_digest is not None and freeze_content.get(
                "protocol_document_sha256"
            ) != protocol_digest:
                issues.append(
                    "current protocol document differs from the training freeze"
                )
            frozen_checkpoints = freeze_content.get("checkpoints")
            frozen_training_jobs = (
                len(frozen_checkpoints)
                if isinstance(frozen_checkpoints, list)
                else None
            )
            observed_frozen_jobs: set[tuple[str, str, int]] = set()
            if isinstance(frozen_checkpoints, list):
                try:
                    observed_frozen_jobs = {
                        (
                            str(checkpoint["scenario"]),
                            str(checkpoint["variant"]),
                            int(checkpoint["policy_seed"]),
                        )
                        for checkpoint in frozen_checkpoints
                        if isinstance(checkpoint, Mapping)
                    }
                except (KeyError, TypeError, ValueError):
                    observed_frozen_jobs = set()
            expected_frozen_jobs = {
                (scenario, configuration, policy_seed)
                for scenario in SCENARIOS
                for configuration in CONFIGURATIONS
                for policy_seed in POLICY_SEEDS
            }
            if (
                not isinstance(frozen_checkpoints, list)
                or len(frozen_checkpoints) != EXPECTED_TRAINING_JOBS
                or observed_frozen_jobs != expected_frozen_jobs
            ):
                issues.append(
                    "training freeze checkpoints are not the unique 9 x 5 x 12 grid"
                )
            elif any(
                not _valid_frozen_checkpoint(checkpoint)
                for checkpoint in frozen_checkpoints
            ):
                issues.append(
                    "training freeze has a checkpoint with incomplete provenance"
                )

    if audit is None:
        issues.append("matrix_audit.json is missing or invalid")
    else:
        if audit.get("matrix_name") != MATRIX_NAME:
            issues.append("matrix audit has the wrong matrix identity")
        if spec_hash is not None and audit.get("spec_sha256") != spec_hash:
            issues.append("matrix audit does not bind to the frozen matrix spec")
        if audit.get("training_freeze_sha256") != training_freeze_hash:
            issues.append("matrix audit does not bind to the training freeze")
        if audit.get("matrix_complete") is not True:
            issues.append("matrix audit does not set matrix_complete=true")
        expected_training = nested_int(audit, "expected_training_jobs")
        expected_evaluation = nested_int(audit, "expected_evaluation_jobs")
        expected_rows = nested_int(audit, "expected_evaluation_rows")
        if expected_training != EXPECTED_TRAINING_JOBS:
            issues.append(
                f"audit expects {expected_training} training jobs, not {EXPECTED_TRAINING_JOBS}"
            )
        if expected_evaluation != EXPECTED_TRAINING_JOBS:
            issues.append(
                f"audit expects {expected_evaluation} evaluation jobs, not {EXPECTED_TRAINING_JOBS}"
            )
        if expected_rows != EXPECTED_TEST_ROWS:
            issues.append(f"audit expects {expected_rows} rows, not {EXPECTED_TEST_ROWS}")

    training_completed = nested_int(audit, "counts", "training", "completed")
    evaluation_completed = nested_int(audit, "counts", "evaluation", "completed")
    validated_rows = nested_int(audit, "validated_evaluation_rows")
    if training_completed != EXPECTED_TRAINING_JOBS:
        issues.append(
            f"validated training count is {training_completed}, not {EXPECTED_TRAINING_JOBS}"
        )
    if evaluation_completed != EXPECTED_TRAINING_JOBS:
        issues.append(
            f"validated evaluation count is {evaluation_completed}, not {EXPECTED_TRAINING_JOBS}"
        )
    if validated_rows != EXPECTED_TEST_ROWS:
        issues.append(f"validated test row count is {validated_rows}, not {EXPECTED_TEST_ROWS}")

    if merge is None:
        issues.append("episode_metrics_manifest.json is missing or invalid")
    else:
        if spec_hash is not None and merge.get("spec_sha256") != spec_hash:
            issues.append("merged episode manifest does not bind to the matrix spec")
        if merge.get("training_freeze_sha256") != training_freeze_hash:
            issues.append("merged episode manifest does not bind to the training freeze")
        merge_checks = {
            "included_job_count": EXPECTED_TRAINING_JOBS,
            "expected_job_count": EXPECTED_TRAINING_JOBS,
            "row_count": EXPECTED_TEST_ROWS,
            "expected_row_count": EXPECTED_TEST_ROWS,
        }
        if merge.get("complete") is not True:
            issues.append("merged episode manifest is not complete")
        for key, expected in merge_checks.items():
            observed = nested_int(merge, key)
            if observed != expected:
                issues.append(f"merged {key} is {observed}, not {expected}")
        episode_path = root / "episode_metrics.csv"
        declared_episode_hash = merge.get("csv_sha256")
        if not episode_path.is_file():
            issues.append("merged episode_metrics.csv is missing")
        elif not isinstance(declared_episode_hash, str):
            issues.append("merged episode manifest lacks csv_sha256")
        elif sha256_file(episode_path) != declared_episode_hash:
            issues.append("merged episode_metrics.csv SHA-256 does not match its manifest")

    if statistics is None:
        issues.append("statistical_analysis_manifest.json is missing or invalid")
    else:
        if statistics.get("matrix_name") != MATRIX_NAME:
            issues.append("statistical analysis has the wrong matrix identity")
        if statistics.get("training_freeze_sha256") != training_freeze_hash:
            issues.append("statistical analysis does not bind to the training freeze")
        statistics_checks = {
            "source_episode_rows": EXPECTED_TEST_ROWS,
            "paired_effect_rows": EXPECTED_EFFECT_ROWS,
            "expected_paired_effect_rows": EXPECTED_EFFECT_ROWS,
            "policy_seed_count": len(POLICY_SEEDS),
            "workload_seed_count": 50,
        }
        if statistics.get("status") != "completed":
            issues.append("statistical analysis status is not completed")
        for key, expected in statistics_checks.items():
            observed = nested_int(statistics, key)
            if observed != expected:
                issues.append(f"statistics {key} is {observed}, not {expected}")
        if merge is not None and statistics.get(
            "source_episode_csv_sha256"
        ) != merge.get("csv_sha256"):
            issues.append("statistics source hash does not match the merged episode CSV")
        if statistics.get("protocol_spec_sha256") != spec_hash:
            issues.append("statistical analysis does not bind to the matrix spec")
        if planned_contrast_names(
            statistics.get("planned_contrasts")
        ) != PLANNED_CONTRASTS:
            issues.append("statistics manifest contrasts differ from the frozen protocol")
        current_analysis_code = source_hashes(project_root, ANALYSIS_SOURCE_FILES)
        recorded_analysis_code = statistics.get("analysis_code_audit")
        if current_analysis_code is None:
            issues.append("current statistical-analysis source set is incomplete")
        elif not isinstance(recorded_analysis_code, Mapping) or any(
            not isinstance(recorded_analysis_code.get(name), Mapping)
            or recorded_analysis_code[name].get("sha256") != digest
            for name, digest in current_analysis_code.items()
        ):
            issues.append(
                "current statistical-analysis sources differ from the analysis manifest"
            )

    if not effect_path.is_file():
        issues.append("paired_ablation_effects.csv is missing")
    elif len(effects) != EXPECTED_EFFECT_ROWS:
        issues.append(f"effect table has {len(effects)} rows, not {EXPECTED_EFFECT_ROWS}")
    if effect_path.is_file() and statistics is not None:
        declared_effect_hash = statistics.get("paired_effects_csv_sha256")
        if not isinstance(declared_effect_hash, str):
            issues.append("statistical analysis manifest lacks paired-effects SHA-256")
        elif sha256_file(effect_path) != declared_effect_hash:
            issues.append("paired effect table SHA-256 does not match its manifest")

    if effects:
        missing_columns = REQUIRED_EFFECT_COLUMNS.difference(effects[0])
        if missing_columns:
            issues.append(f"effect table lacks columns {sorted(missing_columns)}")
        else:
            observed_cells: set[tuple[str, str, str]] = set()
            for index, row in enumerate(effects, start=2):
                cell = (row["scenario"], row["contrast"], row["metric"])
                if cell in observed_cells:
                    issues.append(f"duplicate effect cell {cell!r}")
                    break
                observed_cells.add(cell)
                if row["scenario"] not in SCENARIOS:
                    issues.append(f"unexpected scenario {row['scenario']!r} on row {index}")
                    break
                if row["contrast"] not in PLANNED_CONTRASTS:
                    issues.append(f"unexpected contrast {row['contrast']!r} on row {index}")
                    break
                expected_reference, expected_treatment = CONTRAST_VARIANTS[
                    row["contrast"]
                ]
                if (
                    row["reference_variant"] != expected_reference
                    or row["treatment_variant"] != expected_treatment
                ):
                    issues.append(
                        f"row {index} does not use the frozen reference/treatment direction"
                    )
                    break
                if row["metric"] not in ENDPOINTS:
                    issues.append(f"unexpected endpoint {row['metric']!r} on row {index}")
                    break
                if row["paired_policy_seeds"] != str(len(POLICY_SEEDS)):
                    issues.append(f"row {index} does not contain 12 paired policy seeds")
                    break
                if row["paired_workloads"] != "50":
                    issues.append(f"row {index} does not contain 50 paired workloads")
                    break
                if row["paired_episode_cells"] != "600":
                    issues.append(f"row {index} does not contain 600 paired episode cells")
                    break
                if row["difference_ci_method"] != "crossed_pigeonhole_bootstrap_percentile":
                    issues.append(f"row {index} uses an unplanned confidence interval")
                    break
                if row["ci_resamples"] != "5000":
                    issues.append(f"row {index} does not use 5000 bootstrap resamples")
                    break
                if row["p_value_method"] != "exact_policy_seed_sign_flip_on_workload_means":
                    issues.append(f"row {index} uses an unplanned primary test")
                    break
                if row["sign_flip_permutations"] != "4096":
                    issues.append(f"row {index} does not enumerate all 4096 sign assignments")
                    break
                if row["within_metric_family_size"] != "40":
                    issues.append(f"row {index} does not use the frozen 40-test family")
                    break
                try:
                    for column in (
                        "treatment_minus_reference",
                        "difference_ci95_low",
                        "difference_ci95_high",
                        "raw_p_value",
                        "confirmatory_holm_within_metric_p",
                    ):
                        value = finite_float(row, column)
                        if column.endswith("p_value") or column.endswith("_p"):
                            if not 0.0 <= value <= 1.0:
                                raise ValueError(f"{column!r} lies outside [0, 1]")
                    raw_p_numerator = finite_float(row, "raw_p_value") * 4096
                    if (
                        raw_p_numerator < 2
                        or abs(raw_p_numerator - round(raw_p_numerator)) > 1e-9
                    ):
                        raise ValueError(
                            "raw p-value is not attainable by the frozen 4096-way "
                            "two-sided exact test"
                        )
                except ValueError as error:
                    issues.append(f"row {index}: {error}")
                    break

            expected_cells = {
                (scenario, contrast, endpoint)
                for scenario in SCENARIOS
                for contrast in PLANNED_CONTRASTS
                for endpoint in ENDPOINTS
            }
            if observed_cells != expected_cells:
                missing_count = len(expected_cells.difference(observed_cells))
                extra_count = len(observed_cells.difference(expected_cells))
                issues.append(
                    "effect grid is not the frozen 5 x 8 x 8 design "
                    f"({missing_count} missing, {extra_count} extra cells)"
                )

    # Never expose a stale effect table when any gate fails.
    complete = not issues
    return ConfirmatoryAblation(
        complete=complete,
        root=root,
        training_completed=training_completed,
        frozen_training_jobs=frozen_training_jobs,
        evaluation_completed=evaluation_completed,
        validated_rows=validated_rows,
        effects=tuple(effects) if complete else (),
        issues=tuple(issues),
    )


def _validate_source_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    issues: list[str] = []
    required = {"scenario", "policy", "policy_seed", "workload_seed"}
    if not rows or not required.issubset(rows[0]):
        return ["source CSV lacks seed identity columns"]

    observed_cells: set[tuple[str, str, int, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            cell = (
                row["scenario"],
                row["policy"],
                int(row["policy_seed"]),
                int(row["workload_seed"]),
            )
        except (KeyError, TypeError, ValueError):
            issues.append(f"source row {row_number} has invalid seed identity")
            break
        if cell in observed_cells:
            issues.append(f"source CSV has duplicate cell {cell!r}")
            break
        observed_cells.add(cell)
        try:
            numeric_values = (
                value
                for key, value in row.items()
                if key not in {"scenario", "policy"}
            )
            if any(not math.isfinite(float(value)) for value in numeric_values):
                raise ValueError("non-finite")
        except (TypeError, ValueError):
            issues.append(f"source row {row_number} has invalid numeric data")
            break
    return issues


def _validate_reanalysis_rows(
    rows: Sequence[Mapping[str, str]], kind: str
) -> list[str]:
    issues: list[str] = []
    if kind == "headline":
        identity_fields = ("scenario", "baseline", "metric")
        expected_cells = {
            (scenario, baseline, metric)
            for scenario in SCENARIOS
            for baseline in LEGACY_BASELINES
            for metric in LEGACY_METRICS
        }
        numeric_fields = (
            "mappo_mean",
            "baseline_mean",
            "mean_difference",
            "difference_ci95_low",
            "difference_ci95_high",
            "positive_policy_seed_fraction",
            "raw_p_value",
        )
    elif kind == "ablation":
        identity_fields = ("scenario", "contrast", "metric")
        expected_cells = {
            (scenario, contrast, metric)
            for scenario in SCENARIOS
            for contrast in LEGACY_ABLATION_CONTRASTS
            for metric in ENDPOINTS
        }
        numeric_fields = (
            "reference_mean",
            "treatment_mean",
            "treatment_minus_reference",
            "difference_ci95_low",
            "difference_ci95_high",
            "positive_policy_seed_fraction",
            "raw_p_value",
        )
    else:
        return [f"unsupported reanalysis kind {kind!r}"]

    observed_cells: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            cell = tuple(row[field] for field in identity_fields)
            numeric_values = [finite_float(row, field) for field in numeric_fields]
            policy_seed_count = int(row["paired_policy_seeds"])
            workload_count = int(row["paired_workloads"])
            episode_cell_count = int(row["paired_episode_cells"])
            ci_resamples = int(row["ci_resamples"])
            permutations = int(row["sign_flip_permutations"])
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"reanalysis row {row_number} is invalid: {error}")
            break
        if cell in observed_cells:
            issues.append(f"reanalysis CSV has duplicate cell {cell!r}")
            break
        observed_cells.add(cell)
        raw_p = numeric_values[-1]
        exact_tail_count = raw_p * permutations
        if (
            policy_seed_count != 8
            or workload_count != 50
            or episode_cell_count != 400
            or ci_resamples != 5_000
            or permutations != 256
            or row.get("difference_ci_method")
            != "crossed_pigeonhole_bootstrap_percentile"
            or row.get("p_value_method")
            != "exact_policy_seed_sign_flip_on_workload_means"
            or not 0.0 <= raw_p <= 1.0
            or exact_tail_count < 2.0
            or not math.isclose(exact_tail_count, round(exact_tail_count))
        ):
            issues.append(
                f"reanalysis row {row_number} violates the 8 x 50 seed contract"
            )
            break

    if observed_cells != expected_cells:
        issues.append(
            "reanalysis CSV is not the complete expected grid "
            f"({len(expected_cells - observed_cells)} missing, "
            f"{len(observed_cells - expected_cells)} unexpected)"
        )
    return issues


def inspect_reanalysis_artifact(
    project_root: Path,
    *,
    name: str,
    source_relative: str,
    output_relative: str,
    kind: str,
    artifact_name: str,
    expected_filename: str,
    required_columns: set[str],
    expected_inferential_status: str,
    expected_source_rows: int,
    expected_output_rows: int,
) -> ReanalysisArtifact:
    """Bind a rendered table to its source CSV and reanalysis manifest."""

    project_root = project_root.resolve()
    source = project_root / source_relative
    root = project_root / output_relative
    manifest = load_json(root / "statistical_analysis_manifest.json")
    source_data = load_csv(source)
    output_path = root / expected_filename
    output_rows = load_csv(output_path)
    issues: list[str] = []

    if not source_data:
        issues.append(f"source CSV is missing or empty: {source_relative}")
    elif len(source_data) != expected_source_rows:
        issues.append(
            f"source CSV has {len(source_data)} rows, expected {expected_source_rows}"
        )
    else:
        issues.extend(_validate_source_rows(source_data))
    if manifest is None:
        issues.append("statistical analysis manifest is missing or invalid")
    else:
        if manifest.get("analysis") != "crossed_seed_statistical_reanalysis":
            issues.append("manifest identifies the wrong analysis")
        if manifest.get("kind") != kind:
            issues.append("manifest identifies the wrong analysis kind")
        if manifest.get("inferential_status") != expected_inferential_status:
            issues.append("manifest has the wrong inferential status")
        if manifest.get("confirmatory") is not False:
            issues.append("legacy reanalysis must be marked non-confirmatory")
        if source.is_file():
            if manifest.get("source_sha256") != sha256_file(source):
                issues.append("source CSV SHA-256 does not match the manifest")
            if manifest.get("source_row_count") != len(source_data):
                issues.append("source CSV row count does not match the manifest")

        output_files = manifest.get("output_files")
        if (
            not isinstance(output_files, Mapping)
            or output_files.get(artifact_name) != expected_filename
        ):
            issues.append("manifest output filename is missing or unexpected")

        artifacts = manifest.get("output_artifacts")
        artifact = (
            artifacts.get(artifact_name)
            if isinstance(artifacts, Mapping)
            else None
        )
        if not isinstance(artifact, Mapping):
            issues.append("manifest lacks hashed output artifact metadata")
        else:
            if artifact.get("file") != expected_filename:
                issues.append("hashed artifact filename is unexpected")
            if output_path.is_file():
                if artifact.get("sha256") != sha256_file(output_path):
                    issues.append("output CSV SHA-256 does not match the manifest")
                if artifact.get("row_count") != len(output_rows):
                    issues.append("output CSV row count does not match the manifest")

        statistical_analysis = manifest.get("statistical_analysis")
        hypothesis_test = (
            statistical_analysis.get("hypothesis_test")
            if isinstance(statistical_analysis, Mapping)
            else None
        )
        if (
            not isinstance(hypothesis_test, Mapping)
            or hypothesis_test.get("method")
            != "exact_policy_seed_sign_flip_on_workload_means"
            or hypothesis_test.get("episode_rows_treated_as_independent") is not False
        ):
            issues.append("manifest does not record policy-seed-level inference")

    if not output_rows:
        issues.append(f"reanalysis CSV is missing or empty: {expected_filename}")
    elif len(output_rows) != expected_output_rows:
        issues.append(
            f"reanalysis CSV has {len(output_rows)} rows, "
            f"expected {expected_output_rows}"
        )
    else:
        observed_columns = set(output_rows[0])
        missing_columns = required_columns.difference(observed_columns)
        if missing_columns:
            issues.append(
                "reanalysis CSV lacks required columns: "
                + ", ".join(sorted(missing_columns))
            )
        else:
            issues.extend(_validate_reanalysis_rows(output_rows, kind))

    return ReanalysisArtifact(
        complete=not issues,
        name=name,
        source=source,
        root=root,
        source_rows=len(source_data) if source_data else None,
        output_rows=len(output_rows) if output_rows else None,
        rows=tuple(output_rows) if not issues else (),
        issues=tuple(issues),
    )


def inspect_legacy_headline(
    project_root: Path, name: str
) -> ReanalysisArtifact:
    if name == "eval-main":
        source_relative = "experiments/eval-main/episode_metrics.csv"
    elif name == "train-main":
        source_relative = "experiments/train-main/episode_metrics.csv"
    elif name == "repro-check":
        source_relative = "experiments/archive/repro-check/episode_metrics.csv"
    else:
        raise ValueError(f"unknown legacy headline result {name!r}")
    return inspect_reanalysis_artifact(
        project_root,
        name=name,
        source_relative=source_relative,
        output_relative=f"experiments/legacy-reanalysis/{name}",
        kind="headline",
        artifact_name="comparisons",
        expected_filename="paired_tests.csv",
        required_columns=REQUIRED_HEADLINE_COLUMNS,
        expected_inferential_status="headline_reanalysis",
        expected_source_rows=5_250,
        expected_output_rows=375,
    )


def inspect_legacy_ablation(project_root: Path) -> ReanalysisArtifact:
    return inspect_reanalysis_artifact(
        project_root,
        name="ablation-5k",
        source_relative="experiments/ablation/episode_metrics.csv",
        output_relative="experiments/legacy-reanalysis/ablation-5k",
        kind="ablation",
        artifact_name="comparisons",
        expected_filename="paired_ablation_effects.csv",
        required_columns=REQUIRED_EFFECT_COLUMNS,
        expected_inferential_status="legacy_exploratory_reanalysis",
        expected_source_rows=14_000,
        expected_output_rows=240,
    )


def inspect_legacy_provenance(project_root: Path) -> LegacyProvenance:
    """Verify how the corrected evaluation and archived rerun are related."""

    manifest_paths = {
        "eval-main": project_root / "experiments" / "eval-main" / (
            "experiment_manifest.json"
        ),
        "train-main": project_root / "experiments" / "train-main" / (
            "experiment_manifest.json"
        ),
        "repro-check": project_root / "experiments" / "archive" / (
            "repro-check"
        ) / "experiment_manifest.json",
    }
    manifests = {name: load_json(path) for name, path in manifest_paths.items()}
    issues: list[str] = []
    for name, manifest in manifests.items():
        if manifest is None:
            issues.append(f"{name} experiment manifest is missing or invalid")

    if issues:
        return LegacyProvenance(False, tuple(issues))

    expected_keys = {
        f"{scenario}/seed_{policy_seed}"
        for scenario in SCENARIOS
        for policy_seed in LEGACY_POLICY_SEEDS
    }
    checkpoints: dict[str, Mapping[str, Any]] = {}
    fingerprints: dict[str, Mapping[str, Any]] = {}
    for name, manifest in manifests.items():
        assert manifest is not None
        checkpoint_map = manifest.get("checkpoints")
        code_fingerprint = manifest.get("code_fingerprint")
        config = manifest.get("config")
        try:
            observed_policy_seeds = {int(seed) for seed in manifest["policy_seeds"]}
        except (KeyError, TypeError, ValueError):
            observed_policy_seeds = set()
        if (
            not isinstance(checkpoint_map, Mapping)
            or set(checkpoint_map) != expected_keys
            or not all(isinstance(path, str) and path for path in checkpoint_map.values())
        ):
            issues.append(f"{name} does not record the expected 40 checkpoints")
        else:
            checkpoints[name] = checkpoint_map
        if not isinstance(code_fingerprint, Mapping):
            issues.append(f"{name} lacks code fingerprints")
        else:
            fingerprints[name] = code_fingerprint
        if (
            observed_policy_seeds != set(LEGACY_POLICY_SEEDS)
            or not isinstance(config, Mapping)
            or config.get("timesteps") != 50_000
            or config.get("test_episodes") != 50
        ):
            issues.append(f"{name} has unexpected seed or budget metadata")

    if len(checkpoints) == 3:
        if dict(checkpoints["eval-main"]) != dict(checkpoints["train-main"]):
            issues.append("eval-main and train-main do not name the same checkpoints")
        if dict(checkpoints["repro-check"]) == dict(checkpoints["train-main"]):
            issues.append("repro-check does not name a separate checkpoint set")
        eval_paths = [
            path.replace("\\", "/") for path in checkpoints["eval-main"].values()
        ]
        repro_paths = [
            path.replace("\\", "/")
            for path in checkpoints["repro-check"].values()
        ]
        if not all("/no_lifetime/" in path for path in eval_paths):
            issues.append("eval-main checkpoint paths do not identify no_lifetime")
        if any("/no_lifetime/" in path for path in repro_paths):
            issues.append("repro-check unexpectedly identifies the final variant")

    if len(fingerprints) == 3:
        old_wrapper = fingerprints["train-main"].get("wrapper")
        old_evaluation = fingerprints["train-main"].get("evaluation")
        if (
            fingerprints["repro-check"].get("wrapper") != old_wrapper
            or fingerprints["repro-check"].get("evaluation") != old_evaluation
        ):
            issues.append("repro-check does not use the train-main evaluator version")
        if (
            fingerprints["eval-main"].get("wrapper") == old_wrapper
            or fingerprints["eval-main"].get("evaluation") == old_evaluation
        ):
            issues.append("eval-main does not identify the corrected evaluator version")

    return LegacyProvenance(not issues, tuple(issues))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""

    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    family_size = len(p_values)
    for rank, (index, value) in enumerate(indexed):
        running_max = max(running_max, min(1.0, (family_size - rank) * value))
        adjusted[index] = running_max
    return adjusted


def comparison_rows(
    rows: Sequence[Mapping[str, str]], baseline: str, metric: str
) -> list[Mapping[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("baseline") == baseline and row.get("metric") == metric
    ]
    selected.sort(key=lambda row: SCENARIOS.index(row["scenario"]))
    return selected


def strongest_delivery_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    selected: list[Mapping[str, str]] = []
    for scenario in SCENARIOS:
        candidates = [
            row
            for row in rows
            if row.get("scenario") == scenario
            and row.get("metric") == "delivery_ratio"
        ]
        if not candidates:
            return []
        selected.append(
            max(candidates, key=lambda row: finite_float(row, "baseline_mean"))
        )
    return selected


def format_p(value: float) -> str:
    """Print the observed p-value itself; never substitute a threshold."""

    return repr(value)


def build_summary(project_root: Path) -> str:
    project_root = project_root.resolve()
    headline = inspect_legacy_headline(project_root, "eval-main")
    archived_reproduction = inspect_legacy_headline(project_root, "repro-check")
    legacy_ablation = inspect_legacy_ablation(project_root)
    provenance = inspect_legacy_provenance(project_root)
    confirmatory = inspect_confirmatory_ablation(project_root)
    lines = [
        "# LEO MAPPO Routing - Completed Legacy Results",
        "",
        "> Default evidence package: the completed 8-seed, 50k-step `eval-main` "
        "evaluation. No new training is required to reproduce this summary.",
        "> Evidence status: retrospective reanalysis, not preregistered confirmatory "
        "evidence and not by itself a guarantee of IEEE top-journal acceptance.",
        "> Inference unit: independently trained policy seed. The 50 common workload "
        "seeds are paired repeated measurements, not 50 independent policy runs.",
        "> Historical labels are aliases only: `no_lifetime` means `proposed` (L0), "
        "while `full` means `with_hard_lifetime_mask` (L3).",
        "",
        "## 1. Corrected completed 50k evaluation",
        "",
    ]

    if headline.complete:
        strongest = strongest_delivery_rows(headline.rows)
        if len(strongest) != len(SCENARIOS):
            lines.append(
                "The verified artifact does not contain one delivery comparison per "
                "scenario, so no headline table is rendered."
            )
        else:
            raw_p_values = [finite_float(row, "raw_p_value") for row in strongest]
            adjusted_p_values = holm_adjust(raw_p_values)
            lines += [
                f"Verified source: `{headline.source.relative_to(project_root).as_posix()}` "
                f"({headline.source_rows} episode rows). The table below selects the "
                "highest-delivery baseline in each scenario; in this artifact it is "
                "Q-routing in all five scenarios.",
                "",
                "| scenario | MAPPO | strongest baseline | gap, pp [crossed 95% CI] | MAPPO-higher seeds | exact p | retrospective Holm p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row, adjusted_p in zip(strongest, adjusted_p_values):
                policy_seed_count = int(row["paired_policy_seeds"])
                positive_count = round(
                    policy_seed_count
                    * finite_float(row, "positive_policy_seed_fraction")
                )
                effect = 100.0 * finite_float(row, "mean_difference")
                low = 100.0 * finite_float(row, "difference_ci95_low")
                high = 100.0 * finite_float(row, "difference_ci95_high")
                lines.append(
                    f"| {row['scenario']} | {finite_float(row, 'mappo_mean'):.4f} | "
                    f"{row['baseline']} {finite_float(row, 'baseline_mean'):.4f} | "
                    f"{effect:+.3f} [{low:+.3f}, {high:+.3f}] | "
                    f"{positive_count}/{policy_seed_count} | "
                    f"{format_p(finite_float(row, 'raw_p_value'))} | "
                    f"{format_p(adjusted_p)} |"
                )

            wins = [
                row["scenario"]
                for row in strongest
                if finite_float(row, "difference_ci95_low") > 0.0
            ]
            losses = [
                row["scenario"]
                for row in strongest
                if finite_float(row, "difference_ci95_high") < 0.0
            ]
            unclear = [
                row["scenario"]
                for row in strongest
                if finite_float(row, "difference_ci95_low") <= 0.0
                <= finite_float(row, "difference_ci95_high")
            ]
            lines += [
                "",
                "Interpretation: MAPPO has small delivery gains in "
                f"{', '.join(wins)}, no clear delivery difference in "
                f"{', '.join(unclear)}, and a consistent across-seed loss in "
                f"{', '.join(losses)}. These Holm values adjust the five "
                "retrospectively selected strongest-baseline comparisons; they are "
                "sensitivity values, not confirmatory decisions.",
            ]

        switch_rows = comparison_rows(
            headline.rows, "q_routing", "routing_switches"
        )
        if len(switch_rows) == len(SCENARIOS):
            absolute_reductions = [
                -finite_float(row, "mean_difference") for row in switch_rows
            ]
            relative_reductions = [
                100.0
                * -finite_float(row, "mean_difference")
                / finite_float(row, "baseline_mean")
                for row in switch_rows
            ]
            switch_adjusted = holm_adjust(
                [finite_float(row, "raw_p_value") for row in switch_rows]
            )
            lines += [
                "",
                "Against Q-routing, MAPPO reduces routing switches in all five "
                f"scenarios by {min(absolute_reductions):.2f} to "
                f"{max(absolute_reductions):.2f} switches per episode "
                f"({min(relative_reductions):.1f}% to "
                f"{max(relative_reductions):.1f}%). Every scenario is 8/8 seeds in "
                "the same direction; each exact raw p is 0.0078125 and each "
                f"retrospective five-scenario Holm p is {format_p(switch_adjusted[0])}.",
            ]
    else:
        lines += [
            "**Status: the corrected legacy reanalysis is unavailable or failed its "
            "hash/row-count audit.**",
            "",
        ]
        lines.extend(f"- {issue}" for issue in headline.issues)

    lines += [
        "",
        "## 2. Provenance and archived reproduction check",
        "",
    ]

    if provenance.complete:
        lines += [
            "The three experiment manifests verify that `eval-main` re-evaluates the "
            "same 40 completed `train-main/no_lifetime` checkpoints with the corrected "
            "wrapper/evaluator. It is a corrected evaluation, not an independent "
            "retraining replication.",
            "",
            "The separately trained archived directory named `repro-check` is weaker "
            "evidence than its name suggests: its manifest does not record the method "
            "variant, its checkpoint paths do not identify `no_lifetime`, and it uses "
            "the legacy wrapper/evaluator. It is therefore shown only as a version-"
            "sensitivity check and cannot validate the final method configuration.",
            "",
        ]
    else:
        lines += [
            "Checkpoint/evaluator provenance could not be verified, so no claim of "
            "independent reproduction is made:",
            "",
        ]
        lines.extend(f"- {issue}" for issue in provenance.issues)
        lines.append("")

    if headline.complete and archived_reproduction.complete:
        current_dijkstra = comparison_rows(
            headline.rows, "global_dijkstra", "delivery_ratio"
        )
        archived_dijkstra = comparison_rows(
            archived_reproduction.rows, "global_dijkstra", "delivery_ratio"
        )
        if len(current_dijkstra) == len(SCENARIOS) and len(
            archived_dijkstra
        ) == len(SCENARIOS):
            lines += [
                "| scenario | corrected eval-main gap vs Dijkstra, pp | archived run gap vs Dijkstra, pp | archived exact p |",
                "|---|---:|---:|---:|",
            ]
            for current, archived in zip(current_dijkstra, archived_dijkstra):
                lines.append(
                    f"| {current['scenario']} | "
                    f"{100.0 * finite_float(current, 'mean_difference'):+.2f} | "
                    f"{100.0 * finite_float(archived, 'mean_difference'):+.2f} | "
                    f"{format_p(finite_float(archived, 'raw_p_value'))} |"
                )
            lines += [
                "",
                "The archived run is materially weaker in several scenarios and even "
                "changes direction in low load. This discrepancy must remain visible "
                "in any paper or response to reviewers.",
            ]
    else:
        lines.append(
            "The archived comparison table is not rendered because one of its hashed "
            "reanalysis artifacts is unavailable or invalid."
        )

    lines += [
        "",
        "## 3. Legacy 5k ablation (exploratory only)",
        "",
        "The older `experiments/ablation/` matrix used only 5,000 training steps "
        "(10% of the headline budget). It is a legacy pilot and cannot support a "
        "formal component-necessity claim.",
        "",
    ]

    if legacy_ablation.complete:
        pilot_rows = [
            row
            for row in legacy_ablation.rows
            if row.get("contrast") == "legacy_no_lifetime_vs_full"
            and row.get("metric") == "delivery_ratio"
        ]
        pilot_rows.sort(key=lambda row: SCENARIOS.index(row["scenario"]))
        if len(pilot_rows) == len(SCENARIOS):
            lines += [
                "The most directly relevant old branch comparison is shown only to "
                "document why L0 (`no_lifetime`/`proposed`) was retained. Effects are "
                "L0 minus legacy L3 (`full`).",
                "",
                "| scenario | exploratory delivery effect, pp [crossed 95% CI] | exact raw p |",
                "|---|---:|---:|",
            ]
            for row in pilot_rows:
                effect = 100.0 * finite_float(row, "treatment_minus_reference")
                low = 100.0 * finite_float(row, "difference_ci95_low")
                high = 100.0 * finite_float(row, "difference_ci95_high")
                lines.append(
                    f"| {row['scenario']} | {effect:+.2f} "
                    f"[{low:+.2f}, {high:+.2f}] | "
                    f"{format_p(finite_float(row, 'raw_p_value'))} |"
                )
    else:
        lines.append(
            "The pilot table is not rendered because its hashed reanalysis artifact "
            "is unavailable or invalid."
        )

    lines += [
        "",
        "## 4. Optional paused 50k ablation",
        "",
        "The 9 configurations x 5 scenarios x 12 policy seeds study is not part of "
        "the default workflow. Its partial files are retained only so the study can "
        "be resumed later if explicitly requested.",
        "",
    ]

    if not confirmatory.complete:
        lines += [
            "**Status: incomplete; no confirmatory ablation estimate is reportable.**",
            "",
            "| audited item | observed | required |",
            "|---|---:|---:|",
            f"| completed training jobs | {confirmatory.training_completed if confirmatory.training_completed is not None else 'missing'} | {EXPECTED_TRAINING_JOBS} |",
            f"| frozen checkpoint records | {confirmatory.frozen_training_jobs if confirmatory.frozen_training_jobs is not None else 'missing'} | {EXPECTED_TRAINING_JOBS} |",
            f"| completed evaluations | {confirmatory.evaluation_completed if confirmatory.evaluation_completed is not None else 'missing'} | {EXPECTED_TRAINING_JOBS} |",
            f"| finite, unique test rows | {confirmatory.validated_rows if confirmatory.validated_rows is not None else 'missing'} | {EXPECTED_TEST_ROWS} |",
            "",
            "Publication gates not yet satisfied:",
            "",
        ]
        lines.extend(f"- {issue}" for issue in confirmatory.issues[:8])
        if len(confirmatory.issues) > 8:
            lines.append(
                f"- ... and {len(confirmatory.issues) - 8} additional audit failures"
            )
    else:
        primary = [
            row
            for row in confirmatory.effects
            if row["metric"] == "delivery_ratio"
        ]
        primary.sort(
            key=lambda row: (
                SCENARIOS.index(row["scenario"]),
                PLANNED_CONTRASTS.index(row["contrast"]),
            )
        )
        lines += [
            "**Status: complete and audit-validated.** Effects are treatment minus "
            "reference; all p-values are printed exactly.",
            "",
            "| scenario | planned contrast | reference -> treatment | effect, pp [crossed 95% CI] | exact p | primary Holm p |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in primary:
            effect = 100.0 * finite_float(row, "treatment_minus_reference")
            low = 100.0 * finite_float(row, "difference_ci95_low")
            high = 100.0 * finite_float(row, "difference_ci95_high")
            raw_p = finite_float(row, "raw_p_value")
            holm_p = finite_float(row, "confirmatory_holm_within_metric_p")
            lines.append(
                f"| {row['scenario']} | {row['contrast']} | "
                f"{row['reference_variant']} -> {row['treatment_variant']} | "
                f"{effect:+.2f} [{low:+.2f}, {high:+.2f}] | "
                f"{format_p(raw_p)} | {format_p(holm_p)} |"
            )

    lines += [
        "",
        "## 5. Reporting boundary",
        "",
        "The previous extremely small headline and fairness p-value statements came "
        "from treating repeated workload episodes as independent observations. They "
        "are pseudoreplicated, are not valid policy-seed-level evidence, and are not "
        "reproduced here. The corrected analysis first averages each paired contrast "
        "within policy seed, then applies a two-sided exact sign-flip test across the "
        "eight independently trained seeds. With eight seeds the smallest possible "
        "two-sided exact p-value is 0.0078125.",
        "",
        "The completed data are usable for a manuscript if the claims match the table: "
        "small, scenario-dependent delivery effects; a consistent routing-stability "
        "advantage; an explicit hotspot failure case; no formal 50k component proof; "
        "and no claim that this package alone establishes top-journal-level evidence.",
        "",
        "Reanalysis manifests and hashed outputs: "
        "`experiments/legacy-reanalysis/`.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown path (default: <project-root>/RESULTS_SUMMARY.md)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output = args.output or project_root / "RESULTS_SUMMARY.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_summary(project_root), encoding="utf-8")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
