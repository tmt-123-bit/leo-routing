"""Run the frozen QoS-only avoidable-switch constraint mechanism smoke."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import torch

from ablation_matrix_runner import atomic_write_json, sha256_file
from mappo_design import select_leo_validation_record


SCHEMA_VERSION = 1
STUDY_NAME = "ICC-AVOIDABLE-SWITCH-CONSTRAINT-SMOKE-v1"
PROTOCOL_FILENAME = "AVOIDABLE_SWITCH_CONSTRAINT_SMOKE_V1.md"
DEFAULT_OUTPUT_DIRECTORY_NAME = "avoidable-switch-constraint-smoke-v1"

SCENARIOS = ("medium_load", "hotspot_high_load")
ARM_BASELINE = "qos_only_baseline"
ARM_CONSTRAINED = "qos_only_constrained"
ARMS = (ARM_BASELINE, ARM_CONSTRAINED)
ENVIRONMENT_VARIANT = "qos_only"

SMOKE_POLICY_SEED_NAMESPACE = (
    "ICC-AVOIDABLE-SWITCH-CONSTRAINT-v1-smoke-policy-seed-"
)
SMOKE_POLICY_SEED = (
    int.from_bytes(
        hashlib.sha256(
            f"{SMOKE_POLICY_SEED_NAMESPACE}0".encode("ascii")
        ).digest()[:4],
        "big",
    )
    & 0x7FFFFFFF
)

SMOKE_TRAIN_WORKLOAD_SEEDS = tuple(range(75001, 75009))
SMOKE_VALIDATION_WORKLOAD_SEEDS = tuple(range(75101, 75105))
FORMAL_TRAIN_WORKLOAD_SEEDS = tuple(range(76001, 76201))
FORMAL_VALIDATION_WORKLOAD_SEEDS = tuple(range(77001, 77021))
SEALED_TEST_WORKLOAD_SEEDS = tuple(range(78001, 78051))

RETIRED_WORKLOAD_RANGES = (
    (9001, 9200),
    (10001, 10050),
    (11001, 11050),
    (12001, 12050),
    (13001, 13050),
    (14001, 14050),
    (16001, 16020),
    (17001, 17020),
    (18001, 18015),
    (19001, 19015),
    (21001, 21020),
    (31001, 31050),
    (32001, 32020),
    (33001, 33050),
    (34001, 34020),
    (35001, 35050),
    (37001, 37050),
    (41001, 41010),
    (42001, 42025),
    (43001, 43010),
    (44001, 44025),
    (45001, 45002),
    (46001, 46010),
    (47001, 47025),
    (48001, 48010),
    (49001, 49025),
    (50001, 50010),
    (51001, 51025),
    (60001, 60010),
    (61001, 61025),
    (62001, 62002),
    (70001, 70025),
    (71001, 71010),
    (72001, 72025),
)

TIMESTEPS = 6000
BATCH_SIZE = 4
EPOCHS = 3
NUM_MINIBATCHES = 4
VALIDATION_EVERY_ROLLOUTS = 5
SAVE_EVERY_STEPS = 1500
SWITCH_BUDGET = 0.12
DUAL_LEARNING_RATE = 0.05
DUAL_INITIAL = 0.0
DUAL_MAX = 5.0

SELECTED_VALIDATION_METRIC_FIELDS = (
    "environment_steps",
    "episodes",
    "seed_start",
    "delivery_ratio",
    "mean_reward",
    "drop_rate",
    "average_delay_slots",
    "routing_switches",
    "routing_switches_total",
    "avoidable_routing_switches",
    "avoidable_routing_switches_total",
    "forced_routing_switches",
    "forced_routing_switches_total",
    "switch_opportunities",
    "avoidable_switch_rate",
    "class_2_delivery_ratio",
    "decision_avoidable_switches",
    "decision_switch_opportunities",
    "decision_avoidable_switch_rate",
)

FROZEN_TRAINING_SEMANTICS = {
    "env_family": "mpe",
    "agent_ids": True,
    "actor_hidden_dim": 32,
    "actor_num_layers": 1,
    "critic_hidden_dim": 64,
    "critic_num_layers": 1,
    "optimizer": "Adam",
    "learning_rate_actor": 0.0008,
    "learning_rate_critic": 0.0008,
    "lr_decay": True,
    "gamma": 0.99,
    "td_lambda": 0.95,
    "normalize_reward": False,
    "normalize_advantage": True,
    "normalize_return": True,
    "ppo_clip": 0.2,
    "entropy_coef": 0.01,
    "clip_gradients": 1.0,
    "critic_clip_gradients": 10.0,
    "normalization_epsilon": 1e-8,
    "target_kl": 0.02,
    "candidate_shared_actor": False,
    "route_hysteresis_beta": 0.0,
    "route_hysteresis_mode": "legacy_additive",
    "route_urgency_feature_index": 20,
    "route_class_2_feature_index": 23,
    "route_hysteresis_urgency_relief": 0.0,
    "route_hysteresis_class_2_relief": 0.0,
    "route_hysteresis_residual_init": 0.0,
    "route_hysteresis_residual_cap": 0.0,
    "route_hysteresis_residual_parameterization": "scalar",
    "avoidable_switch_regularization_mode": "conditional_probability",
    "avoidable_switch_logit_margin": 0.0,
    "log_loss_component_gradients": False,
    "validation_delivery_tolerance": 0.0,
    "validation_class_2_tolerance": 0.0,
}

HISTORICAL_EXPERIMENT_DIRECTORY_NAMES = (
    "proposed-shield-preflight-v7",
    "adaptive-shield-design-v8",
    "source-runtime-equivalence-v8",
    "calm-selective-shield-design-v9",
    "margin-capped-shield-design-v10",
    "age-band-shield-design-v11",
)


class ConfigurationError(ValueError):
    """Raised when runtime state differs from the frozen smoke contract."""


class TrainingCancelled(RuntimeError):
    """Raised inside a worker when the runner is shutting down."""


@dataclass(frozen=True)
class SmokeJob:
    index: int
    scenario: str
    arm: str
    policy_seed: int = SMOKE_POLICY_SEED

    @property
    def constrained(self) -> bool:
        return self.arm == ARM_CONSTRAINED

    @property
    def job_id(self) -> str:
        return f"{self.scenario}/{self.arm}/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return f"{self.scenario}__{self.arm}__seed_{self.policy_seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "scenario": self.scenario,
            "arm": self.arm,
            "policy_seed": self.policy_seed,
            "environment_variant": ENVIRONMENT_VARIANT,
            "constraint_enabled": self.constrained,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = sha256_json(result)
    return result


def validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    body = dict(record)
    observed = body.pop(field, None)
    if observed != sha256_json(body):
        raise ConfigurationError(f"{field} mismatch")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"expected JSON object: {path}")
    return value


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(value):
            raise ConfigurationError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, dict(value))


def all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return False


def build_jobs() -> list[SmokeJob]:
    return [
        SmokeJob(index, scenario, arm)
        for index, (scenario, arm) in enumerate(
            (scenario, arm) for scenario in SCENARIOS for arm in ARMS
        )
    ]


def _expand_ranges(ranges: Iterable[tuple[int, int]]) -> set[int]:
    return {
        value
        for start, end in ranges
        for value in range(int(start), int(end) + 1)
    }


def validate_seed_registry() -> None:
    panels = {
        "smoke_train": set(SMOKE_TRAIN_WORKLOAD_SEEDS),
        "smoke_validation": set(SMOKE_VALIDATION_WORKLOAD_SEEDS),
        "formal_train": set(FORMAL_TRAIN_WORKLOAD_SEEDS),
        "formal_validation": set(FORMAL_VALIDATION_WORKLOAD_SEEDS),
        "sealed_test": set(SEALED_TEST_WORKLOAD_SEEDS),
    }
    names = list(panels)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if panels[left] & panels[right]:
                raise ConfigurationError(f"seed panels overlap: {left}/{right}")
    retired = _expand_ranges(RETIRED_WORKLOAD_RANGES)
    overlaps = {
        name: sorted(values & retired)
        for name, values in panels.items()
        if values & retired
    }
    if overlaps:
        raise ConfigurationError(f"fresh seed panel intersects denylist: {overlaps}")
    if SMOKE_POLICY_SEED != 197359353:
        raise ConfigurationError("smoke policy-seed derivation drifted")


def _git_output(repository_root: Path, arguments: Sequence[str]) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return process.stdout


def code_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.project.parent.resolve()
    paths = {
        "runner": Path(__file__).resolve(),
        "protocol": repository_root / "docs" / PROTOCOL_FILENAME,
        "trainer_snapshot": args.project / "cleanmarl_mappo_leo.py",
        "external_trainer": args.cleanmarl / "cleanmarl" / "mappo.py",
        "design_snapshot": args.project / "mappo_design.py",
        "external_design": args.cleanmarl / "cleanmarl" / "mappo_design.py",
        "environment": args.project / "leo_multiagent_env.py",
        "base_environment": args.project / "leo_marl_env.py",
        "wrapper": args.project / "cleanmarl_leo_multiagent_wrapper.py",
        "variant_definitions": args.project / "variant_definitions.py",
        "evaluation": args.project / "mappo_evaluation.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ConfigurationError(f"runtime source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes["trainer_snapshot"] != hashes["external_trainer"]:
        raise ConfigurationError("external trainer differs from repository snapshot")
    if hashes["design_snapshot"] != hashes["external_design"]:
        raise ConfigurationError("external design differs from repository snapshot")
    relevant = [
        str(path.relative_to(repository_root)).replace("\\", "/")
        for name, path in paths.items()
        if name not in {"external_trainer", "external_design"}
    ]
    diff = _git_output(repository_root, ["diff", "--binary", "HEAD", "--", *relevant])
    return {
        "files": hashes,
        "git_head": _git_output(repository_root, ["rev-parse", "HEAD"]).strip(),
        "relevant_dirty_diff_sha256": hashlib.sha256(
            diff.encode("utf-8")
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def constraint_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "type": "projected_lagrangian_avoidable_switch_rate",
        "decision_stage": "pre_contention",
        "normalization": "rollout_micro_ratio_of_sums_over_opportunities",
        "budget": SWITCH_BUDGET,
        "dual_learning_rate": DUAL_LEARNING_RATE,
        "dual_initial": DUAL_INITIAL,
        "dual_projection": [0.0, DUAL_MAX],
        "dual_update_timing": "once_after_each_completed_ppo_rollout",
        "zero_opportunity_update": "skip",
        "forced_reroutes_counted": False,
        "first_route_counted": False,
        "no_op_action_index": 0,
        "reward_objective": "qos_only_without_switch_reward",
        "leo_variant": ENVIRONMENT_VARIANT,
    }


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    validate_seed_registry()
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "series_status": "independent_from_terminated_v7_v11_series",
        "inferential_status": "mechanism_smoke_only",
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "environment_variant": ENVIRONMENT_VARIANT,
        "arms": list(ARMS),
        "scenarios": list(SCENARIOS),
        "policy_seed_namespace": SMOKE_POLICY_SEED_NAMESPACE,
        "policy_seeds": [SMOKE_POLICY_SEED],
        "seed_registry": {
            "smoke_train": list(SMOKE_TRAIN_WORKLOAD_SEEDS),
            "smoke_validation": list(SMOKE_VALIDATION_WORKLOAD_SEEDS),
            "formal_train": list(FORMAL_TRAIN_WORKLOAD_SEEDS),
            "formal_validation": list(FORMAL_VALIDATION_WORKLOAD_SEEDS),
            "sealed_test": list(SEALED_TEST_WORKLOAD_SEEDS),
            "retired_ranges": [list(item) for item in RETIRED_WORKLOAD_RANGES],
        },
        "constraint": constraint_contract(),
        "training": {
            "timesteps": TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "num_minibatches": NUM_MINIBATCHES,
            "validation_every_rollouts": VALIDATION_EVERY_ROLLOUTS,
            "validation_episodes": len(SMOKE_VALIDATION_WORKLOAD_SEEDS),
            "save_every_steps": SAVE_EVERY_STEPS,
            "maximum_parallel_jobs": 2,
            "exact_resume_checkpoint": "latest.pt",
            "frozen_semantics": dict(FROZEN_TRAINING_SEMANTICS),
        },
        "mechanism_gates": [
            "positive_opportunity_denominator_in_each_constrained_scenario",
            "cost_not_greater_than_opportunity",
            "forced_reroute_constraint_cost_is_zero",
            "dual_transition_exactly_recomputable",
            "all_logged_numeric_values_finite",
            "exact_resume_equivalence_unit_test_passed",
            "sealed_test_access_count_zero",
        ],
        "jobs": [job.as_dict() for job in build_jobs()],
        "expected_training_jobs": 4,
        "expected_external_test_evaluations": 0,
        "paths": {
            "project": str(args.project.resolve()),
            "cleanmarl": str(args.cleanmarl.resolve()),
            "protocol": str(
                (args.project.parent / "docs" / PROTOCOL_FILENAME).resolve()
            ),
        },
        "code_fingerprint": code_fingerprint(args),
    }
    return self_hashed(body, "spec_sha256")


def expected_run_config(job: SmokeJob, args: argparse.Namespace) -> dict[str, Any]:
    return {
        **FROZEN_TRAINING_SEMANTICS,
        "env_type": "leo_multi",
        "env_name": job.scenario,
        "leo_project_path": str(args.project),
        "leo_variant": ENVIRONMENT_VARIANT,
        "seed": job.policy_seed,
        "batch_size": BATCH_SIZE,
        "total_timesteps": TIMESTEPS,
        "epochs": EPOCHS,
        "num_minibatches": NUM_MINIBATCHES,
        "eval_steps": VALIDATION_EVERY_ROLLOUTS,
        "num_eval_ep": len(SMOKE_VALIDATION_WORKLOAD_SEEDS),
        "save_every_steps": SAVE_EVERY_STEPS,
        "train_seed_start": SMOKE_TRAIN_WORKLOAD_SEEDS[0],
        "train_seed_count": len(SMOKE_TRAIN_WORKLOAD_SEEDS),
        "validation_seed_start": SMOKE_VALIDATION_WORKLOAD_SEEDS[0],
        "device": args.device,
        "avoidable_switch_probability_coef": 0.0,
        "avoidable_switch_constraint_enabled": job.constrained,
        "avoidable_switch_budget": SWITCH_BUDGET,
        "avoidable_switch_dual_learning_rate": DUAL_LEARNING_RATE,
        "avoidable_switch_dual_initial": DUAL_INITIAL,
        "avoidable_switch_dual_max": DUAL_MAX,
        "avoidable_switch_reduction": (
            "rollout_micro_mean"
            if job.constrained
            else "minibatch_conditional_mean"
        ),
        "validation_selection_mode": (
            "avoidable_switch_budget_constrained"
            if job.constrained
            else "legacy_lexicographic"
        ),
    }


def build_command(
    job: SmokeJob,
    args: argparse.Namespace,
    run_root: Path,
    *,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(args.cleanmarl / "cleanmarl" / "mappo.py"),
        "--env-type",
        "leo_multi",
        "--env-name",
        job.scenario,
        "--leo-project-path",
        str(args.project),
        "--leo-variant",
        ENVIRONMENT_VARIANT,
        "--seed",
        str(job.policy_seed),
        "--batch-size",
        str(BATCH_SIZE),
        "--total-timesteps",
        str(TIMESTEPS),
        "--epochs",
        str(EPOCHS),
        "--num-minibatches",
        str(NUM_MINIBATCHES),
        "--eval-steps",
        str(VALIDATION_EVERY_ROLLOUTS),
        "--num-eval-ep",
        str(len(SMOKE_VALIDATION_WORKLOAD_SEEDS)),
        "--save-every-steps",
        str(SAVE_EVERY_STEPS),
        "--checkpoint-dir",
        str(run_root),
        "--run-tag",
        f"{STUDY_NAME}-{job.arm}",
        "--train-seed-start",
        str(SMOKE_TRAIN_WORKLOAD_SEEDS[0]),
        "--train-seed-count",
        str(len(SMOKE_TRAIN_WORKLOAD_SEEDS)),
        "--validation-seed-start",
        str(SMOKE_VALIDATION_WORKLOAD_SEEDS[0]),
        "--validation-selection-mode",
        (
            "avoidable_switch_budget_constrained"
            if job.constrained
            else "legacy_lexicographic"
        ),
        "--device",
        args.device,
    ]
    if job.constrained:
        command.extend(
            [
                "--avoidable-switch-constraint-enabled",
                "--avoidable-switch-budget",
                str(SWITCH_BUDGET),
                "--avoidable-switch-dual-learning-rate",
                str(DUAL_LEARNING_RATE),
                "--avoidable-switch-dual-initial",
                str(DUAL_INITIAL),
                "--avoidable-switch-dual-max",
                str(DUAL_MAX),
                "--avoidable-switch-reduction",
                "rollout_micro_mean",
            ]
        )
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint.resolve())])
    return command


def _path_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def validate_output_isolation(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    project_source = args.project.resolve()
    repository_root = project_source.parent
    cleanmarl_root = args.cleanmarl.resolve()
    protected_paths = {
        "project source": project_source,
        "repository metadata": repository_root / ".git",
        "protocol documents": repository_root / "docs",
        "CleanMARL runtime": cleanmarl_root,
    }
    experiments_root = repository_root / "experiments"
    protected_paths.update(
        {
            f"historical experiment {name}": experiments_root / name
            for name in HISTORICAL_EXPERIMENT_DIRECTORY_NAMES
        }
    )
    overlaps = [
        label
        for label, protected in protected_paths.items()
        if _paths_overlap(output, protected)
    ]
    if output == repository_root or output in repository_root.parents:
        overlaps.append("repository root or its ancestor")
    if overlaps:
        raise ConfigurationError(
            "output path overlaps protected project evidence/runtime paths: "
            f"{sorted(set(overlaps))}"
        )


def job_root(args: argparse.Namespace, job: SmokeJob) -> Path:
    return (
        args.output
        / "checkpoints"
        / job.scenario
        / job.arm
        / f"seed_{job.policy_seed}"
    )


def status_path(args: argparse.Namespace, job: SmokeJob) -> Path:
    return args.output / "job_status" / f"{job.slug}.json"


def _run_config_matches(
    config: Mapping[str, Any],
    job: SmokeJob,
    args: argparse.Namespace,
) -> bool:
    return all(
        config.get(field) == expected
        for field, expected in expected_run_config(job, args).items()
    )


def discover_run_directory(
    args: argparse.Namespace,
    job: SmokeJob,
) -> tuple[Path | None, Path | None]:
    root = job_root(args, job)
    if not root.exists():
        return None, None
    compatible: list[tuple[Path, bool, Path | None]] = []
    for config_path in sorted(root.glob("*/run_config.json")):
        run_directory = config_path.parent.resolve()
        config = read_json(config_path)
        if not _run_config_matches(config, job, args):
            raise ConfigurationError(
                f"run config drift under frozen job root: {run_directory}"
            )
        manifest_path = run_directory / "run_manifest.json"
        completed = manifest_path.is_file()
        latest = run_directory / "latest.pt"
        compatible.append(
            (run_directory, completed, latest if latest.is_file() else None)
        )
    completed_runs = [item for item in compatible if item[1]]
    if len(completed_runs) > 1:
        raise ConfigurationError(f"multiple completed runs exist for {job.job_id}")
    if completed_runs:
        return completed_runs[0][0], None
    resumable = [item for item in compatible if item[2] is not None]
    if len(resumable) > 1:
        raise ConfigurationError(f"multiple resumable runs exist for {job.job_id}")
    if resumable:
        return resumable[0][0], resumable[0][2]
    # A process can fail before its first completed update. Such a directory
    # has no resumable state and is retained as provenance while a clean
    # attempt starts in a new timestamped directory.
    return None, None


def _manifest_artifact(run_directory: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"run manifest omits {field}")
    path = Path(value)
    if not path.is_absolute():
        path = run_directory / path.name
    path = path.resolve()
    if not _path_within(path, run_directory) or not path.is_file():
        raise ConfigurationError(f"invalid {field}: {path}")
    return path


def read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConfigurationError(
                    f"invalid metrics JSON at line {line_number}"
                ) from error
            if not isinstance(value, dict) or not all_finite(value):
                raise ConfigurationError(
                    f"non-finite or invalid metrics record at line {line_number}"
                )
            records.append(value)
    if not records:
        raise ConfigurationError("training metrics are empty")
    return records


def _close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def audit_constraint_records(
    records: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    updates = [
        record for record in records if record.get("record_type") == "training_update"
    ]
    if not updates:
        raise ConfigurationError("constrained run has no training updates")
    multiplier = DUAL_INITIAL
    update_count = 0
    skipped_count = 0
    cumulative_cost = 0
    cumulative_opportunity = 0
    positive_denominator_updates = 0
    positive_multiplier_updates = 0
    for record in updates:
        if record.get("switch_constraint_schema_version") != 1:
            raise ConfigurationError("training update omits constraint schema")
        cost = int(record["decision_avoidable_switch_cost"])
        opportunity = int(record["decision_switch_opportunities"])
        if cost < 0 or opportunity < 0 or cost > opportunity:
            raise ConfigurationError("constraint cost/opportunity invariant failed")
        used = float(record["dual_multiplier_used"])
        if not _close(used, multiplier):
            raise ConfigurationError("dual multiplier continuity failed")
        if not 0.0 <= used <= DUAL_MAX:
            raise ConfigurationError("dual multiplier left its projection")
        rate = record.get("decision_avoidable_switch_rate")
        violation = record.get("decision_avoidable_switch_violation")
        skipped = bool(record["dual_update_skipped_zero_opportunity"])
        if opportunity == 0:
            if rate is not None or violation is not None or not skipped:
                raise ConfigurationError("zero-opportunity dual update was not skipped")
            expected_next = used
            skipped_count += 1
        else:
            positive_denominator_updates += 1
            expected_rate = cost / opportunity
            if rate is None or not _close(rate, expected_rate):
                raise ConfigurationError("rollout constraint rate is not ratio-of-sums")
            if not 0.0 <= float(rate) <= 1.0:
                raise ConfigurationError("rollout constraint rate is outside [0, 1]")
            if violation is None or not _close(
                violation, expected_rate - SWITCH_BUDGET
            ):
                raise ConfigurationError("rollout constraint violation is invalid")
            expected_next = min(
                DUAL_MAX,
                max(
                    0.0,
                    used + DUAL_LEARNING_RATE * (expected_rate - SWITCH_BUDGET),
                ),
            )
            update_count += 1
            if skipped:
                raise ConfigurationError("positive-opportunity dual update was skipped")
        observed_next = float(record["dual_multiplier_next"])
        if not _close(observed_next, expected_next):
            raise ConfigurationError("dual transition is not exactly recomputable")
        if int(record["dual_update_count"]) != update_count:
            raise ConfigurationError("dual update count is invalid")
        if int(record["dual_skipped_zero_opportunity_count"]) != skipped_count:
            raise ConfigurationError("dual skipped-update count is invalid")
        cumulative_cost += cost
        cumulative_opportunity += opportunity
        if int(record["dual_cumulative_cost"]) != cumulative_cost or int(
            record["dual_cumulative_opportunity"]
        ) != cumulative_opportunity:
            raise ConfigurationError("dual cumulative accounting is invalid")
        surrogate = float(record["switch_constraint_surrogate_rate"])
        if not 0.0 <= surrogate <= 1.0:
            raise ConfigurationError("constraint surrogate is outside [0, 1]")
        if not _close(
            surrogate, record["avoidable_switch_regularization_penalty"], tolerance=1e-7
        ):
            raise ConfigurationError("logged constraint surrogate is inconsistent")
        actor_term = used * surrogate
        if not _close(
            record["switch_constraint_actor_term"], actor_term, tolerance=1e-7
        ):
            raise ConfigurationError("constraint actor term is inconsistent")
        if not _close(
            record["actor_loss"],
            float(record["actor_primary_loss"]) + actor_term,
            tolerance=1e-6,
        ):
            raise ConfigurationError("actor loss omits the constraint surrogate")
        eligible = int(record["rollout_avoidable_switch_eligible_decisions"])
        if eligible > 0 and surrogate <= 0.0:
            raise ConfigurationError("eligible constraint surrogate is identically zero")
        if used > 0.0:
            positive_multiplier_updates += 1
        multiplier = observed_next

    expected_final = {
        "multiplier": multiplier,
        "update_count": update_count,
        "skipped_zero_opportunity_count": skipped_count,
        "cumulative_cost": cumulative_cost,
        "cumulative_opportunity": cumulative_opportunity,
        "last_rollout_cost": int(updates[-1]["decision_avoidable_switch_cost"]),
        "last_rollout_opportunity": int(
            updates[-1]["decision_switch_opportunities"]
        ),
    }
    for field, expected in expected_final.items():
        observed = final_state.get(field)
        if isinstance(expected, float):
            matches = observed is not None and _close(observed, expected)
        else:
            matches = observed == expected
        if not matches:
            raise ConfigurationError(f"final constraint state mismatch: {field}")
    return {
        "training_update_count": len(updates),
        "positive_denominator_updates": positive_denominator_updates,
        "positive_multiplier_updates": positive_multiplier_updates,
        "cumulative_cost": cumulative_cost,
        "cumulative_opportunity": cumulative_opportunity,
        "cumulative_rate": (
            cumulative_cost / cumulative_opportunity
            if cumulative_opportunity > 0
            else None
        ),
        "final_multiplier": multiplier,
        "dual_update_count": update_count,
        "dual_skipped_zero_opportunity_count": skipped_count,
    }


def _legacy_validation_score(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record["delivery_ratio"]),
        float(record["mean_reward"]),
        -float(record["drop_rate"]),
        -float(record["average_delay_slots"]),
    )


def audit_validation_selection(
    job: SmokeJob,
    manifest: Mapping[str, Any],
    validations: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    run_directory: Path,
    selected_path: Path,
    selected_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and cross-check the arm-specific validation selection."""

    candidate_count = manifest.get("validation_candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(validations)
    ):
        raise ConfigurationError("validation candidate count mismatch")

    if job.constrained:
        if len(selections) != 1:
            raise ConfigurationError(
                "constrained run requires exactly one validation selection record"
            )
    elif selections:
        raise ConfigurationError(
            "legacy baseline unexpectedly contains a validation selection record"
        )

    selection_mode = (
        "avoidable_switch_budget_constrained"
        if job.constrained
        else "legacy_lexicographic"
    )
    selection_spec = manifest.get("validation_selection_spec")
    if not isinstance(selection_spec, Mapping):
        raise ConfigurationError("run manifest omits validation selection spec")
    expected_spec_fields = {
        "schema_version": 1,
        "mode": selection_mode,
        "delivery_tolerance": 0.0,
        "class_2_tolerance": 0.0,
        "switch_budget": SWITCH_BUDGET if job.constrained else None,
        "source": "validation_only",
        "test_panel_consulted": False,
        "validation_seed_start": SMOKE_VALIDATION_WORKLOAD_SEEDS[0],
        "validation_episodes": len(SMOKE_VALIDATION_WORKLOAD_SEEDS),
        "selection_timing": (
            "after_all_validation_candidates"
            if job.constrained
            else "online_legacy_compatible"
        ),
    }
    for field, expected in expected_spec_fields.items():
        if selection_spec.get(field) != expected:
            raise ConfigurationError(f"validation selection spec drifted: {field}")
    expected_aggregation = {
        "delivery_ratio": "macro_mean_over_validation_episodes",
        "class_2_delivery_ratio": "macro_mean_over_validation_episodes",
        "routing_switches": "mean_of_episode_total_switch_counts",
    }
    if job.constrained:
        expected_aggregation.update(
            decision_avoidable_switch_rate=(
                "micro_ratio_over_pre_contention_validation_opportunities"
            ),
            decision_avoidable_switches=(
                "sum_over_pre_contention_validation_decisions"
            ),
            decision_switch_opportunities=(
                "sum_over_pre_contention_validation_decisions"
            ),
        )
    if selection_spec.get("metric_aggregation") != expected_aggregation:
        raise ConfigurationError(
            "validation selection metric aggregation drifted"
        )
    if selected_checkpoint.get("validation_selection_spec") != dict(
        selection_spec
    ):
        raise ConfigurationError(
            "selected checkpoint validation selection spec differs from manifest"
        )

    seen_steps: set[int] = set()
    for record in validations:
        missing = set(SELECTED_VALIDATION_METRIC_FIELDS).difference(record)
        if missing:
            raise ConfigurationError(
                "validation record lacks selected metric fields: "
                f"{sorted(missing)}"
            )
        step = record.get("environment_steps")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ConfigurationError("validation candidate step is invalid")
        if step in seen_steps:
            raise ConfigurationError("validation candidate steps are not unique")
        seen_steps.add(step)

    try:
        expected_selected = select_leo_validation_record(
            validations,
            mode=selection_mode,
            delivery_tolerance=0.0,
            class_2_tolerance=0.0,
            switch_budget=SWITCH_BUDGET if job.constrained else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("validation selection cannot be recomputed") from error

    if not job.constrained:
        running_best: tuple[float, ...] | None = None
        for record in validations:
            score = _legacy_validation_score(record)
            if running_best is None or score > running_best:
                running_best = score
            if record.get("is_validation_best") is not (score == running_best):
                raise ConfigurationError(
                    "legacy validation-best marker is inconsistent"
                )

    selected_metrics = manifest.get("selected_validation_metrics")
    if not isinstance(selected_metrics, Mapping):
        raise ConfigurationError("selected validation metrics are missing")
    if set(selected_metrics) != set(SELECTED_VALIDATION_METRIC_FIELDS):
        raise ConfigurationError("selected validation metric schema drifted")
    expected_metrics = {
        field: expected_selected[field]
        for field in SELECTED_VALIDATION_METRIC_FIELDS
    }
    if dict(selected_metrics) != expected_metrics:
        raise ConfigurationError(
            "manifest-selected metrics differ from recomputed validation candidate"
        )

    expected_score = list(_legacy_validation_score(expected_selected))
    manifest_score = manifest.get("best_validation_score")
    if (
        not isinstance(manifest_score, list)
        or len(manifest_score) != len(expected_score)
        or any(
            not _close(observed, expected)
            for observed, expected in zip(manifest_score, expected_score)
        )
    ):
        raise ConfigurationError("manifest best validation score is inconsistent")

    selected_step = expected_selected["environment_steps"]
    checkpoint_step = selected_checkpoint.get("step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step != selected_step
    ):
        raise ConfigurationError(
            "selected checkpoint step differs from recomputed validation candidate"
        )
    selected_sha256 = sha256_file(selected_path)
    if manifest.get("selected_validation_checkpoint_sha256") != selected_sha256:
        raise ConfigurationError("manifest selected checkpoint hash mismatch")

    if job.constrained:
        selection = selections[0]
        if selection.get("selection_spec") != dict(selection_spec):
            raise ConfigurationError(
                "selection-record spec differs from run manifest"
            )
        if selection.get("candidate_count") != len(validations):
            raise ConfigurationError("selection-record candidate count mismatch")
        if selection.get("selected_candidate_step") != selected_step:
            raise ConfigurationError("selection-record candidate step mismatch")
        if selection.get("selected_metrics") != dict(selected_metrics):
            raise ConfigurationError(
                "selection-record metrics differ from run manifest"
            )
        if selection.get("selected_checkpoint_sha256") != selected_sha256:
            raise ConfigurationError("selection-record checkpoint hash mismatch")
        promoted_path = _manifest_artifact(
            run_directory,
            selection.get("validation_best_checkpoint"),
            "selection validation_best_checkpoint",
        )
        if promoted_path != selected_path.resolve():
            raise ConfigurationError("selection-record promoted path mismatch")
        candidate_path = _manifest_artifact(
            run_directory,
            expected_selected.get("candidate_checkpoint"),
            "selected candidate checkpoint",
        )
        recorded_candidate_path = _manifest_artifact(
            run_directory,
            selection.get("selected_candidate_checkpoint"),
            "selection selected_candidate_checkpoint",
        )
        if candidate_path != recorded_candidate_path:
            raise ConfigurationError("selection-record candidate path mismatch")
        if sha256_file(candidate_path) != selected_sha256:
            raise ConfigurationError("promoted checkpoint differs from candidate")
        expected_feasible = bool(
            float(selected_metrics["decision_avoidable_switch_rate"])
            <= SWITCH_BUDGET + 1e-12
        )
        if selection.get("constraint_feasible") is not expected_feasible:
            raise ConfigurationError("selection feasibility flag is inconsistent")

    return dict(selected_metrics)


def _audit_decision_ledger(
    job: SmokeJob,
    metrics: Mapping[str, Any],
    context: str,
) -> float | None:
    opportunity = metrics.get("decision_switch_opportunities")
    cost = metrics.get("decision_avoidable_switches")
    if (
        isinstance(opportunity, bool)
        or not isinstance(opportunity, int)
        or isinstance(cost, bool)
        or not isinstance(cost, int)
        or opportunity < 0
        or cost < 0
        or cost > opportunity
    ):
        raise ConfigurationError(f"{context} decision ledger is invalid")
    logged_rate = metrics.get("decision_avoidable_switch_rate")
    if opportunity == 0:
        if cost != 0 or logged_rate is not None:
            raise ConfigurationError(
                f"zero-opportunity {context} rate must be undefined"
            )
        if job.constrained:
            raise ConfigurationError(
                f"constrained {context} has no switch opportunity"
            )
        return None
    expected_rate = cost / opportunity
    if logged_rate is None or not _close(logged_rate, expected_rate):
        raise ConfigurationError(f"{context} rate is not ratio-of-sums")
    return expected_rate


def audit_selected_decision_ledger(
    job: SmokeJob,
    selected_metrics: Mapping[str, Any],
) -> float | None:
    return _audit_decision_ledger(job, selected_metrics, "selected validation")


def audit_validation_decision_ledgers(
    job: SmokeJob,
    validations: Sequence[Mapping[str, Any]],
) -> int:
    total_opportunities = 0
    for index, record in enumerate(validations):
        _audit_decision_ledger(job, record, f"validation candidate {index}")
        total_opportunities += int(record["decision_switch_opportunities"])
    return total_opportunities


def audit_job(
    args: argparse.Namespace,
    job: SmokeJob,
    run_directory: Path,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if not _path_within(run_directory, job_root(args, job)):
        raise ConfigurationError("training run escaped its frozen job root")
    config_path = run_directory / "run_config.json"
    metrics_path = run_directory / "training_metrics.jsonl"
    manifest_path = run_directory / "run_manifest.json"
    latest_path = run_directory / "latest.pt"
    for path in (config_path, metrics_path, manifest_path, latest_path):
        if not path.is_file():
            raise ConfigurationError(f"required training artifact is missing: {path}")
    config = read_json(config_path)
    if not _run_config_matches(config, job, args):
        raise ConfigurationError("completed run config differs from frozen job")
    manifest = read_json(manifest_path)
    if not all_finite(manifest):
        raise ConfigurationError("run manifest contains non-finite values")
    final_path = _manifest_artifact(
        run_directory, manifest.get("final_checkpoint"), "final_checkpoint"
    )
    selected_path = _manifest_artifact(
        run_directory,
        manifest.get("validation_best_checkpoint"),
        "validation_best_checkpoint",
    )
    records = read_metrics(metrics_path)
    training_updates = [
        record for record in records if record.get("record_type") == "training_update"
    ]
    validations = [
        record for record in records if record.get("record_type") == "validation"
    ]
    selections = [
        record
        for record in records
        if record.get("record_type") == "validation_selection"
    ]
    if not training_updates or not validations:
        raise ConfigurationError("training or validation records are incomplete")
    if any(
        int(record.get("seed_start", -1)) != SMOKE_VALIDATION_WORKLOAD_SEEDS[0]
        or int(record.get("episodes", -1))
        != len(SMOKE_VALIDATION_WORKLOAD_SEEDS)
        for record in validations
    ):
        raise ConfigurationError("validation workload panel drifted")
    validation_opportunities = audit_validation_decision_ledgers(
        job, validations
    )
    environment_steps = int(manifest.get("environment_steps", -1))
    if not TIMESTEPS <= environment_steps <= TIMESTEPS + 200:
        raise ConfigurationError("training did not finish at the expected boundary")

    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    selected_checkpoint = torch.load(
        selected_path, map_location="cpu", weights_only=False
    )
    latest_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    for label, checkpoint in (
        ("final", final_checkpoint),
        ("selected", selected_checkpoint),
        ("latest", latest_checkpoint),
    ):
        if not isinstance(checkpoint, dict):
            raise ConfigurationError(f"{label} checkpoint is not a mapping")
        if checkpoint.get("candidate_feature_dim") != 26:
            raise ConfigurationError(f"{label} candidate feature dimension drifted")
        checkpoint_args = checkpoint.get("args")
        if not isinstance(checkpoint_args, Mapping) or not _run_config_matches(
            checkpoint_args, job, args
        ):
            raise ConfigurationError(f"{label} checkpoint args drifted")
    if latest_checkpoint.get("resume_boundary") != "completed_update":
        raise ConfigurationError("latest checkpoint is not an exact update boundary")

    constraint_summary = None
    if job.constrained:
        expected_contract = constraint_contract()
        for checkpoint in (final_checkpoint, selected_checkpoint, latest_checkpoint):
            spec = checkpoint.get("switch_constraint_spec")
            if not isinstance(spec, Mapping):
                raise ConfigurationError("constrained checkpoint omits constraint spec")
            for field, expected in expected_contract.items():
                if spec.get(field) != expected:
                    raise ConfigurationError(
                        f"checkpoint constraint contract drifted: {field}"
                    )
        final_state = final_checkpoint.get("switch_constraint_state")
        if not isinstance(final_state, Mapping):
            raise ConfigurationError("final checkpoint omits constraint state")
        constraint_summary = audit_constraint_records(records, final_state)
        if constraint_summary["cumulative_opportunity"] <= 0:
            raise ConfigurationError("constrained training had no opportunities")
    else:
        for checkpoint in (final_checkpoint, selected_checkpoint, latest_checkpoint):
            if checkpoint.get("switch_constraint_spec") is not None or checkpoint.get(
                "switch_constraint_state"
            ) is not None:
                raise ConfigurationError("baseline checkpoint contains constraint state")

    selected_metrics = audit_validation_selection(
        job,
        manifest,
        validations,
        selections,
        run_directory,
        selected_path,
        selected_checkpoint,
    )
    selected_opportunities = int(selected_metrics["decision_switch_opportunities"])
    selected_cost = int(selected_metrics["decision_avoidable_switches"])
    selected_rate = audit_selected_decision_ledger(job, selected_metrics)

    return {
        "job": job.as_dict(),
        "run_directory": str(run_directory),
        "environment_steps": environment_steps,
        "validation_candidate_count": len(validations),
        "validation_opportunities": validation_opportunities,
        "selected_validation": {
            "environment_steps": int(selected_metrics["environment_steps"]),
            "delivery_ratio": float(selected_metrics["delivery_ratio"]),
            "class_2_delivery_ratio": float(
                selected_metrics["class_2_delivery_ratio"]
            ),
            "decision_avoidable_switches": selected_cost,
            "decision_switch_opportunities": selected_opportunities,
            "decision_avoidable_switch_rate": selected_rate,
            "constraint_feasible": (
                selected_rate <= SWITCH_BUDGET + 1e-12
                if selected_rate is not None
                else None
            ),
        },
        "constraint_training": constraint_summary,
        "artifacts": {
            "run_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "training_metrics": {
                "path": str(metrics_path),
                "sha256": sha256_file(metrics_path),
            },
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "final_checkpoint": {
                "path": str(final_path),
                "sha256": sha256_file(final_path),
            },
            "selected_checkpoint": {
                "path": str(selected_path),
                "sha256": sha256_file(selected_path),
            },
            "latest_checkpoint": {
                "path": str(latest_path),
                "sha256": sha256_file(latest_path),
            },
        },
    }


def _pid_is_active(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def invocation_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".smoke_runner.lock"
    if path.exists():
        try:
            existing = read_json(path)
            existing_pid = int(existing.get("pid", -1))
        except Exception:
            existing_pid = -1
        if _pid_is_active(existing_pid):
            raise RuntimeError(
                f"another smoke runner is active with pid {existing_pid}"
            )
        path.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    payload = {"pid": os.getpid(), "token": token, "created_at_utc": utc_now()}
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another smoke runner acquired the invocation lock") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        yield
    finally:
        try:
            observed = read_json(path)
        except Exception:
            observed = None
        if isinstance(observed, Mapping) and observed.get("token") == token:
            path.unlink(missing_ok=True)


def load_job_status(
    args: argparse.Namespace,
    job: SmokeJob,
    spec_sha256: str,
) -> dict[str, Any]:
    path = status_path(args, job)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "spec_sha256": spec_sha256,
            "job": job.as_dict(),
            "status": "pending",
            "attempts": [],
        }
    status = read_json(path)
    if status.get("spec_sha256") != spec_sha256 or status.get("job") != job.as_dict():
        raise ConfigurationError(f"job status fingerprint mismatch: {job.job_id}")
    if not isinstance(status.get("attempts"), list):
        raise ConfigurationError(f"job attempt history is invalid: {job.job_id}")
    return status


def write_job_status(
    args: argparse.Namespace,
    job: SmokeJob,
    status: Mapping[str, Any],
) -> None:
    atomic_write_json(status_path(args, job), dict(status))


def audit_no_active_training_children(args: argparse.Namespace) -> None:
    active: list[dict[str, Any]] = []
    status_root = args.output / "job_status"
    if not status_root.exists():
        return
    for path in sorted(status_root.glob("*.json")):
        status = read_json(path)
        if status.get("status") != "running":
            continue
        child_pid = status.get("child_pid")
        if (
            isinstance(child_pid, bool)
            or not isinstance(child_pid, int)
            or not _pid_is_active(child_pid)
        ):
            continue
        job_value = status.get("job")
        active.append(
            {
                "status_path": str(path.resolve()),
                "job_id": (
                    job_value.get("job_id")
                    if isinstance(job_value, Mapping)
                    else None
                ),
                "child_pid": child_pid,
            }
        )
    if active:
        raise RuntimeError(
            "active training children exist before scheduling: "
            f"{active}"
        )


def train_job(
    args: argparse.Namespace,
    job: SmokeJob,
    spec_sha256: str,
    cancellation_event: threading.Event | None = None,
) -> dict[str, Any]:
    if cancellation_event is not None and cancellation_event.is_set():
        raise TrainingCancelled(f"training cancelled before start: {job.job_id}")
    status = load_job_status(args, job, spec_sha256)
    recorded = status.get("artifact")
    if status.get("status") == "completed" and isinstance(recorded, Mapping):
        observed = audit_job(
            args, job, Path(str(recorded["run_directory"]))
        )
        if observed != recorded:
            raise ConfigurationError(f"completed job audit drifted: {job.job_id}")
        return observed

    if status.get("status") == "running":
        child_pid = int(status.get("child_pid", -1))
        if _pid_is_active(child_pid):
            raise RuntimeError(
                f"training process is already active for {job.job_id}: pid={child_pid}"
            )

    discovered_run, resume_checkpoint = discover_run_directory(args, job)
    if discovered_run is not None and resume_checkpoint is None:
        artifact = audit_job(args, job, discovered_run)
        status.update(
            status="completed",
            artifact=artifact,
            recovered_completed_run=True,
            completed_at_utc=utc_now(),
        )
        write_job_status(args, job, status)
        return artifact

    if cancellation_event is not None and cancellation_event.is_set():
        raise TrainingCancelled(f"training cancelled before launch: {job.job_id}")

    root = job_root(args, job)
    root.mkdir(parents=True, exist_ok=True)
    attempt_number = len(status["attempts"]) + 1
    log_path = root / f"runner_attempt_{attempt_number}.log"
    command = build_command(
        job,
        args,
        root,
        resume_checkpoint=resume_checkpoint,
    )
    attempt = {
        "attempt": attempt_number,
        "started_at_utc": utc_now(),
        "status": "running",
        "resume_checkpoint": (
            str(resume_checkpoint.resolve()) if resume_checkpoint is not None else None
        ),
        "command": command,
        "log_path": str(log_path.resolve()),
    }
    status["attempts"].append(attempt)
    status.update(status="running", started_at_utc=attempt["started_at_utc"])
    write_job_status(args, job, status)

    started = time.monotonic()
    mode = "a" if resume_checkpoint is not None else "w"
    try:
        with log_path.open(mode, encoding="utf-8", newline="") as output:
            if cancellation_event is not None and cancellation_event.is_set():
                raise TrainingCancelled(
                    f"training cancelled immediately before launch: {job.job_id}"
                )
            process = subprocess.Popen(
                command,
                cwd=args.cleanmarl,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            status["child_pid"] = process.pid
            attempt["child_pid"] = process.pid
            write_job_status(args, job, status)
            print(
                f"started {job.job_id} pid={process.pid} "
                f"resume={resume_checkpoint is not None}",
                flush=True,
            )
            try:
                next_heartbeat = time.monotonic() + 30.0
                while True:
                    if (
                        cancellation_event is not None
                        and cancellation_event.is_set()
                    ):
                        raise TrainingCancelled(
                            f"training cancelled during shutdown: {job.job_id}"
                        )
                    try:
                        return_code = process.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= next_heartbeat:
                            output.flush()
                            elapsed = (time.monotonic() - started) / 60.0
                            print(
                                f"training heartbeat {job.job_id} pid={process.pid} "
                                f"elapsed={elapsed:.1f} min",
                                flush=True,
                            )
                            next_heartbeat = time.monotonic() + 30.0
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if return_code != 0:
            raise RuntimeError(
                f"trainer exited with code {return_code}; see {log_path}"
            )
        completed_run, remaining_resume = discover_run_directory(args, job)
        if completed_run is None or remaining_resume is not None:
            raise RuntimeError("trainer exited without a completed run manifest")
        artifact = audit_job(args, job, completed_run)
    except BaseException as error:
        attempt.update(
            status=(
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, TrainingCancelled))
                else "failed"
            ),
            finished_at_utc=utc_now(),
            error=repr(error),
        )
        status.update(status=attempt["status"], child_pid=None)
        write_job_status(args, job, status)
        raise

    attempt.update(status="completed", finished_at_utc=utc_now())
    status.update(
        status="completed",
        child_pid=None,
        completed_at_utc=utc_now(),
        artifact=artifact,
    )
    write_job_status(args, job, status)
    return artifact


def run_all_jobs(
    args: argparse.Namespace,
    spec_sha256: str,
) -> dict[str, dict[str, Any]]:
    jobs = build_jobs()
    audit_no_active_training_children(args)
    completed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    cancellation_event = threading.Event()
    remaining = iter(jobs)
    executor = ThreadPoolExecutor(max_workers=args.max_parallel)
    futures: dict[Any, SmokeJob] = {}

    def submit_next() -> bool:
        try:
            job = next(remaining)
        except StopIteration:
            return False
        future = executor.submit(
            train_job,
            args,
            job,
            spec_sha256,
            cancellation_event,
        )
        futures[future] = job
        return True

    try:
        for _ in range(min(args.max_parallel, len(jobs))):
            submit_next()
        while futures:
            future = next(as_completed(tuple(futures)))
            job = futures.pop(future)
            try:
                artifact = future.result()
            except Exception as error:
                if not (
                    isinstance(error, TrainingCancelled) and failures
                ):
                    failures[job.job_id] = repr(error)
                    print(f"failed {job.job_id}: {error}", flush=True)
                cancellation_event.set()
            else:
                completed[job.job_id] = artifact
                print(
                    f"completed {job.job_id} at step "
                    f"{artifact['environment_steps']}",
                    flush=True,
                )
            if not failures and not cancellation_event.is_set():
                submit_next()
    except BaseException:
        cancellation_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        raise RuntimeError(f"{len(failures)} smoke jobs failed: {failures}")
    if len(completed) != len(jobs):
        raise RuntimeError("smoke training grid is incomplete")
    return {job.job_id: completed[job.job_id] for job in jobs}


def _exact_resume_command(
    args: argparse.Namespace,
    checkpoint_root: Path,
    *,
    total_timesteps: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(args.cleanmarl / "cleanmarl" / "mappo.py"),
        "--env-type",
        "leo_multi",
        "--env-name",
        "medium_load",
        "--leo-project-path",
        str(args.project),
        "--leo-variant",
        ENVIRONMENT_VARIANT,
        "--seed",
        "424242",
        "--batch-size",
        "1",
        "--total-timesteps",
        str(total_timesteps),
        "--epochs",
        "1",
        "--num-minibatches",
        "1",
        "--eval-steps",
        "1000",
        "--num-eval-ep",
        "1",
        "--save-every-steps",
        "1",
        "--checkpoint-dir",
        str(checkpoint_root),
        "--run-tag",
        "EXACT-RESUME-EQUIVALENCE-CHECK",
        "--train-seed-start",
        "60001",
        "--train-seed-count",
        "1",
        "--validation-seed-start",
        "60001",
        "--validation-selection-mode",
        "avoidable_switch_budget_constrained",
        "--avoidable-switch-constraint-enabled",
        "--avoidable-switch-budget",
        str(SWITCH_BUDGET),
        "--avoidable-switch-dual-learning-rate",
        str(DUAL_LEARNING_RATE),
        "--avoidable-switch-dual-initial",
        str(DUAL_INITIAL),
        "--avoidable-switch-dual-max",
        str(DUAL_MAX),
        "--avoidable-switch-reduction",
        "rollout_micro_mean",
        "--no-lr-decay",
        "--device",
        "cpu",
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint.resolve())])
    return command


def _run_check_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> None:
    process = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    log_path.write_text(process.stdout, encoding="utf-8", newline="\n")
    if process.returncode != 0:
        raise RuntimeError(
            f"exact-resume verification process failed; see {log_path}"
        )


def _only_run_directory(root: Path) -> Path:
    manifests = list(root.glob("*/run_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one verification run under {root}")
    return manifests[0].parent.resolve()


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _load_final_checkpoint(run_directory: Path) -> tuple[Path, dict[str, Any]]:
    manifest = read_json(run_directory / "run_manifest.json")
    path = _manifest_artifact(
        run_directory, manifest.get("final_checkpoint"), "final_checkpoint"
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("verification final checkpoint is not a mapping")
    return path, checkpoint


def _training_updates(path: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in read_metrics(path)
        if record.get("record_type") == "training_update"
    ]


def run_exact_resume_equivalence(
    args: argparse.Namespace,
    spec_sha256: str,
) -> dict[str, Any]:
    proof_path = args.output / "exact_resume_equivalence.json"
    if proof_path.exists():
        proof = read_json(proof_path)
        validate_self_hash(proof, "proof_sha256")
        if proof.get("spec_sha256") != spec_sha256 or proof.get("status") != "passed":
            raise ConfigurationError("exact-resume proof drifted")
        for artifact in proof.get("artifacts", {}).values():
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or artifact.get("sha256") != sha256_file(path):
                raise ConfigurationError("exact-resume proof artifact drifted")
        return proof

    attempt_root = (
        args.output / "verification" / f"exact_resume_{uuid.uuid4().hex}"
    ).resolve()
    full_root = attempt_root / "uninterrupted"
    resumed_root = attempt_root / "resumed"
    full_root.mkdir(parents=True, exist_ok=False)
    resumed_root.mkdir(parents=True, exist_ok=False)

    _run_check_process(
        _exact_resume_command(args, full_root, total_timesteps=50),
        cwd=args.cleanmarl,
        log_path=attempt_root / "uninterrupted.log",
    )
    full_run = _only_run_directory(full_root)

    _run_check_process(
        _exact_resume_command(args, resumed_root, total_timesteps=1),
        cwd=args.cleanmarl,
        log_path=attempt_root / "partial.log",
    )
    resumed_run = _only_run_directory(resumed_root)
    latest = resumed_run / "latest.pt"
    if not latest.is_file():
        raise RuntimeError("partial verification run produced no latest checkpoint")
    frozen_partial = attempt_root / "partial_latest_frozen.pt"
    shutil.copy2(latest, frozen_partial)
    partial_sha256 = sha256_file(frozen_partial)
    _run_check_process(
        _exact_resume_command(
            args,
            resumed_root,
            total_timesteps=50,
            resume_checkpoint=latest,
        ),
        cwd=args.cleanmarl,
        log_path=attempt_root / "resume.log",
    )

    full_final_path, full_checkpoint = _load_final_checkpoint(full_run)
    resumed_final_path, resumed_checkpoint = _load_final_checkpoint(resumed_run)
    comparison_fields = (
        "step",
        "actor",
        "critic",
        "actor_optimizer",
        "critic_optimizer",
        "obs_size",
        "state_size",
        "action_size",
        "n_agents",
        "candidate_feature_dim",
        "candidate_actor_spec",
        "switch_regularizer_spec",
        "switch_constraint_spec",
        "switch_constraint_state",
        "critic_spec",
        "validation_selection_spec",
    )
    mismatches = [
        field
        for field in comparison_fields
        if not _state_equal(full_checkpoint.get(field), resumed_checkpoint.get(field))
    ]
    full_trainer_state = dict(full_checkpoint["trainer_state"])
    resumed_trainer_state = dict(resumed_checkpoint["trainer_state"])
    full_trainer_state.pop("metrics_file_size_bytes", None)
    resumed_trainer_state.pop("metrics_file_size_bytes", None)
    if not _state_equal(full_trainer_state, resumed_trainer_state):
        mismatches.append("trainer_state")
    full_updates = _training_updates(full_run / "training_metrics.jsonl")
    resumed_updates = _training_updates(resumed_run / "training_metrics.jsonl")
    if not _state_equal(full_updates, resumed_updates):
        mismatches.append("training_update_records")
    if len(full_updates) != 2 or len(resumed_updates) != 2:
        mismatches.append("expected_two_training_updates")
    if mismatches:
        raise RuntimeError(f"exact-resume equivalence failed: {sorted(set(mismatches))}")

    artifact_paths = {
        "uninterrupted_final": full_final_path,
        "resumed_final": resumed_final_path,
        "partial_latest_frozen": frozen_partial,
        "uninterrupted_metrics": full_run / "training_metrics.jsonl",
        "resumed_metrics": resumed_run / "training_metrics.jsonl",
        "uninterrupted_log": attempt_root / "uninterrupted.log",
        "partial_log": attempt_root / "partial.log",
        "resume_log": attempt_root / "resume.log",
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec_sha256,
        "status": "passed",
        "workload_role": "retired_implementation_verification_only",
        "workload_seed": 60001,
        "policy_seed": 424242,
        "uninterrupted_updates": len(full_updates),
        "resumed_updates": len(resumed_updates),
        "partial_checkpoint_sha256_before_resume": partial_sha256,
        "comparison_fields": list(comparison_fields) + [
            "trainer_state_except_metrics_file_offset",
            "training_update_records",
        ],
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "finished_at_utc": utc_now(),
    }
    proof = self_hashed(body, "proof_sha256")
    atomic_write_json(proof_path, proof)
    return proof


def run_preflight_tests(
    args: argparse.Namespace,
    spec_sha256: str,
) -> dict[str, Any]:
    manifest_path = args.output / "preflight_tests.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        validate_self_hash(manifest, "verification_sha256")
        log_path = Path(str(manifest.get("log_path", "")))
        exact_resume = run_exact_resume_equivalence(args, spec_sha256)
        if (
            manifest.get("spec_sha256") != spec_sha256
            or manifest.get("status") != "passed"
            or not log_path.is_file()
            or manifest.get("log_sha256") != sha256_file(log_path)
            or manifest.get("exact_resume_proof_sha256")
            != exact_resume.get("proof_sha256")
        ):
            raise ConfigurationError("preflight test verification drifted")
        return manifest

    log_path = (args.output / "preflight_tests.log").resolve()
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-p",
        "test_*.py",
    ]
    process = subprocess.run(
        command,
        cwd=args.project,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(process.stdout, encoding="utf-8", newline="\n")
    if process.returncode != 0:
        raise RuntimeError(f"preflight tests failed; see {log_path}")
    matches = re.findall(r"^Ran (\d+) tests? in ", process.stdout, flags=re.MULTILINE)
    test_count = int(matches[-1]) if matches else None
    if test_count is None or test_count <= 0:
        raise RuntimeError("preflight test count could not be audited")
    exact_resume = run_exact_resume_equivalence(args, spec_sha256)
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec_sha256,
        "status": "passed",
        "test_count": test_count,
        "exact_resume_status": exact_resume["status"],
        "exact_resume_proof_sha256": exact_resume["proof_sha256"],
        "command": command,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "finished_at_utc": utc_now(),
    }
    manifest = self_hashed(body, "verification_sha256")
    atomic_write_json(manifest_path, manifest)
    return manifest


def write_training_freeze(
    output: Path,
    spec: Mapping[str, Any],
    verification: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {job.job_id for job in build_jobs()}
    if set(artifacts) != expected:
        raise ConfigurationError("training freeze does not contain the exact grid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "verification_sha256": verification["verification_sha256"],
        "training_complete": True,
        "checkpoint_count": len(artifacts),
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "jobs": {
            job.job_id: dict(artifacts[job.job_id]) for job in build_jobs()
        },
    }
    freeze = self_hashed(body, "freeze_sha256")
    ensure_immutable_json(output / "training_freeze.json", freeze)
    return freeze


def write_smoke_report(
    output: Path,
    spec: Mapping[str, Any],
    verification: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    constrained = [
        freeze["jobs"][job.job_id]
        for job in build_jobs()
        if job.constrained
    ]
    positive_denominators = all(
        int(item["constraint_training"]["cumulative_opportunity"]) > 0
        and int(item["validation_opportunities"]) > 0
        for item in constrained
    )
    gates = {
        "all_four_jobs_completed": len(freeze["jobs"]) == 4,
        "positive_opportunity_denominator_in_each_constrained_scenario": (
            positive_denominators
        ),
        "cost_not_greater_than_opportunity": True,
        "forced_reroute_constraint_cost_is_zero": True,
        "dual_transition_exactly_recomputable": True,
        "all_logged_numeric_values_finite": True,
        "exact_resume_equivalence_unit_test_passed": (
            verification.get("status") == "passed"
        ),
        "sealed_test_access_count_zero": True,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "inferential_status": "mechanism_smoke_only",
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "verification_sha256": verification["verification_sha256"],
        "mechanism_gates": gates,
        "mechanism_gates_passed": all(gates.values()),
        "selected_validation_descriptive_only": {
            job.job_id: freeze["jobs"][job.job_id]["selected_validation"]
            for job in build_jobs()
        },
        "method_parameters_may_be_tuned_from_smoke": False,
        "formal_training_started": False,
        "formal_validation_started": False,
        "sealed_test_opened": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "next_step": (
            "separate_formal_preregistration_required"
            if all(gates.values())
            else "stop_and_version_protocol_before_any_method_change"
        ),
    }
    report = self_hashed(body, "report_sha256")
    ensure_immutable_json(output / "smoke_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the frozen four-job avoidable-switch constraint smoke."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "experiments" / DEFAULT_OUTPUT_DIRECTORY_NAME,
    )
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.cleanmarl = args.cleanmarl.resolve()
    args.project = args.project.resolve()
    if args.max_parallel not in {1, 2}:
        raise ConfigurationError("max_parallel must be one or two")
    if args.device != "cuda":
        raise ConfigurationError("the frozen smoke runtime requires CUDA")
    validate_output_isolation(args)
    spec = build_spec(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study_name": STUDY_NAME,
                    "spec_sha256": spec["spec_sha256"],
                    "training_jobs": len(build_jobs()),
                    "test_evaluations": 0,
                    "test_panel_consulted": False,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    ensure_immutable_json(args.output / "preregistration.json", spec)
    invocation_id = uuid.uuid4().hex
    invocation_path = args.output / "invocations" / f"{invocation_id}.json"
    invocation = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "invocation_id": invocation_id,
        "spec_sha256": spec["spec_sha256"],
        "status": "running",
        "started_at_utc": utc_now(),
        "device": args.device,
        "max_parallel": args.max_parallel,
        "test_panel_consulted": False,
    }
    atomic_write_json(invocation_path, invocation)
    try:
        with invocation_lock(args.output):
            verification = run_preflight_tests(args, spec["spec_sha256"])
            artifacts = run_all_jobs(args, spec["spec_sha256"])
            freeze = write_training_freeze(
                args.output, spec, verification, artifacts
            )
            report = write_smoke_report(
                args.output, spec, verification, freeze
            )
        invocation.update(
            status="completed",
            finished_at_utc=utc_now(),
            verification_sha256=verification["verification_sha256"],
            freeze_sha256=freeze["freeze_sha256"],
            report_sha256=report["report_sha256"],
            mechanism_gates_passed=report["mechanism_gates_passed"],
        )
        atomic_write_json(invocation_path, invocation)
        print(
            f"smoke complete: gates_passed={report['mechanism_gates_passed']} "
            f"output={args.output}",
            flush=True,
        )
        return 0
    except BaseException as error:
        invocation.update(
            status=("interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
            finished_at_utc=utc_now(),
            error=repr(error),
        )
        atomic_write_json(invocation_path, invocation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
