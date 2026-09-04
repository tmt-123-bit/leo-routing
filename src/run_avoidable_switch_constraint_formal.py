"""Run the preregistered avoidable-switch formal training/validation study.

This runner deliberately has no sealed-test execution path.  It trains the
frozen three-arm grid, freezes every selected checkpoint, and only then
re-evaluates those checkpoints on the formal validation panel.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np
import torch

from ablation_matrix_runner import atomic_write_csv, atomic_write_json, sha256_file
import formal_avoidable_switch_statistics as formal_statistics
from mappo_design import select_leo_validation_record
from mappo_evaluation import (
    ConstraintEpisodeMetrics,
    evaluate_policy_with_constraint_metrics,
    load_checkpoint_policy,
    metrics_as_dicts,
)
import run_avoidable_switch_constraint_smoke as smoke
from variant_definitions import canonical_variant_name


SCHEMA_VERSION = 1
STUDY_NAME = "ICC-AVOIDABLE-SWITCH-CONSTRAINT-FORMAL-v1"
PROTOCOL_FILENAME = "AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md"
DEFAULT_OUTPUT_DIRECTORY_NAME = "avoidable-switch-constraint-formal-v1"

SCENARIOS = ("medium_load", "hotspot_high_load")
ARM_BASELINE = "qos_only_baseline"
ARM_CONSTRAINED = "qos_only_constrained"
ARM_REWARD_CONTROL = "reward_shaped_control"
ARMS = (ARM_BASELINE, ARM_CONSTRAINED, ARM_REWARD_CONTROL)
ARM_VARIANTS = {
    ARM_BASELINE: "qos_only",
    ARM_CONSTRAINED: "qos_only",
    ARM_REWARD_CONTROL: "proposed",
}

POLICY_SEED_NAMESPACE = (
    "ICC-AVOIDABLE-SWITCH-CONSTRAINT-v1-formal-policy-seed-"
)
POLICY_SEEDS = tuple(
    int.from_bytes(
        hashlib.sha256(f"{POLICY_SEED_NAMESPACE}{index}".encode("ascii")).digest()[
            :4
        ],
        "big",
    )
    & 0x7FFFFFFF
    for index in range(8)
)
EXPECTED_POLICY_SEEDS = (
    179055553,
    626669596,
    746965870,
    183895110,
    310818925,
    2136406109,
    985998595,
    515636025,
)

TRAIN_WORKLOAD_SEEDS = tuple(range(76001, 76201))
SELECTION_WORKLOAD_SEEDS = tuple(range(77001, 77011))
GATE_WORKLOAD_SEEDS = tuple(range(77011, 77021))
VALIDATION_WORKLOAD_SEEDS = SELECTION_WORKLOAD_SEEDS + GATE_WORKLOAD_SEEDS
SEALED_TEST_WORKLOAD_SEEDS = tuple(range(78001, 78051))

TIMESTEPS = 50000
BATCH_SIZE = 4
EPISODE_STEPS = 30
ROLLOUT_STEPS = BATCH_SIZE * EPISODE_STEPS
EXPECTED_FINAL_ENVIRONMENT_STEPS = (
    math.ceil(TIMESTEPS / ROLLOUT_STEPS) * ROLLOUT_STEPS
)
EPOCHS = 3
NUM_MINIBATCHES = 4
VALIDATION_EVERY_ROLLOUTS = 40
SAVE_EVERY_STEPS = 5000
SWITCH_BUDGET = smoke.SWITCH_BUDGET
DUAL_LEARNING_RATE = smoke.DUAL_LEARNING_RATE
DUAL_INITIAL = smoke.DUAL_INITIAL
DUAL_MAX = smoke.DUAL_MAX
DELIVERY_NONINFERIORITY_MARGIN = 0.02
SELECTION_FEASIBILITY_NUMERICAL_TOLERANCE = 1e-12
EXPECTED_TRAINING_JOBS = len(SCENARIOS) * len(ARMS) * len(POLICY_SEEDS)
EXPECTED_SELECTION_RECHECK_ROWS = EXPECTED_TRAINING_JOBS * len(
    SELECTION_WORKLOAD_SEEDS
)
EXPECTED_GATE_ROWS = EXPECTED_TRAINING_JOBS * len(GATE_WORKLOAD_SEEDS)
EXPECTED_VALIDATION_ROWS = EXPECTED_SELECTION_RECHECK_ROWS + EXPECTED_GATE_ROWS
MAX_TRAINING_LAUNCH_ATTEMPTS = 2
MAX_EVALUATION_LAUNCH_ATTEMPTS = 2
RETRYABLE_INFRASTRUCTURE_EXCEPTION_NAMES = ("TimeoutError", "ConnectionError")
RETRYABLE_INFRASTRUCTURE_ERRNO_NAMES = (
    "EAGAIN",
    "EBUSY",
    "ECONNABORTED",
    "ECONNRESET",
    "EHOSTUNREACH",
    "EINTR",
    "EIO",
    "ENETDOWN",
    "ENETRESET",
    "ENETUNREACH",
    "EPIPE",
    "ESTALE",
    "ETIMEDOUT",
)
RETRYABLE_INFRASTRUCTURE_ERRNOS = frozenset(
    getattr(errno, name)
    for name in RETRYABLE_INFRASTRUCTURE_ERRNO_NAMES
    if hasattr(errno, name)
)
RETRYABLE_FAILURE_CATEGORIES = {
    "recognized_infrastructure_failure",
    "recognized_infrastructure_cancellation",
    "orphaned_process_interruption",
}
PREFLIGHT_TEST_MODULES = (
    "test_avoidable_switch_constraint_formal",
    "test_formal_avoidable_switch_statistics",
    "test_avoidable_switch_constraint",
    "test_mappo_design",
    "test_mappo_evaluation_constraint",
    "test_avoidable_switch_constraint_smoke",
)
# Updated only when the frozen, hash-bound allowlist itself changes.
EXPECTED_PREFLIGHT_TEST_COUNT = 163

PANEL_SELECTION_RECHECK = "selection_recheck"
PANEL_INDEPENDENT_GATE = "independent_gate"
VALIDATION_PANELS = (PANEL_SELECTION_RECHECK, PANEL_INDEPENDENT_GATE)
PANEL_WORKLOAD_SEEDS = {
    PANEL_SELECTION_RECHECK: SELECTION_WORKLOAD_SEEDS,
    PANEL_INDEPENDENT_GATE: GATE_WORKLOAD_SEEDS,
}
PANEL_EVALUATION_ROLES = {
    PANEL_SELECTION_RECHECK: "selected_checkpoint_selection_panel_recheck",
    PANEL_INDEPENDENT_GATE: "selected_checkpoint_independent_validation_gate",
}

FROZEN_TRAINING_SEMANTICS = dict(smoke.FROZEN_TRAINING_SEMANTICS)
SELECTED_VALIDATION_METRIC_FIELDS = smoke.SELECTED_VALIDATION_METRIC_FIELDS

SMOKE_EVIDENCE_DIRECTORY_NAME = "avoidable-switch-constraint-smoke-v1-r1"
REJECTED_SMOKE_EVIDENCE_DIRECTORY_NAME = "avoidable-switch-constraint-smoke-v1"
SMOKE_EVIDENCE_FILE_SHA256 = {
    "smoke_report.json": (
        "d107058b5e8f0c6821415c029b07ac9cb436973314c4bf3d8a473a03c7dd5177"
    ),
    "training_freeze.json": (
        "1abb53286b47cd190f029dac82782bedb3d2ac702cb13c090820c7542abfe583"
    ),
    "preregistration.json": (
        "ba2384ff7c10f5acfbe7e792eacfea40963da880c7c7ddfc6e7f8a5ecca31757"
    ),
    "exact_resume_equivalence.json": (
        "48f3c91ca3fafd51890725d3523c3856fc2993c91923b8b0a9c1439d97aaa48c"
    ),
}

EXACT_RESUME_RUNTIME_FILE_BINDINGS = {
    "smoke_runner_dependency": "runner",
    "trainer_snapshot": "trainer_snapshot",
    "external_trainer": "external_trainer",
    "design_snapshot": "design_snapshot",
    "external_design": "external_design",
    "environment": "environment",
    "base_environment": "base_environment",
    "wrapper": "wrapper",
    "variant_definitions": "variant_definitions",
    "evaluation": "evaluation",
}
EXACT_RESUME_RUNTIME_SCALAR_BINDINGS = ("git_head", "python", "torch", "platform")
EXACT_RESUME_RUNTIME_FIELDS_NOT_RECORDED_BY_SMOKE = (
    "python_executable",
    "numpy",
    "torch_cuda",
    "cudnn",
    "cuda_device_name",
    "cuda_compute_capability",
    "python_dependency_inventory",
    "python_dependency_inventory_sha256",
)

HISTORICAL_EXPERIMENT_DIRECTORY_NAMES = tuple(
    dict.fromkeys(
        (*smoke.HISTORICAL_EXPERIMENT_DIRECTORY_NAMES,
         REJECTED_SMOKE_EVIDENCE_DIRECTORY_NAME,
         SMOKE_EVIDENCE_DIRECTORY_NAME)
    )
)


class ConfigurationError(ValueError):
    """Raised when runtime state differs from the formal contract."""


class TrainingCancelled(RuntimeError):
    """Raised inside a worker when fail-fast shutdown cancels training."""

    def __init__(self, message: str, *, infrastructure_cause: bool = False):
        super().__init__(message)
        self.infrastructure_cause = infrastructure_cause


class EvaluationCancelled(RuntimeError):
    """Raised inside a worker when fail-fast shutdown cancels validation."""

    def __init__(self, message: str, *, infrastructure_cause: bool = False):
        super().__init__(message)
        self.infrastructure_cause = infrastructure_cause


def _is_retryable_infrastructure_failure(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return isinstance(error, OSError) and error.errno in RETRYABLE_INFRASTRUCTURE_ERRNOS


def _has_infrastructure_cause(error: BaseException) -> bool:
    return bool(
        isinstance(error, (TrainingCancelled, EvaluationCancelled))
        and error.infrastructure_cause
    ) or _is_retryable_infrastructure_failure(error)


def _failure_category(error: BaseException) -> str:
    if (
        isinstance(error, (TrainingCancelled, EvaluationCancelled))
        and error.infrastructure_cause
    ):
        return "recognized_infrastructure_cancellation"
    if _is_retryable_infrastructure_failure(error):
        return "recognized_infrastructure_failure"
    if isinstance(error, ConfigurationError):
        return "configuration_or_audit_failure"
    if isinstance(
        error, (KeyboardInterrupt, TrainingCancelled, EvaluationCancelled, SystemExit)
    ):
        return "execution_interruption"
    return "scientific_or_runtime_failure"


def _cancelled(
    error_type: type[TrainingCancelled] | type[EvaluationCancelled],
    message: str,
    cancellation_event: threading.Event,
) -> TrainingCancelled | EvaluationCancelled:
    cause = getattr(cancellation_event, "failure_cause", None)
    return error_type(
        message,
        infrastructure_cause=(
            isinstance(cause, BaseException)
            and _has_infrastructure_cause(cause)
        ),
    )


def _bind_cancellation_cause(
    cancellation_event: threading.Event, error: BaseException
) -> None:
    if not hasattr(cancellation_event, "failure_cause"):
        setattr(cancellation_event, "failure_cause", error)


def _retry_classification(error: BaseException) -> tuple[str, bool]:
    category = _failure_category(error)
    return category, category in RETRYABLE_FAILURE_CATEGORIES


def _require_retry_authorization(
    status: Mapping[str, Any], context: str
) -> None:
    attempts = status.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return
    previous = attempts[-1]
    if previous.get("status") not in {"failed", "interrupted"}:
        raise ConfigurationError(f"{context} is not at a retry boundary")
    if (
        previous.get("retry_authorized") is not True
        or previous.get("failure_category") not in RETRYABLE_FAILURE_CATEGORIES
    ):
        raise ConfigurationError(
            f"prior {context} failure is not infrastructure-retry eligible"
        )


def retry_contract() -> dict[str, Any]:
    return {
        "maximum_attempts": 2,
        "automatic_or_reentry_retry_requires_authorization": True,
        "retryable_exception_types": list(
            RETRYABLE_INFRASTRUCTURE_EXCEPTION_NAMES
        ),
        "retryable_oserror_errno_names": list(
            RETRYABLE_INFRASTRUCTURE_ERRNO_NAMES
        ),
        "orphaned_running_attempt_retryable": True,
        "keyboard_interrupt_retryable": False,
        "configuration_audit_or_scientific_failure_retryable": False,
        "retryable_failure_categories": sorted(RETRYABLE_FAILURE_CATEGORIES),
    }


@dataclass(frozen=True)
class FormalJob:
    index: int
    scenario: str
    arm: str
    policy_seed: int

    @property
    def constrained(self) -> bool:
        return self.arm == ARM_CONSTRAINED

    @property
    def environment_variant(self) -> str:
        return ARM_VARIANTS[self.arm]

    @property
    def selection_mode(self) -> str:
        return (
            "avoidable_switch_budget_constrained"
            if self.constrained
            else "legacy_lexicographic"
        )

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
            "environment_variant": self.environment_variant,
            "constraint_enabled": self.constrained,
            "validation_selection_mode": self.selection_mode,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return smoke.canonical_json_bytes(value)


def sha256_json(value: Any) -> str:
    return smoke.sha256_json(value)


def canonical_json_value(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    return smoke.self_hashed(body, field)


def validate_self_hash(record: Mapping[str, Any], field: str) -> None:
    try:
        smoke.validate_self_hash(record, field)
    except smoke.ConfigurationError as error:
        raise ConfigurationError(str(error)) from error


def read_json(path: Path) -> dict[str, Any]:
    try:
        return smoke.read_json(path)
    except smoke.ConfigurationError as error:
        raise ConfigurationError(str(error)) from error


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(value):
            raise ConfigurationError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, dict(value))


def all_finite(value: Any) -> bool:
    return smoke.all_finite(value)


def build_jobs() -> list[FormalJob]:
    cells = (
        (scenario, arm, policy_seed)
        for scenario in SCENARIOS
        for policy_seed in POLICY_SEEDS
        for arm in ARMS
    )
    return [
        FormalJob(index, scenario, arm, policy_seed)
        for index, (scenario, arm, policy_seed) in enumerate(cells)
    ]


def validate_seed_registry() -> None:
    if POLICY_SEEDS != EXPECTED_POLICY_SEEDS:
        raise ConfigurationError("formal policy-seed derivation drifted")
    if len(set(POLICY_SEEDS)) != len(POLICY_SEEDS):
        raise ConfigurationError("formal policy seeds are not unique")
    workload_panels = {
        "formal_train": set(TRAIN_WORKLOAD_SEEDS),
        "formal_selection": set(SELECTION_WORKLOAD_SEEDS),
        "formal_gate": set(GATE_WORKLOAD_SEEDS),
        "sealed_test": set(SEALED_TEST_WORKLOAD_SEEDS),
        "smoke_train": set(smoke.SMOKE_TRAIN_WORKLOAD_SEEDS),
        "smoke_validation": set(smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS),
    }
    names = tuple(workload_panels)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if workload_panels[left] & workload_panels[right]:
                raise ConfigurationError(f"workload panels overlap: {left}/{right}")
    retired = smoke._expand_ranges(smoke.RETIRED_WORKLOAD_RANGES)
    overlaps = {
        name: sorted(panel & retired)
        for name, panel in workload_panels.items()
        if panel & retired
    }
    if overlaps:
        raise ConfigurationError(f"formal seed registry intersects denylist: {overlaps}")
    if (
        formal_statistics.STUDY_NAME != STUDY_NAME
        or tuple(formal_statistics.SCENARIOS) != SCENARIOS
        or tuple(formal_statistics.ARMS) != ARMS
        or tuple(formal_statistics.POLICY_SEEDS) != POLICY_SEEDS
        or tuple(formal_statistics.TRAIN_WORKLOAD_SEEDS) != TRAIN_WORKLOAD_SEEDS
        or tuple(formal_statistics.SELECTION_WORKLOAD_SEEDS)
        != SELECTION_WORKLOAD_SEEDS
        or tuple(formal_statistics.GATE_WORKLOAD_SEEDS) != GATE_WORKLOAD_SEEDS
    ):
        raise ConfigurationError("formal runner/statistics registry drifted")


def validate_validation_workloads(
    workload_seeds: Iterable[int], panel: str
) -> tuple[int, ...]:
    if panel not in VALIDATION_PANELS:
        raise ConfigurationError("unknown formal validation panel")
    observed = tuple(int(seed) for seed in workload_seeds)
    if observed != PANEL_WORKLOAD_SEEDS[panel]:
        raise ConfigurationError(f"evaluation workload panel is not {panel}")
    if set(observed) & set(SEALED_TEST_WORKLOAD_SEEDS):
        raise ConfigurationError("sealed-test workload reached validation execution")
    return observed


def _git_output(repository_root: Path, arguments: Sequence[str]) -> str:
    try:
        return smoke._git_output(repository_root, arguments)
    except subprocess.CalledProcessError as error:
        raise ConfigurationError("git fingerprint command failed") from error


def code_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.project.parent.resolve()
    paths = {
        "runner": Path(__file__).resolve(),
        "protocol": repository_root / "docs" / PROTOCOL_FILENAME,
        "smoke_runner_dependency": Path(smoke.__file__).resolve(),
        "artifact_io_dependency": args.project / "ablation_matrix_runner.py",
        "formal_statistics": args.project / "formal_avoidable_switch_statistics.py",
        "hierarchical_statistics": args.project / "hierarchical_statistics.py",
        "classical_baselines_protocol": (
            repository_root
            / "docs"
            / "AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md"
        ),
        "trainer_snapshot": args.project / "cleanmarl_mappo_leo.py",
        "external_trainer": args.cleanmarl / "cleanmarl" / "mappo.py",
        "design_snapshot": args.project / "mappo_design.py",
        "external_design": args.cleanmarl / "cleanmarl" / "mappo_design.py",
        "environment": args.project / "leo_multiagent_env.py",
        "base_environment": args.project / "leo_marl_env.py",
        "wrapper": args.project / "cleanmarl_leo_multiagent_wrapper.py",
        "variant_definitions": args.project / "variant_definitions.py",
        "evaluation": args.project / "mappo_evaluation.py",
        **{
            f"preflight_test_{module}": args.project / f"{module}.py"
            for module in PREFLIGHT_TEST_MODULES
        },
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ConfigurationError(f"formal runtime source files are missing: {missing}")
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
    cuda_name = None
    cuda_capability = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
        cuda_capability = list(torch.cuda.get_device_capability(0))
    try:
        dependency_process = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise ConfigurationError("cannot capture the Python dependency inventory") from error
    dependency_inventory = sorted(
        {
            line.strip()
            for line in dependency_process.stdout.splitlines()
            if line.strip()
        },
        key=lambda line: (line.casefold(), line),
    )
    dependency_inventory_bytes = (
        ("\n".join(dependency_inventory) + "\n").encode("utf-8")
        if dependency_inventory
        else b""
    )
    return {
        "files": hashes,
        "git_head": _git_output(repository_root, ["rev-parse", "HEAD"]).strip(),
        "relevant_dirty_diff_sha256": hashlib.sha256(
            diff.encode("utf-8")
        ).hexdigest(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_name": cuda_name,
        "cuda_compute_capability": cuda_capability,
        "platform": platform.platform(),
        "python_dependency_inventory_command": [
            str(Path(sys.executable).resolve()),
            "-m",
            "pip",
            "freeze",
            "--all",
        ],
        "python_dependency_inventory": dependency_inventory,
        "python_dependency_inventory_normalization": (
            "trim_nonempty_exact_deduplicate_casefold_then_codepoint_sort_lf"
        ),
        "python_dependency_inventory_sha256": hashlib.sha256(
            dependency_inventory_bytes
        ).hexdigest(),
    }


def assert_runtime_fingerprint(
    args: argparse.Namespace,
    expected: Mapping[str, Any],
) -> None:
    if code_fingerprint(args) != dict(expected):
        raise ConfigurationError("formal runtime fingerprint changed after preregistration")


def constraint_contract() -> dict[str, Any]:
    contract = smoke.constraint_contract()
    if (
        contract.get("budget") != SWITCH_BUDGET
        or contract.get("dual_learning_rate") != DUAL_LEARNING_RATE
        or contract.get("dual_initial") != DUAL_INITIAL
        or contract.get("dual_projection") != [0.0, DUAL_MAX]
    ):
        raise ConfigurationError("formal constraint differs from completed smoke")
    return contract


def _validate_hashed_artifacts(record: Mapping[str, Any], context: str) -> None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ConfigurationError(f"{context} omits artifacts")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ConfigurationError(f"{context} artifact is malformed: {name}")
        path = Path(str(artifact.get("path", ""))).resolve()
        expected = artifact.get("sha256")
        if not path.is_file() or expected != sha256_file(path):
            raise ConfigurationError(f"{context} artifact hash mismatch: {name}")


def validate_exact_resume_runtime_binding(
    smoke_spec: Mapping[str, Any], current_fingerprint: Mapping[str, Any]
) -> dict[str, Any]:
    smoke_fingerprint = smoke_spec.get("code_fingerprint")
    smoke_files = (
        smoke_fingerprint.get("files")
        if isinstance(smoke_fingerprint, Mapping)
        else None
    )
    current_files = current_fingerprint.get("files")
    if not isinstance(smoke_files, Mapping) or not isinstance(current_files, Mapping):
        raise ConfigurationError("exact-resume runtime fingerprint is malformed")
    bound_files = {}
    for current_name, smoke_name in EXACT_RESUME_RUNTIME_FILE_BINDINGS.items():
        current_hash = current_files.get(current_name)
        smoke_hash = smoke_files.get(smoke_name)
        if (
            not isinstance(current_hash, str)
            or len(current_hash) != 64
            or current_hash != smoke_hash
        ):
            raise ConfigurationError(
                f"exact-resume runtime source drifted: {current_name}"
            )
        bound_files[current_name] = current_hash
    bound_runtime = {}
    for field in EXACT_RESUME_RUNTIME_SCALAR_BINDINGS:
        current_value = current_fingerprint.get(field)
        smoke_value = smoke_fingerprint.get(field)
        if current_value is None or current_value != smoke_value:
            raise ConfigurationError(
                f"exact-resume runtime dependency drifted: {field}"
            )
        bound_runtime[field] = current_value
    return {
        "status": "passed_for_fields_recorded_by_smoke_v1_r1",
        "bound_files": bound_files,
        "bound_runtime_fields": bound_runtime,
        "current_fields_not_recorded_by_smoke_v1_r1": list(
            EXACT_RESUME_RUNTIME_FIELDS_NOT_RECORDED_BY_SMOKE
        ),
    }


def validate_smoke_prerequisite(
    repository_root: Path, current_fingerprint: Mapping[str, Any]
) -> dict[str, Any]:
    evidence = (
        repository_root / "experiments" / SMOKE_EVIDENCE_DIRECTORY_NAME
    ).resolve()
    rejected = (
        repository_root
        / "experiments"
        / REJECTED_SMOKE_EVIDENCE_DIRECTORY_NAME
    ).resolve()
    if evidence == rejected or evidence.name != SMOKE_EVIDENCE_DIRECTORY_NAME:
        raise ConfigurationError("formal study must bind only to smoke v1-r1")
    records: dict[str, dict[str, Any]] = {}
    for filename, expected_sha256 in SMOKE_EVIDENCE_FILE_SHA256.items():
        path = evidence / filename
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ConfigurationError(f"completed smoke evidence hash mismatch: {filename}")
        records[filename] = read_json(path)

    report = records["smoke_report.json"]
    freeze = records["training_freeze.json"]
    preregistration = records["preregistration.json"]
    exact_resume = records["exact_resume_equivalence.json"]
    validate_self_hash(report, "report_sha256")
    validate_self_hash(freeze, "freeze_sha256")
    validate_self_hash(preregistration, "spec_sha256")
    validate_self_hash(exact_resume, "proof_sha256")
    if (
        report.get("study_name") != smoke.STUDY_NAME
        or report.get("mechanism_gates_passed") is not True
        or report.get("next_step") != "separate_formal_preregistration_required"
        or report.get("paper_claim_allowed") is not False
        or report.get("promotion_decision_allowed") is not False
        or report.get("test_panel_consulted") is not False
        or report.get("test_access_count") != 0
        or report.get("sealed_test_opened") is not False
    ):
        raise ConfigurationError("completed smoke report does not authorize formal work")
    if not all(report.get("mechanism_gates", {}).values()):
        raise ConfigurationError("completed smoke mechanism gate failed")
    if (
        report.get("freeze_sha256") != freeze.get("freeze_sha256")
        or report.get("spec_sha256") != preregistration.get("spec_sha256")
        or exact_resume.get("spec_sha256") != preregistration.get("spec_sha256")
        or exact_resume.get("status") != "passed"
        or exact_resume.get("proof_sha256")
        != "96ccb677d7299c290570aa058145a676f8319f94430176d8d0b5ab07984e059b"
    ):
        raise ConfigurationError("completed smoke evidence cross-binding failed")
    exact_resume_runtime_binding = validate_exact_resume_runtime_binding(
        preregistration, current_fingerprint
    )
    _validate_hashed_artifacts(exact_resume, "exact-resume proof")
    jobs = freeze.get("jobs")
    if not isinstance(jobs, Mapping) or len(jobs) != 4:
        raise ConfigurationError("completed smoke training freeze is incomplete")
    for job_id, artifact in jobs.items():
        if not isinstance(artifact, Mapping):
            raise ConfigurationError(f"completed smoke job is malformed: {job_id}")
        _validate_hashed_artifacts(artifact, f"completed smoke job {job_id}")
    return {
        "directory": str(evidence),
        "files": {
            filename: {
                "path": str((evidence / filename).resolve()),
                "sha256": expected,
            }
            for filename, expected in SMOKE_EVIDENCE_FILE_SHA256.items()
        },
        "smoke_spec_sha256": preregistration["spec_sha256"],
        "smoke_freeze_sha256": freeze["freeze_sha256"],
        "smoke_report_sha256": report["report_sha256"],
        "exact_resume_proof_sha256": exact_resume["proof_sha256"],
        "exact_resume_runtime_binding": exact_resume_runtime_binding,
    }


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    validate_seed_registry()
    if canonical_variant_name("no_lifetime") != ARM_VARIANTS[ARM_REWARD_CONTROL]:
        raise ConfigurationError("reward-shaped control alias no longer resolves to proposed")
    fingerprint = code_fingerprint(args)
    smoke_evidence = validate_smoke_prerequisite(
        args.project.parent.resolve(), fingerprint
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "stage": "formal_training_and_validation_only",
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "sealed_test_access_authorized": False,
        "retry_contract": retry_contract(),
        "scenarios": list(SCENARIOS),
        "arms": list(ARMS),
        "arm_variants": dict(ARM_VARIANTS),
        "reward_shaped_control_legacy_alias": "no_lifetime",
        "policy_seed_derivation": {
            "namespace": POLICY_SEED_NAMESPACE,
            "indices": list(range(len(POLICY_SEEDS))),
            "algorithm": "sha256_first_four_bytes_big_endian_high_bit_cleared",
        },
        "policy_seeds": list(POLICY_SEEDS),
        "workload_registry": {
            "train": list(TRAIN_WORKLOAD_SEEDS),
            "checkpoint_selection": list(SELECTION_WORKLOAD_SEEDS),
            "independent_validation_gate": list(GATE_WORKLOAD_SEEDS),
            "sealed_test_metadata_only": list(SEALED_TEST_WORKLOAD_SEEDS),
        },
        "constraint": constraint_contract(),
        "training": {
            "target_environment_steps": TIMESTEPS,
            "expected_completed_rollout_boundary": EXPECTED_FINAL_ENVIRONMENT_STEPS,
            "episode_steps": EPISODE_STEPS,
            "batch_size": BATCH_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "epochs": EPOCHS,
            "num_minibatches": NUM_MINIBATCHES,
            "validation_every_rollouts": VALIDATION_EVERY_ROLLOUTS,
            "checkpoint_selection_episodes": len(SELECTION_WORKLOAD_SEEDS),
            "selection_feasibility_numerical_tolerance": (
                SELECTION_FEASIBILITY_NUMERICAL_TOLERANCE
            ),
            "save_every_steps": SAVE_EVERY_STEPS,
            "maximum_parallel_jobs": 2,
            "maximum_launch_attempts_per_job": MAX_TRAINING_LAUNCH_ATTEMPTS,
            "maximum_infrastructure_retries_per_job": (
                MAX_TRAINING_LAUNCH_ATTEMPTS - 1
            ),
            "performance_early_stopping": False,
            "exact_resume_checkpoint": "latest.pt",
            "total_timesteps_mutable_on_resume": False,
            "frozen_semantics": dict(FROZEN_TRAINING_SEMANTICS),
        },
        "post_training_validation": {
            "source_checkpoint": "selected_validation_checkpoint",
            "evaluation_device": args.evaluation_device,
            "selection_recheck_workloads": list(SELECTION_WORKLOAD_SEEDS),
            "selection_recheck_expected_rows": EXPECTED_SELECTION_RECHECK_ROWS,
            "independent_gate_workloads": list(GATE_WORKLOAD_SEEDS),
            "independent_gate_expected_rows": EXPECTED_GATE_ROWS,
            "total_audit_rows": EXPECTED_VALIDATION_ROWS,
            "structured_interface": (
                "mappo_evaluation.evaluate_policy_with_constraint_metrics"
            ),
            "maximum_launch_attempts_per_shard": (
                MAX_EVALUATION_LAUNCH_ATTEMPTS
            ),
            "maximum_infrastructure_retries_per_shard": (
                MAX_EVALUATION_LAUNCH_ATTEMPTS - 1
            ),
            "decision_rate_aggregation": "ratio_of_summed_counts",
            "delivery_noninferiority_margin": DELIVERY_NONINFERIORITY_MARGIN,
            "independent_gate_budget_numerical_tolerance": 0.0,
            "inferential_status": "development_gate_not_paper_evidence",
        },
        "retry_policy": retry_contract(),
        "preflight_tests": {
            "modules": list(PREFLIGHT_TEST_MODULES),
            "expected_test_count": EXPECTED_PREFLIGHT_TEST_COUNT,
            "file_sha256": {
                module: fingerprint["files"][f"preflight_test_{module}"]
                for module in PREFLIGHT_TEST_MODULES
            },
            "discovery_or_globbing_allowed": False,
            "formal_or_sealed_workload_instantiation_allowed": False,
        },
        "jobs": [job.as_dict() for job in build_jobs()],
        "expected_training_jobs": EXPECTED_TRAINING_JOBS,
        "expected_selection_recheck_rows": EXPECTED_SELECTION_RECHECK_ROWS,
        "expected_gate_rows": EXPECTED_GATE_ROWS,
        "expected_validation_rows": EXPECTED_VALIDATION_ROWS,
        "expected_test_evaluations": 0,
        "smoke_prerequisite": smoke_evidence,
        "paths": {
            "project": str(args.project.resolve()),
            "cleanmarl": str(args.cleanmarl.resolve()),
            "protocol": str(
                (args.project.parent / "docs" / PROTOCOL_FILENAME).resolve()
            ),
        },
        "code_fingerprint": fingerprint,
    }
    return self_hashed(body, "spec_sha256")


def validate_runtime_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise ConfigurationError("LEO_REWARD_OVERRIDES must be unset")
    if args.max_parallel not in {1, 2}:
        raise ConfigurationError("max_parallel must be one or two")
    if args.project.resolve() != Path(__file__).resolve().parent:
        raise ConfigurationError(
            "formal --project must be the source directory importing this runner"
        )
    if args.device != "cuda":
        raise ConfigurationError("formal training requires the frozen CUDA device")
    if args.evaluation_device != args.device or args.evaluation_device != "cuda":
        raise ConfigurationError(
            "formal training and validation require the same frozen CUDA device"
        )
    expected_python = Path("F:/leo-venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ConfigurationError(
            f"formal runner requires {expected_python}, observed {sys.executable}"
        )


def expected_run_config(job: FormalJob, args: argparse.Namespace) -> dict[str, Any]:
    return {
        **FROZEN_TRAINING_SEMANTICS,
        "env_type": "leo_multi",
        "env_name": job.scenario,
        "leo_project_path": str(args.project),
        "leo_variant": job.environment_variant,
        "seed": job.policy_seed,
        "batch_size": BATCH_SIZE,
        "total_timesteps": TIMESTEPS,
        "epochs": EPOCHS,
        "num_minibatches": NUM_MINIBATCHES,
        "eval_steps": VALIDATION_EVERY_ROLLOUTS,
        "num_eval_ep": len(SELECTION_WORKLOAD_SEEDS),
        "save_every_steps": SAVE_EVERY_STEPS,
        "train_seed_start": TRAIN_WORKLOAD_SEEDS[0],
        "train_seed_count": len(TRAIN_WORKLOAD_SEEDS),
        "validation_seed_start": SELECTION_WORKLOAD_SEEDS[0],
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
        "validation_selection_mode": job.selection_mode,
    }


def build_command(
    job: FormalJob,
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
        job.environment_variant,
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
        str(len(SELECTION_WORKLOAD_SEEDS)),
        "--save-every-steps",
        str(SAVE_EVERY_STEPS),
        "--checkpoint-dir",
        str(run_root),
        "--run-tag",
        f"{STUDY_NAME}-{job.arm}",
        "--train-seed-start",
        str(TRAIN_WORKLOAD_SEEDS[0]),
        "--train-seed-count",
        str(len(TRAIN_WORKLOAD_SEEDS)),
        "--validation-seed-start",
        str(SELECTION_WORKLOAD_SEEDS[0]),
        "--validation-selection-mode",
        job.selection_mode,
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
    return smoke._path_within(path, root)


def _paths_overlap(left: Path, right: Path) -> bool:
    return smoke._paths_overlap(left, right)


def validate_output_isolation(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    project_source = args.project.resolve()
    repository_root = project_source.parent
    experiments_root = (repository_root / "experiments").resolve()
    if output.parent != experiments_root or output.name in {
        "",
        ".",
        "..",
        SMOKE_EVIDENCE_DIRECTORY_NAME,
        REJECTED_SMOKE_EVIDENCE_DIRECTORY_NAME,
    }:
        raise ConfigurationError(
            "formal output must be one dedicated child of the experiments directory"
        )
    protected_paths = {
        "project source": project_source,
        "repository metadata": repository_root / ".git",
        "protocol documents": repository_root / "docs",
        "CleanMARL runtime": args.cleanmarl.resolve(),
        "completed smoke v1-r1": experiments_root / SMOKE_EVIDENCE_DIRECTORY_NAME,
        "failed smoke v1": experiments_root / REJECTED_SMOKE_EVIDENCE_DIRECTORY_NAME,
    }
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
    if overlaps:
        raise ConfigurationError(
            "formal output overlaps protected evidence/runtime paths: "
            f"{sorted(set(overlaps))}"
        )


def validate_new_output_state(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise ConfigurationError("formal output exists and is not a directory")
    preregistration = output / "preregistration.json"
    if any(output.iterdir()) and not preregistration.is_file():
        raise ConfigurationError(
            "non-empty formal output has no immutable preregistration"
        )


def job_root(args: argparse.Namespace, job: FormalJob) -> Path:
    return (
        args.output
        / "checkpoints"
        / job.scenario
        / job.arm
        / f"seed_{job.policy_seed}"
    )


def training_status_path(args: argparse.Namespace, job: FormalJob) -> Path:
    return args.output / "job_status" / "training" / f"{job.slug}.json"


def evaluation_status_path(
    args: argparse.Namespace, job: FormalJob, panel: str
) -> Path:
    if panel not in VALIDATION_PANELS:
        raise ConfigurationError("unknown formal validation panel")
    return (
        args.output
        / "job_status"
        / "validation"
        / panel
        / f"{job.slug}.json"
    )


def _run_config_matches(
    config: Mapping[str, Any],
    job: FormalJob,
    args: argparse.Namespace,
) -> bool:
    return all(
        config.get(field) == expected
        for field, expected in expected_run_config(job, args).items()
    )


def _manifest_artifact(run_directory: Path, value: Any, field: str) -> Path:
    try:
        return smoke._manifest_artifact(run_directory, value, field)
    except smoke.ConfigurationError as error:
        raise ConfigurationError(str(error)) from error


def read_metrics(path: Path) -> list[dict[str, Any]]:
    try:
        return smoke.read_metrics(path)
    except smoke.ConfigurationError as error:
        raise ConfigurationError(str(error)) from error


def _close(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return smoke._close(left, right, tolerance=tolerance)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("numeric audit comparison failed") from error


def discover_run_directory(
    args: argparse.Namespace,
    job: FormalJob,
) -> tuple[Path | None, Path | None]:
    root = job_root(args, job).resolve()
    if not root.exists():
        return None, None
    compatible: list[tuple[Path, bool, Path | None]] = []
    for config_path in sorted(root.glob("*/run_config.json")):
        run_directory = config_path.parent.resolve()
        if not _path_within(run_directory, root):
            raise ConfigurationError("training run escaped its formal job root")
        config = read_json(config_path)
        if not _run_config_matches(config, job, args):
            raise ConfigurationError(
                f"run config drift under formal job root: {run_directory}"
            )
        manifest = run_directory / "run_manifest.json"
        latest = run_directory / "latest.pt"
        compatible.append(
            (
                run_directory,
                manifest.is_file(),
                latest if latest.is_file() else None,
            )
        )
    completed = [item for item in compatible if item[1]]
    resumable = [item for item in compatible if not item[1] and item[2] is not None]
    if len(completed) > 1:
        raise ConfigurationError(f"multiple completed runs exist for {job.job_id}")
    if completed and resumable:
        raise ConfigurationError(
            f"completed and resumable runs coexist for {job.job_id}"
        )
    if len(resumable) > 1:
        raise ConfigurationError(f"multiple resumable runs exist for {job.job_id}")
    if completed:
        return completed[0][0], None
    if resumable:
        return resumable[0][0], resumable[0][2]
    return None, None


def freeze_resume_source(
    checkpoint: Path,
    run_directory: Path,
    attempt_number: int,
) -> tuple[Path, str]:
    checkpoint = checkpoint.resolve()
    run_directory = run_directory.resolve()
    if checkpoint != run_directory / "latest.pt" or not checkpoint.is_file():
        raise ConfigurationError("resume source is not the same-run latest.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("resume_boundary") != (
        "completed_update"
    ):
        raise ConfigurationError("resume source is not a completed-update checkpoint")
    source_hash = sha256_file(checkpoint)
    destination = run_directory / f"resume_source_attempt_{attempt_number}.pt"
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise ConfigurationError("immutable resume source already differs")
    else:
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            shutil.copy2(checkpoint, temporary)
            if sha256_file(temporary) != source_hash:
                raise ConfigurationError("resume source copy hash mismatch")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination.resolve(), source_hash


def audit_constraint_records(
    records: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return smoke.audit_constraint_records(records, final_state)
    except smoke.ConfigurationError as error:
        raise ConfigurationError(str(error)) from error


def _legacy_validation_score(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(record["delivery_ratio"]),
        float(record["mean_reward"]),
        -float(record["drop_rate"]),
        -float(record["average_delay_slots"]),
    )


def audit_validation_selection(
    job: FormalJob,
    manifest: Mapping[str, Any],
    validations: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    run_directory: Path,
    selected_path: Path,
    selected_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    expected_validation_steps = tuple(
        rollout * ROLLOUT_STEPS
        for rollout in range(
            VALIDATION_EVERY_ROLLOUTS,
            EXPECTED_FINAL_ENVIRONMENT_STEPS // ROLLOUT_STEPS + 1,
            VALIDATION_EVERY_ROLLOUTS,
        )
    )
    observed_steps = tuple(record.get("environment_steps") for record in validations)
    if observed_steps != expected_validation_steps:
        raise ConfigurationError("validation candidate boundaries drifted")
    if manifest.get("validation_candidate_count") != len(validations):
        raise ConfigurationError("validation candidate count mismatch")
    if job.constrained:
        if len(selections) != 1:
            raise ConfigurationError(
                "constrained run requires exactly one validation selection record"
            )
    elif selections:
        raise ConfigurationError(
            "legacy-selection arm unexpectedly contains a selection record"
        )

    selection_spec = manifest.get("validation_selection_spec")
    if not isinstance(selection_spec, Mapping):
        raise ConfigurationError("run manifest omits validation selection spec")
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
    expected_spec = {
        "schema_version": 1,
        "mode": job.selection_mode,
        "delivery_tolerance": 0.0,
        "class_2_tolerance": 0.0,
        "switch_budget": SWITCH_BUDGET if job.constrained else None,
        "source": "validation_only",
        "test_panel_consulted": False,
        "validation_seed_start": SELECTION_WORKLOAD_SEEDS[0],
        "validation_episodes": len(SELECTION_WORKLOAD_SEEDS),
        "metric_aggregation": expected_aggregation,
        "selection_timing": (
            "after_all_validation_candidates"
            if job.constrained
            else "online_legacy_compatible"
        ),
    }
    if dict(selection_spec) != expected_spec:
        raise ConfigurationError("validation selection specification drifted")
    if selected_checkpoint.get("validation_selection_spec") != expected_spec:
        raise ConfigurationError("selected checkpoint selection specification drifted")

    for record in validations:
        missing = set(SELECTED_VALIDATION_METRIC_FIELDS).difference(record)
        if missing:
            raise ConfigurationError(
                f"validation candidate omits metrics: {sorted(missing)}"
            )
        if (
            record.get("seed_start") != SELECTION_WORKLOAD_SEEDS[0]
            or record.get("episodes") != len(SELECTION_WORKLOAD_SEEDS)
        ):
            raise ConfigurationError("validation candidate workload panel drifted")
        _audit_decision_ledger(record, "validation candidate")
        for field in (
            "routing_switches_total",
            "avoidable_routing_switches_total",
            "forced_routing_switches_total",
            "switch_opportunities",
        ):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"validation candidate {field} is invalid")
        if (
            record["routing_switches_total"]
            != record["avoidable_routing_switches_total"]
            + record["forced_routing_switches_total"]
            or record["avoidable_routing_switches_total"]
            > record["switch_opportunities"]
        ):
            raise ConfigurationError("validation candidate accepted-switch ledger drifted")
        expected_accepted_rate = record["avoidable_routing_switches_total"] / max(
            1, record["switch_opportunities"]
        )
        if not _close(record["avoidable_switch_rate"], expected_accepted_rate):
            raise ConfigurationError("validation candidate accepted-switch rate drifted")
        episodes = record["episodes"]
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
            raise ConfigurationError("validation candidate episode count is invalid")
        for mean_field, total_field in (
            ("routing_switches", "routing_switches_total"),
            ("avoidable_routing_switches", "avoidable_routing_switches_total"),
            ("forced_routing_switches", "forced_routing_switches_total"),
        ):
            if not _close(record[mean_field], record[total_field] / episodes):
                raise ConfigurationError(
                    f"validation candidate {mean_field} aggregation drifted"
                )

    try:
        expected_selected = select_leo_validation_record(
            validations,
            mode=job.selection_mode,
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
                raise ConfigurationError("legacy validation-best marker drifted")

    selected_metrics = manifest.get("selected_validation_metrics")
    expected_metrics = {
        field: expected_selected[field]
        for field in SELECTED_VALIDATION_METRIC_FIELDS
    }
    if not isinstance(selected_metrics, Mapping) or dict(selected_metrics) != (
        expected_metrics
    ):
        raise ConfigurationError("manifest-selected validation metrics drifted")
    expected_score = list(_legacy_validation_score(expected_selected))
    observed_score = manifest.get("best_validation_score")
    if (
        not isinstance(observed_score, list)
        or len(observed_score) != len(expected_score)
        or any(
            not _close(left, right)
            for left, right in zip(observed_score, expected_score)
        )
    ):
        raise ConfigurationError("manifest best-validation score drifted")

    selected_step = int(expected_selected["environment_steps"])
    if selected_checkpoint.get("step") != selected_step:
        raise ConfigurationError("selected checkpoint step drifted")
    selected_hash = sha256_file(selected_path)
    if manifest.get("selected_validation_checkpoint_sha256") != selected_hash:
        raise ConfigurationError("selected checkpoint hash drifted")

    if job.constrained:
        selection = selections[0]
        if (
            selection.get("selection_spec") != expected_spec
            or selection.get("candidate_count") != len(validations)
            or selection.get("selected_candidate_step") != selected_step
            or selection.get("selected_metrics") != expected_metrics
            or selection.get("selected_checkpoint_sha256") != selected_hash
        ):
            raise ConfigurationError("constrained validation selection record drifted")
        promoted = _manifest_artifact(
            run_directory,
            selection.get("validation_best_checkpoint"),
            "selection validation_best_checkpoint",
        )
        candidate = _manifest_artifact(
            run_directory,
            selection.get("selected_candidate_checkpoint"),
            "selection selected_candidate_checkpoint",
        )
        expected_candidate = _manifest_artifact(
            run_directory,
            expected_selected.get("candidate_checkpoint"),
            "selected candidate checkpoint",
        )
        if (
            promoted != selected_path.resolve()
            or candidate != expected_candidate
            or sha256_file(candidate) != selected_hash
        ):
            raise ConfigurationError("selected candidate promotion drifted")
        expected_feasible = bool(
            float(expected_metrics["decision_avoidable_switch_rate"])
            <= SWITCH_BUDGET + SELECTION_FEASIBILITY_NUMERICAL_TOLERANCE
        )
        if selection.get("constraint_feasible") is not expected_feasible:
            raise ConfigurationError("selection feasibility flag drifted")
    return expected_metrics


def _audit_decision_ledger(
    metrics: Mapping[str, Any],
    context: str,
) -> float | None:
    cost = metrics.get("decision_avoidable_switches")
    opportunity = metrics.get("decision_switch_opportunities")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, int)
        or isinstance(opportunity, bool)
        or not isinstance(opportunity, int)
        or cost < 0
        or opportunity < 0
        or cost > opportunity
    ):
        raise ConfigurationError(f"{context} decision ledger is invalid")
    rate = metrics.get("decision_avoidable_switch_rate")
    if opportunity == 0:
        if cost != 0 or rate is not None:
            raise ConfigurationError(f"{context} zero-opportunity rate must be undefined")
        return None
    expected = cost / opportunity
    if rate is None or not _close(rate, expected):
        raise ConfigurationError(f"{context} decision rate is not ratio-of-sums")
    return expected


def _checkpoint_state_equal(left: Any, right: Any) -> bool:
    return smoke._state_equal(left, right)


def _checkpoint_inventory(run_directory: Path) -> dict[str, dict[str, Any]]:
    checkpoints = sorted(run_directory.glob("*.pt"), key=lambda path: path.name)
    if not checkpoints:
        raise ConfigurationError("training run contains no checkpoints")
    return {
        path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in checkpoints
    }


def audit_constrained_validation_candidates(
    args: argparse.Namespace,
    job: FormalJob,
    validations: Sequence[Mapping[str, Any]],
    run_directory: Path,
    inventory: Mapping[str, Mapping[str, Any]],
    selected_checkpoint: Mapping[str, Any],
) -> None:
    if not job.constrained:
        raise ConfigurationError("candidate checkpoint audit requires constrained arm")
    expected_names = {
        f"validation_candidate_step_{int(record['environment_steps'])}.pt"
        for record in validations
    }
    observed_names = {
        name for name in inventory if name.startswith("validation_candidate_step_")
    }
    if observed_names != expected_names:
        raise ConfigurationError("constrained validation candidate inventory drifted")
    schema_fields = (
        "obs_size",
        "state_size",
        "action_size",
        "n_agents",
        "candidate_feature_dim",
        "candidate_actor_spec",
        "critic_spec",
        "switch_regularizer_spec",
        "switch_constraint_spec",
        "validation_selection_spec",
    )
    expected_constraint = constraint_contract()
    for record in validations:
        step = record.get("environment_steps")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ConfigurationError("validation candidate step is invalid")
        expected_path = (
            run_directory / f"validation_candidate_step_{step}.pt"
        ).resolve()
        candidate_path = _manifest_artifact(
            run_directory,
            record.get("candidate_checkpoint"),
            f"validation candidate step {step}",
        )
        inventory_entry = inventory.get(expected_path.name)
        if (
            candidate_path != expected_path
            or not isinstance(inventory_entry, Mapping)
            or inventory_entry.get("path") != str(expected_path)
            or inventory_entry.get("sha256") != sha256_file(expected_path)
        ):
            raise ConfigurationError(
                f"validation candidate artifact binding drifted: step {step}"
            )
        payload = torch.load(expected_path, map_location="cpu", weights_only=False)
        checkpoint_args = payload.get("args") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or payload.get("step") != step
            or payload.get("resume_boundary") != "inference_snapshot_only"
            or not isinstance(checkpoint_args, Mapping)
            or not _run_config_matches(checkpoint_args, job, args)
            or any(
                not _checkpoint_state_equal(
                    payload.get(field), selected_checkpoint.get(field)
                )
                for field in schema_fields
            )
        ):
            raise ConfigurationError(
                f"validation candidate checkpoint payload drifted: step {step}"
            )
        contract = payload.get("switch_constraint_spec")
        if not isinstance(contract, Mapping) or any(
            contract.get(field) != value
            for field, value in expected_constraint.items()
        ):
            raise ConfigurationError(
                f"validation candidate constraint schema drifted: step {step}"
            )
        if not isinstance(payload.get("switch_constraint_state"), Mapping):
            raise ConfigurationError(
                f"validation candidate constraint state missing: step {step}"
            )


def _audit_resume_records(
    records: Sequence[Mapping[str, Any]],
    run_directory: Path,
) -> list[dict[str, Any]]:
    audited = []
    for record in records:
        if record.get("record_type") != "resume_event":
            continue
        source = Path(str(record.get("source_checkpoint", ""))).resolve()
        if (
            not _path_within(source, run_directory)
            or not source.name.startswith("resume_source_attempt_")
            or source.suffix != ".pt"
            or not source.is_file()
        ):
            raise ConfigurationError("resume event source is not an immutable same-run copy")
        observed_hash = sha256_file(source)
        if record.get("source_checkpoint_sha256") != observed_hash:
            raise ConfigurationError("resume event source checkpoint hash drifted")
        audited.append(
            {
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": observed_hash,
                "environment_steps": int(record.get("environment_steps", -1)),
                "optimizer_updates": int(record.get("optimizer_updates", -1)),
                "update_round": int(record.get("update_round", -1)),
            }
        )
    return audited


def audit_training_job(
    args: argparse.Namespace,
    job: FormalJob,
    run_directory: Path,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if not _path_within(run_directory, job_root(args, job)):
        raise ConfigurationError("training run escaped its formal job root")
    paths = {
        "run_config": run_directory / "run_config.json",
        "training_metrics": run_directory / "training_metrics.jsonl",
        "run_manifest": run_directory / "run_manifest.json",
        "latest_checkpoint": run_directory / "latest.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ConfigurationError(f"training artifacts are missing: {missing}")
    config = read_json(paths["run_config"])
    if not _run_config_matches(config, job, args):
        raise ConfigurationError("completed run config differs from formal job")
    manifest = read_json(paths["run_manifest"])
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
    records = read_metrics(paths["training_metrics"])
    allowed_record_types = {
        "training_update",
        "validation",
        "validation_selection",
        "resume_event",
    }
    unexpected_types = sorted(
        {
            str(record.get("record_type"))
            for record in records
            if record.get("record_type") not in allowed_record_types
        }
    )
    if unexpected_types:
        raise ConfigurationError(f"unexpected training metric records: {unexpected_types}")
    updates = [
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
    expected_update_count = EXPECTED_FINAL_ENVIRONMENT_STEPS // ROLLOUT_STEPS
    if len(updates) != expected_update_count:
        raise ConfigurationError("training rollout count drifted")
    for index, record in enumerate(updates, start=1):
        if (
            record.get("update_round") != index
            or record.get("environment_steps") != index * ROLLOUT_STEPS
        ):
            raise ConfigurationError("training update boundary or sequence drifted")
        optimizer_updates = record.get("optimizer_updates")
        if (
            isinstance(optimizer_updates, bool)
            or not isinstance(optimizer_updates, int)
            or optimizer_updates <= 0
            or (
                index > 1
                and optimizer_updates <= int(updates[index - 2]["optimizer_updates"])
            )
        ):
            raise ConfigurationError("optimizer update sequence drifted")
    if len(validations) != expected_update_count // VALIDATION_EVERY_ROLLOUTS:
        raise ConfigurationError("formal validation candidate count drifted")

    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    selected_checkpoint = torch.load(
        selected_path, map_location="cpu", weights_only=False
    )
    latest_checkpoint = torch.load(
        paths["latest_checkpoint"], map_location="cpu", weights_only=False
    )
    for label, checkpoint in (
        ("final", final_checkpoint),
        ("selected", selected_checkpoint),
        ("latest", latest_checkpoint),
    ):
        if not isinstance(checkpoint, Mapping):
            raise ConfigurationError(f"{label} checkpoint is not a mapping")
        checkpoint_args = checkpoint.get("args")
        if (
            checkpoint.get("candidate_feature_dim") != 26
            or not isinstance(checkpoint_args, Mapping)
            or not _run_config_matches(checkpoint_args, job, args)
            or canonical_variant_name(str(checkpoint_args.get("leo_variant", "")))
            != job.environment_variant
            or checkpoint.get("switch_regularizer_spec") is not None
        ):
            raise ConfigurationError(f"{label} checkpoint contract drifted")
    for label, checkpoint in (("final", final_checkpoint), ("latest", latest_checkpoint)):
        if checkpoint.get("resume_boundary") != "completed_update":
            raise ConfigurationError(f"{label} is not an exact-resume boundary")
    equal_fields = (
        "step",
        "actor",
        "critic",
        "actor_optimizer",
        "critic_optimizer",
        "switch_constraint_spec",
        "switch_constraint_state",
        "trainer_state",
        "validation_selection_spec",
    )
    if any(
        not _checkpoint_state_equal(final_checkpoint.get(field), latest_checkpoint.get(field))
        for field in equal_fields
    ):
        raise ConfigurationError("final and latest checkpoint states differ")

    trainer_state = final_checkpoint.get("trainer_state")
    if not isinstance(trainer_state, Mapping):
        raise ConfigurationError("final checkpoint omits trainer state")
    final_optimizer_updates = int(updates[-1]["optimizer_updates"])
    if (
        manifest.get("environment_steps") != EXPECTED_FINAL_ENVIRONMENT_STEPS
        or final_checkpoint.get("step") != EXPECTED_FINAL_ENVIRONMENT_STEPS
        or trainer_state.get("step") != EXPECTED_FINAL_ENVIRONMENT_STEPS
        or trainer_state.get("update_round") != expected_update_count
        or trainer_state.get("num_episodes") != expected_update_count * BATCH_SIZE
        or trainer_state.get("training_step") != final_optimizer_updates
        or manifest.get("optimizer_updates") != final_optimizer_updates
    ):
        raise ConfigurationError("final training boundary/state drifted")

    constraint_summary = None
    if job.constrained:
        expected_contract = constraint_contract()
        for checkpoint in (final_checkpoint, selected_checkpoint, latest_checkpoint):
            contract = checkpoint.get("switch_constraint_spec")
            if not isinstance(contract, Mapping) or any(
                contract.get(field) != expected
                for field, expected in expected_contract.items()
            ):
                raise ConfigurationError("constrained checkpoint contract drifted")
        final_state = final_checkpoint.get("switch_constraint_state")
        if not isinstance(final_state, Mapping):
            raise ConfigurationError("constrained final checkpoint omits dual state")
        constraint_summary = audit_constraint_records(records, final_state)
        if constraint_summary["cumulative_opportunity"] <= 0:
            raise ConfigurationError("constrained training has zero opportunities")
    else:
        for checkpoint in (final_checkpoint, selected_checkpoint, latest_checkpoint):
            if (
                checkpoint.get("switch_constraint_spec") is not None
                or checkpoint.get("switch_constraint_state") is not None
            ):
                raise ConfigurationError("non-constrained arm contains constraint state")

    selected_metrics = audit_validation_selection(
        job,
        manifest,
        validations,
        selections,
        run_directory,
        selected_path,
        selected_checkpoint,
    )
    selected_rate = _audit_decision_ledger(
        selected_metrics, "selected validation"
    )
    resume_events = _audit_resume_records(records, run_directory)
    inventory = _checkpoint_inventory(run_directory)
    for required in ("final.pt", "latest.pt", "validation_best.pt"):
        if required not in inventory:
            raise ConfigurationError(f"checkpoint inventory omits {required}")
    if job.constrained:
        audit_constrained_validation_candidates(
            args,
            job,
            validations,
            run_directory,
            inventory,
            selected_checkpoint,
        )
    elif any(
        name.startswith("validation_candidate_step_") for name in inventory
    ):
        raise ConfigurationError("legacy-selection arm has deferred candidate checkpoints")

    attempt_logs = sorted(job_root(args, job).glob("runner_attempt_*.log"))
    return {
        "job": job.as_dict(),
        "run_directory": str(run_directory),
        "environment_steps": EXPECTED_FINAL_ENVIRONMENT_STEPS,
        "rollout_updates": len(updates),
        "optimizer_updates": final_optimizer_updates,
        "validation_candidate_count": len(validations),
        "selected_validation": {
            **canonical_json_value(
                {
                    field: selected_metrics[field]
                    for field in SELECTED_VALIDATION_METRIC_FIELDS
                }
            ),
            "decision_avoidable_switch_rate": selected_rate,
            "constraint_feasible": (
                selected_rate is not None
                and selected_rate
                <= SWITCH_BUDGET + SELECTION_FEASIBILITY_NUMERICAL_TOLERANCE
                if job.constrained
                else None
            ),
        },
        "constraint_training": constraint_summary,
        "resume_events": resume_events,
        "artifacts": {
            **{
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "final_checkpoint": {
                "path": str(final_path),
                "sha256": sha256_file(final_path),
            },
            "selected_checkpoint": {
                "path": str(selected_path),
                "sha256": sha256_file(selected_path),
            },
            "checkpoint_inventory": inventory,
            "runner_logs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in attempt_logs
            ],
        },
        "checkpoint_schema": canonical_json_value(
            dict(
                getattr(
                    load_checkpoint_policy(selected_path, device="cpu")[0],
                    "checkpoint_schema",
                )
            )
        ),
    }


def _pid_is_active(pid: int) -> bool:
    return smoke._pid_is_active(pid)


@contextmanager
def _invocation_lock_guard(output: Path):
    """Serialize stale-lock recovery and main-lock acquisition."""

    guard_path = output / ".formal_runner.lock.guard"
    descriptor = os.open(guard_path, os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ConfigurationError(
                "another formal runner is acquiring or releasing the output lock"
            ) from error
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def invocation_lock(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".formal_runner.lock"
    hostname = socket.gethostname()
    token = uuid.uuid4().hex
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "pid": os.getpid(),
        "host": hostname,
        "output": str(output),
        "token": token,
        "created_at_utc": utc_now(),
    }
    with _invocation_lock_guard(output):
        if path.exists():
            try:
                existing = read_json(path)
            except ConfigurationError as error:
                raise ConfigurationError("existing formal lock is malformed") from error
            existing_pid = existing.get("pid")
            if (
                existing.get("schema_version") != SCHEMA_VERSION
                or existing.get("study_name") != STUDY_NAME
                or isinstance(existing_pid, bool)
                or not isinstance(existing_pid, int)
                or existing_pid <= 0
                or existing.get("output") != str(output)
                or existing.get("host") != hostname
                or not isinstance(existing.get("token"), str)
                or not isinstance(existing.get("created_at_utc"), str)
            ):
                raise ConfigurationError(
                    "existing formal lock is not a recoverable same-output local lock"
                )
            if _pid_is_active(existing_pid):
                raise ConfigurationError(
                    f"another formal runner is active with pid {existing_pid}"
                )
            path.unlink()
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ConfigurationError(
                "another formal runner invocation owns the output"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    try:
        yield
    finally:
        with _invocation_lock_guard(output):
            try:
                observed = read_json(path)
            except ConfigurationError:
                observed = None
            if observed == payload:
                path.unlink(missing_ok=True)


def load_training_status(
    args: argparse.Namespace,
    job: FormalJob,
    spec_sha256: str,
) -> dict[str, Any]:
    path = training_status_path(args, job)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "spec_sha256": spec_sha256,
            "phase": "training",
            "job": job.as_dict(),
            "status": "pending",
            "attempts": [],
        }
    status = read_json(path)
    validate_self_hash(status, "status_sha256")
    _validate_training_status_record(status, job, spec_sha256)
    return status


def _validate_training_status_record(
    status: Mapping[str, Any],
    job: FormalJob,
    spec_sha256: str,
) -> None:
    attempts = status.get("attempts")
    status_value = status.get("status")
    base_fields = {
        "schema_version",
        "study_name",
        "spec_sha256",
        "phase",
        "job",
        "status",
        "attempts",
        "status_sha256",
    }
    allowed_fields = set(base_fields)
    required_fields = set(base_fields)
    if status_value != "pending":
        allowed_fields.update({"started_at_utc", "child_pid"})
        required_fields.add("started_at_utc")
    if status_value == "completed":
        allowed_fields.update(
            {"completed_at_utc", "artifact", "recovered_completed_run"}
        )
        required_fields.update({"completed_at_utc", "artifact"})
    if (
        not required_fields.issubset(status)
        or not set(status).issubset(allowed_fields)
        or status.get("schema_version") != SCHEMA_VERSION
        or status.get("study_name") != STUDY_NAME
        or status.get("spec_sha256") != spec_sha256
        or status.get("phase") != "training"
        or status.get("job") != job.as_dict()
        or status_value
        not in {"pending", "running", "failed", "interrupted", "completed"}
        or not isinstance(attempts, list)
        or len(attempts) > MAX_TRAINING_LAUNCH_ATTEMPTS
    ):
        raise ConfigurationError(f"training status fingerprint drifted: {job.job_id}")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise ConfigurationError(
                f"training attempt history drifted: {job.job_id}"
            )
        attempt_status = attempt.get("status")
        attempt_fields = {
            "attempt",
            "started_at_utc",
            "status",
            "resume_checkpoint",
            "resume_checkpoint_sha256",
            "command",
            "working_directory",
            "log_path",
        }
        if "child_pid" in attempt:
            attempt_fields.add("child_pid")
        if attempt_status in {"failed", "interrupted"}:
            attempt_fields.update(
                {
                    "finished_at_utc",
                    "error",
                    "failure_category",
                    "retry_authorized",
                }
            )
        elif attempt_status == "completed":
            attempt_fields.add("finished_at_utc")
        resume_checkpoint = attempt.get("resume_checkpoint")
        resume_sha256 = attempt.get("resume_checkpoint_sha256")
        if (
            set(attempt) != attempt_fields
            or attempt.get("attempt") != index
            or attempt_status
            not in {"running", "failed", "interrupted", "completed"}
            or not isinstance(attempt.get("started_at_utc"), str)
            or not str(attempt["started_at_utc"]).endswith("Z")
            or not isinstance(attempt.get("command"), list)
            or not attempt["command"]
            or not all(isinstance(value, str) for value in attempt["command"])
            or not isinstance(attempt.get("working_directory"), str)
            or not isinstance(attempt.get("log_path"), str)
            or ((resume_checkpoint is None) != (resume_sha256 is None))
            or (
                resume_checkpoint is not None
                and (
                    not isinstance(resume_checkpoint, str)
                    or not isinstance(resume_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", resume_sha256) is None
                )
            )
            or (
                "child_pid" in attempt
                and (
                    isinstance(attempt["child_pid"], bool)
                    or not isinstance(attempt["child_pid"], int)
                    or attempt["child_pid"] <= 0
                )
            )
            or (
                attempt_status != "running"
                and (
                    not isinstance(attempt.get("finished_at_utc"), str)
                    or not str(attempt["finished_at_utc"]).endswith("Z")
                )
            )
            or (
                attempt_status in {"failed", "interrupted"}
                and (
                    not isinstance(attempt.get("error"), str)
                    or not isinstance(attempt.get("failure_category"), str)
                    or not isinstance(attempt.get("retry_authorized"), bool)
                    or attempt.get("retry_authorized")
                    is not (
                        attempt.get("failure_category")
                        in RETRYABLE_FAILURE_CATEGORIES
                    )
                )
            )
        ):
            raise ConfigurationError(
                f"training attempt history drifted: {job.job_id}"
            )
    if status_value == "pending":
        if attempts:
            raise ConfigurationError(
                f"pending training status has attempts: {job.job_id}"
            )
        return
    if (
        not attempts
        or attempts[-1]["status"] != status_value
        or status.get("started_at_utc") != attempts[-1]["started_at_utc"]
        or any(
            attempt["status"] not in {"failed", "interrupted"}
            for attempt in attempts[:-1]
        )
        or any(
            attempt.get("retry_authorized") is not True
            or attempt.get("failure_category") not in RETRYABLE_FAILURE_CATEGORIES
            for attempt in attempts[:-1]
        )
        or (
            "child_pid" in status
            and status["child_pid"] is not None
            and (
                isinstance(status["child_pid"], bool)
                or not isinstance(status["child_pid"], int)
                or status["child_pid"] <= 0
            )
        )
    ):
        raise ConfigurationError(f"training status/attempt mismatch: {job.job_id}")
    if status_value == "completed":
        if (
            not isinstance(status.get("artifact"), Mapping)
            or not isinstance(status.get("completed_at_utc"), str)
            or not str(status["completed_at_utc"]).endswith("Z")
            or status.get("child_pid") is not None
            or (
                "recovered_completed_run" in status
                and status["recovered_completed_run"] is not True
            )
        ):
            raise ConfigurationError(
                f"completed training status drifted: {job.job_id}"
            )


def write_training_status(
    args: argparse.Namespace,
    job: FormalJob,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(status)
    body.pop("status_sha256", None)
    record = self_hashed(body, "status_sha256")
    _validate_training_status_record(
        record, job, str(record.get("spec_sha256"))
    )
    atomic_write_json(training_status_path(args, job), record)
    return record


def _training_core_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(artifact)
    core.pop("training_status", None)
    return core


def _training_status_reference(
    args: argparse.Namespace,
    job: FormalJob,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    path = training_status_path(args, job)
    if not path.is_file():
        raise ConfigurationError(f"training status is missing: {job.job_id}")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "status_sha256": status["status_sha256"],
        "attempt_count": len(status["attempts"]),
    }


def _bind_training_status(
    args: argparse.Namespace,
    job: FormalJob,
    artifact: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(artifact),
        "training_status": _training_status_reference(args, job, status),
    }


def audit_no_active_training_children(args: argparse.Namespace) -> None:
    status_root = args.output / "job_status" / "training"
    active = []
    if not status_root.exists():
        return
    for path in sorted(status_root.glob("*.json")):
        status = read_json(path)
        pid = status.get("child_pid")
        if (
            status.get("status") == "running"
            and isinstance(pid, int)
            and not isinstance(pid, bool)
            and _pid_is_active(pid)
        ):
            active.append({"path": str(path.resolve()), "pid": pid})
    if active:
        raise RuntimeError(f"active formal training children exist: {active}")


def train_job(
    args: argparse.Namespace,
    job: FormalJob,
    spec: Mapping[str, Any],
    cancellation_event: threading.Event | None = None,
) -> dict[str, Any]:
    if cancellation_event is not None and cancellation_event.is_set():
        raise _cancelled(
            TrainingCancelled,
            f"training cancelled before start: {job.job_id}",
            cancellation_event,
        )
    spec_sha256 = str(spec["spec_sha256"])
    status = load_training_status(args, job, spec_sha256)
    recorded = status.get("artifact")
    if status.get("status") == "completed" and isinstance(recorded, Mapping):
        observed = audit_training_job(
            args, job, Path(str(recorded.get("run_directory", "")))
        )
        if observed != recorded:
            raise ConfigurationError(f"completed training audit drifted: {job.job_id}")
        return _bind_training_status(args, job, observed, status)
    if status.get("status") == "running":
        pid = status.get("child_pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and _pid_is_active(pid):
            raise RuntimeError(f"training already active for {job.job_id}: pid={pid}")

    discovered_run, resume_checkpoint = discover_run_directory(args, job)
    if discovered_run is not None and resume_checkpoint is None:
        if (
            status.get("status") != "running"
            or not status.get("attempts")
            or status["attempts"][-1].get("status") != "running"
        ):
            raise ConfigurationError(
                f"completed run lacks a tracked running attempt: {job.job_id}"
            )
        artifact = audit_training_job(args, job, discovered_run)
        assert_runtime_fingerprint(args, spec["code_fingerprint"])
        finished_at = utc_now()
        status["attempts"][-1].update(
            status="completed", finished_at_utc=finished_at
        )
        status.update(
            status="completed",
            artifact=artifact,
            recovered_completed_run=True,
            child_pid=None,
            completed_at_utc=finished_at,
        )
        completed_status = write_training_status(args, job, status)
        return _bind_training_status(args, job, artifact, completed_status)
    if status.get("status") == "running":
        assert_runtime_fingerprint(args, spec["code_fingerprint"])
        last = status["attempts"][-1]
        finished_at = utc_now()
        last.update(
            status="interrupted",
            finished_at_utc=finished_at,
            error="TrainingCancelled('orphaned running attempt recovered on restart')",
            failure_category="orphaned_process_interruption",
            retry_authorized=True,
        )
        status.update(
            status="interrupted",
            started_at_utc=last["started_at_utc"],
            child_pid=None,
        )
        status.pop("completed_at_utc", None)
        status.pop("artifact", None)
        status = write_training_status(args, job, status)
    if cancellation_event is not None and cancellation_event.is_set():
        raise _cancelled(
            TrainingCancelled,
            f"training cancelled before launch: {job.job_id}",
            cancellation_event,
        )
    if len(status["attempts"]) >= MAX_TRAINING_LAUNCH_ATTEMPTS:
        raise ConfigurationError(
            f"formal training retry limit exhausted: {job.job_id}"
        )
    _require_retry_authorization(status, f"formal training {job.job_id}")
    assert_runtime_fingerprint(args, spec["code_fingerprint"])

    root = job_root(args, job)
    root.mkdir(parents=True, exist_ok=True)
    runtime_cwd = root / "runtime_working_directory"
    runtime_cwd.mkdir(parents=True, exist_ok=True)
    attempt_number = len(status["attempts"]) + 1
    resume_source = None
    resume_source_hash = None
    if resume_checkpoint is not None:
        if discovered_run is None:
            raise ConfigurationError("resumable checkpoint has no run directory")
        resume_source, resume_source_hash = freeze_resume_source(
            resume_checkpoint, discovered_run, attempt_number
        )
    log_path = root / f"runner_attempt_{attempt_number}.log"
    command = build_command(
        job,
        args,
        root,
        resume_checkpoint=resume_source,
    )
    attempt = {
        "attempt": attempt_number,
        "started_at_utc": utc_now(),
        "status": "running",
        "resume_checkpoint": str(resume_source) if resume_source else None,
        "resume_checkpoint_sha256": resume_source_hash,
        "command": command,
        "working_directory": str(runtime_cwd.resolve()),
        "log_path": str(log_path.resolve()),
    }
    status["attempts"].append(attempt)
    status.update(status="running", started_at_utc=attempt["started_at_utc"])
    write_training_status(args, job, status)

    started = time.monotonic()
    mode = "a" if resume_source is not None else "w"
    try:
        with log_path.open(mode, encoding="utf-8", newline="") as output:
            if cancellation_event is not None and cancellation_event.is_set():
                raise _cancelled(
                    TrainingCancelled,
                    f"training cancelled immediately before launch: {job.job_id}",
                    cancellation_event,
                )
            process = subprocess.Popen(
                command,
                cwd=runtime_cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            status["child_pid"] = process.pid
            attempt["child_pid"] = process.pid
            write_training_status(args, job, status)
            print(
                f"started formal training {job.job_id} pid={process.pid} "
                f"resume={resume_source is not None}",
                flush=True,
            )
            try:
                next_heartbeat = time.monotonic() + 30.0
                while True:
                    if cancellation_event is not None and cancellation_event.is_set():
                        raise _cancelled(
                            TrainingCancelled,
                            f"training cancelled during shutdown: {job.job_id}",
                            cancellation_event,
                        )
                    try:
                        return_code = process.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= next_heartbeat:
                            output.flush()
                            elapsed = (time.monotonic() - started) / 60.0
                            print(
                                f"formal training heartbeat {job.job_id} "
                                f"pid={process.pid} elapsed={elapsed:.1f} min",
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
        assert_runtime_fingerprint(args, spec["code_fingerprint"])
        completed_run, remaining_resume = discover_run_directory(args, job)
        if completed_run is None or remaining_resume is not None:
            raise RuntimeError("trainer exited without a completed run manifest")
        artifact = audit_training_job(args, job, completed_run)
    except BaseException as error:
        failure_category, retry_authorized = _retry_classification(error)
        attempt.update(
            status=(
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, TrainingCancelled))
                else "failed"
            ),
            finished_at_utc=utc_now(),
            error=repr(error),
            failure_category=failure_category,
            retry_authorized=retry_authorized,
        )
        status.update(status=attempt["status"], child_pid=None)
        write_training_status(args, job, status)
        raise

    finished_at = utc_now()
    attempt.update(status="completed", finished_at_utc=finished_at)
    status.update(
        status="completed",
        child_pid=None,
        completed_at_utc=finished_at,
        artifact=artifact,
    )
    completed_status = write_training_status(args, job, status)
    return _bind_training_status(args, job, artifact, completed_status)


def run_all_training_jobs(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    jobs = build_jobs()
    audit_no_active_training_children(args)
    for job in jobs:
        status = load_training_status(args, job, str(spec["spec_sha256"]))
        if (
            status.get("status") != "completed"
            and len(status["attempts"]) >= MAX_TRAINING_LAUNCH_ATTEMPTS
        ):
            raise ConfigurationError(
                f"formal training retry limit exhausted: {job.job_id}"
            )
        if status.get("status") in {"failed", "interrupted"}:
            _require_retry_authorization(
                status, f"formal training {job.job_id}"
            )
    completed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    cancellation_event = threading.Event()
    remaining = iter(jobs)
    executor = ThreadPoolExecutor(max_workers=args.max_parallel)
    futures: dict[Any, FormalJob] = {}

    def submit_next() -> bool:
        try:
            job = next(remaining)
        except StopIteration:
            return False
        future = executor.submit(train_job, args, job, spec, cancellation_event)
        futures[future] = job
        return True

    try:
        for _ in range(min(args.max_parallel, len(jobs))):
            submit_next()
        while futures:
            first_done = next(as_completed(tuple(futures)))
            done = {first_done}
            done.update(future for future in futures if future.done())
            for future in sorted(done, key=lambda item: futures[item].index):
                job = futures.pop(future)
                try:
                    artifact = future.result()
                except Exception as error:
                    if not (isinstance(error, TrainingCancelled) and failures):
                        failures[job.job_id] = repr(error)
                        print(
                            f"failed formal training {job.job_id}: {error}",
                            flush=True,
                        )
                    _bind_cancellation_cause(cancellation_event, error)
                    cancellation_event.set()
                else:
                    completed[job.job_id] = artifact
                    print(f"completed formal training {job.job_id}", flush=True)
            while (
                not failures
                and not cancellation_event.is_set()
                and len(futures) < args.max_parallel
                and submit_next()
            ):
                pass
    except BaseException:
        cancellation_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        raise RuntimeError(f"{len(failures)} formal training jobs failed: {failures}")
    if len(completed) != len(jobs):
        raise RuntimeError("formal training grid is incomplete")
    return {job.job_id: completed[job.job_id] for job in jobs}


def _validate_artifact_entry(entry: Mapping[str, Any], context: str) -> None:
    path = Path(str(entry.get("path", ""))).resolve()
    if not path.is_file() or entry.get("sha256") != sha256_file(path):
        raise ConfigurationError(f"artifact hash drifted: {context}")


def validate_training_freeze(
    freeze: Mapping[str, Any],
    spec: Mapping[str, Any],
    args: argparse.Namespace | None = None,
    *,
    reaudit_jobs: bool = False,
) -> None:
    validate_self_hash(freeze, "freeze_sha256")
    expected_ids = [job.job_id for job in build_jobs()]
    jobs = freeze.get("jobs")
    if (
        freeze.get("study_name") != STUDY_NAME
        or freeze.get("spec_sha256") != spec.get("spec_sha256")
        or freeze.get("smoke_exact_resume_proof_sha256")
        != spec.get("smoke_prerequisite", {}).get("exact_resume_proof_sha256")
        or freeze.get("training_complete") is not True
        or freeze.get("checkpoint_count") != EXPECTED_TRAINING_JOBS
        or freeze.get("test_panel_consulted") is not False
        or freeze.get("test_access_count") != 0
        or freeze.get("sealed_test_instantiated") is not False
        or not isinstance(jobs, Mapping)
        or len(jobs) != len(expected_ids)
        or set(jobs) != set(expected_ids)
    ):
        raise ConfigurationError("formal training freeze contract drifted")
    if reaudit_jobs:
        if args is None:
            raise ConfigurationError("full training-freeze audit requires runtime args")
        verification_path = args.output / "preflight_verification.json"
        if not verification_path.is_file():
            raise ConfigurationError("training freeze preflight verification is missing")
        verification = run_preflight_tests(args, spec)
        verification_log = Path(str(verification.get("log_path", ""))).resolve()
        if (
            verification.get("spec_sha256") != spec.get("spec_sha256")
            or verification.get("status") != "passed"
            or freeze.get("verification_sha256")
            != verification.get("verification_sha256")
            or verification.get("exact_resume_proof_sha256")
            != spec.get("smoke_prerequisite", {}).get(
                "exact_resume_proof_sha256"
            )
            or not verification_log.is_file()
            or verification.get("log_sha256") != sha256_file(verification_log)
        ):
            raise ConfigurationError("training freeze preflight binding drifted")
    jobs_by_id = {job.job_id: job for job in build_jobs()}
    for job_id in expected_ids:
        artifact = jobs[job_id]
        job = jobs_by_id[job_id]
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("job") != job.as_dict()
        ):
            raise ConfigurationError(f"training freeze job is malformed: {job_id}")
        if args is None:
            raise ConfigurationError(
                "training freeze status audit requires runtime args"
            )
        core_artifact = _training_core_artifact(artifact)
        status_ref = artifact.get("training_status")
        status_path = training_status_path(args, job)
        if (
            not isinstance(status_ref, Mapping)
            or set(status_ref)
            != {"path", "file_sha256", "status_sha256", "attempt_count"}
            or status_ref.get("path") != str(status_path.resolve())
            or not status_path.is_file()
            or status_ref.get("file_sha256") != sha256_file(status_path)
        ):
            raise ConfigurationError(
                f"training freeze status binding drifted: {job_id}"
            )
        status = load_training_status(
            args, job, str(spec["spec_sha256"])
        )
        if (
            status.get("status") != "completed"
            or status_ref.get("status_sha256") != status.get("status_sha256")
            or status_ref.get("attempt_count") != len(status["attempts"])
            or not 1 <= len(status["attempts"]) <= MAX_TRAINING_LAUNCH_ATTEMPTS
            or canonical_json_value(status.get("artifact"))
            != canonical_json_value(core_artifact)
        ):
            raise ConfigurationError(
                f"training freeze completed status drifted: {job_id}"
            )
        files = core_artifact.get("artifacts")
        if not isinstance(files, Mapping):
            raise ConfigurationError(f"training freeze job omits artifacts: {job_id}")
        for name in (
            "run_config",
            "training_metrics",
            "run_manifest",
            "latest_checkpoint",
            "final_checkpoint",
            "selected_checkpoint",
        ):
            entry = files.get(name)
            if not isinstance(entry, Mapping):
                raise ConfigurationError(f"training freeze omits {job_id}/{name}")
            _validate_artifact_entry(entry, f"{job_id}/{name}")
        inventory = files.get("checkpoint_inventory")
        if not isinstance(inventory, Mapping) or not inventory:
            raise ConfigurationError(f"training freeze checkpoint inventory missing: {job_id}")
        for name, entry in inventory.items():
            if not isinstance(entry, Mapping):
                raise ConfigurationError(f"checkpoint inventory malformed: {job_id}/{name}")
            _validate_artifact_entry(entry, f"{job_id}/checkpoint/{name}")
        logs = files.get("runner_logs")
        if (
            not isinstance(logs, list)
            or len(logs) != len(status["attempts"])
        ):
            raise ConfigurationError(f"training runner logs missing: {job_id}")
        for index, (entry, attempt) in enumerate(
            zip(logs, status["attempts"]), start=1
        ):
            if not isinstance(entry, Mapping):
                raise ConfigurationError(f"training log entry malformed: {job_id}")
            _validate_artifact_entry(entry, f"{job_id}/runner_log/{index}")
            expected_log_path = (
                job_root(args, job) / f"runner_attempt_{index}.log"
            ).resolve()
            if (
                Path(str(entry.get("path", ""))).resolve() != expected_log_path
                or Path(str(attempt.get("log_path", ""))).resolve()
                != expected_log_path
            ):
                raise ConfigurationError(
                    f"training attempt/log binding drifted: {job_id}/{index}"
                )
        if reaudit_jobs:
            assert args is not None
            run_directory = Path(str(artifact.get("run_directory", ""))).resolve()
            if not _path_within(run_directory, job_root(args, job)):
                raise ConfigurationError(
                    f"training freeze run directory escaped job root: {job_id}"
                )
            observed = audit_training_job(args, job, run_directory)
            if canonical_json_value(observed) != canonical_json_value(core_artifact):
                raise ConfigurationError(
                    f"training freeze job audit drifted: {job_id}"
                )


def write_training_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    verification: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = [job.job_id for job in build_jobs()]
    if list(artifacts) != expected_ids:
        raise ConfigurationError("cannot freeze a partial or reordered training grid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "verification_sha256": verification["verification_sha256"],
        "smoke_exact_resume_proof_sha256": spec["smoke_prerequisite"][
            "exact_resume_proof_sha256"
        ],
        "training_complete": True,
        "checkpoint_count": len(artifacts),
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "jobs": {job_id: dict(artifacts[job_id]) for job_id in expected_ids},
    }
    freeze = self_hashed(body, "freeze_sha256")
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    validate_training_freeze(freeze, spec, args, reaudit_jobs=True)
    ensure_immutable_json(args.output / "training_freeze.json", freeze)
    return freeze


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            newline="\n",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_preflight_tests(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = spec.get("preflight_tests")
    expected_file_hashes = {
        module: spec.get("code_fingerprint", {})
        .get("files", {})
        .get(f"preflight_test_{module}")
        for module in PREFLIGHT_TEST_MODULES
    }
    expected_preflight = {
        "modules": list(PREFLIGHT_TEST_MODULES),
        "expected_test_count": EXPECTED_PREFLIGHT_TEST_COUNT,
        "file_sha256": expected_file_hashes,
        "discovery_or_globbing_allowed": False,
        "formal_or_sealed_workload_instantiation_allowed": False,
    }
    if (
        preflight != expected_preflight
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in expected_file_hashes.values()
        )
    ):
        raise ConfigurationError("formal preflight allowlist contract drifted")
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        *PREFLIGHT_TEST_MODULES,
    ]
    manifest_path = args.output / "preflight_verification.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        validate_self_hash(manifest, "verification_sha256")
        log_path = Path(str(manifest.get("log_path", ""))).resolve()
        if (
            set(manifest)
            != {
                "schema_version",
                "study_name",
                "spec_sha256",
                "status",
                "test_count",
                "test_modules",
                "test_file_sha256",
                "command",
                "log_path",
                "log_sha256",
                "exact_resume_proof_sha256",
                "test_panel_consulted",
                "formal_or_sealed_workloads_instantiated",
                "finished_at_utc",
                "verification_sha256",
            }
            or manifest.get("spec_sha256") != spec.get("spec_sha256")
            or manifest.get("status") != "passed"
            or manifest.get("test_count") != EXPECTED_PREFLIGHT_TEST_COUNT
            or manifest.get("test_modules") != list(PREFLIGHT_TEST_MODULES)
            or manifest.get("test_file_sha256") != expected_file_hashes
            or manifest.get("command") != command
            or not log_path.is_file()
            or manifest.get("log_sha256") != sha256_file(log_path)
            or manifest.get("exact_resume_proof_sha256")
            != spec["smoke_prerequisite"]["exact_resume_proof_sha256"]
            or manifest.get("test_panel_consulted") is not False
            or manifest.get("formal_or_sealed_workloads_instantiated") is not False
            or not isinstance(manifest.get("finished_at_utc"), str)
            or not str(manifest["finished_at_utc"]).endswith("Z")
        ):
            raise ConfigurationError("formal preflight verification drifted")
        return manifest
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    process = subprocess.run(
        command,
        cwd=args.project,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path = (args.output / "preflight_tests.log").resolve()
    _atomic_write_text(log_path, process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"formal preflight tests failed; see {log_path}")
    matches = re.findall(r"^Ran (\d+) tests? in ", process.stdout, flags=re.MULTILINE)
    test_count = int(matches[-1]) if matches else 0
    if test_count != EXPECTED_PREFLIGHT_TEST_COUNT:
        raise RuntimeError(
            "formal preflight test count drifted: "
            f"expected {EXPECTED_PREFLIGHT_TEST_COUNT}, observed {test_count}"
        )
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "status": "passed",
        "test_count": test_count,
        "test_modules": list(PREFLIGHT_TEST_MODULES),
        "test_file_sha256": expected_file_hashes,
        "command": command,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "exact_resume_proof_sha256": spec["smoke_prerequisite"][
            "exact_resume_proof_sha256"
        ],
        "test_panel_consulted": False,
        "formal_or_sealed_workloads_instantiated": False,
        "finished_at_utc": utc_now(),
    }
    manifest = self_hashed(body, "verification_sha256")
    atomic_write_json(manifest_path, manifest)
    return manifest


BASE_EVALUATION_FIELDS = tuple(field.name for field in fields(ConstraintEpisodeMetrics))
VALIDATION_ROW_FIELDS = (
    "schema_version",
    "evaluation_role",
    "spec_sha256",
    "training_freeze_sha256",
    "job_id",
    "arm",
    "environment_variant",
    "selected_checkpoint_sha256",
    "decision_forced_switch_cost",
    *BASE_EVALUATION_FIELDS,
)
VALIDATION_INTEGER_FIELDS = {
    "schema_version",
    "policy_seed",
    "workload_seed",
    "generated",
    "delivered",
    "dropped",
    "backlog",
    "max_queue_packets",
    "routing_switches",
    "avoidable_routing_switches",
    "forced_routing_switches",
    "switch_opportunities",
    "decision_avoidable_switches",
    "decision_switch_opportunities",
    "decision_forced_switches",
    "decision_forced_switch_cost",
}
VALIDATION_STRING_FIELDS = set(VALIDATION_ROW_FIELDS) - VALIDATION_INTEGER_FIELDS - {
    "delivery_ratio",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "episode_reward",
    "global_delay_cost",
    "global_queue_cost",
    "global_load_imbalance",
    "global_switch_cost",
    "global_throughput_reward",
    "global_control_overhead_ratio",
    "global_drop_cost",
    "class_0_delivery_ratio",
    "class_1_delivery_ratio",
    "class_2_delivery_ratio",
    "avoidable_switch_rate",
    "decision_avoidable_switch_rate",
}
VALIDATION_FLOAT_FIELDS = (
    set(VALIDATION_ROW_FIELDS)
    - VALIDATION_INTEGER_FIELDS
    - VALIDATION_STRING_FIELDS
)


def validation_shard_paths(
    args: argparse.Namespace, job: FormalJob, panel: str
) -> tuple[Path, Path, Path]:
    if panel not in VALIDATION_PANELS:
        raise ConfigurationError("unknown formal validation panel")
    root = args.output / "validation_shards" / panel
    return (
        root / f"{job.slug}.jsonl",
        root / f"{job.slug}.csv",
        root / f"{job.slug}.manifest.json",
    )


def load_evaluation_status(
    args: argparse.Namespace,
    job: FormalJob,
    panel: str,
    spec_sha256: str,
    training_freeze_sha256: str,
) -> dict[str, Any]:
    path = evaluation_status_path(args, job, panel)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "spec_sha256": spec_sha256,
            "training_freeze_sha256": training_freeze_sha256,
            "phase": "post_training_validation",
            "validation_panel": panel,
            "job": job.as_dict(),
            "status": "pending",
            "attempts": [],
        }
    status = read_json(path)
    validate_self_hash(status, "status_sha256")
    attempts = status.get("attempts")
    status_value = status.get("status")
    base_fields = {
        "schema_version",
        "study_name",
        "spec_sha256",
        "training_freeze_sha256",
        "phase",
        "validation_panel",
        "job",
        "status",
        "attempts",
        "status_sha256",
    }
    expected_fields = set(base_fields)
    if status_value != "pending":
        expected_fields.add("started_at_utc")
    if status_value == "completed":
        expected_fields.update({"completed_at_utc", "artifact"})
    if (
        set(status) != expected_fields
        or status.get("schema_version") != SCHEMA_VERSION
        or status.get("study_name") != STUDY_NAME
        or status.get("spec_sha256") != spec_sha256
        or status.get("training_freeze_sha256") != training_freeze_sha256
        or status.get("phase") != "post_training_validation"
        or status.get("validation_panel") != panel
        or status.get("job") != job.as_dict()
        or status_value
        not in {"pending", "running", "failed", "interrupted", "completed"}
        or not isinstance(attempts, list)
        or len(attempts) > MAX_EVALUATION_LAUNCH_ATTEMPTS
    ):
        raise ConfigurationError(f"evaluation status drifted: {panel}/{job.job_id}")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise ConfigurationError(
                f"evaluation attempt history drifted: {panel}/{job.job_id}"
            )
        attempt_status = attempt.get("status")
        attempt_fields = {
            "attempt",
            "started_at_utc",
            "status",
            "evaluation_device",
            "validation_workload_seeds",
        }
        if attempt_status in {"failed", "interrupted"}:
            attempt_fields.update(
                {
                    "finished_at_utc",
                    "error",
                    "failure_category",
                    "retry_authorized",
                }
            )
        elif attempt_status == "completed":
            attempt_fields.add("finished_at_utc")
        if (
            set(attempt) != attempt_fields
            or attempt.get("attempt") != index
            or attempt_status
            not in {"running", "failed", "interrupted", "completed"}
            or attempt.get("evaluation_device") != args.evaluation_device
            or attempt.get("validation_workload_seeds")
            != list(PANEL_WORKLOAD_SEEDS[panel])
            or not isinstance(attempt.get("started_at_utc"), str)
            or not str(attempt["started_at_utc"]).endswith("Z")
            or (
                attempt_status != "running"
                and (
                    not isinstance(attempt.get("finished_at_utc"), str)
                    or not str(attempt["finished_at_utc"]).endswith("Z")
                )
            )
            or (
                attempt_status in {"failed", "interrupted"}
                and (
                    not isinstance(attempt.get("error"), str)
                    or not isinstance(attempt.get("failure_category"), str)
                    or not isinstance(attempt.get("retry_authorized"), bool)
                    or attempt.get("retry_authorized")
                    is not (
                        attempt.get("failure_category")
                        in RETRYABLE_FAILURE_CATEGORIES
                    )
                )
            )
        ):
            raise ConfigurationError(
                f"evaluation attempt history drifted: {panel}/{job.job_id}"
            )
    if status_value == "pending":
        if attempts:
            raise ConfigurationError(
                f"pending evaluation has attempts: {panel}/{job.job_id}"
            )
        return status
    if (
        not attempts
        or attempts[-1]["status"] != status_value
        or status.get("started_at_utc") != attempts[-1]["started_at_utc"]
        or any(
            attempt["status"] not in {"failed", "interrupted"}
            for attempt in attempts[:-1]
        )
        or any(
            attempt.get("retry_authorized") is not True
            or attempt.get("failure_category") not in RETRYABLE_FAILURE_CATEGORIES
            for attempt in attempts[:-1]
        )
    ):
        raise ConfigurationError(
            f"evaluation status/attempt mismatch: {panel}/{job.job_id}"
        )
    if status_value == "completed":
        artifact = status.get("artifact")
        if (
            status.get("completed_at_utc") != attempts[-1].get("finished_at_utc")
            or not isinstance(artifact, Mapping)
            or set(artifact)
            != {"manifest_path", "manifest_file_sha256", "manifest_sha256"}
        ):
            raise ConfigurationError(
                f"completed evaluation status drifted: {panel}/{job.job_id}"
            )
    return status


def write_evaluation_status(
    args: argparse.Namespace,
    job: FormalJob,
    panel: str,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(status)
    body.pop("status_sha256", None)
    record = self_hashed(body, "status_sha256")
    atomic_write_json(evaluation_status_path(args, job, panel), record)
    return record


def _evaluation_manifest_artifact(
    args: argparse.Namespace,
    job: FormalJob,
    panel: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = validation_shard_paths(args, job, panel)[2]
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(
            dict(row),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConfigurationError(
                    f"invalid validation JSONL line {line_number}"
                ) from error
            if not isinstance(value, dict) or not all_finite(value):
                raise ConfigurationError(
                    f"invalid validation JSONL record {line_number}"
                )
            rows.append(value)
    return rows


def _read_validation_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != VALIDATION_ROW_FIELDS:
            raise ConfigurationError("validation CSV schema drifted")
        for raw in reader:
            row: dict[str, Any] = {}
            try:
                for field in VALIDATION_ROW_FIELDS:
                    value = raw[field]
                    if field in VALIDATION_INTEGER_FIELDS:
                        row[field] = int(value)
                    elif field in VALIDATION_FLOAT_FIELDS:
                        row[field] = float(value)
                    else:
                        row[field] = value
            except (KeyError, TypeError, ValueError) as error:
                raise ConfigurationError("validation CSV value is malformed") from error
            rows.append(row)
    return rows


def _require_nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"validation {field} must be a non-negative integer")
    return value


def validate_validation_rows(
    job: FormalJob,
    rows: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    spec_sha256: str,
    freeze_sha256: str,
    checkpoint_sha256: str,
    checkpoint_step: int,
) -> dict[str, Any]:
    if panel not in VALIDATION_PANELS:
        raise ConfigurationError("unknown formal validation panel")
    workload_seeds = PANEL_WORKLOAD_SEEDS[panel]
    if len(rows) != len(workload_seeds):
        raise ConfigurationError("validation shard row count drifted")
    if tuple(row.get("workload_seed") for row in rows) != workload_seeds:
        raise ConfigurationError("validation shard workload order/panel drifted")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step <= 0
    ):
        raise ConfigurationError("validation checkpoint step is invalid")
    decision_cost = 0
    decision_opportunity = 0
    for row in rows:
        if tuple(row) != VALIDATION_ROW_FIELDS:
            raise ConfigurationError("validation row schema/order drifted")
        if not all_finite(row):
            raise ConfigurationError("validation row contains non-finite values")
        expected_metadata = {
            "schema_version": SCHEMA_VERSION,
            "evaluation_role": PANEL_EVALUATION_ROLES[panel],
            "spec_sha256": spec_sha256,
            "training_freeze_sha256": freeze_sha256,
            "job_id": job.job_id,
            "arm": job.arm,
            "environment_variant": job.environment_variant,
            "selected_checkpoint_sha256": checkpoint_sha256,
            "scenario": job.scenario,
            "policy": job.arm,
            "policy_seed": job.policy_seed,
        }
        if any(row.get(field) != expected for field, expected in expected_metadata.items()):
            raise ConfigurationError("validation row identity/fingerprint drifted")
        for field in VALIDATION_INTEGER_FIELDS:
            _require_nonnegative_integer(row.get(field), field)
        for field in VALIDATION_FLOAT_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"validation {field} must be finite")
            if not math.isfinite(float(value)):
                raise ConfigurationError(f"validation {field} must be finite")
        generated = int(row["generated"])
        delivered = int(row["delivered"])
        dropped = int(row["dropped"])
        backlog = int(row["backlog"])
        if generated != delivered + dropped + backlog:
            raise ConfigurationError("validation packet conservation failed")
        if not _close(row["delivery_ratio"], delivered / max(1, generated)):
            raise ConfigurationError("validation delivery ratio drifted")
        if not _close(row["drop_rate"], dropped / max(1, generated)):
            raise ConfigurationError("validation drop ratio drifted")
        accepted_avoidable = int(row["avoidable_routing_switches"])
        accepted_forced = int(row["forced_routing_switches"])
        accepted_opportunity = int(row["switch_opportunities"])
        if (
            int(row["routing_switches"]) != accepted_avoidable + accepted_forced
            or accepted_avoidable > accepted_opportunity
            or not _close(
                row["avoidable_switch_rate"],
                accepted_avoidable / max(1, accepted_opportunity),
            )
        ):
            raise ConfigurationError("validation accepted-switch ledger drifted")
        cost = int(row["decision_avoidable_switches"])
        opportunity = int(row["decision_switch_opportunities"])
        forced = int(row["decision_forced_switches"])
        if (
            cost > opportunity
            or row["decision_forced_switch_cost"] != 0
            or not _close(
                row["decision_avoidable_switch_rate"],
                cost / max(1, opportunity),
            )
            or accepted_avoidable > cost
            or accepted_opportunity > opportunity
            or accepted_forced > forced
        ):
            raise ConfigurationError("validation decision-switch ledger drifted")
        if job.environment_variant == "qos_only" and not _close(
            row["global_switch_cost"], 0.0
        ):
            raise ConfigurationError("QoS-only validation contains switch reward cost")
        decision_cost += cost
        decision_opportunity += opportunity
    routing_switches_total = sum(int(row["routing_switches"]) for row in rows)
    avoidable_routing_switches_total = sum(
        int(row["avoidable_routing_switches"]) for row in rows
    )
    forced_routing_switches_total = sum(
        int(row["forced_routing_switches"]) for row in rows
    )
    switch_opportunities = sum(int(row["switch_opportunities"]) for row in rows)
    return {
        "environment_steps": checkpoint_step,
        "episodes": len(rows),
        "seed_start": workload_seeds[0],
        "delivery_ratio": float(np.mean([row["delivery_ratio"] for row in rows])),
        "mean_reward": float(np.mean([row["episode_reward"] for row in rows])),
        "drop_rate": float(np.mean([row["drop_rate"] for row in rows])),
        "average_delay_slots": float(
            np.mean([row["average_delay_slots"] for row in rows])
        ),
        "routing_switches": routing_switches_total / len(rows),
        "routing_switches_total": routing_switches_total,
        "avoidable_routing_switches": (
            avoidable_routing_switches_total / len(rows)
        ),
        "avoidable_routing_switches_total": avoidable_routing_switches_total,
        "forced_routing_switches": forced_routing_switches_total / len(rows),
        "forced_routing_switches_total": forced_routing_switches_total,
        "switch_opportunities": switch_opportunities,
        "avoidable_switch_rate": (
            avoidable_routing_switches_total / max(1, switch_opportunities)
        ),
        "class_2_delivery_ratio": float(
            np.mean([row["class_2_delivery_ratio"] for row in rows])
        ),
        "decision_avoidable_switches": decision_cost,
        "decision_switch_opportunities": decision_opportunity,
        "decision_avoidable_switch_rate": (
            decision_cost / decision_opportunity
            if decision_opportunity > 0
            else None
        ),
        "row_count": len(rows),
    }


def _audit_recheck_against_selected(
    job: FormalJob,
    summary: Mapping[str, Any],
    training_artifact: Mapping[str, Any],
) -> None:
    selected = training_artifact.get("selected_validation")
    if not isinstance(selected, Mapping):
        raise ConfigurationError("training freeze omits selected validation metrics")
    integer_fields = {
        "environment_steps",
        "episodes",
        "seed_start",
        "routing_switches_total",
        "avoidable_routing_switches_total",
        "forced_routing_switches_total",
        "switch_opportunities",
        "decision_avoidable_switches",
        "decision_switch_opportunities",
    }
    missing = set(SELECTED_VALIDATION_METRIC_FIELDS).difference(selected)
    if missing:
        raise ConfigurationError(
            f"training freeze omits selected metrics: {sorted(missing)}"
        )
    for field in SELECTED_VALIDATION_METRIC_FIELDS:
        observed = summary.get(field)
        expected = selected[field]
        if field in integer_fields:
            matches = observed == expected
        elif observed is None or expected is None:
            matches = observed is None and expected is None
        else:
            matches = _close(observed, expected, tolerance=1e-8)
        if not matches:
            raise ConfigurationError(
                f"selected-checkpoint validation recheck drifted: {job.job_id}/{field}"
            )


def validate_evaluation_shard(
    args: argparse.Namespace,
    job: FormalJob,
    spec: Mapping[str, Any],
    freeze: Mapping[str, Any],
    panel: str,
    *,
    require_completed_status: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jsonl_path, csv_path, manifest_path = validation_shard_paths(args, job, panel)
    if not manifest_path.is_file():
        raise ConfigurationError(f"validation shard manifest is missing: {job.job_id}")
    manifest = read_json(manifest_path)
    validate_self_hash(manifest, "manifest_sha256")
    training_artifact = freeze["jobs"][job.job_id]
    selected_entry = training_artifact["artifacts"]["selected_checkpoint"]
    checkpoint_schema = training_artifact.get("checkpoint_schema")
    selected_validation = training_artifact.get("selected_validation")
    if not isinstance(checkpoint_schema, Mapping) or not isinstance(
        selected_validation, Mapping
    ):
        raise ConfigurationError(f"training checkpoint metadata missing: {job.job_id}")
    selected_step = selected_validation.get("environment_steps")
    if isinstance(selected_step, bool) or not isinstance(selected_step, int):
        raise ConfigurationError(f"selected checkpoint step is malformed: {job.job_id}")
    evaluation_attempt = manifest.get("evaluation_attempt")
    if (
        isinstance(evaluation_attempt, bool)
        or not isinstance(evaluation_attempt, int)
        or not 1 <= evaluation_attempt <= MAX_EVALUATION_LAUNCH_ATTEMPTS
    ):
        raise ConfigurationError(f"validation shard attempt drifted: {job.job_id}")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": freeze["freeze_sha256"],
        "job": job.as_dict(),
        "evaluation_role": PANEL_EVALUATION_ROLES[panel],
        "validation_panel": panel,
        "validation_workload_seeds": list(PANEL_WORKLOAD_SEEDS[panel]),
        "selected_checkpoint_path": str(Path(selected_entry["path"]).resolve()),
        "selected_checkpoint_sha256": selected_entry["sha256"],
        "checkpoint_schema": canonical_json_value(dict(checkpoint_schema)),
        "checkpoint_step": selected_step,
        "evaluation_attempt": evaluation_attempt,
        "evaluation_device": args.evaluation_device,
        "evaluation_status_path": str(
            evaluation_status_path(args, job, panel).resolve()
        ),
        "row_fields": list(VALIDATION_ROW_FIELDS),
        "row_count": len(PANEL_WORKLOAD_SEEDS[panel]),
        "jsonl_path": str(jsonl_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "sealed_test_access_authorized": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
    }
    expected_fields = set(expected) | {
        "summary",
        "jsonl_sha256",
        "csv_sha256",
        "completed_at_utc",
        "manifest_sha256",
    }
    if (
        set(manifest) != expected_fields
        or any(manifest.get(field) != value for field, value in expected.items())
        or not isinstance(manifest.get("completed_at_utc"), str)
        or not str(manifest["completed_at_utc"]).endswith("Z")
    ):
        raise ConfigurationError(f"validation shard contract drifted: {job.job_id}")
    for path, field in ((jsonl_path, "jsonl_sha256"), (csv_path, "csv_sha256")):
        if not path.is_file() or manifest.get(field) != sha256_file(path):
            raise ConfigurationError(f"validation shard file hash drifted: {job.job_id}")
    rows = _read_jsonl(jsonl_path)
    csv_rows = _read_validation_csv(csv_path)
    if rows != csv_rows:
        raise ConfigurationError(f"validation JSONL/CSV disagree: {job.job_id}")
    summary = validate_validation_rows(
        job,
        rows,
        panel=panel,
        spec_sha256=spec["spec_sha256"],
        freeze_sha256=freeze["freeze_sha256"],
        checkpoint_sha256=selected_entry["sha256"],
        checkpoint_step=selected_step,
    )
    if manifest.get("summary") != summary:
        raise ConfigurationError(f"validation shard summary drifted: {job.job_id}")
    if panel == PANEL_SELECTION_RECHECK:
        _audit_recheck_against_selected(job, summary, training_artifact)
    if require_completed_status:
        status = load_evaluation_status(
            args,
            job,
            panel,
            str(spec["spec_sha256"]),
            str(freeze["freeze_sha256"]),
        )
        if (
            status.get("status") != "completed"
            or len(status["attempts"]) != evaluation_attempt
            or status["attempts"][-1].get("status") != "completed"
            or status.get("artifact")
            != _evaluation_manifest_artifact(args, job, panel, manifest)
        ):
            raise ConfigurationError(
                f"validation shard status binding drifted: {panel}/{job.job_id}"
            )
    return manifest, rows


def evaluate_validation_job(
    args: argparse.Namespace,
    job: FormalJob,
    spec: Mapping[str, Any],
    freeze: Mapping[str, Any],
    panel: str,
    cancellation_event: threading.Event | None = None,
) -> dict[str, Any]:
    if cancellation_event is not None and cancellation_event.is_set():
        raise _cancelled(
            EvaluationCancelled,
            f"validation cancelled before start: {job.job_id}",
            cancellation_event,
        )
    jsonl_path, csv_path, manifest_path = validation_shard_paths(args, job, panel)
    spec_sha256 = str(spec["spec_sha256"])
    freeze_sha256 = str(freeze["freeze_sha256"])
    status = load_evaluation_status(
        args, job, panel, spec_sha256, freeze_sha256
    )
    if manifest_path.exists():
        manifest = validate_evaluation_shard(
            args,
            job,
            spec,
            freeze,
            panel,
            require_completed_status=False,
        )[0]
        attempt_number = int(manifest["evaluation_attempt"])
        if (
            len(status["attempts"]) != attempt_number
            or status["attempts"][-1].get("attempt") != attempt_number
        ):
            raise ConfigurationError(
                f"completed validation has no matching attempt: {panel}/{job.job_id}"
            )
        if status.get("status") == "completed":
            return validate_evaluation_shard(args, job, spec, freeze, panel)[0]
        if (
            status.get("status") != "running"
            or status["attempts"][-1].get("status") != "running"
        ):
            raise ConfigurationError(
                "completed validation manifest cannot replace a non-running "
                f"attempt: {panel}/{job.job_id}"
            )
        else:
            assert_runtime_fingerprint(args, spec["code_fingerprint"])
            last = status["attempts"][-1]
            finished_at = str(manifest["completed_at_utc"])
            status["attempts"][-1] = {
                "attempt": attempt_number,
                "started_at_utc": last["started_at_utc"],
                "status": "completed",
                "evaluation_device": last["evaluation_device"],
                "validation_workload_seeds": last["validation_workload_seeds"],
                "finished_at_utc": finished_at,
            }
            status.update(
                status="completed",
                started_at_utc=last["started_at_utc"],
                completed_at_utc=finished_at,
                artifact=_evaluation_manifest_artifact(
                    args, job, panel, manifest
                ),
            )
            write_evaluation_status(args, job, panel, status)
        return validate_evaluation_shard(args, job, spec, freeze, panel)[0]
    if status.get("status") == "completed":
        raise ConfigurationError(
            f"completed evaluation manifest is missing: {panel}/{job.job_id}"
        )
    if cancellation_event is not None and cancellation_event.is_set():
        raise _cancelled(
            EvaluationCancelled,
            f"validation cancelled before launch: {job.job_id}",
            cancellation_event,
        )
    if status.get("status") in {"failed", "interrupted"}:
        if len(status["attempts"]) >= MAX_EVALUATION_LAUNCH_ATTEMPTS:
            raise ConfigurationError(
                f"formal validation retry limit exhausted: {panel}/{job.job_id}"
            )
        _require_retry_authorization(
            status, f"formal {panel} validation {job.job_id}"
        )
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    if status.get("status") == "running":
        previous = status["attempts"][-1]
        status["attempts"][-1] = {
            "attempt": previous["attempt"],
            "started_at_utc": previous["started_at_utc"],
            "status": "interrupted",
            "evaluation_device": previous["evaluation_device"],
            "validation_workload_seeds": previous["validation_workload_seeds"],
            "finished_at_utc": utc_now(),
            "error": "EvaluationCancelled('stale running attempt recovered on restart')",
            "failure_category": "orphaned_process_interruption",
            "retry_authorized": True,
        }
        status.update(
            status="interrupted", started_at_utc=previous["started_at_utc"]
        )
        status = write_evaluation_status(args, job, panel, status)
    if len(status["attempts"]) >= MAX_EVALUATION_LAUNCH_ATTEMPTS:
        raise ConfigurationError(
            f"formal validation retry limit exhausted: {panel}/{job.job_id}"
        )
    _require_retry_authorization(
        status, f"formal {panel} validation {job.job_id}"
    )
    attempt_number = len(status["attempts"]) + 1
    started_at = utc_now()
    attempt = {
        "attempt": attempt_number,
        "started_at_utc": started_at,
        "status": "running",
        "evaluation_device": args.evaluation_device,
        "validation_workload_seeds": list(PANEL_WORKLOAD_SEEDS[panel]),
    }
    status["attempts"].append(attempt)
    status.update(status="running", started_at_utc=started_at)
    write_evaluation_status(args, job, panel, status)

    try:
        training_artifact = freeze["jobs"][job.job_id]
        selected_entry = training_artifact["artifacts"]["selected_checkpoint"]
        selected_path = Path(selected_entry["path"]).resolve()
        run_directory = Path(training_artifact["run_directory"]).resolve()
        if (
            not _path_within(selected_path, run_directory)
            or not selected_path.is_file()
            or sha256_file(selected_path) != selected_entry["sha256"]
        ):
            raise ConfigurationError(
                "selected checkpoint escaped or changed before validation"
            )
        policy, checkpoint = load_checkpoint_policy(
            selected_path, device=args.evaluation_device
        )
        schema = getattr(policy, "checkpoint_schema", None)
        expected_schema = training_artifact.get("checkpoint_schema")
        if (
            not isinstance(schema, Mapping)
            or not isinstance(expected_schema, Mapping)
            or canonical_json_value(dict(schema))
            != canonical_json_value(dict(expected_schema))
            or schema.get("variant") != job.environment_variant
        ):
            raise ConfigurationError("selected checkpoint/environment schema drifted")
        selected_validation = training_artifact.get("selected_validation")
        expected_step = (
            selected_validation.get("environment_steps")
            if isinstance(selected_validation, Mapping)
            else None
        )
        observed_step = (
            checkpoint.get("step") if isinstance(checkpoint, Mapping) else None
        )
        if (
            isinstance(expected_step, bool)
            or not isinstance(expected_step, int)
            or observed_step != expected_step
        ):
            raise ConfigurationError(
                "selected checkpoint step drifted before validation"
            )
        has_constraint = getattr(policy, "switch_constraint_spec", None) is not None
        if has_constraint is not job.constrained:
            raise ConfigurationError("selected checkpoint constraint arm drifted")
        if job.constrained:
            contract = getattr(policy, "switch_constraint_spec")
            if any(
                contract.get(field) != value
                for field, value in constraint_contract().items()
            ):
                raise ConfigurationError(
                    "selected constrained checkpoint contract drifted"
                )
        workloads = validate_validation_workloads(
            PANEL_WORKLOAD_SEEDS[panel], panel
        )
        evaluated: list[ConstraintEpisodeMetrics] = []
        for workload_seed in workloads:
            if cancellation_event is not None and cancellation_event.is_set():
                raise _cancelled(
                    EvaluationCancelled,
                    f"validation cancelled between workloads: {job.job_id}",
                    cancellation_event,
                )
            result = evaluate_policy_with_constraint_metrics(
                scenario=job.scenario,
                policy_name=job.arm,
                policy=policy,
                policy_seed=job.policy_seed,
                workload_seeds=(workload_seed,),
                variant=job.environment_variant,
            )
            if (
                len(result) != 1
                or type(result[0]) is not ConstraintEpisodeMetrics
            ):
                raise ConfigurationError(
                    "structured evaluator returned an unexpected row type"
                )
            evaluated.append(result[0])
        if cancellation_event is not None and cancellation_event.is_set():
            raise _cancelled(
                EvaluationCancelled,
                f"validation cancelled before artifact commit: {job.job_id}",
                cancellation_event,
            )
        rows = []
        for metric in metrics_as_dicts(evaluated):
            row = {
                "schema_version": SCHEMA_VERSION,
                "evaluation_role": PANEL_EVALUATION_ROLES[panel],
                "spec_sha256": spec_sha256,
                "training_freeze_sha256": freeze_sha256,
                "job_id": job.job_id,
                "arm": job.arm,
                "environment_variant": job.environment_variant,
                "selected_checkpoint_sha256": selected_entry["sha256"],
                "decision_forced_switch_cost": 0,
                **metric,
            }
            rows.append(row)
        summary = validate_validation_rows(
            job,
            rows,
            panel=panel,
            spec_sha256=spec_sha256,
            freeze_sha256=freeze_sha256,
            checkpoint_sha256=selected_entry["sha256"],
            checkpoint_step=expected_step,
        )
        if panel == PANEL_SELECTION_RECHECK:
            _audit_recheck_against_selected(job, summary, training_artifact)
        if sha256_file(selected_path) != selected_entry["sha256"]:
            raise ConfigurationError("selected checkpoint changed during validation")
        assert_runtime_fingerprint(args, spec["code_fingerprint"])
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_jsonl(jsonl_path, rows)
        atomic_write_csv(csv_path, rows, fieldnames=VALIDATION_ROW_FIELDS)
        body = {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "spec_sha256": spec_sha256,
            "training_freeze_sha256": freeze_sha256,
            "job": job.as_dict(),
            "evaluation_role": PANEL_EVALUATION_ROLES[panel],
            "validation_panel": panel,
            "validation_workload_seeds": list(PANEL_WORKLOAD_SEEDS[panel]),
            "selected_checkpoint_path": str(selected_path),
            "selected_checkpoint_sha256": selected_entry["sha256"],
            "checkpoint_schema": canonical_json_value(dict(schema)),
            "checkpoint_step": int(checkpoint["step"]),
            "evaluation_attempt": attempt_number,
            "evaluation_device": args.evaluation_device,
            "evaluation_status_path": str(
                evaluation_status_path(args, job, panel).resolve()
            ),
            "row_fields": list(VALIDATION_ROW_FIELDS),
            "row_count": len(rows),
            "summary": summary,
            "jsonl_path": str(jsonl_path.resolve()),
            "jsonl_sha256": sha256_file(jsonl_path),
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
            "sealed_test_access_authorized": False,
            "paper_claim_allowed": False,
            "promotion_decision_allowed": False,
            "test_panel_consulted": False,
            "test_access_count": 0,
            "sealed_test_instantiated": False,
            "completed_at_utc": utc_now(),
        }
        manifest = self_hashed(body, "manifest_sha256")
        atomic_write_json(manifest_path, manifest)
        manifest = validate_evaluation_shard(
            args,
            job,
            spec,
            freeze,
            panel,
            require_completed_status=False,
        )[0]
    except BaseException as error:
        finished_at = utc_now()
        failure_category, retry_authorized = _retry_classification(error)
        attempt.update(
            status=(
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, EvaluationCancelled))
                else "failed"
            ),
            finished_at_utc=finished_at,
            error=repr(error),
            failure_category=failure_category,
            retry_authorized=retry_authorized,
        )
        status.update(status=attempt["status"], started_at_utc=started_at)
        status.pop("completed_at_utc", None)
        status.pop("artifact", None)
        write_evaluation_status(args, job, panel, status)
        raise

    finished_at = utc_now()
    attempt.update(status="completed", finished_at_utc=finished_at)
    status.update(
        status="completed",
        started_at_utc=started_at,
        completed_at_utc=finished_at,
        artifact=_evaluation_manifest_artifact(args, job, panel, manifest),
    )
    write_evaluation_status(args, job, panel, status)
    return validate_evaluation_shard(args, job, spec, freeze, panel)[0]


def run_all_validation_jobs(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    validate_training_freeze(freeze, spec, args)
    jobs = build_jobs()
    for panel in VALIDATION_PANELS:
        for job in jobs:
            status = load_evaluation_status(
                args,
                job,
                panel,
                str(spec["spec_sha256"]),
                str(freeze["freeze_sha256"]),
            )
            if (
                status.get("status") != "completed"
                and len(status["attempts"]) >= MAX_EVALUATION_LAUNCH_ATTEMPTS
            ):
                raise ConfigurationError(
                    "formal validation retry limit exhausted: "
                    f"{panel}/{job.job_id}"
                )
            if status.get("status") in {"failed", "interrupted"}:
                _require_retry_authorization(
                    status, f"formal {panel} validation {job.job_id}"
                )
    panels: dict[str, dict[str, dict[str, Any]]] = {}
    for panel in VALIDATION_PANELS:
        completed: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}
        cancellation_event = threading.Event()
        remaining = iter(jobs)
        executor = ThreadPoolExecutor(max_workers=args.max_parallel)
        futures: dict[Any, FormalJob] = {}

        def submit_next() -> bool:
            try:
                job = next(remaining)
            except StopIteration:
                return False
            future = executor.submit(
                evaluate_validation_job,
                args,
                job,
                spec,
                freeze,
                panel,
                cancellation_event,
            )
            futures[future] = job
            return True

        try:
            for _ in range(min(args.max_parallel, len(jobs))):
                submit_next()
            while futures:
                first_done = next(as_completed(tuple(futures)))
                done = {first_done}
                done.update(future for future in futures if future.done())
                for future in sorted(done, key=lambda item: futures[item].index):
                    job = futures.pop(future)
                    try:
                        manifest = future.result()
                    except Exception as error:
                        if not (isinstance(error, EvaluationCancelled) and failures):
                            failures[job.job_id] = repr(error)
                            print(
                                f"failed formal {panel} validation "
                                f"{job.job_id}: {error}",
                                flush=True,
                            )
                        _bind_cancellation_cause(cancellation_event, error)
                        cancellation_event.set()
                    else:
                        completed[job.job_id] = manifest
                        print(
                            f"completed formal {panel} validation {job.job_id}",
                            flush=True,
                        )
                while (
                    not failures
                    and not cancellation_event.is_set()
                    and len(futures) < args.max_parallel
                    and submit_next()
                ):
                    pass
        except BaseException:
            cancellation_event.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if failures:
            raise RuntimeError(
                f"{len(failures)} formal {panel} jobs failed: {failures}"
            )
        if len(completed) != EXPECTED_TRAINING_JOBS:
            raise RuntimeError(f"formal {panel} shard grid is incomplete")
        panels[panel] = {
            job.job_id: completed[job.job_id]
            for job in jobs
        }
    return panels


def merged_validation_paths(
    args: argparse.Namespace, panel: str
) -> tuple[Path, Path]:
    if panel not in VALIDATION_PANELS:
        raise ConfigurationError("unknown formal validation panel")
    stem = f"formal_{panel}_rows"
    return args.output / f"{stem}.jsonl", args.output / f"{stem}.csv"


def _validate_shard_manifest_grid(
    shard_manifests: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    expected_ids = [job.job_id for job in build_jobs()]
    if (
        not isinstance(shard_manifests, Mapping)
        or len(shard_manifests) != len(VALIDATION_PANELS)
        or set(shard_manifests) != set(VALIDATION_PANELS)
    ):
        raise ConfigurationError("cannot merge a partial/reordered validation panel grid")
    for panel in VALIDATION_PANELS:
        manifests = shard_manifests.get(panel)
        if (
            not isinstance(manifests, Mapping)
            or len(manifests) != len(expected_ids)
            or set(manifests) != set(expected_ids)
        ):
            raise ConfigurationError(
                f"cannot merge a partial/reordered {panel} validation grid"
            )


def _validate_merged_panel_rows(
    rows: Sequence[Mapping[str, Any]], panel: str
) -> None:
    expected_order = tuple(
        (job.scenario, job.arm, job.policy_seed, workload_seed)
        for job in build_jobs()
        for workload_seed in PANEL_WORKLOAD_SEEDS[panel]
    )
    observed_order = tuple(
        (
            row.get("scenario"),
            row.get("arm"),
            row.get("policy_seed"),
            row.get("workload_seed"),
        )
        for row in rows
    )
    if observed_order != expected_order or len(set(observed_order)) != len(rows):
        raise ConfigurationError(
            f"merged {panel} grid has reordered, duplicate, or missing cells"
        )


def _collect_validated_panel_rows(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    panel: str,
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in build_jobs():
        observed_manifest, shard_rows = validate_evaluation_shard(
            args, job, spec, training_freeze, panel
        )
        if observed_manifest != manifests[job.job_id]:
            raise ConfigurationError(
                f"validation shard changed: {panel}/{job.job_id}"
            )
        rows.extend(shard_rows)
    expected_count = len(build_jobs()) * len(PANEL_WORKLOAD_SEEDS[panel])
    if len(rows) != expected_count:
        raise ConfigurationError(f"merged {panel} row count is incomplete")
    _validate_merged_panel_rows(rows, panel)
    return rows


def _selection_constraint_gate(
    training_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    constrained_jobs = [job for job in build_jobs() if job.constrained]
    failed = []
    for job in constrained_jobs:
        artifact = training_freeze["jobs"][job.job_id]
        selected = artifact.get("selected_validation")
        if not isinstance(selected, Mapping):
            raise ConfigurationError(
                f"constrained selection metrics are missing: {job.job_id}"
            )
        selected_rate = _audit_decision_ledger(
            selected, f"constrained selected validation {job.job_id}"
        )
        expected_feasible = bool(
            selected_rate is not None
            and selected_rate
            <= SWITCH_BUDGET + SELECTION_FEASIBILITY_NUMERICAL_TOLERANCE
        )
        if selected.get("constraint_feasible") is not expected_feasible:
            raise ConfigurationError(
                f"constrained selection feasibility drifted: {job.job_id}"
            )
        if not expected_feasible:
            failed.append(job.job_id)
    return {
        "selection_budget": SWITCH_BUDGET,
        "selection_feasibility_numerical_tolerance": (
            SELECTION_FEASIBILITY_NUMERICAL_TOLERANCE
        ),
        "independent_gate_budget_numerical_tolerance": 0.0,
        "expected_constrained_jobs": len(constrained_jobs),
        "feasible_constrained_jobs": len(constrained_jobs) - len(failed),
        "failed_job_ids": failed,
        "passed": not failed,
    }


def _selection_recheck_gate(
    training_freeze: Mapping[str, Any],
    selection_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checked = 0
    for job in build_jobs():
        manifest = selection_manifests.get(job.job_id)
        artifact = training_freeze["jobs"][job.job_id]
        summary = manifest.get("summary") if isinstance(manifest, Mapping) else None
        if not isinstance(summary, Mapping):
            raise ConfigurationError(
                f"selection-recheck summary is missing: {job.job_id}"
            )
        _audit_recheck_against_selected(job, summary, artifact)
        checked += 1
    return {
        "expected_jobs": EXPECTED_TRAINING_JOBS,
        "matching_jobs": checked,
        "passed": checked == EXPECTED_TRAINING_JOBS,
    }


def _analyze_gate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        report = formal_statistics.analyze_validation_rows(rows)
        formal_statistics.validate_report_hash(report, rows=rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("formal gate statistics validation failed") from error
    return report


def _artifact_file_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _evaluation_shard_freeze_reference(
    args: argparse.Namespace,
    job: FormalJob,
    panel: str,
    manifest: Mapping[str, Any],
    spec_sha256: str,
    training_freeze_sha256: str,
) -> dict[str, Any]:
    status = load_evaluation_status(
        args,
        job,
        panel,
        spec_sha256,
        training_freeze_sha256,
    )
    status_path = evaluation_status_path(args, job, panel)
    if status.get("status") != "completed" or not status_path.is_file():
        raise ConfigurationError(
            f"evaluation status is not complete: {panel}/{job.job_id}"
        )
    return {
        **_evaluation_manifest_artifact(args, job, panel, manifest),
        "evaluation_status_path": str(status_path.resolve()),
        "evaluation_status_file_sha256": sha256_file(status_path),
        "evaluation_status_sha256": status["status_sha256"],
    }


def validate_validation_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    expected_shard_manifests: (
        Mapping[str, Mapping[str, Mapping[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    validate_training_freeze(
        training_freeze, spec, args, reaudit_jobs=True
    )
    validate_self_hash(freeze, "validation_freeze_sha256")
    expected_fields = {
        "schema_version",
        "study_name",
        "spec_sha256",
        "training_freeze_sha256",
        "validation_complete",
        "selection_recheck_row_count",
        "independent_gate_row_count",
        "validation_row_count",
        "selection_constraint_gate",
        "selection_recheck_gate",
        "integrity_gates",
        "integrity_gates_passed",
        "statistical_validation_gate_passed",
        "formal_validation_gate_passed",
        "method_validation_gate_passed",
        "classical_baseline_freeze_bound",
        "joint_authorization_prerequisites_complete",
        "next_required_stage",
        "eligible_for_separate_test_authorization",
        "sealed_test_access_authorized",
        "paper_claim_allowed",
        "promotion_decision_allowed",
        "test_panel_consulted",
        "test_access_count",
        "sealed_test_instantiated",
        "merged_panels",
        "statistics",
        "shards",
        "validation_freeze_sha256",
    }
    if set(freeze) != expected_fields:
        raise ConfigurationError("formal validation freeze schema drifted")

    nested_manifests: dict[str, dict[str, dict[str, Any]]] = {}
    nested_rows: dict[str, list[dict[str, Any]]] = {}
    frozen_shards = freeze.get("shards")
    if (
        not isinstance(frozen_shards, Mapping)
        or len(frozen_shards) != len(VALIDATION_PANELS)
        or set(frozen_shards) != set(VALIDATION_PANELS)
    ):
        raise ConfigurationError("formal validation freeze shard panels drifted")
    for panel in VALIDATION_PANELS:
        panel_refs = frozen_shards.get(panel)
        if not isinstance(panel_refs, Mapping):
            raise ConfigurationError(f"formal validation freeze omits {panel} shards")
        nested_manifests[panel] = {}
        nested_rows[panel] = []
        expected_ids = [job.job_id for job in build_jobs()]
        if len(panel_refs) != len(expected_ids) or set(panel_refs) != set(expected_ids):
            raise ConfigurationError(f"formal validation freeze {panel} grid drifted")
        for job in build_jobs():
            manifest, rows = validate_evaluation_shard(
                args, job, spec, training_freeze, panel
            )
            expected_ref = _evaluation_shard_freeze_reference(
                args,
                job,
                panel,
                manifest,
                str(spec["spec_sha256"]),
                str(training_freeze["freeze_sha256"]),
            )
            if panel_refs.get(job.job_id) != expected_ref:
                raise ConfigurationError(
                    f"formal validation freeze shard binding drifted: {panel}/{job.job_id}"
                )
            nested_manifests[panel][job.job_id] = manifest
            nested_rows[panel].extend(rows)
        _validate_merged_panel_rows(nested_rows[panel], panel)

    if expected_shard_manifests is not None:
        _validate_shard_manifest_grid(expected_shard_manifests)
        if nested_manifests != {
            panel: dict(expected_shard_manifests[panel])
            for panel in VALIDATION_PANELS
        }:
            raise ConfigurationError("completed validation shard manifests changed")

    merged_panels = freeze.get("merged_panels")
    if (
        not isinstance(merged_panels, Mapping)
        or len(merged_panels) != len(VALIDATION_PANELS)
        or set(merged_panels) != set(VALIDATION_PANELS)
    ):
        raise ConfigurationError("formal validation merged-panel registry drifted")
    for panel in VALIDATION_PANELS:
        jsonl_path, csv_path = merged_validation_paths(args, panel)
        entry = merged_panels.get(panel)
        expected_entry = {
            "evaluation_role": PANEL_EVALUATION_ROLES[panel],
            "workload_seeds": list(PANEL_WORKLOAD_SEEDS[panel]),
            "row_count": len(nested_rows[panel]),
            "jsonl": _artifact_file_entry(jsonl_path),
            "csv": _artifact_file_entry(csv_path),
        }
        if entry != expected_entry:
            raise ConfigurationError(f"formal merged {panel} artifact binding drifted")
        if _read_jsonl(jsonl_path) != nested_rows[panel]:
            raise ConfigurationError(f"formal merged {panel} JSONL rows drifted")
        if _read_validation_csv(csv_path) != nested_rows[panel]:
            raise ConfigurationError(f"formal merged {panel} CSV rows drifted")

    gate_rows = nested_rows[PANEL_INDEPENDENT_GATE]
    statistics_path = args.output / "formal_validation_statistics.json"
    persisted_statistics = read_json(statistics_path)
    try:
        formal_statistics.validate_report_hash(persisted_statistics, rows=gate_rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError("persisted formal statistics report drifted") from error
    recomputed_statistics = _analyze_gate_rows(gate_rows)
    if persisted_statistics != recomputed_statistics:
        raise ConfigurationError("persisted formal statistics do not recompute")
    expected_statistics = {
        "path": str(statistics_path.resolve()),
        "sha256": sha256_file(statistics_path),
        "report_sha256": persisted_statistics["report_sha256"],
        "input_panel": PANEL_INDEPENDENT_GATE,
        "input_evaluation_role": PANEL_EVALUATION_ROLES[PANEL_INDEPENDENT_GATE],
        "input_row_count": EXPECTED_GATE_ROWS,
        "selection_rows_included": False,
    }

    selection_gate = _selection_constraint_gate(training_freeze)
    selection_recheck_gate = _selection_recheck_gate(
        training_freeze, nested_manifests[PANEL_SELECTION_RECHECK]
    )
    integrity_gates = {
        "all_training_jobs_frozen": len(training_freeze["jobs"])
        == EXPECTED_TRAINING_JOBS,
        "all_selection_recheck_shards_frozen": len(
            nested_manifests[PANEL_SELECTION_RECHECK]
        )
        == EXPECTED_TRAINING_JOBS,
        "all_independent_gate_shards_frozen": len(
            nested_manifests[PANEL_INDEPENDENT_GATE]
        )
        == EXPECTED_TRAINING_JOBS,
        "all_selection_recheck_rows_present": len(
            nested_rows[PANEL_SELECTION_RECHECK]
        )
        == EXPECTED_SELECTION_RECHECK_ROWS,
        "all_independent_gate_rows_present": len(gate_rows) == EXPECTED_GATE_ROWS,
        "selected_checkpoint_rechecks_match": selection_recheck_gate["passed"],
        "all_constrained_selections_feasible": selection_gate["passed"],
        "all_artifact_hashes_verified": True,
        "statistics_use_independent_gate_only": True,
        "test_panel_consulted_false": True,
        "test_access_count_zero": True,
        "sealed_test_instantiated_false": True,
    }
    integrity_passed = all(integrity_gates.values())
    statistics_passed = persisted_statistics.get("validation_gate_passed") is True
    formal_gate_passed = selection_gate["passed"] and statistics_passed
    method_gate_passed = bool(integrity_passed and formal_gate_passed)
    semantic_expected = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "validation_complete": True,
        "selection_recheck_row_count": EXPECTED_SELECTION_RECHECK_ROWS,
        "independent_gate_row_count": EXPECTED_GATE_ROWS,
        "validation_row_count": EXPECTED_VALIDATION_ROWS,
        "selection_constraint_gate": selection_gate,
        "selection_recheck_gate": selection_recheck_gate,
        "integrity_gates": integrity_gates,
        "integrity_gates_passed": integrity_passed,
        "statistical_validation_gate_passed": statistics_passed,
        "formal_validation_gate_passed": formal_gate_passed,
        "method_validation_gate_passed": method_gate_passed,
        "classical_baseline_freeze_bound": False,
        "joint_authorization_prerequisites_complete": False,
        "next_required_stage": (
            "classical_baseline_freeze_and_joint_protocol"
            if method_gate_passed
            else "closed_method_validation_gate_failed"
        ),
        "eligible_for_separate_test_authorization": False,
        "sealed_test_access_authorized": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "statistics": expected_statistics,
    }
    if any(freeze.get(field) != value for field, value in semantic_expected.items()):
        raise ConfigurationError("formal validation freeze contract drifted")
    return dict(freeze)


def load_validation_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    *,
    expected_shard_manifests: (
        Mapping[str, Mapping[str, Mapping[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    path = args.output / "validation_freeze.json"
    if not path.is_file():
        raise ConfigurationError("completed formal validation freeze is missing")
    return validate_validation_freeze(
        args,
        spec,
        training_freeze,
        read_json(path),
        expected_shard_manifests=expected_shard_manifests,
    )


def write_validation_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    shard_manifests: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    completed_path = args.output / "validation_freeze.json"
    if completed_path.exists():
        return load_validation_freeze(
            args,
            spec,
            training_freeze,
            expected_shard_manifests=shard_manifests,
        )
    validate_training_freeze(training_freeze, spec, args)
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    _validate_shard_manifest_grid(shard_manifests)
    panel_rows = {
        panel: _collect_validated_panel_rows(
            args,
            spec,
            training_freeze,
            panel,
            shard_manifests[panel],
        )
        for panel in VALIDATION_PANELS
    }
    merged_panels: dict[str, dict[str, Any]] = {}
    for panel in VALIDATION_PANELS:
        jsonl_path, csv_path = merged_validation_paths(args, panel)
        rows = panel_rows[panel]
        _atomic_write_jsonl(jsonl_path, rows)
        atomic_write_csv(csv_path, rows, fieldnames=VALIDATION_ROW_FIELDS)
        merged_panels[panel] = {
            "evaluation_role": PANEL_EVALUATION_ROLES[panel],
            "workload_seeds": list(PANEL_WORKLOAD_SEEDS[panel]),
            "row_count": len(rows),
            "jsonl": _artifact_file_entry(jsonl_path),
            "csv": _artifact_file_entry(csv_path),
        }
    gate_rows = panel_rows[PANEL_INDEPENDENT_GATE]
    statistics = _analyze_gate_rows(gate_rows)
    statistics_path = args.output / "formal_validation_statistics.json"
    atomic_write_json(statistics_path, statistics)
    selection_gate = _selection_constraint_gate(training_freeze)
    selection_recheck_gate = _selection_recheck_gate(
        training_freeze, shard_manifests[PANEL_SELECTION_RECHECK]
    )
    integrity_gates = {
        "all_training_jobs_frozen": len(training_freeze["jobs"])
        == EXPECTED_TRAINING_JOBS,
        "all_selection_recheck_shards_frozen": len(
            shard_manifests[PANEL_SELECTION_RECHECK]
        )
        == EXPECTED_TRAINING_JOBS,
        "all_independent_gate_shards_frozen": len(
            shard_manifests[PANEL_INDEPENDENT_GATE]
        )
        == EXPECTED_TRAINING_JOBS,
        "all_selection_recheck_rows_present": len(
            panel_rows[PANEL_SELECTION_RECHECK]
        )
        == EXPECTED_SELECTION_RECHECK_ROWS,
        "all_independent_gate_rows_present": len(gate_rows) == EXPECTED_GATE_ROWS,
        "selected_checkpoint_rechecks_match": selection_recheck_gate["passed"],
        "all_constrained_selections_feasible": selection_gate["passed"],
        "all_artifact_hashes_verified": True,
        "statistics_use_independent_gate_only": True,
        "test_panel_consulted_false": True,
        "test_access_count_zero": True,
        "sealed_test_instantiated_false": True,
    }
    statistical_gate_passed = statistics["validation_gate_passed"] is True
    formal_gate_passed = selection_gate["passed"] and statistical_gate_passed
    integrity_passed = all(integrity_gates.values())
    method_gate_passed = bool(integrity_passed and formal_gate_passed)
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "validation_complete": True,
        "selection_recheck_row_count": len(
            panel_rows[PANEL_SELECTION_RECHECK]
        ),
        "independent_gate_row_count": len(gate_rows),
        "validation_row_count": sum(len(rows) for rows in panel_rows.values()),
        "selection_constraint_gate": selection_gate,
        "selection_recheck_gate": selection_recheck_gate,
        "integrity_gates": integrity_gates,
        "integrity_gates_passed": integrity_passed,
        "statistical_validation_gate_passed": statistical_gate_passed,
        "formal_validation_gate_passed": formal_gate_passed,
        "method_validation_gate_passed": method_gate_passed,
        "classical_baseline_freeze_bound": False,
        "joint_authorization_prerequisites_complete": False,
        "next_required_stage": (
            "classical_baseline_freeze_and_joint_protocol"
            if method_gate_passed
            else "closed_method_validation_gate_failed"
        ),
        "eligible_for_separate_test_authorization": False,
        "sealed_test_access_authorized": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "test_panel_consulted": False,
        "test_access_count": 0,
        "sealed_test_instantiated": False,
        "merged_panels": merged_panels,
        "statistics": {
            "path": str(statistics_path.resolve()),
            "sha256": sha256_file(statistics_path),
            "report_sha256": statistics["report_sha256"],
            "input_panel": PANEL_INDEPENDENT_GATE,
            "input_evaluation_role": PANEL_EVALUATION_ROLES[
                PANEL_INDEPENDENT_GATE
            ],
            "input_row_count": len(gate_rows),
            "selection_rows_included": False,
        },
        "shards": {
            panel: {
                job.job_id: _evaluation_shard_freeze_reference(
                    args,
                    job,
                    panel,
                    shard_manifests[panel][job.job_id],
                    str(spec["spec_sha256"]),
                    str(training_freeze["freeze_sha256"]),
                )
                for job in build_jobs()
            }
            for panel in VALIDATION_PANELS
        },
    }
    freeze = self_hashed(body, "validation_freeze_sha256")
    assert_runtime_fingerprint(args, spec["code_fingerprint"])
    ensure_immutable_json(completed_path, freeze)
    return validate_validation_freeze(
        args,
        spec,
        training_freeze,
        freeze,
        expected_shard_manifests=shard_manifests,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the frozen formal avoidable-switch training/validation grid."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "experiments" / DEFAULT_OUTPUT_DIRECTORY_NAME,
    )
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.cleanmarl = args.cleanmarl.resolve()
    args.project = args.project.resolve()
    validate_runtime_environment(args)
    validate_output_isolation(args)
    validate_new_output_state(args.output)
    spec = build_spec(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "study_name": STUDY_NAME,
                    "spec_sha256": spec["spec_sha256"],
                    "training_jobs": EXPECTED_TRAINING_JOBS,
                    "selection_recheck_rows": EXPECTED_SELECTION_RECHECK_ROWS,
                    "independent_gate_rows": EXPECTED_GATE_ROWS,
                    "validation_rows": EXPECTED_VALIDATION_ROWS,
                    "test_evaluations": 0,
                    "test_panel_consulted": False,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    preregistration_path = args.output / "preregistration.json"
    training_freeze_path = args.output / "training_freeze.json"
    completed_validation_path = args.output / "validation_freeze.json"
    if completed_validation_path.exists():
        if not preregistration_path.is_file() or not training_freeze_path.is_file():
            raise ConfigurationError(
                "completed validation state omits preregistration or training freeze"
            )
        if read_json(preregistration_path) != spec:
            raise ConfigurationError("completed validation preregistration drifted")
        training_freeze = read_json(training_freeze_path)
        validation_freeze = load_validation_freeze(args, spec, training_freeze)
        print(
            "formal validation already complete: "
            f"gate_passed={validation_freeze['formal_validation_gate_passed']} "
            f"output={args.output}",
            flush=True,
        )
        return 0
    existing_training_freeze = None
    if training_freeze_path.exists():
        if not training_freeze_path.is_file() or not preregistration_path.is_file():
            raise ConfigurationError(
                "completed training state omits a regular freeze or preregistration"
            )
        if read_json(preregistration_path) != spec:
            raise ConfigurationError("completed training preregistration drifted")
        existing_training_freeze = read_json(training_freeze_path)
        validate_training_freeze(
            existing_training_freeze, spec, args, reaudit_jobs=True
        )
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
        "max_parallel": args.max_parallel,
        "test_panel_consulted": False,
        "test_access_count": 0,
    }
    atomic_write_json(invocation_path, invocation)
    try:
        with invocation_lock(args.output):
            verification = run_preflight_tests(args, spec)
            if existing_training_freeze is None:
                training_artifacts = run_all_training_jobs(args, spec)
                training_freeze = write_training_freeze(
                    args, spec, verification, training_artifacts
                )
            else:
                training_freeze = existing_training_freeze
            validation_shards = run_all_validation_jobs(
                args, spec, training_freeze
            )
            validation_freeze = write_validation_freeze(
                args, spec, training_freeze, validation_shards
            )
        invocation.update(
            status="completed",
            finished_at_utc=utc_now(),
            verification_sha256=verification["verification_sha256"],
            training_freeze_sha256=training_freeze["freeze_sha256"],
            validation_freeze_sha256=validation_freeze[
                "validation_freeze_sha256"
            ],
            formal_validation_gate_passed=validation_freeze[
                "formal_validation_gate_passed"
            ],
            eligible_for_separate_test_authorization=validation_freeze[
                "eligible_for_separate_test_authorization"
            ],
            sealed_test_access_authorized=False,
        )
        atomic_write_json(invocation_path, invocation)
        print(
            "formal validation complete: "
            f"gate_passed={validation_freeze['formal_validation_gate_passed']} "
            f"output={args.output}",
            flush=True,
        )
        return 0
    except BaseException as error:
        invocation.update(
            status=("interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
            finished_at_utc=utc_now(),
            error=repr(error),
            sealed_test_access_authorized=False,
        )
        atomic_write_json(invocation_path, invocation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
