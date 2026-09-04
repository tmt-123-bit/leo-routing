"""Run the frozen exploratory integrated route-hysteresis screen."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, get_type_hints
import uuid

import torch

from ablation_matrix_runner import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from hierarchical_statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    build_crossed_matrix,
    crossed_mean_summary,
    pair_crossed_matrices,
    paired_crossed_summary,
    statistical_analysis_manifest,
)
from hysteresis_policy import load_hysteresis_policy
from leo_multiagent_env import ROUTE_SWITCH_FEATURE_INDEX
from mappo_evaluation import EpisodeMetrics, evaluate_policy, load_checkpoint_policy
from run_congestion_screen import screen_job_lock
from run_exp004_mappo import code_fingerprint, train_one
from run_hysteresis_screen import (
    audit_source_screen as audit_frozen_context_screen,
    invocation_lock,
)
from variant_definitions import canonical_variant_name, resolve_variant


SCREEN_NAME = "INTEGRATED-HYSTERESIS-SCREEN-v1"
SCHEMA_VERSION = 1
SOURCE_SCREEN_NAME = "CONGESTION-CONTEXT-SCREEN-v1"
ENVIRONMENT_VARIANT = "with_congestion_context"
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072, 1047581915, 1245825580)
TRAIN_WORKLOAD_SEEDS = tuple(range(9001, 9201))
VALIDATION_WORKLOAD_SEEDS = tuple(range(32001, 32021))
TEST_WORKLOAD_SEEDS = tuple(range(37001, 37051))
INTEGRATED_BETA = 0.20

ARM_PROPOSED = "source_proposed"
ARM_RAW_CONTEXT = "source_raw_context"
ARM_POSTHOC = "source_posthoc_hysteresis_beta_0p20"
ARM_INTEGRATED = "integrated_hysteresis_beta_0p20"
EVALUATION_ARMS = (
    ARM_PROPOSED,
    ARM_RAW_CONTEXT,
    ARM_POSTHOC,
    ARM_INTEGRATED,
)

POLICY_NAMES = {
    ARM_PROPOSED: "mappo_proposed_source",
    ARM_RAW_CONTEXT: "mappo_context_raw_source",
    ARM_POSTHOC: "mappo_context_posthoc_hysteresis_beta_0p20",
    ARM_INTEGRATED: "mappo_context_integrated_hysteresis_beta_0p20",
}

CONFIG = {
    "scenarios": list(SCENARIOS),
    "timesteps": 20000,
    "validation_episodes": len(VALIDATION_WORKLOAD_SEEDS),
    "test_episodes": len(TEST_WORKLOAD_SEEDS),
    "eval_every_rollouts": 20,
    "save_every_steps": 2500,
    "batch_size": 4,
    "q_routing_train_episodes": 0,
}

TRAINER_OVERRIDES = (
    "--validation-seed-start",
    str(VALIDATION_WORKLOAD_SEEDS[0]),
    "--run-tag",
    SCREEN_NAME,
    "--route-hysteresis-beta",
    str(INTEGRATED_BETA),
)

METRICS = (
    "delivery_ratio",
    "backlog",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "max_queue_packets",
    "routing_switches",
    "global_control_overhead_ratio",
    "class_0_delivery_ratio",
    "class_1_delivery_ratio",
    "class_2_delivery_ratio",
)
LOWER_IS_BETTER_COSTS = (
    "average_delay_slots",
    "p95_delay_slots",
    "routing_switches",
    "global_control_overhead_ratio",
)
CLASS_DELIVERY_METRICS = (
    "class_0_delivery_ratio",
    "class_1_delivery_ratio",
    "class_2_delivery_ratio",
)
HARD_GATES = {
    "hotspot_delivery_vs_proposed_min": 0.010,
    "hotspot_positive_seed_fraction_min": 1.00,
    "hotspot_delivery_vs_posthoc_min": 0.005,
    "medium_delivery_vs_proposed_min": -0.003,
    "medium_seed_noninferiority_margin": -0.010,
    "medium_seed_noninferiority_count_min": 3,
    "switch_mean_ratio_vs_proposed_max": 1.02,
    "switch_seed_ratio_preferred_max": 1.05,
    "switch_seed_ratio_preferred_count_min": 3,
    "switch_seed_ratio_absolute_max": 1.15,
    "switch_mean_ratio_vs_raw_max": 0.90,
    "switch_seed_vs_raw_improvement_fraction_min": 1.00,
    "medium_average_delay_mean_ratio_max": 1.01,
    "medium_average_delay_seed_ratio_preferred_max": 1.02,
    "medium_average_delay_seed_preferred_count_min": 3,
    "medium_average_delay_seed_ratio_absolute_max": 1.03,
    "medium_p95_delay_mean_ratio_max": 1.02,
    "medium_class_2_delivery_mean_min": -0.002,
    "medium_class_2_seed_margin": -0.010,
    "medium_class_2_seed_count_min": 3,
    "medium_class_2_seed_absolute_min": -0.015,
    "cost_relative_regression_max": 0.10,
    "class_delivery_regression_max": 0.020,
}

EXPECTED_ACTOR_SPEC = {
    "schema_version": 1,
    "type": "shared_candidate_actor",
    "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
    "route_hysteresis_beta": INTEGRATED_BETA,
}
EXPECTED_SWITCH_REGULARIZER_SPEC: dict[str, Any] | None = None
LEGACY_ACTOR_SPEC = {
    "schema_version": 0,
    "type": "shared_candidate_actor",
    "route_switch_feature_index": None,
    "route_hysteresis_beta": 0.0,
}


@dataclass(frozen=True)
class TrainingJob:
    index: int
    scenario: str
    policy_seed: int

    @property
    def variant(self) -> str:
        return ENVIRONMENT_VARIANT

    @property
    def job_id(self) -> str:
        return f"{self.scenario}/integrated_beta_0p20/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return (
            f"train__{self.scenario}__integrated_beta_0p20__"
            f"seed_{self.policy_seed}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "scenario": self.scenario,
            "arm": ARM_INTEGRATED,
            "policy_seed": self.policy_seed,
            "environment_variant": self.variant,
            "route_hysteresis_beta": INTEGRATED_BETA,
            "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
        }


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    scenario: str
    arm: str
    policy_seed: int

    @property
    def variant(self) -> str:
        if self.arm == ARM_PROPOSED:
            return "proposed"
        return ENVIRONMENT_VARIANT

    @property
    def policy_name(self) -> str:
        return POLICY_NAMES[self.arm]

    @property
    def source_kind(self) -> str:
        return "integrated_training" if self.arm == ARM_INTEGRATED else "frozen_source"

    @property
    def source_job_id(self) -> str:
        if self.arm == ARM_INTEGRATED:
            return f"{self.scenario}/integrated_beta_0p20/seed_{self.policy_seed}"
        source_variant = "proposed" if self.arm == ARM_PROPOSED else ENVIRONMENT_VARIANT
        return f"{self.scenario}/{source_variant}/seed_{self.policy_seed}"

    @property
    def job_id(self) -> str:
        return f"test/{self.scenario}/{self.arm}/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return f"eval__{self.scenario}__{self.arm}__seed_{self.policy_seed}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "job_id": self.job_id,
            "scenario": self.scenario,
            "arm": self.arm,
            "policy_name": self.policy_name,
            "policy_seed": self.policy_seed,
            "environment_variant": self.variant,
            "source_kind": self.source_kind,
            "source_job_id": self.source_job_id,
        }


def build_training_jobs() -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    for scenario in SCENARIOS:
        for policy_seed in POLICY_SEEDS:
            jobs.append(TrainingJob(len(jobs), scenario, policy_seed))
    return jobs


def build_evaluation_jobs() -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    for scenario in SCENARIOS:
        for arm in EVALUATION_ARMS:
            for policy_seed in POLICY_SEEDS:
                jobs.append(EvaluationJob(len(jobs), scenario, arm, policy_seed))
    return jobs


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _self_hash(record: Mapping[str, Any], field: str) -> str:
    body = dict(record)
    observed = body.pop(field, None)
    expected = sha256_json(body)
    if observed != expected:
        raise ValueError(f"invalid {field}: observed={observed!r}, expected={expected}")
    return expected


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if existing != dict(value):
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, dict(value))


def _path_is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return False


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_module_path(module_name: str, expected: Path) -> None:
    actual = Path(inspect.getfile(importlib.import_module(module_name))).resolve()
    if actual != expected.resolve():
        raise RuntimeError(
            f"runtime module path mismatch for {module_name}: {actual} != {expected}"
        )


def screen_code_fingerprint(args: argparse.Namespace) -> dict[str, str]:
    training_fingerprint = code_fingerprint(args)
    external_design = args.cleanmarl / "cleanmarl" / "mappo_design.py"
    repository_design = args.project / "mappo_design.py"
    external_design_sha = sha256_file(external_design)
    if external_design_sha != sha256_file(repository_design):
        raise RuntimeError(
            "external CleanMARL design module differs from the repository snapshot"
        )
    repository_root = args.project.parent.resolve()
    files = {
        "integrated_runner": Path(__file__).resolve(),
        "integrated_protocol": (
            repository_root / "docs" / "INTEGRATED_HYSTERESIS_SCREEN_V1.md"
        ),
        "source_audit_runner": args.project / "run_hysteresis_screen.py",
        "job_lock_runner": args.project / "run_congestion_screen.py",
        "posthoc_policy": args.project / "hysteresis_policy.py",
        "statistics": args.project / "hierarchical_statistics.py",
        "artifact_helpers": args.project / "ablation_matrix_runner.py",
        "external_design": external_design,
    }
    return {
        **training_fingerprint,
        **{name: sha256_file(path) for name, path in files.items()},
    }


def audit_source_screen(source_path: Path) -> dict[str, Any]:
    source = audit_frozen_context_screen(source_path)
    if Path(str(source["source_path"])).resolve() != source_path.resolve():
        raise ValueError("source audit returned a different source directory")
    if len(source["checkpoints"]) != 16:
        raise ValueError("frozen source checkpoint grid is incomplete")
    for job_id, record in source["checkpoints"].items():
        checkpoint_path = Path(str(record["checkpoint_path"])).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_args = checkpoint.get("args")
        if not isinstance(actor_args, dict):
            raise ValueError(f"source checkpoint actor args are invalid: {job_id}")
        beta = float(actor_args.get("route_hysteresis_beta", 0.0))
        if not math.isfinite(beta) or beta != 0.0:
            raise ValueError(f"source checkpoint is not a legacy beta-zero policy: {job_id}")
        actor_spec = checkpoint.get("candidate_actor_spec")
        if actor_spec is not None:
            if not isinstance(actor_spec, dict):
                raise ValueError(f"source candidate actor spec is invalid: {job_id}")
            spec_beta = float(actor_spec.get("route_hysteresis_beta", math.nan))
            if not math.isfinite(spec_beta) or spec_beta != 0.0:
                raise ValueError(f"source candidate actor beta is not zero: {job_id}")
    return source


def build_screen_spec(
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    training_jobs = build_training_jobs()
    evaluation_jobs = build_evaluation_jobs()
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_engineering_screen",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "design": "reuse_frozen_references_train_integrated_treatment_only",
        "environment_variant": ENVIRONMENT_VARIANT,
        "integrated_policy_contract": dict(EXPECTED_ACTOR_SPEC),
        "posthoc_stay_bonus": INTEGRATED_BETA,
        "config": dict(CONFIG),
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "train_workload_seeds": list(TRAIN_WORKLOAD_SEEDS),
        "validation_workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
        "validation_panel_status": (
            "reused_from_source_checkpoint_selection_to_isolate_integrated_beta"
        ),
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "test_panel_status": "fresh_frozen_holdout_unseen_by_prior_screens",
        "prohibited_prior_test_panels": [
            [33001, 33050],
            [34001, 34020],
            [35001, 35050],
        ],
        "trainer_overrides": list(TRAINER_OVERRIDES),
        "evaluation_arms": list(EVALUATION_ARMS),
        "policy_names": dict(POLICY_NAMES),
        "metrics": list(METRICS),
        "hard_gates": dict(HARD_GATES),
        "expected_training_jobs": len(training_jobs),
        "expected_evaluation_jobs": len(evaluation_jobs),
        "expected_evaluation_rows": len(evaluation_jobs) * len(TEST_WORKLOAD_SEEDS),
        "training_jobs": [job.as_dict() for job in training_jobs],
        "evaluation_jobs": [job.as_dict() for job in evaluation_jobs],
        "source": dict(source),
        "code_fingerprint": screen_code_fingerprint(args),
        "paths": {
            "project": str(args.project.resolve()),
            "cleanmarl": str(args.cleanmarl.resolve()),
            "source_screen": str(args.source.resolve()),
            "protocol": str(
                (
                    args.project.parent
                    / "docs"
                    / "INTEGRATED_HYSTERESIS_SCREEN_V1.md"
                ).resolve()
            ),
        },
    }
    body["spec_sha256"] = sha256_json(body)
    return body


def _status_path(output: Path, job: TrainingJob | EvaluationJob) -> Path:
    return output / "job_status" / f"{job.slug}.json"


def _new_status(
    job: TrainingJob | EvaluationJob,
    spec_sha256: str,
    phase: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "phase": phase,
        "job": job.as_dict(),
        "status": "pending",
        "attempts": [],
        "artifact": None,
    }


def load_job_status(
    output: Path,
    job: TrainingJob | EvaluationJob,
    spec_sha256: str,
    phase: str,
) -> dict[str, Any]:
    path = _status_path(output, job)
    if not path.is_file():
        return _new_status(job, spec_sha256, phase)
    status = _load_json(path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "phase": phase,
    }
    mismatches = {
        key: {"observed": status.get(key), "expected": value}
        for key, value in expected.items()
        if status.get(key) != value
    }
    if mismatches or status.get("job", {}).get("job_id") != job.job_id:
        raise RuntimeError(f"job status contract mismatch for {job.job_id}: {mismatches}")
    status.setdefault("attempts", [])
    status.setdefault("artifact", None)
    return status


def write_job_status(
    output: Path,
    job: TrainingJob | EvaluationJob,
    status: dict[str, Any],
) -> None:
    status["updated_at_utc"] = utc_now()
    atomic_write_json(_status_path(output, job), status)


def _start_attempt(status: dict[str, Any]) -> dict[str, Any]:
    attempt = {
        "attempt": len(status.setdefault("attempts", [])) + 1,
        "started_at_utc": utc_now(),
        "status": "running",
    }
    status["attempts"].append(attempt)
    status["status"] = "running"
    return attempt


def _train_args(args: argparse.Namespace, *, skip_training: bool) -> SimpleNamespace:
    return SimpleNamespace(
        output=args.output,
        cleanmarl=args.cleanmarl,
        project=args.project,
        skip_training=skip_training,
        device=args.device,
    )


def _manifest_path_value(run_directory: Path, manifest: Mapping[str, Any], name: str) -> Path:
    value = manifest.get(name)
    if not value:
        raise ValueError(f"run manifest lacks {name}")
    path = Path(str(value))
    if not path.is_absolute():
        path = run_directory / path.name
    return path.resolve()


def _read_training_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid training metrics JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict) or not _all_finite(record):
                raise ValueError(
                    f"training metrics contain invalid values at line {line_number}"
                )
            records.append(record)
    if not records:
        raise ValueError("training metrics are empty")
    return records


def _audit_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    job: TrainingJob,
    *,
    expected_step: int | None,
) -> None:
    if checkpoint.get("candidate_actor_spec") != EXPECTED_ACTOR_SPEC:
        raise ValueError(f"candidate actor spec mismatch for {job.job_id}")
    actor_args = checkpoint.get("args")
    if not isinstance(actor_args, dict):
        raise ValueError(f"checkpoint actor args are invalid for {job.job_id}")
    if canonical_variant_name(str(actor_args.get("leo_variant"))) != ENVIRONMENT_VARIANT:
        raise ValueError(f"checkpoint environment variant mismatch for {job.job_id}")
    if float(actor_args.get("route_hysteresis_beta", math.nan)) != INTEGRATED_BETA:
        raise ValueError(f"checkpoint integrated beta mismatch for {job.job_id}")
    if int(actor_args.get("seed", -1)) != job.policy_seed:
        raise ValueError(f"checkpoint policy seed mismatch for {job.job_id}")
    if int(checkpoint.get("candidate_feature_dim", -1)) != 28:
        raise ValueError(f"checkpoint candidate schema mismatch for {job.job_id}")
    if int(checkpoint.get("action_size", -1)) != 7:
        raise ValueError(f"checkpoint action schema mismatch for {job.job_id}")
    if int(checkpoint.get("n_agents", -1)) != 24:
        raise ValueError(f"checkpoint agent schema mismatch for {job.job_id}")
    if int(checkpoint.get("obs_size", -1)) != 196:
        raise ValueError(f"checkpoint observation schema mismatch for {job.job_id}")
    critic_spec = checkpoint.get("critic_spec") or {}
    if int(critic_spec.get("node_feature_dim", -1)) != 26:
        raise ValueError(f"checkpoint critic schema mismatch for {job.job_id}")
    if expected_step is not None and int(checkpoint.get("step", -1)) != expected_step:
        raise ValueError(f"checkpoint step mismatch for {job.job_id}")


def audit_integrated_checkpoint(
    checkpoint_path: Path,
    job: TrainingJob,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file() or not _path_is_within(checkpoint_path, args.output):
        raise FileNotFoundError(checkpoint_path)
    run_directory = checkpoint_path.parent
    run_root = run_directory.parent
    paths = {
        "run_manifest": run_directory / "run_manifest.json",
        "run_config": run_directory / "run_config.json",
        "training_metrics": run_directory / "training_metrics.jsonl",
        "final_checkpoint": run_directory / "final.pt",
        "job_fingerprint": run_root / "code_fingerprint.json",
        "trainer_stdout": run_root / "trainer_stdout.log",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training artifacts missing: {missing}")

    run_manifest = _load_json(paths["run_manifest"])
    run_config = _load_json(paths["run_config"])
    expected_config = {
        "env_type": "leo_multi",
        "env_name": job.scenario,
        "leo_variant": ENVIRONMENT_VARIANT,
        "route_hysteresis_beta": INTEGRATED_BETA,
        "batch_size": CONFIG["batch_size"],
        "total_timesteps": CONFIG["timesteps"],
        "epochs": 3,
        "num_minibatches": 4,
        "eval_steps": CONFIG["eval_every_rollouts"],
        "num_eval_ep": len(VALIDATION_WORKLOAD_SEEDS),
        "save_every_steps": CONFIG["save_every_steps"],
        "train_seed_start": TRAIN_WORKLOAD_SEEDS[0],
        "train_seed_count": len(TRAIN_WORKLOAD_SEEDS),
        "validation_seed_start": VALIDATION_WORKLOAD_SEEDS[0],
        "seed": job.policy_seed,
        "run_tag": SCREEN_NAME,
        "device": args.device,
    }
    mismatches = {
        key: {"observed": run_config.get(key), "expected": value}
        for key, value in expected_config.items()
        if run_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint run configuration mismatch: {mismatches}")
    if run_manifest.get("candidate_actor_spec") != EXPECTED_ACTOR_SPEC:
        raise ValueError("run manifest candidate actor spec mismatch")
    if _manifest_path_value(
        run_directory, run_manifest, "validation_best_checkpoint"
    ) != checkpoint_path:
        raise ValueError("run manifest does not select the audited checkpoint")
    if _manifest_path_value(
        run_directory, run_manifest, "final_checkpoint"
    ) != paths["final_checkpoint"].resolve():
        raise ValueError("run manifest final checkpoint path mismatch")

    expected_fingerprint = {
        "code": code_fingerprint(args),
        "scenario": job.scenario,
        "policy_seed": job.policy_seed,
        "experiment_variant": ENVIRONMENT_VARIANT,
        "environment_variant": ENVIRONMENT_VARIANT,
        "variant_definition": resolve_variant(ENVIRONMENT_VARIANT).as_dict(),
        "config": dict(CONFIG),
        "trainer_overrides": list(TRAINER_OVERRIDES),
    }
    observed_fingerprint = _load_json(paths["job_fingerprint"])
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(f"per-job training fingerprint mismatch for {job.job_id}")

    actual_steps = int(run_manifest.get("environment_steps", -1))
    step_span = CONFIG["batch_size"] * 30
    if not CONFIG["timesteps"] <= actual_steps < CONFIG["timesteps"] + step_span:
        raise ValueError(f"actual training steps out of range: {actual_steps}")
    best_score = run_manifest.get("best_validation_score")
    if not isinstance(best_score, list) or len(best_score) != 4 or not _all_finite(best_score):
        raise ValueError("best validation score is missing or non-finite")

    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(selected, dict):
        raise ValueError("selected checkpoint payload is not a mapping")
    selected_step = int(selected.get("step", -1))
    if selected_step > actual_steps:
        raise ValueError("selected checkpoint step exceeds completed training")
    _audit_checkpoint_payload(selected, job, expected_step=selected_step)
    if selected.get("args") != run_config:
        raise ValueError("selected checkpoint args differ from run config")

    final = torch.load(
        paths["final_checkpoint"], map_location="cpu", weights_only=False
    )
    if not isinstance(final, dict):
        raise ValueError("final checkpoint payload is not a mapping")
    _audit_checkpoint_payload(final, job, expected_step=actual_steps)
    if final.get("args") != run_config:
        raise ValueError("final checkpoint args differ from run config")

    metric_records = _read_training_metrics(paths["training_metrics"])
    recorded_steps = [
        int(record["environment_steps"])
        for record in metric_records
        if "environment_steps" in record
    ]
    if not recorded_steps or max(recorded_steps) != actual_steps:
        raise ValueError("training metrics and manifest step counts differ")
    validation_records = [
        record
        for record in metric_records
        if record.get("record_type") == "validation"
        and int(record.get("environment_steps", -1)) == selected_step
    ]
    if not validation_records:
        raise ValueError("selected checkpoint has no validation record")
    expected_score = tuple(float(value) for value in best_score)
    for record in validation_records:
        if int(record.get("episodes", -1)) != len(VALIDATION_WORKLOAD_SEEDS):
            raise ValueError("validation episode count mismatch")
        if int(record.get("seed_start", -1)) != VALIDATION_WORKLOAD_SEEDS[0]:
            raise ValueError("validation workload panel mismatch")
        observed_score = (
            float(record["delivery_ratio"]),
            float(record["mean_reward"]),
            -float(record["drop_rate"]),
            -float(record["average_delay_slots"]),
        )
        if not all(
            math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in zip(observed_score, expected_score)
        ):
            raise ValueError("selected validation score differs from run manifest")
    if not any(record.get("is_validation_best") is True for record in validation_records):
        raise ValueError("selected checkpoint is not marked as validation best")

    artifact_paths = {
        "selected_checkpoint": checkpoint_path,
        **paths,
    }
    return {
        "job_id": job.job_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_step": selected_step,
        "actual_environment_steps": actual_steps,
        "best_validation_score": [float(value) for value in best_score],
        "candidate_feature_dim": 28,
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "critic_spec": dict(selected["critic_spec"]),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
    }


def train_job(
    args: argparse.Namespace,
    job: TrainingJob,
    spec_sha256: str,
) -> dict[str, Any]:
    with screen_job_lock(args.output, job):
        status = load_job_status(args.output, job, spec_sha256, "training")
        recorded = status.get("artifact")
        if status.get("status") == "completed" and isinstance(recorded, dict):
            observed = audit_integrated_checkpoint(
                Path(str(recorded["checkpoint_path"])), job, args
            )
            if observed != recorded:
                raise RuntimeError(f"completed checkpoint audit drifted: {job.job_id}")
            return observed

        attempt = _start_attempt(status)
        write_job_status(args.output, job, status)
        try:
            checkpoint = train_one(
                _train_args(args, skip_training=False),
                dict(CONFIG),
                job.scenario,
                job.policy_seed,
                variant=ENVIRONMENT_VARIANT,
                trainer_overrides=list(TRAINER_OVERRIDES),
            )
            artifact = audit_integrated_checkpoint(checkpoint, job, args)
        except BaseException as error:
            attempt.update(
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "finished_at_utc": utc_now(),
                    "error": repr(error),
                }
            )
            status["status"] = attempt["status"]
            write_job_status(args.output, job, status)
            raise
        attempt.update({"status": "completed", "finished_at_utc": utc_now()})
        status.update(
            {
                "status": "completed",
                "artifact": artifact,
                "completed_at_utc": utc_now(),
            }
        )
        write_job_status(args.output, job, status)
        return artifact


def train_all(
    args: argparse.Namespace,
    jobs: Sequence[TrainingJob],
    spec_sha256: str,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    result_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(train_job, args, job, spec_sha256): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                artifact = future.result()
            except Exception as error:
                failures[job.job_id] = repr(error)
                print(f"failed training {job.job_id}: {error}", flush=True)
            else:
                with result_lock:
                    completed[job.job_id] = artifact
                print(
                    f"completed training {job.job_id} at step "
                    f"{artifact['actual_environment_steps']}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"{len(failures)} integrated training jobs failed: {failures}")
    if len(completed) != len(jobs):
        raise RuntimeError("integrated training grid is incomplete")
    return {job.job_id: completed[job.job_id] for job in jobs}


def resolve_integrated_checkpoints(
    args: argparse.Namespace,
    jobs: Sequence[TrainingJob],
    spec_sha256: str,
) -> dict[str, dict[str, Any]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for job in jobs:
        status = load_job_status(args.output, job, spec_sha256, "training")
        recorded = status.get("artifact")
        if status.get("status") != "completed" or not isinstance(recorded, dict):
            checkpoint = train_one(
                _train_args(args, skip_training=True),
                dict(CONFIG),
                job.scenario,
                job.policy_seed,
                variant=ENVIRONMENT_VARIANT,
                trainer_overrides=list(TRAINER_OVERRIDES),
            )
            recorded = audit_integrated_checkpoint(checkpoint, job, args)
            status.update({"status": "completed", "artifact": recorded})
            write_job_status(args.output, job, status)
        observed = audit_integrated_checkpoint(
            Path(str(recorded["checkpoint_path"])), job, args
        )
        if observed != recorded:
            raise RuntimeError(f"training recovery audit mismatch: {job.job_id}")
        checkpoints[job.job_id] = observed
    return checkpoints


def ensure_training_freeze(
    output: Path,
    spec_sha256: str,
    source_freeze_sha256: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_ids = {job.job_id for job in build_training_jobs()}
    if set(checkpoints) != expected_ids:
        raise RuntimeError("training freeze does not contain the exact 8-job grid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "source_training_freeze_sha256": source_freeze_sha256,
        "training_complete": True,
        "checkpoint_count": len(checkpoints),
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "checkpoints": {job.job_id: dict(checkpoints[job.job_id]) for job in build_training_jobs()},
    }
    body["freeze_sha256"] = sha256_json(body)
    path = output / "training_freeze.json"
    ensure_immutable_json(path, body)
    return body


def _episode_rows_from_csv(path: Path) -> list[EpisodeMetrics]:
    field_types = get_type_hints(EpisodeMetrics)
    rows: list[EpisodeMetrics] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(field_types):
            raise ValueError(f"evaluation shard schema mismatch: {path}")
        for raw in reader:
            converted: dict[str, Any] = {}
            for field, field_type in field_types.items():
                value = raw[field]
                if field_type is int:
                    converted[field] = int(value)
                elif field_type is float:
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError(f"non-finite {field} in {path}")
                    converted[field] = numeric
                elif field_type is str:
                    converted[field] = value
                else:
                    raise TypeError(f"unsupported EpisodeMetrics type: {field_type!r}")
            rows.append(EpisodeMetrics(**converted))
    return rows


def validate_evaluation_shard(
    path: Path,
    job: EvaluationJob,
) -> list[EpisodeMetrics]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _episode_rows_from_csv(path)
    if len(rows) != len(TEST_WORKLOAD_SEEDS):
        raise ValueError(
            f"evaluation shard has {len(rows)} rows, expected "
            f"{len(TEST_WORKLOAD_SEEDS)}"
        )
    observed_workloads = [row.workload_seed for row in rows]
    if len(set(observed_workloads)) != len(observed_workloads):
        raise ValueError(f"duplicate workloads in {path}")
    if set(observed_workloads) != set(TEST_WORKLOAD_SEEDS):
        raise ValueError(f"workload panel mismatch in {path}")
    for row in rows:
        if row.scenario != job.scenario:
            raise ValueError(f"scenario mismatch in {path}")
        if row.policy != job.policy_name:
            raise ValueError(f"policy mismatch in {path}")
        if row.policy_seed != job.policy_seed:
            raise ValueError(f"policy-seed mismatch in {path}")
    return rows


def _evaluation_paths(output: Path, job: EvaluationJob) -> tuple[Path, Path]:
    directory = output / "evaluation_shards"
    return directory / f"{job.slug}.csv", directory / f"{job.slug}.json"


def _checkpoint_record_for_evaluation(
    job: EvaluationJob,
    source: Mapping[str, Any],
    integrated: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    records = integrated if job.source_kind == "integrated_training" else source["checkpoints"]
    try:
        return records[job.source_job_id]
    except KeyError as error:
        raise ValueError(f"missing checkpoint for {job.job_id}") from error


def _expected_policy_schema(job: EvaluationJob) -> dict[str, Any]:
    feature_dim = 26 if job.arm == ARM_PROPOSED else 28
    base = {
        "candidate_feature_dim": feature_dim,
        "action_size": 7,
        "obs_size": 7 * feature_dim,
        "n_agents": 24,
        "variant": job.variant,
    }
    if job.arm == ARM_INTEGRATED:
        expected = {**base, "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC)}
        if EXPECTED_SWITCH_REGULARIZER_SPEC is not None:
            expected["switch_regularizer_spec"] = dict(
                EXPECTED_SWITCH_REGULARIZER_SPEC
            )
        return expected
    if job.arm == ARM_POSTHOC:
        return {
            **base,
            "checkpoint_candidate_actor_spec": dict(LEGACY_ACTOR_SPEC),
            "posthoc_controller": {
                "type": "actor_score_hysteresis",
                "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
                "stay_bonus": INTEGRATED_BETA,
            },
        }
    return {**base, "candidate_actor_spec": dict(LEGACY_ACTOR_SPEC)}


def _load_evaluation_policy(
    job: EvaluationJob,
    checkpoint_path: Path,
    device: str,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    if job.arm == ARM_POSTHOC:
        policy, _ = load_hysteresis_policy(
            checkpoint_path,
            stay_bonus=INTEGRATED_BETA,
            device=device,
        )
        observed = {
            **policy.checkpoint_schema,
            "checkpoint_candidate_actor_spec": dict(LEGACY_ACTOR_SPEC),
            "posthoc_controller": {
                "type": "actor_score_hysteresis",
                "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
                "stay_bonus": policy.stay_bonus,
            },
        }
        diagnostics = policy.diagnostics()
    else:
        policy, _ = load_checkpoint_policy(checkpoint_path, device=device)
        observed = dict(getattr(policy, "checkpoint_schema", {}))
        diagnostics = None
    expected = _expected_policy_schema(job)
    if observed != expected:
        raise ValueError(
            f"loaded policy schema mismatch for {job.job_id}: "
            f"{observed} != {expected}"
        )
    return policy, observed, diagnostics


def _expected_shard_metadata(
    job: EvaluationJob,
    spec_sha256: str,
    training_freeze_sha256: str,
    source_freeze_sha256: str,
    checkpoint_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "training_freeze_sha256": training_freeze_sha256,
        "source_training_freeze_sha256": source_freeze_sha256,
        "job": job.as_dict(),
        "checkpoint_path": str(Path(str(checkpoint_record["checkpoint_path"])).resolve()),
        "checkpoint_sha256": str(checkpoint_record["checkpoint_sha256"]),
        "policy_schema": _expected_policy_schema(job),
        "workload_seeds": list(TEST_WORKLOAD_SEEDS),
    }


def evaluate_job(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec_sha256: str,
    training_freeze_sha256: str,
    source: Mapping[str, Any],
    integrated: Mapping[str, Mapping[str, Any]],
) -> list[EpisodeMetrics]:
    checkpoint_record = _checkpoint_record_for_evaluation(job, source, integrated)
    expected_metadata = _expected_shard_metadata(
        job,
        spec_sha256,
        training_freeze_sha256,
        str(source["training_freeze_sha256"]),
        checkpoint_record,
    )
    shard_path, metadata_path = _evaluation_paths(args.output, job)
    with screen_job_lock(args.output, job):
        status = load_job_status(args.output, job, spec_sha256, "evaluation")
        if shard_path.is_file() and metadata_path.is_file():
            metadata = _load_json(metadata_path)
            if (
                all(metadata.get(key) == value for key, value in expected_metadata.items())
                and metadata.get("csv_sha256") == sha256_file(shard_path)
                and metadata.get("row_count") == len(TEST_WORKLOAD_SEEDS)
            ):
                rows = validate_evaluation_shard(shard_path, job)
                artifact = {
                    "csv_path": str(shard_path.resolve()),
                    "csv_sha256": metadata["csv_sha256"],
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": sha256_file(metadata_path),
                    "row_count": len(rows),
                }
                status.update({"status": "completed", "artifact": artifact})
                write_job_status(args.output, job, status)
                return rows

        attempt = _start_attempt(status)
        write_job_status(args.output, job, status)
        try:
            checkpoint_path = Path(str(checkpoint_record["checkpoint_path"])).resolve()
            allowed_root = args.output if job.source_kind == "integrated_training" else args.source
            if not checkpoint_path.is_file() or not _path_is_within(
                checkpoint_path, allowed_root
            ):
                raise FileNotFoundError(checkpoint_path)
            if sha256_file(checkpoint_path) != checkpoint_record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash changed before {job.job_id}")
            policy, observed_schema, _ = _load_evaluation_policy(
                job, checkpoint_path, args.device
            )
            rows = evaluate_policy(
                job.scenario,
                job.policy_name,
                policy,
                job.policy_seed,
                TEST_WORKLOAD_SEEDS,
                variant=job.variant,
            )
            if len(rows) != len(TEST_WORKLOAD_SEEDS):
                raise ValueError("evaluation returned the wrong row count")
            diagnostics_getter = getattr(policy, "diagnostics", None)
            diagnostics = (
                diagnostics_getter() if callable(diagnostics_getter) else None
            )
            atomic_write_csv(shard_path, (asdict(row) for row in rows))
            validated = validate_evaluation_shard(shard_path, job)
            metadata = {
                **expected_metadata,
                "policy_schema": observed_schema,
                "csv_sha256": sha256_file(shard_path),
                "row_count": len(validated),
                "policy_diagnostics": diagnostics,
            }
            atomic_write_json(metadata_path, metadata)
        except BaseException as error:
            attempt.update(
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "finished_at_utc": utc_now(),
                    "error": repr(error),
                }
            )
            status["status"] = attempt["status"]
            write_job_status(args.output, job, status)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        attempt.update({"status": "completed", "finished_at_utc": utc_now()})
        artifact = {
            "csv_path": str(shard_path.resolve()),
            "csv_sha256": sha256_file(shard_path),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "row_count": len(validated),
        }
        status.update(
            {
                "status": "completed",
                "artifact": artifact,
                "completed_at_utc": utc_now(),
            }
        )
        write_job_status(args.output, job, status)
        return validated


def evaluate_all(
    args: argparse.Namespace,
    jobs: Sequence[EvaluationJob],
    spec_sha256: str,
    training_freeze: Mapping[str, Any],
    source: Mapping[str, Any],
    integrated: Mapping[str, Mapping[str, Any]],
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for completed, job in enumerate(jobs, start=1):
        rows.extend(
            evaluate_job(
                args,
                job,
                spec_sha256,
                str(training_freeze["freeze_sha256"]),
                source,
                integrated,
            )
        )
        print(f"[{completed}/{len(jobs)}] completed {job.job_id}", flush=True)
    expected_rows = len(jobs) * len(TEST_WORKLOAD_SEEDS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"evaluation has {len(rows)} rows, expected {expected_rows}")
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed) for row in rows
    }
    if len(keys) != expected_rows:
        raise RuntimeError("evaluation contains duplicate episode cells")
    return rows


def aggregate_rows(rows: Sequence[EpisodeMetrics]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({(row.scenario, row.policy) for row in rows})
    for group_index, (scenario, policy) in enumerate(groups):
        selected = [
            row for row in rows if row.scenario == scenario and row.policy == policy
        ]
        for metric_index, metric in enumerate(METRICS):
            summary = crossed_mean_summary(
                build_crossed_matrix(selected, metric),
                rng_seed=61000 + 100 * group_index + metric_index,
                resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            )
            output.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "metric": metric,
                    **summary,
                }
            )
    return output


def paired_rows(rows: Sequence[EpisodeMetrics]) -> list[dict[str, Any]]:
    comparisons = (
        ("raw_context_minus_proposed", ARM_RAW_CONTEXT, ARM_PROPOSED),
        ("posthoc_minus_proposed", ARM_POSTHOC, ARM_PROPOSED),
        ("integrated_minus_proposed", ARM_INTEGRATED, ARM_PROPOSED),
        ("integrated_minus_raw_context", ARM_INTEGRATED, ARM_RAW_CONTEXT),
        ("integrated_minus_posthoc", ARM_INTEGRATED, ARM_POSTHOC),
    )
    output: list[dict[str, Any]] = []
    for contrast_index, (contrast, treatment_arm, reference_arm) in enumerate(
        comparisons
    ):
        treatment_name = POLICY_NAMES[treatment_arm]
        reference_name = POLICY_NAMES[reference_arm]
        for scenario_index, scenario in enumerate(SCENARIOS):
            treatment = [
                row
                for row in rows
                if row.scenario == scenario and row.policy == treatment_name
            ]
            reference = [
                row
                for row in rows
                if row.scenario == scenario and row.policy == reference_name
            ]
            for metric_index, metric in enumerate(METRICS):
                paired = pair_crossed_matrices(treatment, reference, metric)
                summary = paired_crossed_summary(
                    paired,
                    rng_seed=(
                        62000
                        + 1000 * contrast_index
                        + 100 * scenario_index
                        + metric_index
                    ),
                    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                )
                summary["policy_seed_treatment_means"] = [
                    float(value) for value in paired.treatment.mean(axis=1)
                ]
                summary["policy_seed_reference_means"] = [
                    float(value) for value in paired.reference.mean(axis=1)
                ]
                output.append(
                    {
                        "analysis": "exploratory_integrated_hysteresis_holdout",
                        "contrast": contrast,
                        "scenario": scenario,
                        "reference": reference_name,
                        "treatment": treatment_name,
                        "metric": metric,
                        **summary,
                    }
                )
    return output


def _effect_row(
    rows: Sequence[Mapping[str, Any]],
    contrast: str,
    scenario: str,
    metric: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["contrast"] == contrast
        and row["scenario"] == scenario
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one effect row for {(contrast, scenario, metric)}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _seed_ratios(effect: Mapping[str, Any]) -> list[float]:
    treatment = [float(value) for value in effect["policy_seed_treatment_means"]]
    reference = [float(value) for value in effect["policy_seed_reference_means"]]
    if len(treatment) != len(POLICY_SEEDS) or len(reference) != len(POLICY_SEEDS):
        raise ValueError("effect row has an incomplete policy-seed mean grid")
    return [
        treatment_value / max(abs(reference_value), 1e-12)
        for treatment_value, reference_value in zip(treatment, reference)
    ]


def decide_screen(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integrated_vs_proposed = "integrated_minus_proposed"
    integrated_vs_raw = "integrated_minus_raw_context"
    integrated_vs_posthoc = "integrated_minus_posthoc"
    switch_ratios_vs_proposed: dict[str, float] = {}
    switch_seed_ratios_vs_proposed: dict[str, list[float]] = {}
    switch_ratios_vs_raw: dict[str, float] = {}
    switch_seed_ratios_vs_raw: dict[str, list[float]] = {}
    for scenario in SCENARIOS:
        proposed_effect = _effect_row(
            rows, integrated_vs_proposed, scenario, "routing_switches"
        )
        raw_effect = _effect_row(
            rows, integrated_vs_raw, scenario, "routing_switches"
        )
        switch_ratios_vs_proposed[scenario] = float(
            proposed_effect["treatment_mean"]
        ) / max(
            abs(float(proposed_effect["reference_mean"])), 1e-12
        )
        switch_seed_ratios_vs_proposed[scenario] = _seed_ratios(proposed_effect)
        switch_ratios_vs_raw[scenario] = float(raw_effect["treatment_mean"]) / max(
            abs(float(raw_effect["reference_mean"])), 1e-12
        )
        switch_seed_ratios_vs_raw[scenario] = _seed_ratios(raw_effect)

    hotspot = _effect_row(
        rows, integrated_vs_proposed, "hotspot_high_load", "delivery_ratio"
    )
    medium = _effect_row(
        rows, integrated_vs_proposed, "medium_load", "delivery_ratio"
    )
    hotspot_vs_posthoc = float(
        _effect_row(
            rows,
            integrated_vs_posthoc,
            "hotspot_high_load",
            "delivery_ratio",
        )["mean_difference"]
    )
    medium_difference = float(medium["mean_difference"])
    medium_seed_differences = [
        float(value) for value in medium["policy_seed_mean_differences"]
    ]
    medium_noninferior_count = sum(
        value >= HARD_GATES["medium_seed_noninferiority_margin"]
        for value in medium_seed_differences
    )

    medium_delay = _effect_row(
        rows, integrated_vs_proposed, "medium_load", "average_delay_slots"
    )
    medium_delay_ratio = float(medium_delay["treatment_mean"]) / max(
        abs(float(medium_delay["reference_mean"])), 1e-12
    )
    medium_delay_seed_ratios = _seed_ratios(medium_delay)
    medium_p95 = _effect_row(
        rows, integrated_vs_proposed, "medium_load", "p95_delay_slots"
    )
    medium_p95_ratio = float(medium_p95["treatment_mean"]) / max(
        abs(float(medium_p95["reference_mean"])), 1e-12
    )
    medium_class_2 = _effect_row(
        rows, integrated_vs_proposed, "medium_load", "class_2_delivery_ratio"
    )
    medium_class_2_difference = float(medium_class_2["mean_difference"])
    medium_class_2_seed_differences = [
        float(value) for value in medium_class_2["policy_seed_mean_differences"]
    ]

    cost_regressions: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for metric in LOWER_IS_BETTER_COSTS:
            effect = _effect_row(rows, integrated_vs_proposed, scenario, metric)
            relative = float(effect["mean_difference"]) / max(
                abs(float(effect["reference_mean"])), 1e-12
            )
            if relative > HARD_GATES["cost_relative_regression_max"]:
                cost_regressions.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "relative_regression": relative,
                    }
                )
        for metric in CLASS_DELIVERY_METRICS:
            difference = float(
                _effect_row(rows, integrated_vs_proposed, scenario, metric)[
                    "mean_difference"
                ]
            )
            if difference < -HARD_GATES["class_delivery_regression_max"]:
                cost_regressions.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "absolute_regression": -difference,
                    }
                )

    gate_results = {
        "hotspot_effect_size_vs_proposed": float(hotspot["mean_difference"])
        >= HARD_GATES["hotspot_delivery_vs_proposed_min"],
        "hotspot_seed_consistency": float(hotspot["positive_policy_seed_fraction"])
        >= HARD_GATES["hotspot_positive_seed_fraction_min"],
        "hotspot_training_advantage_vs_posthoc": hotspot_vs_posthoc
        >= HARD_GATES["hotspot_delivery_vs_posthoc_min"],
        "medium_delivery_noninferiority": medium_difference
        >= HARD_GATES["medium_delivery_vs_proposed_min"],
        "medium_seed_consistency": medium_noninferior_count
        >= HARD_GATES["medium_seed_noninferiority_count_min"],
        "switch_mean_control_vs_proposed": all(
            ratio <= HARD_GATES["switch_mean_ratio_vs_proposed_max"]
            for ratio in switch_ratios_vs_proposed.values()
        ),
        "switch_seed_safety_vs_proposed": all(
            sum(
                ratio <= HARD_GATES["switch_seed_ratio_preferred_max"]
                for ratio in ratios
            )
            >= HARD_GATES["switch_seed_ratio_preferred_count_min"]
            and max(ratios) <= HARD_GATES["switch_seed_ratio_absolute_max"]
            for ratios in switch_seed_ratios_vs_proposed.values()
        ),
        "switch_mean_reduction_vs_raw": all(
            ratio <= HARD_GATES["switch_mean_ratio_vs_raw_max"]
            for ratio in switch_ratios_vs_raw.values()
        ),
        "switch_seed_reduction_vs_raw": all(
            sum(ratio < 1.0 for ratio in ratios) / len(ratios)
            >= HARD_GATES["switch_seed_vs_raw_improvement_fraction_min"]
            for ratios in switch_seed_ratios_vs_raw.values()
        ),
        "medium_average_delay_mean_safety": medium_delay_ratio
        <= HARD_GATES["medium_average_delay_mean_ratio_max"],
        "medium_average_delay_seed_safety": sum(
            ratio <= HARD_GATES["medium_average_delay_seed_ratio_preferred_max"]
            for ratio in medium_delay_seed_ratios
        )
        >= HARD_GATES["medium_average_delay_seed_preferred_count_min"]
        and max(medium_delay_seed_ratios)
        <= HARD_GATES["medium_average_delay_seed_ratio_absolute_max"],
        "medium_p95_delay_safety": medium_p95_ratio
        <= HARD_GATES["medium_p95_delay_mean_ratio_max"],
        "medium_class_2_mean_safety": medium_class_2_difference
        >= HARD_GATES["medium_class_2_delivery_mean_min"],
        "medium_class_2_seed_safety": sum(
            difference >= HARD_GATES["medium_class_2_seed_margin"]
            for difference in medium_class_2_seed_differences
        )
        >= HARD_GATES["medium_class_2_seed_count_min"]
        and min(medium_class_2_seed_differences)
        >= HARD_GATES["medium_class_2_seed_absolute_min"],
        "cost_and_class_safety": not cost_regressions,
    }
    passed = all(gate_results.values())
    decision = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_engineering_screen",
        "decision": (
            "eligible_for_separately_frozen_50k_experiment"
            if passed
            else "do_not_advance"
        ),
        "hard_gates_passed": passed,
        "gate_results": gate_results,
        "thresholds": dict(HARD_GATES),
        "evidence": {
            "switch_ratios_vs_proposed": switch_ratios_vs_proposed,
            "switch_seed_ratios_vs_proposed": switch_seed_ratios_vs_proposed,
            "switch_ratios_vs_raw": switch_ratios_vs_raw,
            "switch_seed_ratios_vs_raw": switch_seed_ratios_vs_raw,
            "hotspot_delivery_difference_vs_proposed": float(
                hotspot["mean_difference"]
            ),
            "hotspot_delivery_difference_vs_posthoc": hotspot_vs_posthoc,
            "hotspot_positive_policy_seed_fraction": float(
                hotspot["positive_policy_seed_fraction"]
            ),
            "medium_delivery_difference_vs_proposed": medium_difference,
            "medium_noninferior_policy_seed_count": medium_noninferior_count,
            "medium_average_delay_ratio_vs_proposed": medium_delay_ratio,
            "medium_average_delay_seed_ratios_vs_proposed": (
                medium_delay_seed_ratios
            ),
            "medium_p95_delay_ratio_vs_proposed": medium_p95_ratio,
            "medium_class_2_delivery_difference_vs_proposed": (
                medium_class_2_difference
            ),
            "medium_class_2_seed_differences_vs_proposed": (
                medium_class_2_seed_differences
            ),
            "cost_regressions": cost_regressions,
        },
        "minimum_attainable_two_sided_exact_p": 0.125,
        "paper_claim_allowed": False,
    }
    decision["decision_sha256"] = sha256_json(decision)
    return decision


def write_summary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    lines = [
        "# Integrated Hysteresis Screen v1",
        "",
        "> Exploratory engineering screen only. This is not final IEEE evidence.",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "The beta, training seeds, reused validation panel, fresh test panel, and "
        "hard gates were frozen before test evaluation.",
        "",
        "| Scenario | Contrast | Reference | Treatment | Difference | 95% crossed CI | Positive seeds | Exact p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for contrast in (
            "integrated_minus_proposed",
            "integrated_minus_raw_context",
            "integrated_minus_posthoc",
        ):
            row = _effect_row(rows, contrast, scenario, "delivery_ratio")
            lines.append(
                "| {scenario} | {contrast} | {reference:.4f} | {treatment:.4f} | "
                "{difference:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                "{positive:.0%} | {p_value:.4f} |".format(
                    scenario=scenario,
                    contrast=contrast,
                    reference=float(row["reference_mean"]),
                    treatment=float(row["treatment_mean"]),
                    difference=float(row["mean_difference"]),
                    low=float(row["difference_ci95_low"]),
                    high=float(row["difference_ci95_high"]),
                    positive=float(row["positive_policy_seed_fraction"]),
                    p_value=float(row["raw_p_value"]),
                )
            )
    lines.extend(
        [
            "",
            "Four reused policy seeds imply a minimum two-sided exact p-value of "
            "0.125. Passing this screen only permits a separately frozen larger "
            "experiment; it does not permit a paper claim.",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def _evaluation_artifact_index(
    output: Path,
    jobs: Sequence[EvaluationJob],
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for job in jobs:
        csv_path, metadata_path = _evaluation_paths(output, job)
        metadata = _load_json(metadata_path)
        if metadata.get("csv_sha256") != sha256_file(csv_path):
            raise ValueError(f"evaluation shard hash mismatch: {job.job_id}")
        artifacts[job.job_id] = {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256_file(metadata_path),
            "row_count": int(metadata["row_count"]),
        }
    return artifacts


def write_final_artifacts(
    output: Path,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    jobs: Sequence[EvaluationJob],
    rows: Sequence[EpisodeMetrics],
) -> dict[str, Any]:
    episode_path = output / "final_episode_metrics.csv"
    aggregate_path = output / "final_aggregate_metrics.csv"
    paired_path = output / "final_paired_effects.csv"
    decision_path = output / "final_decision.json"
    summary_path = output / "INTEGRATED_HYSTERESIS_SUMMARY.md"

    episode_rows = [asdict(row) for row in rows]
    aggregate = aggregate_rows(rows)
    paired = paired_rows(rows)
    decision = decide_screen(paired)
    atomic_write_csv(episode_path, episode_rows)
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(paired_path, paired)
    atomic_write_json(decision_path, decision)
    write_summary(summary_path, paired, decision)

    artifacts = {
        "episode_metrics": {
            "path": str(episode_path.resolve()),
            "sha256": sha256_file(episode_path),
            "row_count": len(episode_rows),
        },
        "aggregate_metrics": {
            "path": str(aggregate_path.resolve()),
            "sha256": sha256_file(aggregate_path),
            "row_count": len(aggregate),
        },
        "paired_effects": {
            "path": str(paired_path.resolve()),
            "sha256": sha256_file(paired_path),
            "row_count": len(paired),
        },
        "decision": {
            "path": str(decision_path.resolve()),
            "sha256": sha256_file(decision_path),
        },
        "summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_engineering_screen",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "source_training_freeze_sha256": training_freeze[
            "source_training_freeze_sha256"
        ],
        "training_jobs": len(build_training_jobs()),
        "evaluation_jobs": len(jobs),
        "evaluation_rows": len(rows),
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(TEST_WORKLOAD_SEEDS),
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "statistical_analysis": statistical_analysis_manifest(),
        "decision": decision,
        "evaluation_shards": _evaluation_artifact_index(output, jobs),
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    ensure_immutable_json(output / "integrated_hysteresis_manifest.json", manifest)
    return manifest


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset for this screen")
    repository_root = args.project.parent.resolve()
    experiments_root = (repository_root / "experiments").resolve()
    archive_root = (experiments_root / "archive").resolve()
    output = args.output.resolve()
    source = args.source.resolve()
    broad_outputs = {
        repository_root,
        args.project.resolve(),
        experiments_root,
        archive_root,
        source,
    }
    if output in broad_outputs:
        raise RuntimeError("output must be a new dedicated experiment directory")
    if source in output.parents or output in source.parents:
        raise RuntimeError("output and frozen source must be in disjoint directory trees")
    formal_output = (experiments_root / "ablation-50k-v2").resolve()
    if output == formal_output or formal_output in output.parents:
        raise RuntimeError("output must not be inside the frozen ablation experiment")
    if output.is_dir() and not (
        output / "integrated_hysteresis_spec.json"
    ).is_file():
        conflicting = [
            name
            for name in (
                "screen_spec.json",
                "hysteresis_spec.json",
                "matrix_spec.json",
                "experiment_manifest.json",
                "screen_manifest.json",
            )
            if (output / name).exists()
        ]
        if conflicting:
            raise RuntimeError(f"output contains another experiment: {conflicting}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    if canonical_variant_name(ENVIRONMENT_VARIANT) != ENVIRONMENT_VARIANT:
        raise RuntimeError("integrated environment variant is not canonical")
    if ROUTE_SWITCH_FEATURE_INDEX != 17:
        raise RuntimeError("route-switch feature contract changed")
    prohibited_workloads = (
        set(range(33001, 33051))
        | set(range(34001, 34021))
        | set(range(35001, 35051))
    )
    if set(TEST_WORKLOAD_SEEDS) & prohibited_workloads:
        raise RuntimeError("fresh test panel overlaps a prohibited prior panel")
    if set(TEST_WORKLOAD_SEEDS) & set(VALIDATION_WORKLOAD_SEEDS):
        raise RuntimeError("test and validation workload panels overlap")

    module_files = {
        "ablation_matrix_runner": "ablation_matrix_runner.py",
        "hierarchical_statistics": "hierarchical_statistics.py",
        "hysteresis_policy": "hysteresis_policy.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "mappo_design": "mappo_design.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "run_congestion_screen": "run_congestion_screen.py",
        "run_exp004_mappo": "run_exp004_mappo.py",
        "run_hysteresis_screen": "run_hysteresis_screen.py",
        "variant_definitions": "variant_definitions.py",
    }
    for module_name, filename in module_files.items():
        _runtime_module_path(module_name, args.project / filename)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Train eight integrated beta=0.20 jobs and evaluate four frozen arms "
            "on a fresh 50-workload panel."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            repository_root
            / "experiments"
            / "archive"
            / "congestion-context-screen-20k-v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root
            / "experiments"
            / "archive"
            / "integrated-hysteresis-screen-20k-v1"
        ),
    )
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args(argv)


def _write_plans(output: Path) -> None:
    atomic_write_csv(
        output / "training_plan.csv",
        (job.as_dict() for job in build_training_jobs()),
    )
    atomic_write_csv(
        output / "evaluation_plan.csv",
        (job.as_dict() for job in build_evaluation_jobs()),
    )


def _dry_run_summary(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_engineering_screen",
        "paper_claim_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "training_jobs": len(build_training_jobs()),
        "evaluation_jobs": len(build_evaluation_jobs()),
        "evaluation_rows": len(build_evaluation_jobs()) * len(TEST_WORKLOAD_SEEDS),
        "training_environment_variant": ENVIRONMENT_VARIANT,
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "validation_workloads": [
            VALIDATION_WORKLOAD_SEEDS[0],
            VALIDATION_WORKLOAD_SEEDS[-1],
        ],
        "test_workloads": [TEST_WORKLOAD_SEEDS[0], TEST_WORKLOAD_SEEDS[-1]],
        "output": str(args.output),
        "dry_run_writes_output": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.cleanmarl = args.cleanmarl.resolve()
    args.project = args.project.resolve()
    validate_environment(args)
    source = audit_source_screen(args.source)
    spec = build_screen_spec(args, source)

    if args.dry_run:
        print(json.dumps(_dry_run_summary(args, spec, source), indent=2))
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    ensure_immutable_json(
        args.output / "integrated_hysteresis_spec.json",
        spec,
    )
    _write_plans(args.output)

    invocation_id = uuid.uuid4().hex
    invocation_path = args.output / "invocations" / f"{invocation_id}.json"
    invocation = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "invocation_id": invocation_id,
        "spec_sha256": spec["spec_sha256"],
        "started_at_utc": utc_now(),
        "status": "running",
        "mode": (
            "train_only"
            if args.train_only
            else "evaluate_only"
            if args.evaluate_only
            else "train_and_evaluate"
        ),
        "device": args.device,
        "max_parallel": args.max_parallel,
    }
    atomic_write_json(invocation_path, invocation)

    try:
        with invocation_lock(args.output):
            training_jobs = build_training_jobs()
            if args.evaluate_only:
                checkpoints = resolve_integrated_checkpoints(
                    args, training_jobs, spec["spec_sha256"]
                )
            else:
                checkpoints = train_all(args, training_jobs, spec["spec_sha256"])
            training_freeze = ensure_training_freeze(
                args.output,
                spec["spec_sha256"],
                str(source["training_freeze_sha256"]),
                checkpoints,
            )
            if args.train_only:
                invocation.update(
                    {
                        "status": "completed",
                        "finished_at_utc": utc_now(),
                        "training_freeze_sha256": training_freeze["freeze_sha256"],
                    }
                )
                atomic_write_json(invocation_path, invocation)
                print(f"training freeze complete: {args.output}", flush=True)
                return 0

            evaluation_jobs = build_evaluation_jobs()
            rows = evaluate_all(
                args,
                evaluation_jobs,
                spec["spec_sha256"],
                training_freeze,
                source,
                checkpoints,
            )
            manifest = write_final_artifacts(
                args.output,
                spec,
                training_freeze,
                evaluation_jobs,
                rows,
            )
            invocation.update(
                {
                    "status": "completed",
                    "finished_at_utc": utc_now(),
                    "decision": manifest["decision"]["decision"],
                    "manifest_sha256": manifest["manifest_sha256"],
                }
            )
            atomic_write_json(invocation_path, invocation)
            print(json.dumps(manifest["decision"], indent=2))
            print(f"integrated hysteresis screen complete: {args.output}", flush=True)
            return 0
    except BaseException as error:
        invocation.update(
            {
                "status": (
                    "interrupted"
                    if isinstance(error, (KeyboardInterrupt, SystemExit))
                    else "failed"
                ),
                "finished_at_utc": utc_now(),
                "error": repr(error),
            }
        )
        atomic_write_json(invocation_path, invocation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
