"""Run the frozen exploratory congestion-context candidate screen."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, get_type_hints
import uuid

import torch

from ablation_matrix_runner import (
    JobLockBusy,
    atomic_write_csv,
    atomic_write_json,
    job_lock,
    lock_path,
    sha256_file,
    utc_now,
    validate_evaluation_shard,
)
from hierarchical_statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    build_crossed_matrix,
    crossed_mean_summary,
    pair_crossed_matrices,
    paired_crossed_summary,
    statistical_analysis_manifest,
)
from mappo_evaluation import EpisodeMetrics, evaluate_policy, load_checkpoint_policy
from run_exp004_mappo import code_fingerprint, train_one
from variant_definitions import canonical_variant_name, resolve_variant


SCREEN_NAME = "CONGESTION-CONTEXT-SCREEN-v1"
SCHEMA_VERSION = 1
SCENARIOS = ("medium_load", "hotspot_high_load")
VARIANTS = ("proposed", "with_congestion_context")
POLICY_SEED_NAMESPACE = f"{SCREEN_NAME}-policy-seed-"
VALIDATION_WORKLOAD_SEEDS = tuple(range(32001, 32021))
TEST_WORKLOAD_SEEDS = tuple(range(33001, 33051))
REFERENCE_VARIANT = "proposed"
TREATMENT_VARIANT = "with_congestion_context"
TRAINER_OVERRIDES = (
    "--validation-seed-start",
    str(VALIDATION_WORKLOAD_SEEDS[0]),
    "--run-tag",
    SCREEN_NAME,
)
SCREEN_METRICS = (
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
DECISION_THRESHOLDS = {
    "hotspot_delivery_promote_min": 0.010,
    "hotspot_positive_seed_fraction_min": 0.75,
    "medium_delivery_promote_min": -0.003,
    "medium_seed_noninferiority_margin": -0.010,
    "medium_seed_noninferiority_count_min": 3,
    "cost_relative_regression_max": 0.10,
    "class_delivery_regression_max": 0.020,
    "medium_delivery_reject_below": -0.010,
}


def derive_policy_seeds(namespace: str, count: int) -> tuple[int, ...]:
    seeds = []
    for index in range(count):
        digest = hashlib.sha256(f"{namespace}{index}".encode("ascii")).digest()
        seeds.append(int.from_bytes(digest[:4], "big") & 0x7FFFFFFF)
    return tuple(seeds)


POLICY_SEEDS = derive_policy_seeds(POLICY_SEED_NAMESPACE, 4)
EXPECTED_POLICY_SEEDS = (1710210210, 2078783072, 1047581915, 1245825580)
if POLICY_SEEDS != EXPECTED_POLICY_SEEDS:
    raise RuntimeError("screening policy-seed derivation changed")


@dataclass(frozen=True)
class ScreenJob:
    index: int
    scenario: str
    variant: str
    policy_seed: int

    @property
    def job_id(self) -> str:
        return f"{self.scenario}/{self.variant}/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return f"{self.scenario}__{self.variant}__seed_{self.policy_seed}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"job_id": self.job_id}


def build_jobs() -> list[ScreenJob]:
    jobs = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            for policy_seed in POLICY_SEEDS:
                jobs.append(
                    ScreenJob(
                        index=len(jobs),
                        scenario=scenario,
                        variant=variant,
                        policy_seed=policy_seed,
                    )
                )
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


def screen_code_fingerprint(args: argparse.Namespace) -> dict[str, str]:
    fingerprints = code_fingerprint(args)
    executed_design = args.cleanmarl / "cleanmarl" / "mappo_design.py"
    executed_design_sha = sha256_file(executed_design)
    if executed_design_sha != fingerprints["design"]:
        raise RuntimeError(
            "executed CleanMARL design module differs from the repository snapshot"
        )
    return {**fingerprints, "executed_design": executed_design_sha}


def _expected_feature_dim(variant: str) -> int:
    return 28 if variant == TREATMENT_VARIANT else 26


def build_screen_spec(args: argparse.Namespace, jobs: Sequence[ScreenJob]) -> dict[str, Any]:
    protocol_path = args.project.parent / "docs" / "CONGESTION_CONTEXT_SCREEN_V1.md"
    runner_path = Path(__file__).resolve()
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_candidate_screening",
        "confirmatory": False,
        "contrast": {
            "reference": REFERENCE_VARIANT,
            "treatment": TREATMENT_VARIANT,
        },
        "training_device": args.device,
        "config": CONFIG,
        "scenarios": list(SCENARIOS),
        "variants": list(VARIANTS),
        "policy_seed_derivation": {
            "namespace": POLICY_SEED_NAMESPACE,
            "digest": "sha256",
            "extraction": "first_4_bytes_big_endian_bitand_0x7fffffff",
            "indices": list(range(len(POLICY_SEEDS))),
        },
        "policy_seeds": list(POLICY_SEEDS),
        "train_workload_seeds": list(range(9001, 9201)),
        "validation_workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "trainer_overrides": list(TRAINER_OVERRIDES),
        "metrics": list(SCREEN_METRICS),
        "decision_thresholds": DECISION_THRESHOLDS,
        "expected_training_jobs": len(jobs),
        "expected_evaluation_rows": len(jobs) * len(TEST_WORKLOAD_SEEDS),
        "jobs": [job.as_dict() for job in jobs],
        "code_fingerprint": {
            **screen_code_fingerprint(args),
            "screen_runner": sha256_file(runner_path),
            "screen_protocol": sha256_file(protocol_path),
            "screen_statistics": sha256_file(
                args.project / "hierarchical_statistics.py"
            ),
            "screen_artifact_helpers": sha256_file(
                args.project / "ablation_matrix_runner.py"
            ),
        },
        "paths": {
            "project": str(args.project.resolve()),
            "cleanmarl": str(args.cleanmarl.resolve()),
            "protocol": str(protocol_path.resolve()),
        },
    }
    body["spec_sha256"] = sha256_json(body)
    return body


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(value):
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, dict(value))


def _status_path(output: Path, job: ScreenJob) -> Path:
    return output / "job_status" / f"{job.slug}.json"


def _new_phase() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": [],
        "artifact": None,
    }


def load_job_status(output: Path, job: ScreenJob, spec_sha256: str) -> dict[str, Any]:
    path = _status_path(output, job)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "spec_sha256": spec_sha256,
            "job": job.as_dict(),
            "training": _new_phase(),
            "evaluation": _new_phase(),
        }
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported status schema: {path}")
    if status.get("spec_sha256") != spec_sha256:
        raise RuntimeError(f"status/spec mismatch: {path}")
    if status.get("job", {}).get("job_id") != job.job_id:
        raise RuntimeError(f"status/job mismatch: {path}")
    status.setdefault("training", _new_phase())
    status.setdefault("evaluation", _new_phase())
    return status


def write_job_status(output: Path, job: ScreenJob, status: dict[str, Any]) -> None:
    status["updated_at_utc"] = utc_now()
    atomic_write_json(_status_path(output, job), status)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        # Windows reports access denial for protected live processes.
        return getattr(error, "winerror", None) == 5
    return True


def _matching_trainer_pids(output: Path, job: ScreenJob) -> tuple[int, ...] | None:
    try:
        import psutil
    except ImportError:
        return None

    expected_checkpoint_dir = (
        output
        / "checkpoints"
        / job.scenario
        / job.variant
        / f"seed_{job.policy_seed}"
    ).resolve()
    matches = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = process.info.get("cmdline") or []
            if "--checkpoint-dir" not in command:
                continue
            index = command.index("--checkpoint-dir")
            if index + 1 >= len(command):
                continue
            observed_checkpoint_dir = Path(command[index + 1]).resolve()
            is_trainer = any(
                Path(token).name.lower() == "mappo.py" for token in command
            )
            if is_trainer and observed_checkpoint_dir == expected_checkpoint_dir:
                matches.append(int(process.info["pid"]))
        except (OSError, ValueError, psutil.Error):
            continue
    return tuple(sorted(matches))


def _recover_orphaned_job_lock(output: Path, job: ScreenJob) -> None:
    path = lock_path(output, job)
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        owner_pid = int(payload.get("pid", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        trainers = _matching_trainer_pids(output, job)
        if trainers is None or trainers:
            raise JobLockBusy(
                f"cannot safely recover malformed lock for {job.job_id}; "
                f"matching_trainers={trainers}"
            )
        try:
            age_seconds = time.time() - path.stat().st_mtime
        except OSError:
            return
        if age_seconds > 300.0:
            path.unlink(missing_ok=True)
        return
    if payload.get("hostname") != socket.gethostname():
        return
    if not _process_is_alive(owner_pid):
        trainers = _matching_trainer_pids(output, job)
        if trainers is None or trainers:
            raise JobLockBusy(
                f"runner lock is orphaned but its trainer may still be active for "
                f"{job.job_id}; matching_trainers={trainers}"
            )
        path.unlink(missing_ok=True)


@contextmanager
def screen_job_lock(output: Path, job: ScreenJob):
    _recover_orphaned_job_lock(output, job)
    # This screen runs on one host. A live local owner never expires merely
    # because a long validation pass crossed a wall-clock threshold.
    with job_lock(output, job, stale_after_seconds=float("inf")):
        yield


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


def audit_checkpoint(
    checkpoint_path: Path,
    job: ScreenJob,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    run_directory = checkpoint_path.parent
    run_root = run_directory.parent
    manifest_path = run_directory / "run_manifest.json"
    config_path = run_directory / "run_config.json"
    metrics_path = run_directory / "training_metrics.jsonl"
    final_path = run_directory / "final.pt"
    fingerprint_path = run_root / "code_fingerprint.json"
    stdout_path = run_root / "trainer_stdout.log"
    required = (
        manifest_path,
        config_path,
        metrics_path,
        final_path,
        fingerprint_path,
        stdout_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training artifacts missing: {missing}")

    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "env_type": "leo_multi",
        "env_name": job.scenario,
        "batch_size": CONFIG["batch_size"],
        "total_timesteps": CONFIG["timesteps"],
        "epochs": 3,
        "num_minibatches": 4,
        "eval_steps": CONFIG["eval_every_rollouts"],
        "num_eval_ep": len(VALIDATION_WORKLOAD_SEEDS),
        "save_every_steps": CONFIG["save_every_steps"],
        "train_seed_start": 9001,
        "train_seed_count": 200,
        "validation_seed_start": VALIDATION_WORKLOAD_SEEDS[0],
        "seed": job.policy_seed,
        "run_tag": SCREEN_NAME,
        "device": args.device,
    }
    mismatches = {
        key: {"observed": run_config.get(key), "expected": expected}
        for key, expected in expected_config.items()
        if run_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint run configuration mismatch: {mismatches}")
    if canonical_variant_name(str(run_config.get("leo_variant"))) != job.variant:
        raise ValueError("checkpoint variant mismatch")

    def manifest_path_value(name: str) -> Path:
        value = run_manifest.get(name)
        if not value:
            raise ValueError(f"run manifest lacks {name}")
        path = Path(str(value))
        if not path.is_absolute():
            path = run_directory / path.name
        return path.resolve()

    if manifest_path_value("validation_best_checkpoint") != checkpoint_path:
        raise ValueError("run manifest does not select the audited checkpoint")
    if manifest_path_value("final_checkpoint") != final_path.resolve():
        raise ValueError("run manifest final-checkpoint path mismatch")

    observed_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    current_screen_fingerprint = screen_code_fingerprint(args)
    current_training_fingerprint = {
        key: value
        for key, value in current_screen_fingerprint.items()
        if key != "executed_design"
    }
    expected_fingerprint = {
        "code": current_training_fingerprint,
        "scenario": job.scenario,
        "policy_seed": job.policy_seed,
        "experiment_variant": job.variant,
        "environment_variant": canonical_variant_name(job.variant),
        "variant_definition": resolve_variant(job.variant).as_dict(),
        "config": dict(CONFIG),
        "trainer_overrides": list(TRAINER_OVERRIDES),
    }
    expected_fingerprint = json.loads(canonical_json_bytes(expected_fingerprint))
    if observed_fingerprint != expected_fingerprint:
        raise ValueError("per-job training fingerprint differs from the frozen code")

    actual_steps = int(run_manifest.get("environment_steps", -1))
    step_span = CONFIG["batch_size"] * 30
    if not CONFIG["timesteps"] <= actual_steps < CONFIG["timesteps"] + step_span:
        raise ValueError(f"actual training steps out of range: {actual_steps}")
    best_score = run_manifest.get("best_validation_score")
    if not isinstance(best_score, list) or len(best_score) != 4 or not _all_finite(best_score):
        raise ValueError("best validation score is missing or non-finite")

    metric_records = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid training metrics JSON at line {line_number}"
                ) from error
            if not isinstance(record, Mapping) or not _all_finite(record):
                raise ValueError(
                    f"training metrics contain invalid values at line {line_number}"
                )
            metric_records.append(record)
    if not metric_records:
        raise ValueError("training metrics are empty")
    recorded_steps = [
        int(record["environment_steps"])
        for record in metric_records
        if "environment_steps" in record
    ]
    if not recorded_steps or max(recorded_steps) != actual_steps:
        raise ValueError("training metrics and manifest step counts differ")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) > actual_steps:
        raise ValueError("selected checkpoint step exceeds completed training")
    if int(checkpoint.get("candidate_feature_dim", -1)) != _expected_feature_dim(job.variant):
        raise ValueError("selected checkpoint candidate schema mismatch")
    if int(checkpoint.get("action_size", -1)) != 7:
        raise ValueError("selected checkpoint action schema mismatch")
    if int(checkpoint.get("n_agents", -1)) != 24:
        raise ValueError("selected checkpoint agent schema mismatch")
    if int(checkpoint.get("obs_size", -1)) != 7 * _expected_feature_dim(job.variant):
        raise ValueError("selected checkpoint observation schema mismatch")
    checkpoint_variant = checkpoint.get("args", {}).get("leo_variant")
    if canonical_variant_name(str(checkpoint_variant)) != job.variant:
        raise ValueError("selected checkpoint variant metadata mismatch")
    critic_spec = checkpoint.get("critic_spec") or {}
    expected_node_dim = 26 if job.variant == TREATMENT_VARIANT else 25
    if int(critic_spec.get("node_feature_dim", -1)) != expected_node_dim:
        raise ValueError("selected checkpoint critic schema mismatch")

    selected_step = int(checkpoint["step"])
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
            raise ValueError("selected checkpoint validation episode count mismatch")
        if int(record.get("seed_start", -1)) != VALIDATION_WORKLOAD_SEEDS[0]:
            raise ValueError("selected checkpoint validation panel mismatch")
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
            raise ValueError("selected validation score differs from the run manifest")
    if not any(record.get("is_validation_best") is True for record in validation_records):
        raise ValueError("selected checkpoint is not marked as a validation best")

    artifacts = {
        "selected_checkpoint": checkpoint_path,
        "final_checkpoint": final_path,
        "run_manifest": manifest_path,
        "run_config": config_path,
        "training_metrics": metrics_path,
        "job_fingerprint": fingerprint_path,
        "trainer_stdout": stdout_path,
    }
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_step": selected_step,
        "actual_environment_steps": actual_steps,
        "best_validation_score": [float(value) for value in best_score],
        "candidate_feature_dim": int(checkpoint["candidate_feature_dim"]),
        "critic_spec": critic_spec,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }


def _phase_attempt(phase: dict[str, Any]) -> dict[str, Any]:
    attempt = {
        "attempt": len(phase.setdefault("attempts", [])) + 1,
        "started_at_utc": utc_now(),
        "status": "running",
    }
    phase["attempts"].append(attempt)
    phase["status"] = "running"
    return attempt


def _train_args(args: argparse.Namespace, *, skip_training: bool) -> SimpleNamespace:
    return SimpleNamespace(
        output=args.output,
        cleanmarl=args.cleanmarl,
        project=args.project,
        skip_training=skip_training,
        device=args.device,
    )


def train_job(
    args: argparse.Namespace,
    job: ScreenJob,
    spec_sha256: str,
) -> dict[str, Any]:
    with screen_job_lock(args.output, job):
        status = load_job_status(args.output, job, spec_sha256)
        recorded = status["training"].get("artifact")
        if status["training"].get("status") == "completed" and recorded:
            observed = audit_checkpoint(Path(recorded["checkpoint_path"]), job, args)
            if observed != recorded:
                raise RuntimeError(f"completed checkpoint audit drifted: {job.job_id}")
            return observed

        attempt = _phase_attempt(status["training"])
        write_job_status(args.output, job, status)
        try:
            checkpoint = train_one(
                _train_args(args, skip_training=False),
                dict(CONFIG),
                job.scenario,
                job.policy_seed,
                variant=job.variant,
                trainer_overrides=list(TRAINER_OVERRIDES),
            )
            artifact = audit_checkpoint(checkpoint, job, args)
        except BaseException as error:
            attempt.update(
                {
                    "status": "failed",
                    "finished_at_utc": utc_now(),
                    "error": repr(error),
                }
            )
            status["training"]["status"] = "failed"
            write_job_status(args.output, job, status)
            raise
        attempt.update({"status": "completed", "finished_at_utc": utc_now()})
        status["training"].update(
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
    jobs: Sequence[ScreenJob],
    spec_sha256: str,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    output_lock = threading.Lock()
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
                with output_lock:
                    completed[job.job_id] = artifact
                print(
                    f"completed training {job.job_id} "
                    f"at step {artifact['actual_environment_steps']}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"{len(failures)} screening training jobs failed: {failures}")
    if len(completed) != len(jobs):
        raise RuntimeError("screening training grid is incomplete")
    return {job.job_id: completed[job.job_id] for job in jobs}


def resolve_training_freeze(
    args: argparse.Namespace,
    jobs: Sequence[ScreenJob],
    spec_sha256: str,
) -> dict[str, dict[str, Any]]:
    checkpoints = {}
    for job in jobs:
        status = load_job_status(args.output, job, spec_sha256)
        recorded = status["training"].get("artifact")
        if status["training"].get("status") != "completed" or not recorded:
            checkpoint = train_one(
                _train_args(args, skip_training=True),
                dict(CONFIG),
                job.scenario,
                job.policy_seed,
                variant=job.variant,
                trainer_overrides=list(TRAINER_OVERRIDES),
            )
            recorded = audit_checkpoint(checkpoint, job, args)
            status["training"].update(
                {"status": "completed", "artifact": recorded}
            )
            write_job_status(args.output, job, status)
        observed = audit_checkpoint(Path(recorded["checkpoint_path"]), job, args)
        if observed != recorded:
            raise RuntimeError(f"training freeze audit mismatch: {job.job_id}")
        checkpoints[job.job_id] = observed
    return checkpoints


def ensure_training_freeze(
    output: Path,
    spec_sha256: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_job_ids = {job.job_id for job in build_jobs()}
    if set(checkpoints) != expected_job_ids:
        raise RuntimeError("training freeze does not contain the exact 16-job grid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "training_complete": True,
        "checkpoint_count": len(checkpoints),
        "checkpoints": dict(checkpoints),
    }
    body["freeze_sha256"] = sha256_json(body)
    path = output / "training_freeze.json"
    ensure_immutable_json(path, body)
    return body


def _episode_rows_from_csv(path: Path) -> list[EpisodeMetrics]:
    field_types = get_type_hints(EpisodeMetrics)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            converted = {}
            for field, field_type in field_types.items():
                if field_type is int:
                    converted[field] = int(raw[field])
                elif field_type is float:
                    converted[field] = float(raw[field])
                elif field_type is str:
                    converted[field] = raw[field]
                else:
                    raise TypeError(f"unsupported EpisodeMetrics type: {field_type!r}")
            rows.append(EpisodeMetrics(**converted))
    return rows


def _evaluation_paths(output: Path, job: ScreenJob) -> tuple[Path, Path]:
    directory = output / "evaluation_shards"
    return directory / f"{job.slug}.csv", directory / f"{job.slug}.json"


def evaluate_job(
    args: argparse.Namespace,
    job: ScreenJob,
    spec_sha256: str,
    training_artifact: Mapping[str, Any],
) -> list[EpisodeMetrics]:
    shard_path, metadata_path = _evaluation_paths(args.output, job)
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "spec_sha256": spec_sha256,
        "job": job.as_dict(),
        "checkpoint_path": training_artifact["checkpoint_path"],
        "checkpoint_sha256": training_artifact["checkpoint_sha256"],
        "workload_seeds": list(TEST_WORKLOAD_SEEDS),
    }
    with screen_job_lock(args.output, job):
        status = load_job_status(args.output, job, spec_sha256)
        if shard_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            csv_sha = metadata.pop("csv_sha256", None)
            row_count = metadata.pop("row_count", None)
            if (
                metadata == expected_metadata
                and csv_sha == sha256_file(shard_path)
                and row_count == len(TEST_WORKLOAD_SEEDS)
            ):
                validate_evaluation_shard(shard_path, job, TEST_WORKLOAD_SEEDS)
                rows = _episode_rows_from_csv(shard_path)
                status["evaluation"].update(
                    {
                        "status": "completed",
                        "artifact": {
                            "path": str(shard_path.resolve()),
                            "sha256": csv_sha,
                            "row_count": row_count,
                        },
                    }
                )
                write_job_status(args.output, job, status)
                return rows

        attempt = _phase_attempt(status["evaluation"])
        write_job_status(args.output, job, status)
        try:
            checkpoint_path = Path(training_artifact["checkpoint_path"])
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if sha256_file(checkpoint_path) != training_artifact["checkpoint_sha256"]:
                raise ValueError("checkpoint hash changed after the training freeze")
            policy, _ = load_checkpoint_policy(
                checkpoint_path,
                device=args.device,
            )
            rows = evaluate_policy(
                job.scenario,
                f"mappo_{job.variant}",
                policy,
                job.policy_seed,
                TEST_WORKLOAD_SEEDS,
                variant=job.variant,
            )
            if len(rows) != len(TEST_WORKLOAD_SEEDS):
                raise ValueError("evaluation returned the wrong row count")
            atomic_write_csv(shard_path, (asdict(row) for row in rows))
            validate_evaluation_shard(shard_path, job, TEST_WORKLOAD_SEEDS)
            metadata = {
                **expected_metadata,
                "csv_sha256": sha256_file(shard_path),
                "row_count": len(rows),
            }
            atomic_write_json(metadata_path, metadata)
        except BaseException as error:
            attempt.update(
                {
                    "status": "failed",
                    "finished_at_utc": utc_now(),
                    "error": repr(error),
                }
            )
            status["evaluation"]["status"] = "failed"
            write_job_status(args.output, job, status)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        attempt.update({"status": "completed", "finished_at_utc": utc_now()})
        artifact = {
            "path": str(shard_path.resolve()),
            "sha256": sha256_file(shard_path),
            "row_count": len(rows),
        }
        status["evaluation"].update(
            {
                "status": "completed",
                "artifact": artifact,
                "completed_at_utc": utc_now(),
            }
        )
        write_job_status(args.output, job, status)
        return rows


def evaluate_all(
    args: argparse.Namespace,
    jobs: Sequence[ScreenJob],
    spec_sha256: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> list[EpisodeMetrics]:
    rows = []
    for job in jobs:
        job_rows = evaluate_job(
            args,
            job,
            spec_sha256,
            checkpoints[job.job_id],
        )
        rows.extend(job_rows)
        print(f"completed evaluation {job.job_id}", flush=True)
    expected_rows = len(jobs) * len(TEST_WORKLOAD_SEEDS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"evaluation has {len(rows)} rows, expected {expected_rows}")
    keys = {(row.scenario, row.policy, row.policy_seed, row.workload_seed) for row in rows}
    if len(keys) != expected_rows:
        raise RuntimeError("evaluation contains duplicate episode cells")
    return rows


def aggregate_screen_rows(rows: Sequence[EpisodeMetrics]) -> list[dict[str, Any]]:
    output = []
    groups = sorted({(row.scenario, row.policy) for row in rows})
    for group_index, (scenario, policy) in enumerate(groups):
        selected = [
            row for row in rows if row.scenario == scenario and row.policy == policy
        ]
        for metric_index, metric in enumerate(SCREEN_METRICS):
            summary = crossed_mean_summary(
                build_crossed_matrix(selected, metric),
                rng_seed=41000 + 100 * group_index + metric_index,
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


def paired_screen_rows(rows: Sequence[EpisodeMetrics]) -> list[dict[str, Any]]:
    output = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        treatment = [
            row
            for row in rows
            if row.scenario == scenario
            and row.policy == f"mappo_{TREATMENT_VARIANT}"
        ]
        reference = [
            row
            for row in rows
            if row.scenario == scenario
            and row.policy == f"mappo_{REFERENCE_VARIANT}"
        ]
        for metric_index, metric in enumerate(SCREEN_METRICS):
            paired = pair_crossed_matrices(treatment, reference, metric)
            summary = paired_crossed_summary(
                paired,
                rng_seed=42000 + 100 * scenario_index + metric_index,
                resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            )
            output.append(
                {
                    "analysis": "exploratory_candidate_screening",
                    "scenario": scenario,
                    "reference": REFERENCE_VARIANT,
                    "treatment": TREATMENT_VARIANT,
                    "metric": metric,
                    **summary,
                }
            )
    return output


def decide_screen(paired_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["scenario"]), str(row["metric"])): row for row in paired_rows
    }
    hotspot = indexed[("hotspot_high_load", "delivery_ratio")]
    medium = indexed[("medium_load", "delivery_ratio")]
    hotspot_delta = float(hotspot["mean_difference"])
    medium_delta = float(medium["mean_difference"])
    hotspot_positive_fraction = float(hotspot["positive_policy_seed_fraction"])
    medium_seed_differences = [
        float(value) for value in medium["policy_seed_mean_differences"]
    ]
    medium_noninferior_count = sum(
        value > DECISION_THRESHOLDS["medium_seed_noninferiority_margin"]
        for value in medium_seed_differences
    )

    cost_regressions = []
    for scenario in SCENARIOS:
        for metric in LOWER_IS_BETTER_COSTS:
            row = indexed[(scenario, metric)]
            reference = abs(float(row["reference_mean"]))
            difference = float(row["mean_difference"])
            relative = difference / max(reference, 1e-12)
            if relative > DECISION_THRESHOLDS["cost_relative_regression_max"]:
                cost_regressions.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "relative_regression": relative,
                    }
                )
        for metric in CLASS_DELIVERY_METRICS:
            difference = float(indexed[(scenario, metric)]["mean_difference"])
            if difference < -DECISION_THRESHOLDS["class_delivery_regression_max"]:
                cost_regressions.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "absolute_regression": -difference,
                    }
                )

    reject_reasons = []
    if hotspot_delta <= 0.0:
        reject_reasons.append("hotspot delivery did not improve")
    if medium_delta < DECISION_THRESHOLDS["medium_delivery_reject_below"]:
        reject_reasons.append("medium-load delivery crossed the hard reject margin")
    if hotspot_positive_fraction <= 0.50:
        reject_reasons.append("hotspot direction was supported by at most 2 of 4 seeds")
    if medium_noninferior_count < DECISION_THRESHOLDS[
        "medium_seed_noninferiority_count_min"
    ]:
        reject_reasons.append("medium-load seed consistency gate failed")
    if cost_regressions:
        reject_reasons.append("one or more predeclared cost gates failed")

    promote = (
        hotspot_delta >= DECISION_THRESHOLDS["hotspot_delivery_promote_min"]
        and hotspot_positive_fraction
        >= DECISION_THRESHOLDS["hotspot_positive_seed_fraction_min"]
        and medium_delta >= DECISION_THRESHOLDS["medium_delivery_promote_min"]
        and medium_noninferior_count
        >= DECISION_THRESHOLDS["medium_seed_noninferiority_count_min"]
        and not cost_regressions
    )
    if reject_reasons:
        decision = "reject"
    elif promote:
        decision = "promote_to_separately_frozen_50k_head_to_head"
    else:
        decision = "inconclusive"
    return {
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_candidate_screening",
        "decision": decision,
        "thresholds": DECISION_THRESHOLDS,
        "evidence": {
            "hotspot_delivery_difference": hotspot_delta,
            "hotspot_positive_policy_seed_fraction": hotspot_positive_fraction,
            "medium_delivery_difference": medium_delta,
            "medium_noninferior_policy_seed_count": medium_noninferior_count,
            "cost_regressions": cost_regressions,
        },
        "reject_reasons": reject_reasons,
        "minimum_attainable_two_sided_exact_p": 0.125,
        "paper_claim_allowed": False,
    }


def write_summary(
    path: Path,
    paired_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    indexed = {
        (str(row["scenario"]), str(row["metric"])): row for row in paired_rows
    }
    lines = [
        "# Congestion Context Screen v1",
        "",
        "> Exploratory candidate screen only. This is not final IEEE evidence.",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "| Scenario | Proposed | Congestion context | Difference | 95% crossed CI | Positive seeds | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        row = indexed[(scenario, "delivery_ratio")]
        lines.append(
            "| {scenario} | {reference:.4f} | {treatment:.4f} | "
            "{difference:+.4f} | [{low:+.4f}, {high:+.4f}] | {positive:.0%} | "
            "{p_value:.4f} |".format(
                scenario=scenario,
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
            "Four policy seeds imply a minimum two-sided exact p-value of 0.125. "
            "The decision uses the effect-size, direction, non-inferiority, and cost "
            "gates frozen before evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 2 x 2 x 4, 20k congestion-context screen."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "archive"
            / "congestion-context-screen-20k-v1"
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
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    return args


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset for the screen")
    repository_root = args.project.parent.resolve()
    experiments_root = (repository_root / "experiments").resolve()
    archive_root = (experiments_root / "archive").resolve()
    broad_outputs = {repository_root, args.project.resolve(), experiments_root, archive_root}
    if args.output in broad_outputs:
        raise RuntimeError("screen output must be a dedicated experiment directory")
    formal_output = (experiments_root / "ablation-50k-v2").resolve()
    if args.output == formal_output or formal_output in args.output.parents:
        raise RuntimeError("screen output must not be inside the frozen ablation output")
    if args.output.is_dir() and not (args.output / "screen_spec.json").is_file():
        conflicting_markers = [
            name
            for name in (
                "matrix_spec.json",
                "experiment_manifest.json",
                "screen_manifest.json",
            )
            if (args.output / name).exists()
        ]
        if conflicting_markers:
            raise RuntimeError(
                f"screen output contains artifacts from another experiment: "
                f"{conflicting_markers}"
            )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for variant in VARIANTS:
        if resolve_variant(variant).name != variant:
            raise RuntimeError(f"screen variant does not resolve canonically: {variant}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    args.cleanmarl = args.cleanmarl.resolve()
    args.project = args.project.resolve()
    validate_environment(args)
    jobs = build_jobs()
    spec = build_screen_spec(args, jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_immutable_json(args.output / "screen_spec.json", spec)
    atomic_write_csv(
        args.output / "screen_plan.csv",
        (job.as_dict() for job in jobs),
        ("index", "scenario", "variant", "policy_seed", "job_id"),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "screen_name": SCREEN_NAME,
                    "spec_sha256": spec["spec_sha256"],
                    "training_jobs": len(jobs),
                    "evaluation_rows": len(jobs) * len(TEST_WORKLOAD_SEEDS),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

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
        if not args.evaluate_only:
            checkpoints = train_all(args, jobs, spec["spec_sha256"])
        else:
            checkpoints = resolve_training_freeze(args, jobs, spec["spec_sha256"])
        freeze = ensure_training_freeze(
            args.output,
            spec["spec_sha256"],
            checkpoints,
        )
        if args.train_only:
            invocation.update(
                {"status": "completed", "finished_at_utc": utc_now()}
            )
            atomic_write_json(invocation_path, invocation)
            print(f"training freeze complete: {args.output}")
            return 0

        rows = evaluate_all(
            args,
            jobs,
            spec["spec_sha256"],
            checkpoints,
        )
        episode_rows = [asdict(row) for row in rows]
        episode_path = args.output / "episode_metrics.csv"
        aggregate_path = args.output / "aggregate_metrics.csv"
        paired_path = args.output / "paired_effects.csv"
        decision_path = args.output / "screen_decision.json"
        atomic_write_csv(episode_path, episode_rows)
        aggregate_rows = aggregate_screen_rows(rows)
        paired_rows = paired_screen_rows(rows)
        atomic_write_csv(aggregate_path, aggregate_rows)
        atomic_write_csv(paired_path, paired_rows)
        decision = decide_screen(paired_rows)
        atomic_write_json(decision_path, decision)
        summary_path = args.output / "SCREENING_SUMMARY.md"
        write_summary(summary_path, paired_rows, decision)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "screen_name": SCREEN_NAME,
            "inferential_status": "exploratory_candidate_screening",
            "confirmatory": False,
            "spec_sha256": spec["spec_sha256"],
            "training_freeze_sha256": freeze["freeze_sha256"],
            "training_jobs": len(jobs),
            "evaluation_rows": len(rows),
            "policy_seed_count": len(POLICY_SEEDS),
            "workload_seed_count": len(TEST_WORKLOAD_SEEDS),
            "statistical_analysis": statistical_analysis_manifest(),
            "decision": decision,
            "artifacts": {
                "episode_metrics": {
                    "path": str(episode_path.resolve()),
                    "sha256": sha256_file(episode_path),
                    "row_count": len(episode_rows),
                },
                "aggregate_metrics": {
                    "path": str(aggregate_path.resolve()),
                    "sha256": sha256_file(aggregate_path),
                    "row_count": len(aggregate_rows),
                },
                "paired_effects": {
                    "path": str(paired_path.resolve()),
                    "sha256": sha256_file(paired_path),
                    "row_count": len(paired_rows),
                },
                "decision": {
                    "path": str(decision_path.resolve()),
                    "sha256": sha256_file(decision_path),
                },
                "summary": {
                    "path": str(summary_path.resolve()),
                    "sha256": sha256_file(summary_path),
                },
            },
            "paper_claim_allowed": False,
        }
        atomic_write_json(args.output / "screen_manifest.json", manifest)
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
    invocation.update(
        {
            "status": "completed",
            "finished_at_utc": utc_now(),
            "decision": decision["decision"],
        }
    )
    atomic_write_json(invocation_path, invocation)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"screen complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
