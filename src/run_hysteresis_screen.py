"""Tune and evaluate actor-score hysteresis on frozen context checkpoints."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
from typing import Any, Iterable, Mapping, Sequence, get_type_hints
import uuid

import numpy as np
import torch

from ablation_matrix_runner import (
    JobLockBusy,
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
from hysteresis_policy import SWITCH_FEATURE_INDEX, load_hysteresis_policy
from mappo_evaluation import EpisodeMetrics, evaluate_policy
from variant_definitions import canonical_variant_name


SCREEN_NAME = "ACTOR-SCORE-HYSTERESIS-SCREEN-v1"
SCHEMA_VERSION = 1
SOURCE_SCREEN_NAME = "CONGESTION-CONTEXT-SCREEN-v1"
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072, 1047581915, 1245825580)
SOURCE_VARIANTS = ("proposed", "with_congestion_context")
TUNING_WORKLOAD_SEEDS = tuple(range(34001, 34021))
FINAL_WORKLOAD_SEEDS = tuple(range(35001, 35051))
TUNING_BETAS = (0.02, 0.05, 0.10, 0.20, 0.40)
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
TUNING_GATES = {
    "switch_ratio_max": 1.05,
    "hotspot_delivery_vs_raw_min": -0.002,
    "medium_delivery_vs_proposed_min": -0.003,
}
ADVANCEMENT_GATES = {
    "hotspot_delivery_vs_proposed_min": 0.010,
    "hotspot_positive_seed_fraction_min": 0.75,
    "medium_delivery_vs_proposed_min": -0.003,
    "medium_seed_noninferiority_margin": -0.010,
    "medium_seed_noninferiority_count_min": 3,
    "cost_relative_regression_max": 0.10,
    "class_delivery_regression_max": 0.020,
}


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


def ensure_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(value):
            raise RuntimeError(f"immutable artifact mismatch: {path}")
        return
    atomic_write_json(path, dict(value))


def _self_hash(record: Mapping[str, Any], field: str) -> str:
    body = dict(record)
    observed = body.pop(field, None)
    expected = sha256_json(body)
    if observed != expected:
        raise ValueError(f"invalid {field}: observed={observed!r}, expected={expected}")
    return expected


def _path_is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def beta_token(beta: float) -> str:
    return f"{float(beta):.2f}".replace(".", "p")


def proposed_policy_name() -> str:
    return "mappo_proposed"


def raw_context_policy_name() -> str:
    return "mappo_with_congestion_context"


def hysteresis_policy_name(beta: float, *, final: bool) -> str:
    if final:
        return "mappo_context_hysteresis"
    return f"mappo_context_hysteresis_beta_{beta_token(beta)}"


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    phase: str
    scenario: str
    arm: str
    policy_seed: int
    stay_bonus: float

    @property
    def source_variant(self) -> str:
        return "proposed" if self.arm == "proposed" else "with_congestion_context"

    @property
    def policy_name(self) -> str:
        if self.arm == "proposed":
            return proposed_policy_name()
        if self.arm == "raw_context":
            return raw_context_policy_name()
        return hysteresis_policy_name(self.stay_bonus, final=self.phase == "final")

    @property
    def source_job_id(self) -> str:
        return f"{self.scenario}/{self.source_variant}/seed_{self.policy_seed}"

    @property
    def job_id(self) -> str:
        return (
            f"{self.phase}/{self.scenario}/{self.arm}/"
            f"beta_{beta_token(self.stay_bonus)}/seed_{self.policy_seed}"
        )

    @property
    def slug(self) -> str:
        return "__".join(
            (
                self.scenario,
                self.arm,
                f"beta_{beta_token(self.stay_bonus)}",
                f"seed_{self.policy_seed}",
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "phase": self.phase,
            "scenario": self.scenario,
            "arm": self.arm,
            "policy_seed": self.policy_seed,
            "stay_bonus": self.stay_bonus,
            "policy_name": self.policy_name,
            "source_variant": self.source_variant,
            "source_job_id": self.source_job_id,
            "job_id": self.job_id,
        }


def build_tuning_jobs() -> list[EvaluationJob]:
    arms = [("proposed", 0.0), ("raw_context", 0.0)] + [
        ("hysteresis", beta) for beta in TUNING_BETAS
    ]
    jobs: list[EvaluationJob] = []
    for scenario in SCENARIOS:
        for arm, beta in arms:
            for policy_seed in POLICY_SEEDS:
                jobs.append(
                    EvaluationJob(
                        index=len(jobs),
                        phase="tuning",
                        scenario=scenario,
                        arm=arm,
                        policy_seed=policy_seed,
                        stay_bonus=beta,
                    )
                )
    return jobs


def build_final_jobs(selected_beta: float) -> list[EvaluationJob]:
    arms = (
        ("proposed", 0.0),
        ("raw_context", 0.0),
        ("hysteresis", float(selected_beta)),
    )
    jobs: list[EvaluationJob] = []
    for scenario in SCENARIOS:
        for arm, beta in arms:
            for policy_seed in POLICY_SEEDS:
                jobs.append(
                    EvaluationJob(
                        index=len(jobs),
                        phase="final",
                        scenario=scenario,
                        arm=arm,
                        policy_seed=policy_seed,
                        stay_bonus=beta,
                    )
                )
    return jobs


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def audit_source_screen(source: Path) -> dict[str, Any]:
    source = source.resolve()
    spec_path = source / "screen_spec.json"
    freeze_path = source / "training_freeze.json"
    manifest_path = source / "screen_manifest.json"
    spec = _load_json(spec_path)
    freeze = _load_json(freeze_path)
    manifest = _load_json(manifest_path)
    spec_sha = _self_hash(spec, "spec_sha256")
    freeze_sha = _self_hash(freeze, "freeze_sha256")

    if spec.get("screen_name") != SOURCE_SCREEN_NAME:
        raise ValueError("source screen name mismatch")
    if spec.get("confirmatory") is not False:
        raise ValueError("source screen must remain exploratory")
    if freeze.get("screen_name") != SOURCE_SCREEN_NAME:
        raise ValueError("source training freeze name mismatch")
    if freeze.get("spec_sha256") != spec_sha:
        raise ValueError("source training freeze/spec mismatch")
    if manifest.get("spec_sha256") != spec_sha:
        raise ValueError("source manifest/spec mismatch")
    if manifest.get("training_freeze_sha256") != freeze_sha:
        raise ValueError("source manifest/training-freeze mismatch")

    expected_ids = {
        f"{scenario}/{variant}/seed_{seed}"
        for scenario in SCENARIOS
        for variant in SOURCE_VARIANTS
        for seed in POLICY_SEEDS
    }
    checkpoints = freeze.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_ids:
        raise ValueError("source training freeze does not contain the exact 16-job grid")

    audited_checkpoints: dict[str, dict[str, Any]] = {}
    for job_id in sorted(expected_ids):
        record = checkpoints[job_id]
        checkpoint_path = Path(str(record["checkpoint_path"])).resolve()
        checkpoint_sha = str(record["checkpoint_sha256"])
        if not _path_is_within(checkpoint_path, source):
            raise ValueError(f"source checkpoint escapes source directory: {job_id}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"source checkpoint SHA-256 mismatch: {job_id}")
        scenario, variant, seed_label = job_id.split("/")
        policy_seed = int(seed_label.removeprefix("seed_"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_variant = canonical_variant_name(
            str(checkpoint.get("args", {}).get("leo_variant"))
        )
        expected_dim = 26 if variant == "proposed" else 28
        if checkpoint_variant != variant:
            raise ValueError(f"source checkpoint variant mismatch: {job_id}")
        if int(checkpoint.get("candidate_feature_dim", -1)) != expected_dim:
            raise ValueError(f"source checkpoint feature schema mismatch: {job_id}")
        if int(checkpoint.get("action_size", -1)) != 7:
            raise ValueError(f"source checkpoint action schema mismatch: {job_id}")
        if int(checkpoint.get("n_agents", -1)) != 24:
            raise ValueError(f"source checkpoint agent schema mismatch: {job_id}")
        if int(checkpoint.get("obs_size", -1)) != 7 * expected_dim:
            raise ValueError(f"source checkpoint observation schema mismatch: {job_id}")
        if int(checkpoint.get("args", {}).get("seed", -1)) != policy_seed:
            raise ValueError(f"source checkpoint policy seed mismatch: {job_id}")
        if int(checkpoint.get("step", -1)) != int(record["selected_checkpoint_step"]):
            raise ValueError(f"source checkpoint selected-step mismatch: {job_id}")
        selected_artifact = record.get("artifacts", {}).get("selected_checkpoint", {})
        if Path(str(selected_artifact.get("path", ""))).resolve() != checkpoint_path:
            raise ValueError(f"source selected-checkpoint path mismatch: {job_id}")
        if selected_artifact.get("sha256") != checkpoint_sha:
            raise ValueError(f"source selected-checkpoint hash mismatch: {job_id}")
        audited_checkpoints[job_id] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "candidate_feature_dim": expected_dim,
            "selected_checkpoint_step": int(record["selected_checkpoint_step"]),
        }

    return {
        "source_path": str(source),
        "screen_spec_path": str(spec_path.resolve()),
        "screen_spec_file_sha256": sha256_file(spec_path),
        "spec_sha256": spec_sha,
        "training_freeze_path": str(freeze_path.resolve()),
        "training_freeze_file_sha256": sha256_file(freeze_path),
        "training_freeze_sha256": freeze_sha,
        "screen_manifest_path": str(manifest_path.resolve()),
        "screen_manifest_sha256": sha256_file(manifest_path),
        "source_code_fingerprint": dict(spec["code_fingerprint"]),
        "checkpoints": audited_checkpoints,
    }


def build_screen_spec(args: argparse.Namespace, source: Mapping[str, Any]) -> dict[str, Any]:
    repository_root = args.project.parent.resolve()
    protocol_path = repository_root / "docs" / "HYSTERESIS_SCREEN_V1.md"
    runner_path = Path(__file__).resolve()
    policy_path = args.project / "hysteresis_policy.py"
    code_files = {
        "runner": runner_path,
        "protocol": protocol_path,
        "hysteresis_policy": policy_path,
        "design": args.project / "mappo_design.py",
        "evaluation": args.project / "mappo_evaluation.py",
        "wrapper": args.project / "cleanmarl_leo_multiagent_wrapper.py",
        "multiagent_environment": args.project / "leo_multiagent_env.py",
        "base_environment": args.project / "leo_marl_env.py",
        "variants": args.project / "variant_definitions.py",
        "statistics": args.project / "hierarchical_statistics.py",
        "artifact_helpers": args.project / "ablation_matrix_runner.py",
    }
    module_files = {
        "hysteresis_policy": "hysteresis_policy.py",
        "mappo_design": "mappo_design.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "cleanmarl_leo_multiagent_wrapper": "cleanmarl_leo_multiagent_wrapper.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "leo_marl_env": "leo_marl_env.py",
        "variant_definitions": "variant_definitions.py",
        "hierarchical_statistics": "hierarchical_statistics.py",
        "ablation_matrix_runner": "ablation_matrix_runner.py",
    }
    for module_name, filename in module_files.items():
        actual = Path(inspect.getfile(importlib.import_module(module_name))).resolve()
        expected = (args.project / filename).resolve()
        if actual != expected:
            raise RuntimeError(
                f"runtime module path mismatch for {module_name}: {actual} != {expected}"
            )
    code_fingerprint = {name: sha256_file(path) for name, path in code_files.items()}
    source_fingerprint = source["source_code_fingerprint"]
    source_contract = {
        "design": "design",
        "evaluation": "evaluation",
        "wrapper": "wrapper",
        "multiagent_environment": "environment",
        "base_environment": "base_environment",
        "variants": "variant_definitions",
        "statistics": "screen_statistics",
        "artifact_helpers": "screen_artifact_helpers",
    }
    drift = {
        current_name: {
            "current": code_fingerprint[current_name],
            "source": source_fingerprint[source_name],
        }
        for current_name, source_name in source_contract.items()
        if code_fingerprint[current_name] != source_fingerprint[source_name]
    }
    if drift:
        raise RuntimeError(f"source v1 runtime fingerprint drift: {drift}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_two_stage_method_optimization",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "device": args.device,
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "tuning_workload_seeds": list(TUNING_WORKLOAD_SEEDS),
        "final_workload_seeds": list(FINAL_WORKLOAD_SEEDS),
        "tuning_betas": list(TUNING_BETAS),
        "tuning_gates": TUNING_GATES,
        "advancement_gates": ADVANCEMENT_GATES,
        "metrics": list(METRICS),
        "selection_rule": "smallest_beta_passing_every_tuning_gate",
        "expected_tuning_jobs": len(build_tuning_jobs()),
        "expected_tuning_rows": len(build_tuning_jobs())
        * len(TUNING_WORKLOAD_SEEDS),
        "expected_final_jobs_after_selection": 24,
        "expected_final_rows_after_selection": 24 * len(FINAL_WORKLOAD_SEEDS),
        "source": dict(source),
        "code_fingerprint": code_fingerprint,
        "source_runtime_contract": source_contract,
        "paths": {
            "project": str(args.project.resolve()),
            "protocol": str(protocol_path.resolve()),
            "source_screen": str(args.source.resolve()),
        },
    }
    body["spec_sha256"] = sha256_json(body)
    return body


def _process_create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except ImportError as error:
        raise RuntimeError("psutil is required for safe runner-lock recovery") from error
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, psutil.Error):
        return float("nan")


def _existing_lock_is_live(payload: Mapping[str, Any]) -> bool | None:
    if payload.get("hostname") != socket.gethostname():
        return None
    try:
        pid = int(payload["pid"])
        recorded_create_time = float(payload["process_create_time"])
    except (KeyError, TypeError, ValueError):
        return None
    observed_create_time = _process_create_time(pid)
    if observed_create_time is None:
        return False
    if not math.isfinite(observed_create_time):
        return None
    return abs(observed_create_time - recorded_create_time) < 1e-3


@contextmanager
def invocation_lock(output: Path):
    path = output / "runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    create_time = _process_create_time(os.getpid())
    if create_time is None or not math.isfinite(create_time):
        raise RuntimeError("cannot determine the current process identity")
    payload = {
        "pid": os.getpid(),
        "process_create_time": create_time,
        "hostname": socket.gethostname(),
        "token": token,
        "created_at_utc": utc_now(),
    }
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            try:
                existing = _load_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise JobLockBusy(
                    f"cannot safely recover malformed runner lock: {path}"
                ) from error
            live = _existing_lock_is_live(existing)
            if live is not False:
                raise JobLockBusy(f"hysteresis runner is already active: {existing}")
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        break
    else:
        raise JobLockBusy(f"could not acquire runner lock: {path}")
    try:
        yield
    finally:
        try:
            existing = _load_json(path)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            existing = {}
        if existing.get("token") == token:
            path.unlink(missing_ok=True)


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
    workload_seeds: Sequence[int],
) -> list[EpisodeMetrics]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _episode_rows_from_csv(path)
    if len(rows) != len(workload_seeds):
        raise ValueError(f"evaluation shard has {len(rows)} rows")
    expected_workloads = {int(seed) for seed in workload_seeds}
    observed_workloads = [row.workload_seed for row in rows]
    if len(set(observed_workloads)) != len(observed_workloads):
        raise ValueError(f"duplicate workloads in {path}")
    if set(observed_workloads) != expected_workloads:
        raise ValueError(f"workload grid mismatch in {path}")
    for row in rows:
        if row.scenario != job.scenario:
            raise ValueError(f"scenario mismatch in {path}")
        if row.policy != job.policy_name:
            raise ValueError(f"policy mismatch in {path}")
        if row.policy_seed != job.policy_seed:
            raise ValueError(f"policy-seed mismatch in {path}")
    return rows


def _evaluation_paths(output: Path, job: EvaluationJob) -> tuple[Path, Path]:
    directory = output / "evaluation_shards" / job.phase
    return directory / f"{job.slug}.csv", directory / f"{job.slug}.json"


def _expected_shard_metadata(
    job: EvaluationJob,
    spec_sha256: str,
    source_checkpoint: Mapping[str, Any],
    workload_seeds: Sequence[int],
    selection_freeze_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "selection_freeze_sha256": selection_freeze_sha256,
        "job": job.as_dict(),
        "checkpoint_path": source_checkpoint["checkpoint_path"],
        "checkpoint_sha256": source_checkpoint["checkpoint_sha256"],
        "policy_schema": {
            "candidate_feature_dim": int(
                source_checkpoint["candidate_feature_dim"]
            ),
            "action_size": 7,
            "obs_size": 7 * int(source_checkpoint["candidate_feature_dim"]),
            "n_agents": 24,
            "variant": job.source_variant,
            "switch_feature_index": SWITCH_FEATURE_INDEX,
            "stay_bonus": job.stay_bonus,
        },
        "workload_seeds": list(workload_seeds),
    }


def evaluate_job(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec_sha256: str,
    source_checkpoint: Mapping[str, Any],
    workload_seeds: Sequence[int],
    *,
    selection_freeze_sha256: str | None = None,
) -> list[EpisodeMetrics]:
    shard_path, metadata_path = _evaluation_paths(args.output, job)
    expected_metadata = _expected_shard_metadata(
        job,
        spec_sha256,
        source_checkpoint,
        workload_seeds,
        selection_freeze_sha256,
    )
    if shard_path.is_file() and metadata_path.is_file():
        metadata = _load_json(metadata_path)
        if (
            all(metadata.get(key) == value for key, value in expected_metadata.items())
            and metadata.get("csv_sha256") == sha256_file(shard_path)
            and metadata.get("row_count") == len(workload_seeds)
        ):
            return validate_evaluation_shard(shard_path, job, workload_seeds)

    checkpoint_path = Path(str(source_checkpoint["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != source_checkpoint["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash changed before {job.job_id}")
    policy, _ = load_hysteresis_policy(
        checkpoint_path,
        stay_bonus=job.stay_bonus,
        device=args.device,
    )
    expected_policy_schema = dict(expected_metadata["policy_schema"])
    observed_policy_schema = {
        **policy.checkpoint_schema,
        "switch_feature_index": SWITCH_FEATURE_INDEX,
        "stay_bonus": policy.stay_bonus,
    }
    if observed_policy_schema != expected_policy_schema:
        raise ValueError(
            f"loaded policy schema mismatch for {job.job_id}: "
            f"{observed_policy_schema} != {expected_policy_schema}"
        )
    rows = evaluate_policy(
        job.scenario,
        job.policy_name,
        policy,
        job.policy_seed,
        workload_seeds,
        variant=job.source_variant,
    )
    if len(rows) != len(workload_seeds):
        raise ValueError("evaluation returned the wrong row count")
    atomic_write_csv(shard_path, (asdict(row) for row in rows))
    validated_rows = validate_evaluation_shard(shard_path, job, workload_seeds)
    metadata = {
        **expected_metadata,
        "csv_sha256": sha256_file(shard_path),
        "row_count": len(validated_rows),
        "policy_diagnostics": policy.diagnostics(),
    }
    atomic_write_json(metadata_path, metadata)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return validated_rows


def evaluate_jobs(
    args: argparse.Namespace,
    jobs: Sequence[EvaluationJob],
    spec_sha256: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
    workload_seeds: Sequence[int],
    *,
    selection_freeze_sha256: str | None = None,
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for completed, job in enumerate(jobs, start=1):
        rows.extend(
            evaluate_job(
                args,
                job,
                spec_sha256,
                checkpoints[job.source_job_id],
                workload_seeds,
                selection_freeze_sha256=selection_freeze_sha256,
            )
        )
        print(
            f"[{completed}/{len(jobs)}] completed {job.job_id}",
            flush=True,
        )
    expected_rows = len(jobs) * len(workload_seeds)
    if len(rows) != expected_rows:
        raise RuntimeError(f"evaluation has {len(rows)} rows, expected {expected_rows}")
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed) for row in rows
    }
    if len(keys) != expected_rows:
        raise RuntimeError("evaluation contains duplicate episode cells")
    return rows


def aggregate_rows(
    rows: Sequence[EpisodeMetrics],
    *,
    rng_base: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({(row.scenario, row.policy) for row in rows})
    for group_index, (scenario, policy) in enumerate(groups):
        selected = [
            row for row in rows if row.scenario == scenario and row.policy == policy
        ]
        for metric_index, metric in enumerate(METRICS):
            summary = crossed_mean_summary(
                build_crossed_matrix(selected, metric),
                rng_seed=rng_base + 100 * group_index + metric_index,
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


def _metric_mean(
    rows: Sequence[EpisodeMetrics],
    scenario: str,
    policy: str,
    metric: str,
) -> float:
    selected = [
        float(getattr(row, metric))
        for row in rows
        if row.scenario == scenario and row.policy == policy
    ]
    expected = len(POLICY_SEEDS) * (
        len(TUNING_WORKLOAD_SEEDS)
        if len(rows) == len(build_tuning_jobs()) * len(TUNING_WORKLOAD_SEEDS)
        else len(FINAL_WORKLOAD_SEEDS)
    )
    if len(selected) != expected:
        raise ValueError(
            f"incomplete mean cell {scenario}/{policy}/{metric}: {len(selected)}"
        )
    return float(np.mean(selected))


def select_tuning_beta(rows: Sequence[EpisodeMetrics]) -> tuple[list[dict[str, Any]], float | None]:
    proposed = proposed_policy_name()
    raw = raw_context_policy_name()
    selection_rows: list[dict[str, Any]] = []
    for beta in TUNING_BETAS:
        candidate = hysteresis_policy_name(beta, final=False)
        medium_switch_ratio = _metric_mean(
            rows, "medium_load", candidate, "routing_switches"
        ) / max(
            _metric_mean(rows, "medium_load", proposed, "routing_switches"),
            1e-12,
        )
        hotspot_switch_ratio = _metric_mean(
            rows, "hotspot_high_load", candidate, "routing_switches"
        ) / max(
            _metric_mean(
                rows, "hotspot_high_load", proposed, "routing_switches"
            ),
            1e-12,
        )
        hotspot_delivery_vs_raw = _metric_mean(
            rows, "hotspot_high_load", candidate, "delivery_ratio"
        ) - _metric_mean(rows, "hotspot_high_load", raw, "delivery_ratio")
        medium_delivery_vs_proposed = _metric_mean(
            rows, "medium_load", candidate, "delivery_ratio"
        ) - _metric_mean(rows, "medium_load", proposed, "delivery_ratio")
        medium_switch_pass = medium_switch_ratio <= TUNING_GATES["switch_ratio_max"]
        hotspot_switch_pass = hotspot_switch_ratio <= TUNING_GATES["switch_ratio_max"]
        hotspot_delivery_pass = (
            hotspot_delivery_vs_raw
            >= TUNING_GATES["hotspot_delivery_vs_raw_min"]
        )
        medium_delivery_pass = (
            medium_delivery_vs_proposed
            >= TUNING_GATES["medium_delivery_vs_proposed_min"]
        )
        eligible = all(
            (
                medium_switch_pass,
                hotspot_switch_pass,
                hotspot_delivery_pass,
                medium_delivery_pass,
            )
        )
        selection_rows.append(
            {
                "stay_bonus": beta,
                "medium_switch_ratio_vs_proposed": medium_switch_ratio,
                "hotspot_switch_ratio_vs_proposed": hotspot_switch_ratio,
                "hotspot_delivery_difference_vs_raw": hotspot_delivery_vs_raw,
                "medium_delivery_difference_vs_proposed": medium_delivery_vs_proposed,
                "medium_switch_gate_pass": medium_switch_pass,
                "hotspot_switch_gate_pass": hotspot_switch_pass,
                "hotspot_delivery_gate_pass": hotspot_delivery_pass,
                "medium_delivery_gate_pass": medium_delivery_pass,
                "eligible": eligible,
            }
        )
    selected = next(
        (float(row["stay_bonus"]) for row in selection_rows if row["eligible"]),
        None,
    )
    return selection_rows, selected


def write_tuning_artifacts(
    output: Path,
    spec_sha256: str,
    source_freeze_sha256: str,
    rows: Sequence[EpisodeMetrics],
) -> dict[str, Any]:
    episode_path = output / "tuning_episode_metrics.csv"
    aggregate_path = output / "tuning_aggregate_metrics.csv"
    selection_path = output / "tuning_selection.csv"
    episode_rows = [asdict(row) for row in rows]
    aggregate = aggregate_rows(rows, rng_base=51000)
    selection_rows, selected_beta = select_tuning_beta(rows)
    atomic_write_csv(episode_path, episode_rows)
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(selection_path, selection_rows)
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "source_training_freeze_sha256": source_freeze_sha256,
        "selection_status": (
            "selected" if selected_beta is not None else "no_eligible_candidate"
        ),
        "selected_beta": selected_beta,
        "selection_rule": "smallest_beta_passing_every_tuning_gate",
        "tuning_gates": TUNING_GATES,
        "candidate_betas": list(TUNING_BETAS),
        "tuning_jobs": len(build_tuning_jobs()),
        "tuning_rows": len(rows),
        "artifacts": {
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
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
                "row_count": len(selection_rows),
            },
        },
    }
    body["selection_freeze_sha256"] = sha256_json(body)
    ensure_immutable_json(output / "selection_freeze.json", body)
    return body


def _read_tuning_selection(path: Path) -> list[dict[str, Any]]:
    float_fields = {
        "stay_bonus",
        "medium_switch_ratio_vs_proposed",
        "hotspot_switch_ratio_vs_proposed",
        "hotspot_delivery_difference_vs_raw",
        "medium_delivery_difference_vs_proposed",
    }
    bool_fields = {
        "medium_switch_gate_pass",
        "hotspot_switch_gate_pass",
        "hotspot_delivery_gate_pass",
        "medium_delivery_gate_pass",
        "eligible",
    }
    expected_fields = [
        "stay_bonus",
        "medium_switch_ratio_vs_proposed",
        "hotspot_switch_ratio_vs_proposed",
        "hotspot_delivery_difference_vs_raw",
        "medium_delivery_difference_vs_proposed",
        "medium_switch_gate_pass",
        "hotspot_switch_gate_pass",
        "hotspot_delivery_gate_pass",
        "medium_delivery_gate_pass",
        "eligible",
    ]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError("selection table schema mismatch")
        for raw in reader:
            converted: dict[str, Any] = {}
            for field in expected_fields:
                if field in float_fields:
                    value = float(raw[field])
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite selection field: {field}")
                    converted[field] = value
                elif field in bool_fields:
                    if raw[field] not in ("True", "False"):
                        raise ValueError(f"invalid selection Boolean: {field}")
                    converted[field] = raw[field] == "True"
            rows.append(converted)
    return rows


def load_selection_freeze(output: Path, spec_sha256: str) -> dict[str, Any]:
    path = output / "selection_freeze.json"
    freeze = _load_json(path)
    _self_hash(freeze, "selection_freeze_sha256")
    if freeze.get("screen_name") != SCREEN_NAME:
        raise ValueError("selection freeze screen mismatch")
    if freeze.get("spec_sha256") != spec_sha256:
        raise ValueError("selection freeze spec mismatch")
    if freeze.get("selection_rule") != "smallest_beta_passing_every_tuning_gate":
        raise ValueError("selection rule mismatch")
    if freeze.get("candidate_betas") != list(TUNING_BETAS):
        raise ValueError("selection candidate grid mismatch")
    artifacts = freeze.get("artifacts")
    required_artifacts = {"episode_metrics", "aggregate_metrics", "selection"}
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
        raise ValueError("selection freeze artifact set mismatch")
    artifact_paths: dict[str, Path] = {}
    for name, artifact in artifacts.items():
        artifact_path = Path(str(artifact["path"])).resolve()
        if not _path_is_within(artifact_path, output):
            raise ValueError(f"selection artifact escapes output: {name}")
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"selection artifact hash mismatch: {name}")
        with artifact_path.open("r", encoding="utf-8-sig", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        if row_count != int(artifact["row_count"]):
            raise ValueError(f"selection artifact row count mismatch: {name}")
        artifact_paths[name] = artifact_path

    tuning_rows = _episode_rows_from_csv(artifact_paths["episode_metrics"])
    expected_keys = {
        (job.scenario, job.policy_name, job.policy_seed, int(workload_seed))
        for job in build_tuning_jobs()
        for workload_seed in TUNING_WORKLOAD_SEEDS
    }
    observed_keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in tuning_rows
    }
    if len(tuning_rows) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError("selection freeze tuning grid is incomplete or duplicated")
    recomputed_rows, recomputed_beta = select_tuning_beta(tuning_rows)
    recorded_selection = _read_tuning_selection(artifact_paths["selection"])
    if len(recorded_selection) != len(recomputed_rows):
        raise ValueError("selection table candidate count mismatch")
    for recorded, recomputed in zip(recorded_selection, recomputed_rows, strict=True):
        if recorded.keys() != recomputed.keys():
            raise ValueError("selection table schema mismatch")
        for key, expected in recomputed.items():
            observed = recorded[key]
            if isinstance(expected, bool):
                matches = observed is expected
            else:
                matches = math.isclose(
                    float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
                )
            if not matches:
                raise ValueError(
                    f"selection replay mismatch for beta={recomputed['stay_bonus']}, "
                    f"field={key}: {observed!r} != {expected!r}"
                )
    selected = freeze.get("selected_beta")
    if selected is not None and float(selected) not in TUNING_BETAS:
        raise ValueError("selected beta is outside the frozen grid")
    expected_status = "selected" if recomputed_beta is not None else "no_eligible_candidate"
    if freeze.get("selection_status") != expected_status:
        raise ValueError("selection status does not match replayed tuning result")
    if selected is None:
        selected_matches = recomputed_beta is None
    else:
        selected_matches = recomputed_beta is not None and math.isclose(
            float(selected), recomputed_beta, rel_tol=0.0, abs_tol=1e-12
        )
    if not selected_matches:
        raise ValueError("selected beta is not the replayed smallest eligible candidate")
    return freeze


def paired_effect_rows(rows: Sequence[EpisodeMetrics]) -> list[dict[str, Any]]:
    proposed = proposed_policy_name()
    raw = raw_context_policy_name()
    optimized = hysteresis_policy_name(0.0, final=True)
    contrasts = (
        ("raw_context_minus_proposed", raw, proposed),
        ("hysteresis_minus_proposed", optimized, proposed),
        ("hysteresis_minus_raw_context", optimized, raw),
    )
    output: list[dict[str, Any]] = []
    for contrast_index, (contrast, treatment_name, reference_name) in enumerate(
        contrasts
    ):
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
                        52000
                        + 1000 * contrast_index
                        + 100 * scenario_index
                        + metric_index
                    ),
                    resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                )
                output.append(
                    {
                        "analysis": "exploratory_hysteresis_final_holdout",
                        "contrast": contrast,
                        "scenario": scenario,
                        "reference": reference_name,
                        "treatment": treatment_name,
                        "metric": metric,
                        **summary,
                    }
                )
    return output


def decide_final(paired_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["contrast"]), str(row["scenario"]), str(row["metric"])): row
        for row in paired_rows
    }

    def row(contrast: str, scenario: str, metric: str) -> Mapping[str, Any]:
        return indexed[(contrast, scenario, metric)]

    switch_ratios: dict[str, float] = {}
    for scenario in SCENARIOS:
        effect = row("hysteresis_minus_proposed", scenario, "routing_switches")
        switch_ratios[scenario] = float(effect["treatment_mean"]) / max(
            float(effect["reference_mean"]), 1e-12
        )
    hotspot_vs_raw = float(
        row(
            "hysteresis_minus_raw_context",
            "hotspot_high_load",
            "delivery_ratio",
        )["mean_difference"]
    )
    medium_vs_proposed = float(
        row(
            "hysteresis_minus_proposed", "medium_load", "delivery_ratio"
        )["mean_difference"]
    )
    optimization_gates = {
        "medium_switch_ratio": switch_ratios["medium_load"]
        <= TUNING_GATES["switch_ratio_max"],
        "hotspot_switch_ratio": switch_ratios["hotspot_high_load"]
        <= TUNING_GATES["switch_ratio_max"],
        "hotspot_delivery_preservation": hotspot_vs_raw
        >= TUNING_GATES["hotspot_delivery_vs_raw_min"],
        "medium_delivery_noninferiority": medium_vs_proposed
        >= TUNING_GATES["medium_delivery_vs_proposed_min"],
    }
    optimization_accepted = all(optimization_gates.values())

    hotspot = row(
        "hysteresis_minus_proposed", "hotspot_high_load", "delivery_ratio"
    )
    medium = row(
        "hysteresis_minus_proposed", "medium_load", "delivery_ratio"
    )
    medium_seed_differences = [
        float(value) for value in medium["policy_seed_mean_differences"]
    ]
    medium_seed_noninferior_count = sum(
        value > ADVANCEMENT_GATES["medium_seed_noninferiority_margin"]
        for value in medium_seed_differences
    )
    cost_regressions: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for metric in LOWER_IS_BETTER_COSTS:
            effect = row("hysteresis_minus_proposed", scenario, metric)
            relative = float(effect["mean_difference"]) / max(
                abs(float(effect["reference_mean"])), 1e-12
            )
            if relative > ADVANCEMENT_GATES["cost_relative_regression_max"]:
                cost_regressions.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "relative_regression": relative,
                    }
                )
        for metric in CLASS_DELIVERY_METRICS:
            difference = float(
                row("hysteresis_minus_proposed", scenario, metric)[
                    "mean_difference"
                ]
            )
            if difference < -ADVANCEMENT_GATES["class_delivery_regression_max"]:
                cost_regressions.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "absolute_regression": -difference,
                    }
                )
    advancement_gates = {
        "hotspot_effect_size": float(hotspot["mean_difference"])
        >= ADVANCEMENT_GATES["hotspot_delivery_vs_proposed_min"],
        "hotspot_seed_consistency": float(
            hotspot["positive_policy_seed_fraction"]
        )
        >= ADVANCEMENT_GATES["hotspot_positive_seed_fraction_min"],
        "medium_delivery_noninferiority": medium_vs_proposed
        >= ADVANCEMENT_GATES["medium_delivery_vs_proposed_min"],
        "medium_seed_consistency": medium_seed_noninferior_count
        >= ADVANCEMENT_GATES["medium_seed_noninferiority_count_min"],
        "cost_and_class_safety": not cost_regressions,
    }
    advancement_passed = all(advancement_gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_two_stage_method_optimization",
        "optimization_decision": (
            "accept_for_later_method_development"
            if optimization_accepted
            else "reject_hysteresis_optimization"
        ),
        "optimization_accepted": optimization_accepted,
        "optimization_gates": optimization_gates,
        "advancement_decision": (
            "promote_to_separately_frozen_training_experiment"
            if advancement_passed
            else "do_not_promote"
        ),
        "advancement_passed": advancement_passed,
        "advancement_gates": advancement_gates,
        "evidence": {
            "switch_ratios_vs_proposed": switch_ratios,
            "hotspot_delivery_difference_vs_raw": hotspot_vs_raw,
            "hotspot_delivery_difference_vs_proposed": float(
                hotspot["mean_difference"]
            ),
            "hotspot_positive_policy_seed_fraction": float(
                hotspot["positive_policy_seed_fraction"]
            ),
            "medium_delivery_difference_vs_proposed": medium_vs_proposed,
            "medium_noninferior_policy_seed_count": medium_seed_noninferior_count,
            "cost_regressions": cost_regressions,
        },
        "tuning_gates": TUNING_GATES,
        "advancement_thresholds": ADVANCEMENT_GATES,
        "paper_claim_allowed": False,
    }


def _paired_row(
    rows: Sequence[Mapping[str, Any]],
    contrast: str,
    scenario: str,
    metric: str,
) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["contrast"] == contrast
        and row["scenario"] == scenario
        and row["metric"] == metric
    )


def write_summary(
    path: Path,
    selection_freeze: Mapping[str, Any],
    paired_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    beta = float(selection_freeze["selected_beta"])
    lines = [
        "# Actor-Score Hysteresis Screen v1",
        "",
        "> Exploratory two-stage optimization only. This is not final IEEE evidence.",
        "",
        f"Frozen stay bonus: **beta = {beta:.2f}**",
        "",
        f"Optimization decision: **{decision['optimization_decision']}**",
        "",
        f"Advancement decision: **{decision['advancement_decision']}**",
        "",
        "## Final delivery results",
        "",
        "| Scenario | Contrast | Reference | Treatment | Difference | 95% crossed CI | Positive seeds | Exact p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for contrast, label in (
            ("raw_context_minus_proposed", "Raw context - proposed"),
            ("hysteresis_minus_proposed", "Hysteresis - proposed"),
            ("hysteresis_minus_raw_context", "Hysteresis - raw context"),
        ):
            row = _paired_row(paired_rows, contrast, scenario, "delivery_ratio")
            lines.append(
                "| {scenario} | {label} | {reference:.4f} | {treatment:.4f} | "
                "{difference:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                "{positive:.0%} | {p_value:.4f} |".format(
                    scenario=scenario,
                    label=label,
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
            "## Final routing stability",
            "",
            "| Scenario | Proposed switches | Hysteresis switches | Ratio | Gate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    switch_ratios = decision["evidence"]["switch_ratios_vs_proposed"]
    for scenario in SCENARIOS:
        row = _paired_row(
            paired_rows,
            "hysteresis_minus_proposed",
            scenario,
            "routing_switches",
        )
        ratio = float(switch_ratios[scenario])
        lines.append(
            f"| {scenario} | {float(row['reference_mean']):.2f} | "
            f"{float(row['treatment_mean']):.2f} | {ratio:.3f} | "
            f"{'pass' if ratio <= TUNING_GATES['switch_ratio_max'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The beta grid was selected only on workloads 34001..34020. These "
            "tables use the untouched 35001..35050 panel. Four policy seeds imply "
            "a minimum attainable two-sided exact p-value of 0.125.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_artifacts(
    output: Path,
    spec: Mapping[str, Any],
    selection_freeze: Mapping[str, Any],
    rows: Sequence[EpisodeMetrics],
) -> dict[str, Any]:
    episode_path = output / "final_episode_metrics.csv"
    aggregate_path = output / "final_aggregate_metrics.csv"
    paired_path = output / "final_paired_effects.csv"
    decision_path = output / "final_decision.json"
    summary_path = output / "HYSTERESIS_SUMMARY.md"
    episode_rows = [asdict(row) for row in rows]
    aggregate = aggregate_rows(rows, rng_base=53000)
    paired = paired_effect_rows(rows)
    decision = decide_final(paired)
    decision["selected_beta"] = float(selection_freeze["selected_beta"])
    atomic_write_csv(episode_path, episode_rows)
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(paired_path, paired)
    atomic_write_json(decision_path, decision)
    write_summary(summary_path, selection_freeze, paired, decision)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exploratory_two_stage_method_optimization",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "source_training_freeze_sha256": spec["source"][
            "training_freeze_sha256"
        ],
        "selection_freeze_sha256": selection_freeze[
            "selection_freeze_sha256"
        ],
        "selected_beta": float(selection_freeze["selected_beta"]),
        "final_jobs": len(build_final_jobs(float(selection_freeze["selected_beta"]))),
        "final_rows": len(rows),
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(FINAL_WORKLOAD_SEEDS),
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
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    ensure_immutable_json(output / "hysteresis_manifest.json", manifest)
    return manifest


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset for this screen")
    repository_root = args.project.parent.resolve()
    experiments_root = (repository_root / "experiments").resolve()
    archive_root = (experiments_root / "archive").resolve()
    broad_outputs = {
        repository_root,
        args.project.resolve(),
        experiments_root,
        archive_root,
        args.source.resolve(),
    }
    output = args.output.resolve()
    source = args.source.resolve()
    if output in broad_outputs:
        raise RuntimeError("output must be a new dedicated experiment directory")
    if source in output.parents or output in source.parents:
        raise RuntimeError("output and frozen source must be in disjoint directory trees")
    formal_output = (experiments_root / "ablation-50k-v2").resolve()
    if output == formal_output or formal_output in output.parents:
        raise RuntimeError("output must not be inside the frozen ablation experiment")
    if args.output.is_dir() and not (args.output / "hysteresis_spec.json").is_file():
        conflicting = [
            name
            for name in (
                "screen_spec.json",
                "matrix_spec.json",
                "experiment_manifest.json",
                "screen_manifest.json",
            )
            if (args.output / name).exists()
        ]
        if conflicting:
            raise RuntimeError(f"output contains another experiment: {conflicting}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Tune and test actor-score hysteresis on frozen checkpoints."
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
            / "congestion-hysteresis-screen-v1"
        ),
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--tune-only", action="store_true")
    modes.add_argument("--final-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.project = args.project.resolve()
    validate_environment(args)
    source = audit_source_screen(args.source)
    spec = build_screen_spec(args, source)
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_immutable_json(args.output / "hysteresis_spec.json", spec)
    atomic_write_csv(
        args.output / "tuning_plan.csv",
        (job.as_dict() for job in build_tuning_jobs()),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "screen_name": SCREEN_NAME,
                    "spec_sha256": spec["spec_sha256"],
                    "source_training_freeze_sha256": source[
                        "training_freeze_sha256"
                    ],
                    "tuning_jobs": len(build_tuning_jobs()),
                    "tuning_rows": len(build_tuning_jobs())
                    * len(TUNING_WORKLOAD_SEEDS),
                    "final_jobs_after_selection": 24,
                    "final_rows_after_selection": 24 * len(FINAL_WORKLOAD_SEEDS),
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
            "tune_only"
            if args.tune_only
            else "final_only"
            if args.final_only
            else "tune_and_final"
        ),
        "device": args.device,
    }
    atomic_write_json(invocation_path, invocation)
    try:
        with invocation_lock(args.output):
            if args.final_only:
                selection_freeze = load_selection_freeze(
                    args.output, spec["spec_sha256"]
                )
            else:
                tuning_rows = evaluate_jobs(
                    args,
                    build_tuning_jobs(),
                    spec["spec_sha256"],
                    source["checkpoints"],
                    TUNING_WORKLOAD_SEEDS,
                )
                selection_freeze = write_tuning_artifacts(
                    args.output,
                    spec["spec_sha256"],
                    source["training_freeze_sha256"],
                    tuning_rows,
                )
            if selection_freeze["selection_status"] != "selected":
                invocation.update(
                    {
                        "status": "completed_no_eligible_candidate",
                        "finished_at_utc": utc_now(),
                    }
                )
                atomic_write_json(invocation_path, invocation)
                print("no hysteresis beta passed the frozen tuning gates", flush=True)
                return 2
            if args.tune_only:
                invocation.update(
                    {"status": "completed", "finished_at_utc": utc_now()}
                )
                atomic_write_json(invocation_path, invocation)
                print(
                    f"selected beta={float(selection_freeze['selected_beta']):.2f}",
                    flush=True,
                )
                return 0

            selected_beta = float(selection_freeze["selected_beta"])
            final_jobs = build_final_jobs(selected_beta)
            atomic_write_csv(
                args.output / "final_plan.csv",
                (job.as_dict() for job in final_jobs),
            )
            final_rows = evaluate_jobs(
                args,
                final_jobs,
                spec["spec_sha256"],
                source["checkpoints"],
                FINAL_WORKLOAD_SEEDS,
                selection_freeze_sha256=selection_freeze[
                    "selection_freeze_sha256"
                ],
            )
            manifest = write_final_artifacts(
                args.output, spec, selection_freeze, final_rows
            )
            invocation.update(
                {
                    "status": "completed",
                    "finished_at_utc": utc_now(),
                    "selected_beta": selected_beta,
                    "manifest_sha256": manifest["manifest_sha256"],
                }
            )
            atomic_write_json(invocation_path, invocation)
            print(f"hysteresis screen complete: {args.output}", flush=True)
            return 0
    except BaseException as error:
        invocation.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "error": repr(error),
            }
        )
        atomic_write_json(invocation_path, invocation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
