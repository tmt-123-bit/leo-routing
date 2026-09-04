"""Resumable and auditable runner for the 50k-step MAPPO ablation matrix.

The runner deliberately keeps orchestration separate from the experiment and
environment modules.  Every matrix cell has an atomic JSON status record and
an atomic evaluation CSV.  A cell is resumable only when its phase fingerprint
and artifact SHA256 still match, which prevents silently mixing configurations
or code revisions in one reported matrix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence, get_type_hints
import uuid


SCHEMA_VERSION = 1
MATRIX_NAME = "controlled_ablation_50k_v2"

VARIANTS = (
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
SCENARIOS = (
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links",
)
# Frozen before the 50k rerun.  Twelve independent training seeds are needed
# because an exact two-sided sign-flip test with only eight seeds cannot attain
# the resolution needed for a 30--40 hypothesis Holm family.  Derivation is
# deterministic and preregistered so seed selection cannot depend on outcomes.
POLICY_SEED_NAMESPACE = "ABLATION-50K-v2.2-policy-seed-"
POLICY_SEEDS = tuple(
    int.from_bytes(
        hashlib.sha256(f"{POLICY_SEED_NAMESPACE}{index}".encode("ascii")).digest()[
            :4
        ],
        "big",
    )
    & 0x7FFFFFFF
    for index in range(12)
)
# The historical 13001..13050 panel was inspected during earlier 5k ablations.
# This untouched panel was selected by a repository-wide seed-usage audit.
TEST_WORKLOAD_SEEDS = tuple(range(31001, 31051))

# This is a protocol invariant, not an independently chosen runner preset.
# The actual dictionary passed to train_one is imported from
# run_exp004_mappo.mode_config("full") and must match this contract exactly.
EXPECTED_MAIN_FULL_CONFIG = {
    "scenarios": list(SCENARIOS),
    "timesteps": 50000,
    "validation_episodes": 50,
    "test_episodes": 50,
    "eval_every_rollouts": 40,
    "save_every_steps": 5000,
    "batch_size": 4,
    "q_routing_train_episodes": 500,
}

NO_PPO_PROTECTION_OVERRIDES = (
    "--clip-gradients",
    "0",
    "--target-kl",
    "0",
    "--no-normalize-advantage",
)

STATUS_DIR = "job_status"
LOCK_DIR = "job_locks"
EVALUATION_SHARD_DIR = "evaluation_shards"
TRAINING_RUN_DIR = "training_runs"
TRAINING_FREEZE_MANIFEST = "training_freeze_manifest.json"
HISTORICAL_WORKLOAD_SEED_AUDIT = "historical_workload_seed_audit.json"
REQUIRED_TRAINING_AUDIT_ARTIFACTS = (
    "run_manifest",
    "run_config",
    "training_metrics",
    "trainer_log",
    "training_code_fingerprint",
)

TRAINING_CODE_COMPONENTS = frozenset(
    {
        "runner",
        "protocol_document",
        "training_entry",
        "base_environment",
        "environment",
        "wrapper",
        "variant_definitions",
        "design",
        "evaluation",  # Included by run_exp004_mappo.code_fingerprint.
        "trainer",
        "trainer_snapshot",
    }
)
EVALUATION_CODE_COMPONENTS = frozenset(
    {
        "runner",
        "protocol_document",
        "base_environment",
        "environment",
        "wrapper",
        "variant_definitions",
        "design",
        "evaluation",
    }
)


class ConfigurationError(ValueError):
    """Raised when the requested run is not the preregistered 50k matrix."""


class JobLockBusy(RuntimeError):
    """Raised when another process currently owns a matrix cell."""


@dataclass(frozen=True)
class MatrixJob:
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
        return {
            "index": self.index,
            "job_id": self.job_id,
            "scenario": self.scenario,
            "variant": self.variant,
            "policy_seed": self.policy_seed,
        }


@dataclass(frozen=True)
class PhaseOutcome:
    job_id: str
    phase: str
    status: str
    attempts: int = 0
    message: str = ""


@dataclass(frozen=True)
class RuntimeAPI:
    train_one: Callable[..., Path]
    mode_config: Callable[[str], Mapping[str, Any]]
    runtime_scenarios: Sequence[str]
    runtime_policy_seeds: Sequence[int]
    load_checkpoint_policy: Callable[..., tuple[Any, Mapping[str, Any]]]
    evaluate_policy: Callable[..., Iterable[Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, writer: Callable[[Any], None], *, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            **kwargs,
        ) as temporary:
            temporary_name = temporary.name
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(path: Path, value: Any) -> None:
    def write(handle: Any) -> None:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    _atomic_replace(path, write, binary=False)


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> int:
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        if not materialized:
            raise ValueError("fieldnames are required when writing an empty CSV")
        fieldnames = tuple(materialized[0])
    fieldnames = tuple(fieldnames)

    def write(handle: Any) -> None:
        # utf-8-sig keeps compatibility with the existing experiment tables.
        handle.write("\ufeff")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)

    _atomic_replace(path, write, binary=False)
    return len(materialized)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _compress_integer_ranges(values: Iterable[int]) -> list[dict[str, int]]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    ranges: list[dict[str, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(
            {"start": start, "end": previous, "count": previous - start + 1}
        )
        start = previous = value
    ranges.append(
        {"start": start, "end": previous, "count": previous - start + 1}
    )
    return ranges


def validate_historical_workload_seed_audit(output: Path) -> dict[str, Any]:
    """Validate the immutable pre-run audit without rescanning experiment data."""

    path = output / HISTORICAL_WORKLOAD_SEED_AUDIT
    if not path.is_file():
        raise ConfigurationError(
            f"historical workload-seed audit is missing: {path.resolve()}"
        )
    manifest = read_json(path)
    content = manifest.get("content")
    if not isinstance(content, Mapping):
        raise ConfigurationError("historical workload-seed audit has no content")
    if manifest.get("status") != "passed_no_selected_workload_seed_overlap":
        raise ConfigurationError("historical workload-seed audit did not pass")
    content_sha256 = manifest.get("content_sha256")
    if content_sha256 != sha256_json(content):
        raise ConfigurationError("historical workload-seed audit self-hash mismatch")
    if content.get("selected_workload_seeds") != list(TEST_WORKLOAD_SEEDS):
        raise ConfigurationError(
            "historical workload-seed audit targets a different test panel"
        )
    if content.get("selected_policy_seeds") != list(POLICY_SEEDS):
        raise ConfigurationError(
            "historical workload-seed audit targets a different policy-seed panel"
        )
    if (
        content.get("selected_prior_overlap") != []
        or content.get("selected_workload_seed_prior_overlap") != []
        or content.get("selected_policy_seed_prior_overlap") != []
    ):
        raise ConfigurationError(
            "historical seed audit reports prior use of a selected seed panel"
        )
    scanned = content.get("scanned_csv_files")
    workload_ranges = content.get("historical_workload_seed_ranges")
    policy_seeds = content.get("historical_policy_seeds")
    if (
        not isinstance(scanned, list)
        or not isinstance(workload_ranges, list)
        or not isinstance(policy_seeds, list)
    ):
        raise ConfigurationError("historical workload-seed audit is malformed")
    return manifest


def ensure_historical_workload_seed_audit(
    output: Path,
    experiments_root: Path,
) -> dict[str, Any]:
    """Freeze a header/seed-only audit of historical experiment CSV files."""

    output = output.resolve()
    experiments_root = experiments_root.resolve()
    manifest_path = output / HISTORICAL_WORKLOAD_SEED_AUDIT
    if manifest_path.is_file():
        return validate_historical_workload_seed_audit(output)
    if output == experiments_root:
        raise ConfigurationError(
            "formal output cannot be the experiments root because the historical "
            "seed audit must exclude only its own output subtree"
        )

    observed_workload_seeds: set[int] = set()
    observed_policy_seeds: set[int] = set()
    scanned_files: list[dict[str, Any]] = []
    if experiments_root.is_dir():
        candidates = sorted(
            experiments_root.rglob("*.csv"),
            key=lambda item: item.as_posix().casefold(),
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if _path_is_within(resolved, output):
                continue
            if not _path_is_within(resolved, experiments_root):
                raise ConfigurationError(
                    f"historical CSV resolves outside experiments root: {candidate}"
                )
            relative_path = resolved.relative_to(experiments_root).as_posix()
            before_sha256 = sha256_file(resolved)
            row_count = 0
            workload_seed_row_count = 0
            policy_seed_row_count = 0
            has_workload_seed_column = False
            has_policy_seed_column = False
            with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    header = []
                workload_seed_indices = [
                    index
                    for index, field in enumerate(header)
                    if field.strip() == "workload_seed"
                ]
                policy_seed_indices = [
                    index
                    for index, field in enumerate(header)
                    if field.strip() == "policy_seed"
                ]
                if len(workload_seed_indices) > 1:
                    raise ConfigurationError(
                        f"historical CSV has duplicate workload_seed columns: {resolved}"
                    )
                if len(policy_seed_indices) > 1:
                    raise ConfigurationError(
                        f"historical CSV has duplicate policy_seed columns: {resolved}"
                    )
                has_workload_seed_column = len(workload_seed_indices) == 1
                has_policy_seed_column = len(policy_seed_indices) == 1
                workload_seed_index = (
                    workload_seed_indices[0]
                    if has_workload_seed_column
                    else None
                )
                policy_seed_index = (
                    policy_seed_indices[0] if has_policy_seed_column else None
                )
                for line_number, row in enumerate(reader, start=2):
                    row_count += 1
                    for field_name, field_index, destination in (
                        (
                            "workload_seed",
                            workload_seed_index,
                            observed_workload_seeds,
                        ),
                        ("policy_seed", policy_seed_index, observed_policy_seeds),
                    ):
                        if field_index is None:
                            continue
                        if field_index >= len(row):
                            raise ConfigurationError(
                                f"historical CSV row {line_number} is shorter than "
                                f"its {field_name} column: {resolved}"
                            )
                        raw_seed = row[field_index].strip()
                        if not raw_seed:
                            continue
                        try:
                            seed_value = int(raw_seed)
                        except ValueError as error:
                            raise ConfigurationError(
                                f"historical CSV has non-integer {field_name} "
                                f"{raw_seed!r} at {resolved}:{line_number}"
                            ) from error
                        if field_name == "workload_seed":
                            destination.add(seed_value)
                            workload_seed_row_count += 1
                        elif seed_value >= 0:
                            destination.add(seed_value)
                            policy_seed_row_count += 1
            after_sha256 = sha256_file(resolved)
            if after_sha256 != before_sha256:
                raise ConfigurationError(
                    f"historical CSV changed during seed audit: {resolved}"
                )
            scanned_files.append(
                {
                    "relative_path": relative_path,
                    "sha256": after_sha256,
                    "has_workload_seed_column": has_workload_seed_column,
                    "has_policy_seed_column": has_policy_seed_column,
                    "data_row_count": row_count,
                    "workload_seed_row_count": workload_seed_row_count,
                    "nonnegative_policy_seed_row_count": policy_seed_row_count,
                }
            )

    workload_overlap = sorted(
        observed_workload_seeds.intersection(TEST_WORKLOAD_SEEDS)
    )
    policy_overlap = sorted(observed_policy_seeds.intersection(POLICY_SEEDS))
    if workload_overlap or policy_overlap:
        raise ConfigurationError(
            "selected formal seed panels have prior experiment overlap: "
            f"policy={policy_overlap}, workload={workload_overlap}"
        )
    content = {
        "matrix_name": MATRIX_NAME,
        "experiments_root": str(experiments_root),
        "excluded_output_subtree": str(output),
        "selected_policy_seeds": list(POLICY_SEEDS),
        "selected_policy_panel_sha256": sha256_json(list(POLICY_SEEDS)),
        "selected_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "selected_workload_panel_sha256": sha256_json(list(TEST_WORKLOAD_SEEDS)),
        "scanned_csv_files": scanned_files,
        "scanned_csv_file_count": len(scanned_files),
        "historical_policy_seeds": sorted(observed_policy_seeds),
        "historical_workload_seed_ranges": _compress_integer_ranges(
            observed_workload_seeds
        ),
        "historical_unique_workload_seed_count": len(observed_workload_seeds),
        "selected_policy_seed_prior_overlap": [],
        "selected_workload_seed_prior_overlap": [],
        "selected_prior_overlap": [],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_no_selected_workload_seed_overlap",
        "created_at_utc": utc_now(),
        "content_sha256": sha256_json(content),
        "content": content,
    }
    atomic_write_json(manifest_path, manifest)
    return validate_historical_workload_seed_audit(output)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_full_budget_contract(
    config: Mapping[str, Any],
    runtime_scenarios: Sequence[str],
    runtime_policy_seeds: Sequence[int],
) -> dict[str, Any]:
    actual = dict(config)
    errors: list[str] = []
    if actual != EXPECTED_MAIN_FULL_CONFIG:
        expected_keys = set(EXPECTED_MAIN_FULL_CONFIG)
        actual_keys = set(actual)
        for key in sorted(expected_keys | actual_keys):
            expected = EXPECTED_MAIN_FULL_CONFIG.get(key, "<missing>")
            observed = actual.get(key, "<missing>")
            if observed != expected:
                errors.append(f"config[{key!r}]={observed!r}, expected {expected!r}")
    if tuple(runtime_scenarios) != SCENARIOS:
        errors.append(
            f"main scenarios={tuple(runtime_scenarios)!r}, expected {SCENARIOS!r}"
        )
    if tuple(runtime_policy_seeds) != POLICY_SEEDS:
        errors.append(
            "main policy seeds="
            f"{tuple(runtime_policy_seeds)!r}, expected {POLICY_SEEDS!r}"
        )
    if len(TEST_WORKLOAD_SEEDS) != int(actual.get("test_episodes", -1)):
        errors.append(
            "held-out workload count does not match main full test_episodes: "
            f"{len(TEST_WORKLOAD_SEEDS)} != {actual.get('test_episodes')!r}"
        )
    if errors:
        raise ConfigurationError(
            "refusing to run a non-canonical full-budget matrix:\n- "
            + "\n- ".join(errors)
        )
    # JSON round-tripping also removes aliases to mutable lists owned by the
    # imported module without changing the exact values passed to train_one.
    return json.loads(json.dumps(actual))


def build_matrix(
    scenarios: Sequence[str] = SCENARIOS,
    variants: Sequence[str] = VARIANTS,
    policy_seeds: Sequence[int] = POLICY_SEEDS,
) -> list[MatrixJob]:
    if len(set(scenarios)) != len(scenarios):
        raise ConfigurationError("scenario names must be unique")
    if len(set(variants)) != len(variants):
        raise ConfigurationError("variant names must be unique")
    if len(set(policy_seeds)) != len(policy_seeds):
        raise ConfigurationError("policy seeds must be unique")
    jobs: list[MatrixJob] = []
    for scenario in scenarios:
        for variant in variants:
            for policy_seed in policy_seeds:
                jobs.append(
                    MatrixJob(
                        index=len(jobs),
                        scenario=str(scenario),
                        variant=str(variant),
                        policy_seed=int(policy_seed),
                    )
                )
    return jobs


def select_jobs(
    jobs: Sequence[MatrixJob],
    *,
    scenarios: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    policy_seeds: Sequence[int] | None = None,
    indices: Sequence[int] | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> list[MatrixJob]:
    if (shard_index is None) != (shard_count is None):
        raise ConfigurationError("--shard-index and --shard-count must be set together")
    if shard_count is not None:
        if shard_count < 1:
            raise ConfigurationError("--shard-count must be at least 1")
        if shard_index is None or not 0 <= shard_index < shard_count:
            raise ConfigurationError(
                f"--shard-index must be in [0, {shard_count - 1}]"
            )

    known_indices = {job.index for job in jobs}
    selected_indices = None if indices is None else {int(index) for index in indices}
    if selected_indices is not None:
        unknown = sorted(selected_indices - known_indices)
        if unknown:
            raise ConfigurationError(f"unknown global job indices: {unknown}")

    scenario_filter = None if scenarios is None else set(scenarios)
    variant_filter = None if variants is None else set(variants)
    seed_filter = None if policy_seeds is None else {int(seed) for seed in policy_seeds}
    selected = []
    for job in jobs:
        if scenario_filter is not None and job.scenario not in scenario_filter:
            continue
        if variant_filter is not None and job.variant not in variant_filter:
            continue
        if seed_filter is not None and job.policy_seed not in seed_filter:
            continue
        if selected_indices is not None and job.index not in selected_indices:
            continue
        if shard_count is not None and job.index % shard_count != shard_index:
            continue
        selected.append(job)
    return selected


def matrix_spec(config: Mapping[str, Any], jobs: Sequence[MatrixJob]) -> dict[str, Any]:
    from variant_definitions import planned_contrasts_manifest

    body = {
        "schema_version": SCHEMA_VERSION,
        "matrix_name": MATRIX_NAME,
        "design": "component_removals_plus_adjacent_lifetime_ladder",
        "planned_contrasts": planned_contrasts_manifest(),
        "full_budget_steps": 50000,
        "config_source": "run_exp004_mappo.mode_config('full')",
        "config": dict(config),
        "scenarios": list(SCENARIOS),
        "variants": list(VARIANTS),
        "policy_seeds": list(POLICY_SEEDS),
        "policy_seed_derivation": {
            "namespace": POLICY_SEED_NAMESPACE,
            "digest": "sha256",
            "extraction": "first_4_bytes_big_endian_bitand_0x7fffffff",
            "indices": list(range(len(POLICY_SEEDS))),
        },
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "expected_training_jobs": len(jobs),
        "expected_evaluation_rows": len(jobs) * len(TEST_WORKLOAD_SEEDS),
        "confirmatory_statistics_requires_complete_matrix": True,
        "jobs": [job.as_dict() for job in jobs],
    }
    # Normalize tuples and other JSON sequences before comparing a resumed
    # invocation with the spec that was read back from disk.
    body = json.loads(canonical_json_bytes(body))
    body["spec_sha256"] = sha256_json(body)
    return body


def ensure_matrix_spec(output: Path, spec: Mapping[str, Any]) -> Path:
    path = output / "matrix_spec.json"
    if path.exists():
        existing = read_json(path)
        if existing != dict(spec):
            raise ConfigurationError(
                f"{path} describes a different matrix; use a new output directory"
            )
    else:
        atomic_write_json(path, dict(spec))
    return path


def status_path(output: Path, job: MatrixJob) -> Path:
    return output / STATUS_DIR / f"{job.slug}.json"


def lock_path(output: Path, job: MatrixJob) -> Path:
    return output / LOCK_DIR / f"{job.slug}.lock"


def evaluation_shard_path(output: Path, job: MatrixJob) -> Path:
    return output / EVALUATION_SHARD_DIR / f"{job.slug}.csv"


def _new_phase_record() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts_total": 0,
        "attempt_history": [],
    }


def new_job_state(job: MatrixJob, spec_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_sha256": spec_sha256,
        "job": job.as_dict(),
        "training": _new_phase_record(),
        "evaluation": _new_phase_record(),
        "updated_at_utc": utc_now(),
    }


def load_job_state(output: Path, job: MatrixJob, spec_sha256: str) -> dict[str, Any]:
    path = status_path(output, job)
    if not path.exists():
        return new_job_state(job, spec_sha256)
    state = read_json(path)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(f"unsupported status schema in {path}")
    if state.get("spec_sha256") != spec_sha256:
        raise ConfigurationError(f"matrix spec mismatch in {path}")
    recorded_job = state.get("job", {})
    if recorded_job.get("job_id") != job.job_id or recorded_job.get("index") != job.index:
        raise ConfigurationError(f"job identity mismatch in {path}")
    for phase in ("training", "evaluation"):
        state.setdefault(phase, _new_phase_record())
    return state


def write_job_state(output: Path, job: MatrixJob, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_write_json(status_path(output, job), state)


@contextmanager
def job_lock(
    output: Path,
    job: MatrixJob,
    stale_after_seconds: float,
    *,
    wait_seconds: float = 0.0,
):
    path = lock_path(output, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = canonical_json_bytes(
        {
            "token": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "job_id": job.job_id,
            "created_at_utc": utc_now(),
        }
    )

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= stale_after_seconds:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise JobLockBusy(
                        f"job is locked by another runner: {job.job_id}"
                    )
                time.sleep(min(0.5, remaining))
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            break

    try:
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                path.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass


def trainer_overrides(variant: str) -> list[str]:
    return list(NO_PPO_PROTECTION_OVERRIDES) if variant == "no_ppo_protection" else []


def phase_fingerprint(
    phase: str,
    job: MatrixJob,
    *,
    config: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    checkpoint_sha256: str | None = None,
    training_freeze_sha256: str | None = None,
) -> str:
    if phase == "training":
        relevant_names = TRAINING_CODE_COMPONENTS
    elif phase == "evaluation":
        relevant_names = EVALUATION_CODE_COMPONENTS
    else:
        raise ValueError(f"unknown phase {phase!r}")
    relevant_code = {
        name: digest
        for name, digest in code_sha256.items()
        if name in relevant_names
    }
    if not relevant_code:
        raise ValueError(f"no code hashes supplied for {phase} fingerprint")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "job": job.as_dict(),
        "config": dict(config),
        "code_sha256": dict(sorted(relevant_code.items())),
    }
    if phase == "training":
        payload["trainer_overrides"] = trainer_overrides(job.variant)
    elif phase == "evaluation":
        from variant_definitions import canonical_variant_name

        if not checkpoint_sha256:
            raise ValueError("evaluation fingerprint requires checkpoint_sha256")
        if not training_freeze_sha256:
            raise ValueError("evaluation fingerprint requires training freeze SHA256")
        payload["checkpoint_sha256"] = checkpoint_sha256
        payload["training_freeze_sha256"] = training_freeze_sha256
        payload["test_workload_seeds"] = list(TEST_WORKLOAD_SEEDS)
        payload["policy_name"] = f"mappo_{job.variant}"
        payload["environment_variant"] = canonical_variant_name(job.variant)
    return sha256_json(payload)


def validate_training_record(
    record: Mapping[str, Any], expected_fingerprint: str
) -> tuple[bool, str]:
    if record.get("status") != "completed":
        return False, f"status={record.get('status', 'missing')}"
    if record.get("fingerprint") != expected_fingerprint:
        return False, "phase fingerprint changed"
    checkpoint_value = record.get("checkpoint_path")
    expected_sha = record.get("checkpoint_sha256")
    if not checkpoint_value or not expected_sha:
        return False, "checkpoint path or SHA256 missing"
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        return False, f"checkpoint missing: {checkpoint}"
    try:
        actual_sha = sha256_file(checkpoint)
    except OSError as error:
        return False, f"checkpoint cannot be hashed: {error}"
    if actual_sha != expected_sha:
        return False, "checkpoint SHA256 mismatch"
    training_audit = record.get("training_audit")
    if not isinstance(training_audit, Mapping) or not training_audit.get(
        "budget_verified"
    ):
        return False, "50k training budget is not verified"
    try:
        requested_steps = int(
            training_audit.get("requested_environment_steps", -1)
        )
        actual_steps = int(training_audit.get("actual_environment_steps", -1))
    except (TypeError, ValueError):
        return False, "training step audit is malformed"
    if requested_steps != 50000:
        return False, "requested training budget is not 50000 steps"
    if not 50000 <= actual_steps <= 50119:
        return False, f"actual environment steps out of protocol range: {actual_steps}"
    try:
        selected_step = int(training_audit["selected_checkpoint_step"])
        earliest_best_step = int(training_audit["earliest_best_validation_step"])
        selected_score = [
            float(value) for value in training_audit["selected_validation_score"]
        ]
    except (KeyError, TypeError, ValueError):
        return False, "selected checkpoint step or validation score is malformed"
    if not 0 <= selected_step <= actual_steps:
        return False, "selected checkpoint step is outside the completed run"
    if earliest_best_step != selected_step:
        return False, "selected checkpoint is not the earliest best validation step"
    if len(selected_score) != 4 or not all(
        math.isfinite(value) for value in selected_score
    ):
        return False, "selected validation score is not a finite four-vector"
    selected_metrics = training_audit.get("selected_validation_metrics")
    required_metrics = {
        "delivery_ratio",
        "mean_reward",
        "drop_rate",
        "average_delay_slots",
    }
    if (
        not isinstance(selected_metrics, Mapping)
        or set(selected_metrics) != required_metrics
    ):
        return False, "selected validation metrics are missing or malformed"
    try:
        metric_values = {
            name: float(selected_metrics[name]) for name in required_metrics
        }
        selected_record_count = int(
            training_audit["selected_validation_records_at_step"]
        )
        validation_tie_count = int(training_audit["best_validation_tie_count"])
    except (KeyError, TypeError, ValueError):
        return False, "selected validation audit counts or values are malformed"
    if not all(math.isfinite(value) for value in metric_values.values()):
        return False, "selected validation metrics are non-finite"
    derived_score = (
        metric_values["delivery_ratio"],
        metric_values["mean_reward"],
        -metric_values["drop_rate"],
        -metric_values["average_delay_slots"],
    )
    if not all(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(derived_score, selected_score)
    ):
        return False, "selected validation metrics do not reproduce the score"
    if selected_record_count < 1 or validation_tie_count < 1:
        return False, "selected validation record or best-score tie count is invalid"
    artifacts = training_audit.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return False, "training artifact audit is missing"
    for name in REQUIRED_TRAINING_AUDIT_ARTIFACTS:
        metadata = artifacts.get(name)
        if not isinstance(metadata, Mapping):
            return False, f"training artifact {name!r} is not audited"
        artifact_path = Path(str(metadata.get("path", "")))
        expected_artifact_sha = metadata.get("sha256")
        if not artifact_path.is_file() or not expected_artifact_sha:
            return False, f"training artifact {name!r} is missing"
        try:
            actual_artifact_sha = sha256_file(artifact_path)
        except OSError as error:
            return False, f"training artifact {name!r} cannot be hashed: {error}"
        if actual_artifact_sha != expected_artifact_sha:
            return False, f"training artifact {name!r} SHA256 mismatch"
    return True, "valid"


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_evaluation_shard(
    path: Path,
    job: MatrixJob,
    workload_seeds: Sequence[int] = TEST_WORKLOAD_SEEDS,
) -> tuple[list[str], list[dict[str, str]]]:
    from mappo_evaluation import EpisodeMetrics

    if not path.is_file():
        raise FileNotFoundError(path)
    fieldnames, rows = _read_csv_rows(path)
    field_types = get_type_hints(EpisodeMetrics)
    expected_fieldnames = list(field_types)
    if fieldnames != expected_fieldnames:
        missing = sorted(set(expected_fieldnames) - set(fieldnames))
        unexpected = sorted(set(fieldnames) - set(expected_fieldnames))
        raise ValueError(
            "evaluation shard schema mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"order_matches={fieldnames == expected_fieldnames}"
        )
    if len(rows) != len(workload_seeds):
        raise ValueError(
            f"evaluation shard has {len(rows)} rows, expected {len(workload_seeds)}"
        )
    expected_workloads = {int(seed) for seed in workload_seeds}
    observed_workloads: list[int] = []
    expected_policy = f"mappo_{job.variant}"
    for row_index, row in enumerate(rows, start=2):
        for field, field_type in field_types.items():
            raw = row[field]
            try:
                if field_type is int:
                    int(raw)
                elif field_type is float:
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError("non-finite")
                elif field_type is not str:
                    raise TypeError(f"unsupported type {field_type!r}")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid {field_type!r} value for {field!r} at "
                    f"CSV row {row_index}: {raw!r}"
                ) from error
        if row["scenario"] != job.scenario:
            raise ValueError(f"scenario mismatch in {path}")
        if row["policy"] != expected_policy:
            raise ValueError(f"policy mismatch in {path}")
        if int(row["policy_seed"]) != job.policy_seed:
            raise ValueError(f"policy seed mismatch in {path}")
        observed_workloads.append(int(row["workload_seed"]))
    if len(set(observed_workloads)) != len(observed_workloads):
        raise ValueError(f"duplicate workload seed in {path}")
    if set(observed_workloads) != expected_workloads:
        raise ValueError(f"workload seed set mismatch in {path}")
    return fieldnames, rows


def validate_evaluation_record(
    record: Mapping[str, Any],
    expected_fingerprint: str,
    job: MatrixJob,
) -> tuple[bool, str]:
    if record.get("status") != "completed":
        return False, f"status={record.get('status', 'missing')}"
    if record.get("fingerprint") != expected_fingerprint:
        return False, "phase fingerprint changed"
    shard_value = record.get("shard_path")
    expected_sha = record.get("shard_sha256")
    if not shard_value or not expected_sha:
        return False, "evaluation shard path or SHA256 missing"
    shard = Path(shard_value)
    if not shard.is_file():
        return False, f"evaluation shard missing: {shard}"
    try:
        actual_sha = sha256_file(shard)
    except OSError as error:
        return False, f"evaluation shard cannot be hashed: {error}"
    if actual_sha != expected_sha:
        return False, "evaluation shard SHA256 mismatch"
    try:
        validate_evaluation_shard(shard, job)
    except (OSError, ValueError) as error:
        return False, str(error)
    return True, "valid"


def _begin_attempt(
    state: dict[str, Any],
    phase: str,
    fingerprint: str,
    invocation_id: str,
) -> tuple[dict[str, Any], float]:
    record = state[phase]
    if record.get("fingerprint") != fingerprint:
        record["attempts_for_fingerprint"] = 0
    record["attempts_total"] = int(record.get("attempts_total", 0)) + 1
    record["attempts_for_fingerprint"] = int(
        record.get("attempts_for_fingerprint", 0)
    ) + 1
    started_at = utc_now()
    monotonic_start = time.monotonic()
    record.update(
        {
            "status": "running",
            "fingerprint": fingerprint,
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "invocation_id": invocation_id,
            "worker_pid": os.getpid(),
            "worker_thread": threading.current_thread().name,
            "last_error": None,
        }
    )
    return record, monotonic_start


def _finish_attempt(
    record: dict[str, Any],
    *,
    status: str,
    monotonic_start: float,
    result: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    finished_at = utc_now()
    duration = max(0.0, time.monotonic() - monotonic_start)
    attempt = {
        "attempt_number": record["attempts_total"],
        "attempt_for_fingerprint": record["attempts_for_fingerprint"],
        "fingerprint": record.get("fingerprint"),
        "started_at_utc": record["started_at_utc"],
        "finished_at_utc": finished_at,
        "duration_seconds": duration,
        "status": status,
        "invocation_id": record.get("invocation_id"),
    }
    if error is not None:
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        attempt["error"] = error_payload
        record["last_error"] = error_payload
    record.setdefault("attempt_history", []).append(attempt)
    record["status"] = status
    record["finished_at_utc"] = finished_at
    record["duration_seconds"] = duration
    if result:
        record.update(dict(result))


def load_runtime_api() -> RuntimeAPI:
    # Heavy torch/environment imports are intentionally delayed so matrix and
    # persistence tests do not require CUDA or instantiate an environment.
    from mappo_evaluation import evaluate_policy, load_checkpoint_policy
    from run_exp004_mappo import (
        ALL_SCENARIOS,
        POLICY_SEEDS as RUNTIME_POLICY_SEEDS,
        mode_config,
        train_one,
    )

    return RuntimeAPI(
        train_one=train_one,
        mode_config=mode_config,
        runtime_scenarios=tuple(ALL_SCENARIOS),
        runtime_policy_seeds=tuple(RUNTIME_POLICY_SEEDS),
        load_checkpoint_policy=load_checkpoint_policy,
        evaluate_policy=evaluate_policy,
    )


def ensure_device(device: str, torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        import torch as torch_module

    requested = torch_module.device(device)
    metadata: dict[str, Any] = {
        "requested_device": str(requested),
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_device_count": int(torch_module.cuda.device_count()),
    }
    if requested.type != "cuda":
        metadata["selected_device"] = str(requested)
        return metadata
    if not metadata["cuda_available"]:
        raise RuntimeError(
            f"device={device!r} requires CUDA, but torch.cuda.is_available() is False"
        )
    index = requested.index
    if index is None:
        index = int(torch_module.cuda.current_device())
    if not 0 <= index < metadata["cuda_device_count"]:
        raise RuntimeError(
            f"device={device!r} selects CUDA index {index}, but only "
            f"{metadata['cuda_device_count']} device(s) are visible"
        )
    properties = torch_module.cuda.get_device_properties(index)
    metadata.update(
        {
            "selected_device": f"cuda:{index}",
            "device_name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    )
    return metadata


def _source_paths(project: Path, cleanmarl: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "training_entry": project / "run_exp004_mappo.py",
        "base_environment": project / "leo_marl_env.py",
        "environment": project / "leo_multiagent_env.py",
        "wrapper": project / "cleanmarl_leo_multiagent_wrapper.py",
        "variant_definitions": project / "variant_definitions.py",
        "design": project / "mappo_design.py",
        "evaluation": project / "mappo_evaluation.py",
        "ablation_analysis": project / "run_ablation_experiments.py",
        "statistics": project / "hierarchical_statistics.py",
        "protocol_document": project.parent / "docs" / "ABLATION_50K_PROTOCOL.md",
        "trainer": cleanmarl / "cleanmarl" / "mappo.py",
        "trainer_snapshot": project / "cleanmarl_mappo_leo.py",
    }


def collect_code_audit(project: Path, cleanmarl: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in _source_paths(project.resolve(), cleanmarl.resolve()).items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"required {name} source file is missing: {resolved}")
        result[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    if result["trainer"]["sha256"] != result["trainer_snapshot"]["sha256"]:
        raise ConfigurationError(
            "executed CleanMARL trainer differs from the repository snapshot: "
            f"{result['trainer']['path']}={result['trainer']['sha256']}, "
            f"{result['trainer_snapshot']['path']}="
            f"{result['trainer_snapshot']['sha256']}"
        )
    return result


def _git_output(repository: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_environment_audit(
    project: Path,
    device_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "torch", "tensorboard", "tyro"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    repository = project.resolve().parent
    git_status = _git_output(repository, "status", "--short")
    return {
        "captured_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "packages": package_versions,
        "device": dict(device_metadata),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "LEO_REWARD_OVERRIDES": os.environ.get("LEO_REWARD_OVERRIDES"),
        },
        "git": {
            "repository": str(repository),
            "commit": _git_output(repository, "rev-parse", "HEAD"),
            "branch": _git_output(repository, "branch", "--show-current"),
            "status_short": git_status,
            "dirty": bool(git_status),
        },
    }


def validate_formal_environment() -> None:
    reward_overrides = os.environ.get("LEO_REWARD_OVERRIDES", "").strip()
    if reward_overrides:
        raise ConfigurationError(
            "LEO_REWARD_OVERRIDES is set; refusing to contaminate the formal "
            "ablation matrix with reward-sensitivity overrides"
        )


def validate_variant_contract() -> None:
    from variant_definitions import resolve_variant
    from run_ablation_experiments import (
        TEST_WORKLOAD_SEEDS as ANALYSIS_TEST_WORKLOAD_SEEDS,
        VARIANTS as ANALYSIS_VARIANTS,
    )

    expected_environment_names = {
        variant: ("proposed" if variant == "no_ppo_protection" else variant)
        for variant in VARIANTS
    }
    for variant, expected in expected_environment_names.items():
        observed = resolve_variant(variant).name
        if observed != expected:
            raise ConfigurationError(
                f"variant {variant!r} resolves to {observed!r}, expected {expected!r}"
            )
    if tuple(ANALYSIS_VARIANTS) != VARIANTS:
        raise ConfigurationError(
            "runner and paired-analysis variant order differ: "
            f"{VARIANTS!r} != {tuple(ANALYSIS_VARIANTS)!r}"
        )
    if tuple(ANALYSIS_TEST_WORKLOAD_SEEDS) != TEST_WORKLOAD_SEEDS:
        raise ConfigurationError(
            "runner and paired-analysis held-out workload panels differ: "
            f"{TEST_WORKLOAD_SEEDS!r} != {tuple(ANALYSIS_TEST_WORKLOAD_SEEDS)!r}"
        )


def _code_sha_map(code_audit: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: str(metadata["sha256"]) for name, metadata in code_audit.items()}


def _training_args(
    output: Path,
    cleanmarl: Path,
    project: Path,
    device: str,
    *,
    skip_training: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        cleanmarl=cleanmarl,
        project=project,
        device=device,
        skip_training=skip_training,
    )


def _training_output_root(output: Path, fingerprint: str) -> Path:
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("training fingerprint must be a lowercase SHA-256 digest")
    return output / TRAINING_RUN_DIR / fingerprint


def _checkpoint_result(checkpoint: Path, *, discovered: bool = False) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"training returned a missing checkpoint: {checkpoint}")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_discovered_without_training": discovered,
    }


def _artifact_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def audit_training_artifacts(
    checkpoint: Path,
    job: MatrixJob,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that a selected checkpoint came from the formal 50k protocol."""

    from variant_definitions import canonical_variant_name

    checkpoint = checkpoint.resolve()
    run_directory = checkpoint.parent
    checkpoint_root = run_directory.parent
    paths = {
        "run_manifest": run_directory / "run_manifest.json",
        "run_config": run_directory / "run_config.json",
        "training_metrics": run_directory / "training_metrics.jsonl",
        "trainer_log": checkpoint_root / "trainer_stdout.log",
        "training_code_fingerprint": checkpoint_root / "code_fingerprint.json",
    }
    artifacts = {name: _artifact_metadata(path) for name, path in paths.items()}
    run_manifest = read_json(paths["run_manifest"])
    run_config = read_json(paths["run_config"])

    requested_steps = int(config["timesteps"])
    actual_steps = int(run_manifest.get("environment_steps", -1))
    max_overshoot = int(config["batch_size"]) * 30 - 1
    if requested_steps != 50000:
        raise ConfigurationError(f"formal training request is {requested_steps}, not 50000")
    if not requested_steps <= actual_steps <= requested_steps + max_overshoot:
        raise ValueError(
            f"run manifest reports {actual_steps} steps; expected {requested_steps} "
            f"through {requested_steps + max_overshoot}"
        )

    selected_value = run_manifest.get("validation_best_checkpoint")
    if not selected_value or Path(selected_value).resolve() != checkpoint:
        raise ValueError("selected checkpoint does not match run_manifest.json")
    expected_run_config = {
        "env_type": "leo_multi",
        "env_name": job.scenario,
        "batch_size": int(config["batch_size"]),
        "total_timesteps": requested_steps,
        "epochs": 3,
        "num_minibatches": 4,
        "eval_steps": int(config["eval_every_rollouts"]),
        "num_eval_ep": int(config["validation_episodes"]),
        "save_every_steps": int(config["save_every_steps"]),
        "train_seed_start": 9001,
        "train_seed_count": 200,
        "validation_seed_start": 10001,
        "seed": job.policy_seed,
        "leo_variant": canonical_variant_name(job.variant),
    }
    mismatches = {
        key: {"observed": run_config.get(key), "expected": expected}
        for key, expected in expected_run_config.items()
        if run_config.get(key) != expected
    }
    expected_protections = (
        {
            "clip_gradients": 0.0,
            "target_kl": 0.0,
            "normalize_advantage": False,
        }
        if job.variant == "no_ppo_protection"
        else {
            "clip_gradients": 1.0,
            "target_kl": 0.02,
            "normalize_advantage": True,
        }
    )
    for key, expected in expected_protections.items():
        if run_config.get(key) != expected:
            mismatches[key] = {
                "observed": run_config.get(key),
                "expected": expected,
            }
    if mismatches:
        raise ValueError(f"run_config.json violates the formal protocol: {mismatches}")

    import torch

    try:
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(f"cannot load selected checkpoint {checkpoint}") from error
    if not isinstance(checkpoint_payload, Mapping) or "step" not in checkpoint_payload:
        raise ValueError("selected checkpoint does not contain a training step")
    try:
        selected_checkpoint_step = int(checkpoint_payload["step"])
    except (TypeError, ValueError) as error:
        raise ValueError("selected checkpoint step is not an integer") from error
    if selected_checkpoint_step < 0 or selected_checkpoint_step > actual_steps:
        raise ValueError(
            f"selected checkpoint step {selected_checkpoint_step} is outside "
            f"the completed training range [0, {actual_steps}]"
        )

    manifest_score_value = run_manifest.get("best_validation_score")
    if not isinstance(manifest_score_value, list) or len(manifest_score_value) != 4:
        raise ValueError("run manifest best_validation_score must be a four-vector")
    try:
        manifest_score = tuple(float(value) for value in manifest_score_value)
    except (TypeError, ValueError) as error:
        raise ValueError("run manifest best_validation_score is not numeric") from error
    if not all(math.isfinite(value) for value in manifest_score):
        raise ValueError("run manifest best_validation_score is non-finite")

    metric_records = 0
    maximum_metric_step = -1
    validation_records: list[dict[str, Any]] = []
    with paths["training_metrics"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid training_metrics.jsonl line {line_number}"
                ) from error
            metric_records += 1
            if "environment_steps" in record:
                maximum_metric_step = max(
                    maximum_metric_step, int(record["environment_steps"])
                )
            if record.get("record_type") == "validation":
                required_validation_fields = (
                    "environment_steps",
                    "delivery_ratio",
                    "mean_reward",
                    "drop_rate",
                    "average_delay_slots",
                    "is_validation_best",
                )
                missing_fields = [
                    field
                    for field in required_validation_fields
                    if field not in record
                ]
                if missing_fields:
                    raise ValueError(
                        "validation metrics record lacks fields "
                        f"{missing_fields} at line {line_number}"
                    )
                raw_metrics = {
                    "delivery_ratio": float(record["delivery_ratio"]),
                    "mean_reward": float(record["mean_reward"]),
                    "drop_rate": float(record["drop_rate"]),
                    "average_delay_slots": float(record["average_delay_slots"]),
                }
                if not all(math.isfinite(value) for value in raw_metrics.values()):
                    raise ValueError(
                        f"non-finite validation metric at line {line_number}"
                    )
                validation_records.append(
                    {
                        "line_number": line_number,
                        "environment_steps": int(record["environment_steps"]),
                        "episodes": int(record.get("episodes", -1)),
                        "seed_start": int(record.get("seed_start", -1)),
                        "is_validation_best": record["is_validation_best"] is True,
                        "metrics": raw_metrics,
                        "score": (
                            raw_metrics["delivery_ratio"],
                            raw_metrics["mean_reward"],
                            -raw_metrics["drop_rate"],
                            -raw_metrics["average_delay_slots"],
                        ),
                    }
                )
    if not metric_records or maximum_metric_step != actual_steps:
        raise ValueError(
            "training metrics do not corroborate run-manifest steps: "
            f"records={metric_records}, max_step={maximum_metric_step}, "
            f"manifest_step={actual_steps}"
        )
    if not validation_records:
        raise ValueError("training metrics contain no validation records")

    selected_records = [
        record
        for record in validation_records
        if record["environment_steps"] == selected_checkpoint_step
    ]
    if not selected_records:
        raise ValueError(
            "training metrics contain no validation record at selected "
            f"checkpoint step {selected_checkpoint_step}"
        )
    if any(
        record["episodes"] != int(config["validation_episodes"])
        or record["seed_start"] != 10001
        for record in selected_records
    ):
        raise ValueError(
            "selected validation record uses the wrong episode count or seed panel"
        )

    def scores_close(left: Sequence[float], right: Sequence[float]) -> bool:
        return all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(left, right)
        )

    selected_score = selected_records[0]["score"]
    if any(
        not scores_close(record["score"], selected_score)
        for record in selected_records[1:]
    ):
        lines = [record["line_number"] for record in selected_records]
        raise ValueError(
            "conflicting validation records at selected checkpoint step "
            f"{selected_checkpoint_step}: lines={lines}"
        )
    if not any(record["is_validation_best"] for record in selected_records):
        raise ValueError(
            "selected checkpoint step has no is_validation_best=true record"
        )
    if not scores_close(selected_score, manifest_score):
        raise ValueError(
            "selected validation score does not match run manifest: "
            f"metrics={list(selected_score)}, manifest={list(manifest_score)}"
        )
    for record in validation_records:
        score = record["score"]
        if not scores_close(score, manifest_score) and score > manifest_score:
            raise ValueError(
                "run manifest best_validation_score is not the lexicographic "
                f"maximum; line {record['line_number']} has {list(score)}"
            )
    validation_tie_count = sum(
        scores_close(record["score"], manifest_score)
        for record in validation_records
    )
    earliest_best_validation_step = min(
        record["environment_steps"]
        for record in validation_records
        if scores_close(record["score"], manifest_score)
    )
    if selected_checkpoint_step != earliest_best_validation_step:
        raise ValueError(
            "selected checkpoint does not retain the earliest validation-score "
            f"tie: selected={selected_checkpoint_step}, "
            f"earliest={earliest_best_validation_step}"
        )

    final_checkpoint = Path(str(run_manifest.get("final_checkpoint", "")))
    final_checkpoint_metadata = _artifact_metadata(final_checkpoint)
    return {
        "budget_verified": True,
        "requested_environment_steps": requested_steps,
        "actual_environment_steps": actual_steps,
        "maximum_allowed_environment_steps": requested_steps + max_overshoot,
        "rollout_overshoot_steps": actual_steps - requested_steps,
        "optimizer_updates": int(run_manifest.get("optimizer_updates", -1)),
        "training_metric_records": metric_records,
        "maximum_training_metric_step": maximum_metric_step,
        "selected_checkpoint_step": selected_checkpoint_step,
        "earliest_best_validation_step": earliest_best_validation_step,
        "selected_validation_score": list(manifest_score),
        "selected_validation_metrics": dict(selected_records[0]["metrics"]),
        "selected_validation_records_at_step": len(selected_records),
        "best_validation_tie_count": validation_tie_count,
        "run_name": run_manifest.get("run_name"),
        "environment_variant": canonical_variant_name(job.variant),
        "trainer_protection_package_enabled": job.variant
        != "no_ppo_protection",
        "artifacts": artifacts,
        "final_checkpoint": final_checkpoint_metadata,
    }


def ensure_training_freeze(
    output: Path,
    jobs: Sequence[MatrixJob],
    *,
    spec_sha256: str,
    config: Mapping[str, Any],
    code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Atomically freeze every current checkpoint before test evaluation."""

    historical_seed_audit = validate_historical_workload_seed_audit(output)
    historical_seed_audit_path = output / HISTORICAL_WORKLOAD_SEED_AUDIT
    training_code = {
        name: digest
        for name, digest in code_sha256.items()
        if name in TRAINING_CODE_COMPONENTS
    }
    checkpoints = []
    for job in jobs:
        fingerprint = phase_fingerprint(
            "training", job, config=config, code_sha256=code_sha256
        )
        state = load_job_state(output, job, spec_sha256)
        valid, reason = validate_training_record(state["training"], fingerprint)
        if not valid:
            raise ConfigurationError(
                f"cannot freeze incomplete/stale training job {job.job_id}: {reason}"
            )
        record = state["training"]
        audit = record["training_audit"]
        checkpoints.append(
            {
                "index": job.index,
                "job_id": job.job_id,
                "scenario": job.scenario,
                "variant": job.variant,
                "policy_seed": job.policy_seed,
                "training_fingerprint": fingerprint,
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "selected_checkpoint_step": audit["selected_checkpoint_step"],
                "earliest_best_validation_step": audit[
                    "earliest_best_validation_step"
                ],
                "selected_validation_score": audit["selected_validation_score"],
                "actual_environment_steps": audit["actual_environment_steps"],
                "run_manifest_sha256": audit["artifacts"]["run_manifest"]["sha256"],
                "run_config_sha256": audit["artifacts"]["run_config"]["sha256"],
                "training_metrics_sha256": audit["artifacts"]["training_metrics"][
                    "sha256"
                ],
                "trainer_log_sha256": audit["artifacts"]["trainer_log"]["sha256"],
                "training_code_fingerprint_sha256": audit["artifacts"][
                    "training_code_fingerprint"
                ]["sha256"],
            }
        )
    content = {
        "matrix_name": MATRIX_NAME,
        "protocol_spec_sha256": spec_sha256,
        "expected_training_jobs": len(jobs),
        "training_code_sha256": dict(sorted(training_code.items())),
        "all_code_sha256": dict(sorted(code_sha256.items())),
        "protocol_document_sha256": code_sha256.get("protocol_document"),
        "historical_workload_seed_audit_sha256": sha256_file(
            historical_seed_audit_path
        ),
        "historical_workload_seed_audit_content_sha256": historical_seed_audit[
            "content_sha256"
        ],
        "checkpoints": checkpoints,
    }
    content_sha256 = sha256_json(content)
    path = output / TRAINING_FREEZE_MANIFEST
    if path.is_file():
        existing = read_json(path)
        if (
            existing.get("content") != content
            or existing.get("content_sha256") != content_sha256
            or sha256_json(existing.get("content")) != content_sha256
        ):
            raise ConfigurationError(
                "existing training freeze differs from current checkpoints; "
                "use a new protocol output directory"
            )
        return existing
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_test_evaluation",
        "frozen_at_utc": utc_now(),
        "content_sha256": content_sha256,
        "content": content,
    }
    atomic_write_json(path, manifest)
    return manifest


def validate_training_freeze_for_job(
    freeze: Mapping[str, Any],
    job: MatrixJob,
    *,
    spec_sha256: str,
    code_sha256: Mapping[str, str],
    historical_seed_audit_sha256: str,
    historical_seed_audit_content_sha256: str,
) -> dict[str, Any]:
    content = freeze.get("content")
    if not isinstance(content, Mapping):
        raise ConfigurationError("training freeze has no content object")
    if freeze.get("status") != "frozen_before_test_evaluation":
        raise ConfigurationError("training freeze is not active")
    if freeze.get("content_sha256") != sha256_json(content):
        raise ConfigurationError("training freeze content SHA256 mismatch")
    if content.get("protocol_spec_sha256") != spec_sha256:
        raise ConfigurationError("training freeze protocol SHA256 mismatch")
    expected_code = {
        name: digest
        for name, digest in code_sha256.items()
        if name in TRAINING_CODE_COMPONENTS
    }
    if content.get("training_code_sha256") != dict(sorted(expected_code.items())):
        raise ConfigurationError("training code changed after checkpoint freeze")
    if content.get("all_code_sha256") != dict(sorted(code_sha256.items())):
        raise ConfigurationError("code changed after checkpoint freeze")
    if content.get("protocol_document_sha256") != code_sha256.get(
        "protocol_document"
    ):
        raise ConfigurationError("protocol document changed after checkpoint freeze")
    if (
        content.get("historical_workload_seed_audit_sha256")
        != historical_seed_audit_sha256
        or content.get("historical_workload_seed_audit_content_sha256")
        != historical_seed_audit_content_sha256
    ):
        raise ConfigurationError(
            "historical seed audit differs from the checkpoint freeze"
        )
    checkpoints = content.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != content.get(
        "expected_training_jobs"
    ):
        raise ConfigurationError("training freeze checkpoint matrix is incomplete")
    matches = [entry for entry in checkpoints if entry.get("job_id") == job.job_id]
    if len(matches) != 1:
        raise ConfigurationError(
            f"training freeze has {len(matches)} entries for {job.job_id}"
        )
    return dict(matches[0])


def run_training_job(
    job: MatrixJob,
    *,
    output: Path,
    cleanmarl: Path,
    project: Path,
    device: str,
    config: Mapping[str, Any],
    spec_sha256: str,
    code_sha256: Mapping[str, str],
    invocation_id: str,
    max_retries: int,
    stale_lock_seconds: float,
    train_one: Callable[..., Path],
    artifact_auditor: Callable[[Path, MatrixJob, Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
) -> PhaseOutcome:
    fingerprint = phase_fingerprint(
        "training", job, config=config, code_sha256=code_sha256
    )
    try:
        with job_lock(output, job, stale_lock_seconds):
            state = load_job_state(output, job, spec_sha256)
            valid, reason = validate_training_record(state["training"], fingerprint)
            if valid:
                return PhaseOutcome(job.job_id, "training", "skipped", message="resume hit")
            state["training"]["stale_reason"] = reason

            maximum_attempts = max_retries + 1
            attempts_used = (
                int(state["training"].get("attempts_for_fingerprint", 0))
                if state["training"].get("fingerprint") == fingerprint
                else 0
            )
            remaining_attempts = maximum_attempts - attempts_used
            if remaining_attempts <= 0:
                return PhaseOutcome(
                    job.job_id,
                    "training",
                    "failed",
                    message=(
                        f"retry budget exhausted for fingerprint after "
                        f"{attempts_used}/{maximum_attempts} attempts"
                    ),
                )
            for local_attempt in range(remaining_attempts):
                record, started = _begin_attempt(
                    state, "training", fingerprint, invocation_id
                )
                write_job_state(output, job, state)
                try:
                    checkpoint = train_one(
                        _training_args(
                            _training_output_root(output, fingerprint),
                            cleanmarl,
                            project,
                            device,
                            skip_training=False,
                        ),
                        dict(config),
                        job.scenario,
                        job.policy_seed,
                        variant=job.variant,
                        trainer_overrides=trainer_overrides(job.variant),
                    )
                    result = _checkpoint_result(Path(checkpoint))
                    auditor = artifact_auditor or audit_training_artifacts
                    result["training_audit"] = dict(
                        auditor(Path(checkpoint), job, config)
                    )
                except BaseException as error:
                    _finish_attempt(
                        record,
                        status="failed",
                        monotonic_start=started,
                        error=error,
                    )
                    write_job_state(output, job, state)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    if local_attempt + 1 >= remaining_attempts:
                        return PhaseOutcome(
                            job.job_id,
                            "training",
                            "failed",
                            attempts=local_attempt + 1,
                            message=str(error),
                        )
                else:
                    _finish_attempt(
                        record,
                        status="completed",
                        monotonic_start=started,
                        result=result,
                    )
                    record.pop("stale_reason", None)
                    write_job_state(output, job, state)
                    return PhaseOutcome(
                        job.job_id,
                        "training",
                        "completed",
                        attempts=local_attempt + 1,
                    )
    except JobLockBusy as error:
        return PhaseOutcome(job.job_id, "training", "locked", message=str(error))
    raise AssertionError("unreachable training state")


def _discover_checkpoint_for_evaluation(
    job: MatrixJob,
    *,
    output: Path,
    cleanmarl: Path,
    project: Path,
    device: str,
    config: Mapping[str, Any],
    training_fingerprint: str,
    train_one: Callable[..., Path],
    artifact_auditor: Callable[[Path, MatrixJob, Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint = train_one(
        _training_args(
            _training_output_root(output, training_fingerprint),
            cleanmarl,
            project,
            device,
            skip_training=True,
        ),
        dict(config),
        job.scenario,
        job.policy_seed,
        variant=job.variant,
        trainer_overrides=trainer_overrides(job.variant),
    )
    result = _checkpoint_result(Path(checkpoint), discovered=True)
    result["training_audit"] = dict(artifact_auditor(Path(checkpoint), job, config))
    return result


def _rows_as_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if is_dataclass(row) and not isinstance(row, type):
            result.append(asdict(row))
        elif isinstance(row, Mapping):
            result.append(dict(row))
        else:
            raise TypeError(f"evaluation returned unsupported row type {type(row)!r}")
    return result


def run_evaluation_job(
    job: MatrixJob,
    *,
    output: Path,
    cleanmarl: Path,
    project: Path,
    device: str,
    config: Mapping[str, Any],
    spec_sha256: str,
    code_sha256: Mapping[str, str],
    invocation_id: str,
    max_retries: int,
    stale_lock_seconds: float,
    load_checkpoint_policy: Callable[..., tuple[Any, Mapping[str, Any]]],
    evaluate_policy: Callable[..., Iterable[Any]],
    training_freeze: Mapping[str, Any],
) -> PhaseOutcome:
    from variant_definitions import canonical_variant_name

    training_fingerprint = phase_fingerprint(
        "training", job, config=config, code_sha256=code_sha256
    )
    try:
        historical_seed_audit = validate_historical_workload_seed_audit(output)
        historical_seed_audit_sha256 = sha256_file(
            output / HISTORICAL_WORKLOAD_SEED_AUDIT
        )
        frozen_checkpoint = validate_training_freeze_for_job(
            training_freeze,
            job,
            spec_sha256=spec_sha256,
            code_sha256=code_sha256,
            historical_seed_audit_sha256=historical_seed_audit_sha256,
            historical_seed_audit_content_sha256=str(
                historical_seed_audit["content_sha256"]
            ),
        )
    except (OSError, ValueError, ConfigurationError) as error:
        return PhaseOutcome(
            job.job_id,
            "evaluation",
            "failed",
            message=f"test evaluation is locked: {error}",
        )
    try:
        with job_lock(output, job, stale_lock_seconds):
            state = load_job_state(output, job, spec_sha256)
            training_valid, training_reason = validate_training_record(
                state["training"], training_fingerprint
            )
            if not training_valid:
                return PhaseOutcome(
                    job.job_id,
                    "evaluation",
                    "failed",
                    message=(
                        "frozen training job is no longer current: "
                        f"{training_reason}"
                    ),
                )

            checkpoint = Path(state["training"]["checkpoint_path"])
            checkpoint_sha = str(state["training"]["checkpoint_sha256"])
            if (
                checkpoint.resolve()
                != Path(str(frozen_checkpoint["checkpoint_path"])).resolve()
                or checkpoint_sha != frozen_checkpoint["checkpoint_sha256"]
            ):
                return PhaseOutcome(
                    job.job_id,
                    "evaluation",
                    "failed",
                    message="current checkpoint differs from the training freeze",
                )
            fingerprint = phase_fingerprint(
                "evaluation",
                job,
                config=config,
                code_sha256=code_sha256,
                checkpoint_sha256=checkpoint_sha,
                training_freeze_sha256=str(
                    training_freeze["content_sha256"]
                ),
            )
            valid, reason = validate_evaluation_record(
                state["evaluation"], fingerprint, job
            )
            if valid:
                return PhaseOutcome(job.job_id, "evaluation", "skipped", message="resume hit")
            state["evaluation"]["stale_reason"] = reason

            maximum_attempts = max_retries + 1
            attempts_used = (
                int(state["evaluation"].get("attempts_for_fingerprint", 0))
                if state["evaluation"].get("fingerprint") == fingerprint
                else 0
            )
            remaining_attempts = maximum_attempts - attempts_used
            if remaining_attempts <= 0:
                return PhaseOutcome(
                    job.job_id,
                    "evaluation",
                    "failed",
                    message=(
                        f"retry budget exhausted for fingerprint after "
                        f"{attempts_used}/{maximum_attempts} attempts"
                    ),
                )
            for local_attempt in range(remaining_attempts):
                record, started = _begin_attempt(
                    state, "evaluation", fingerprint, invocation_id
                )
                write_job_state(output, job, state)
                try:
                    policy, checkpoint_metadata = load_checkpoint_policy(
                        checkpoint, device=device
                    )
                    rows = _rows_as_dicts(
                        evaluate_policy(
                            job.scenario,
                            f"mappo_{job.variant}",
                            policy,
                            job.policy_seed,
                            TEST_WORKLOAD_SEEDS,
                            variant=canonical_variant_name(job.variant),
                        )
                    )
                    shard = evaluation_shard_path(output, job).resolve()
                    atomic_write_csv(shard, rows)
                    fieldnames, validated_rows = validate_evaluation_shard(shard, job)
                    result = {
                        "shard_path": str(shard),
                        "shard_sha256": sha256_file(shard),
                        "row_count": len(validated_rows),
                        "fieldnames": fieldnames,
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_metadata_keys": sorted(
                            str(key) for key in checkpoint_metadata.keys()
                        ),
                    }
                except BaseException as error:
                    _finish_attempt(
                        record,
                        status="failed",
                        monotonic_start=started,
                        error=error,
                    )
                    write_job_state(output, job, state)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    if local_attempt + 1 >= remaining_attempts:
                        return PhaseOutcome(
                            job.job_id,
                            "evaluation",
                            "failed",
                            attempts=local_attempt + 1,
                            message=str(error),
                        )
                else:
                    _finish_attempt(
                        record,
                        status="completed",
                        monotonic_start=started,
                        result=result,
                    )
                    record.pop("stale_reason", None)
                    write_job_state(output, job, state)
                    return PhaseOutcome(
                        job.job_id,
                        "evaluation",
                        "completed",
                        attempts=local_attempt + 1,
                    )
    except JobLockBusy as error:
        return PhaseOutcome(job.job_id, "evaluation", "locked", message=str(error))
    raise AssertionError("unreachable evaluation state")


def _run_parallel(
    jobs: Sequence[MatrixJob],
    worker: Callable[[MatrixJob], PhaseOutcome],
    max_parallel: int,
) -> list[PhaseOutcome]:
    outcomes: list[PhaseOutcome] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_job = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                outcome = future.result()
            except BaseException:
                for pending in future_to_job:
                    pending.cancel()
                raise
            outcomes.append(outcome)
            detail = f": {outcome.message}" if outcome.message else ""
            print(
                f"[{outcome.phase}] {outcome.status} {job.index:03d} "
                f"{outcome.job_id}{detail}",
                flush=True,
            )
    return sorted(outcomes, key=lambda item: item.job_id)


def expected_phase_fingerprints(
    output: Path,
    jobs: Sequence[MatrixJob],
    spec_sha256: str,
    config: Mapping[str, Any],
    code_sha256: Mapping[str, str],
    training_freeze: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str | None]]:
    result: dict[str, dict[str, str | None]] = {}
    for job in jobs:
        training = phase_fingerprint(
            "training", job, config=config, code_sha256=code_sha256
        )
        evaluation: str | None = None
        try:
            state = load_job_state(output, job, spec_sha256)
            training_valid, _ = validate_training_record(state["training"], training)
            if training_valid and training_freeze is not None:
                evaluation = phase_fingerprint(
                    "evaluation",
                    job,
                    config=config,
                    code_sha256=code_sha256,
                    checkpoint_sha256=state["training"]["checkpoint_sha256"],
                    training_freeze_sha256=str(
                        training_freeze["content_sha256"]
                    ),
                )
        except (OSError, ValueError, ConfigurationError, json.JSONDecodeError):
            pass
        result[job.job_id] = {"training": training, "evaluation": evaluation}
    return result


def audit_matrix(
    output: Path,
    jobs: Sequence[MatrixJob],
    spec_sha256: str,
    fingerprints: Mapping[str, Mapping[str, str | None]],
    training_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    categories = (
        "completed",
        "missing",
        "pending",
        "running",
        "failed",
        "stale_or_invalid",
    )
    details: dict[str, dict[str, list[str]]] = {
        phase: {category: [] for category in categories}
        for phase in ("training", "evaluation")
    }
    evaluation_rows = 0
    actual_steps_by_job: dict[str, int] = {}
    malformed_status: dict[str, str] = {}

    for job in jobs:
        path = status_path(output, job)
        if not path.exists():
            details["training"]["missing"].append(job.job_id)
            details["evaluation"]["missing"].append(job.job_id)
            continue
        try:
            state = load_job_state(output, job, spec_sha256)
        except Exception as error:
            malformed_status[job.job_id] = str(error)
            details["training"]["stale_or_invalid"].append(job.job_id)
            details["evaluation"]["stale_or_invalid"].append(job.job_id)
            continue

        training_fp = fingerprints[job.job_id]["training"]
        training_valid, _ = validate_training_record(
            state["training"], str(training_fp)
        )
        if training_valid:
            details["training"]["completed"].append(job.job_id)
            actual_steps_by_job[job.job_id] = int(
                state["training"]["training_audit"]["actual_environment_steps"]
            )
        else:
            raw_status = state["training"].get("status", "pending")
            category = (
                raw_status
                if raw_status in {"pending", "running", "failed"}
                else "stale_or_invalid"
            )
            if raw_status == "completed":
                category = "stale_or_invalid"
            details["training"][category].append(job.job_id)

        evaluation_fp = fingerprints[job.job_id].get("evaluation")
        evaluation_valid = False
        if evaluation_fp is not None:
            evaluation_valid, _ = validate_evaluation_record(
                state["evaluation"], str(evaluation_fp), job
            )
        if evaluation_valid:
            details["evaluation"]["completed"].append(job.job_id)
            evaluation_rows += int(state["evaluation"].get("row_count", 0))
        else:
            raw_status = state["evaluation"].get("status", "pending")
            category = (
                raw_status
                if raw_status in {"pending", "running", "failed"}
                else "stale_or_invalid"
            )
            if raw_status == "completed":
                category = "stale_or_invalid"
            details["evaluation"][category].append(job.job_id)

    counts = {
        phase: {category: len(job_ids) for category, job_ids in phase_details.items()}
        for phase, phase_details in details.items()
    }
    expected_jobs = len(jobs)
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_name": MATRIX_NAME,
        "audited_at_utc": utc_now(),
        "spec_sha256": spec_sha256,
        "training_freeze_sha256": training_freeze_sha256,
        "expected_training_jobs": expected_jobs,
        "expected_evaluation_jobs": expected_jobs,
        "expected_evaluation_rows": expected_jobs * len(TEST_WORKLOAD_SEEDS),
        "validated_evaluation_rows": evaluation_rows,
        "validated_actual_environment_steps": {
            "by_job": actual_steps_by_job,
            "minimum": min(actual_steps_by_job.values())
            if actual_steps_by_job
            else None,
            "maximum": max(actual_steps_by_job.values())
            if actual_steps_by_job
            else None,
            "total": sum(actual_steps_by_job.values()),
        },
        "training_complete": counts["training"]["completed"] == expected_jobs,
        "evaluation_complete": counts["evaluation"]["completed"] == expected_jobs,
        "matrix_complete": (
            training_freeze_sha256 is not None
            and
            counts["training"]["completed"] == expected_jobs
            and counts["evaluation"]["completed"] == expected_jobs
            and evaluation_rows == expected_jobs * len(TEST_WORKLOAD_SEEDS)
        ),
        "counts": counts,
        "jobs": details,
        "malformed_status": malformed_status,
    }


def merge_evaluation_shards(
    output: Path,
    jobs: Sequence[MatrixJob],
    spec_sha256: str,
    fingerprints: Mapping[str, Mapping[str, str | None]],
    training_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    fieldnames: list[str] | None = None
    all_rows: list[dict[str, str]] = []
    included_jobs: list[str] = []
    for job in jobs:
        try:
            state = load_job_state(output, job, spec_sha256)
        except (OSError, ValueError, ConfigurationError, json.JSONDecodeError):
            continue
        expected = fingerprints[job.job_id].get("evaluation")
        if expected is None:
            continue
        valid, _ = validate_evaluation_record(state["evaluation"], expected, job)
        if not valid:
            continue
        current_fields, rows = validate_evaluation_shard(
            Path(state["evaluation"]["shard_path"]), job
        )
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise ValueError(
                f"evaluation schema mismatch for {job.job_id}: "
                f"{current_fields!r} != {fieldnames!r}"
            )
        all_rows.extend(rows)
        included_jobs.append(job.job_id)

    csv_path = output / "episode_metrics.csv"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "spec_sha256": spec_sha256,
        "training_freeze_sha256": training_freeze_sha256,
        "included_jobs": included_jobs,
        "included_job_count": len(included_jobs),
        "row_count": len(all_rows),
        "expected_job_count": len(jobs),
        "expected_row_count": len(jobs) * len(TEST_WORKLOAD_SEEDS),
        "complete": (
            len(included_jobs) == len(jobs)
            and len(all_rows) == len(jobs) * len(TEST_WORKLOAD_SEEDS)
        ),
    }
    if all_rows:
        atomic_write_csv(csv_path, all_rows, fieldnames)
        manifest.update(
            {
                "csv_path": str(csv_path.resolve()),
                "csv_sha256": sha256_file(csv_path),
                "fieldnames": fieldnames,
            }
        )
    else:
        manifest["csv_path"] = None
        manifest["csv_sha256"] = None
        manifest["fieldnames"] = None
    atomic_write_json(output / "episode_metrics_manifest.json", manifest)
    return manifest


def load_episode_metrics_csv(path: Path, row_type: type[Any]) -> list[Any]:
    """Load the merged CSV using the declared EpisodeMetrics field types."""

    field_types = get_type_hints(row_type)
    _, rows = _read_csv_rows(path)
    expected_fields = set(field_types)
    output = []
    for row_index, row in enumerate(rows, start=2):
        missing = expected_fields - set(row)
        if missing:
            raise ValueError(
                f"merged episode CSV row {row_index} lacks {sorted(missing)}"
            )
        converted: dict[str, Any] = {}
        for field, field_type in field_types.items():
            raw = row[field]
            if field_type is int:
                converted[field] = int(raw)
            elif field_type is float:
                converted[field] = float(raw)
            elif field_type is str:
                converted[field] = raw
            else:
                raise TypeError(
                    f"unsupported EpisodeMetrics field type {field_type!r} "
                    f"for {field!r}"
                )
        output.append(row_type(**converted))
    return output


def generate_final_statistics(
    output: Path,
    merge_manifest: Mapping[str, Any],
    *,
    code_audit: Mapping[str, Mapping[str, Any]],
    paired_effects_fn: Callable[..., list[dict[str, Any]]] | None = None,
    analysis_manifest_fn: Callable[[], dict[str, Any]] | None = None,
    episode_row_type: type[Any] | None = None,
    contrasts: Sequence[Any] | None = None,
    primary_metrics: Sequence[str] | None = None,
    scenarios: Sequence[str] = SCENARIOS,
) -> dict[str, Any]:
    """Generate predeclared paired effects only from a complete merged matrix."""

    from run_ablation_experiments import (
        PLANNED_CONTRASTS,
        PRIMARY_METRICS,
        paired_effects,
    )
    from hierarchical_statistics import statistical_analysis_manifest
    from mappo_evaluation import EpisodeMetrics
    from variant_definitions import planned_contrasts_manifest

    manifest_path = output / "statistical_analysis_manifest.json"
    if not merge_manifest.get("complete"):
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "not_generated_incomplete_matrix",
            "created_at_utc": utc_now(),
            "matrix_name": MATRIX_NAME,
            "merged_job_count": int(merge_manifest.get("included_job_count", 0)),
            "expected_job_count": int(merge_manifest.get("expected_job_count", 0)),
            "merged_row_count": int(merge_manifest.get("row_count", 0)),
            "expected_row_count": int(merge_manifest.get("expected_row_count", 0)),
            "planned_contrasts": planned_contrasts_manifest(),
            "protocol_spec_sha256": merge_manifest.get("spec_sha256"),
            "confirmatory_statistics_requires_complete_matrix": True,
        }
        atomic_write_json(manifest_path, manifest)
        return manifest

    csv_value = merge_manifest.get("csv_path")
    if not merge_manifest.get("training_freeze_sha256"):
        raise ValueError("complete merge has no pre-evaluation training freeze")
    if not csv_value:
        raise ValueError("complete merge manifest has no episode CSV path")
    episode_csv = Path(csv_value)
    if sha256_file(episode_csv) != merge_manifest.get("csv_sha256"):
        raise ValueError("merged episode CSV SHA256 changed before statistical analysis")

    analysis_code = {
        name: metadata
        for name, metadata in code_audit.items()
        if name in {"ablation_analysis", "statistics"}
    }
    if manifest_path.is_file():
        try:
            existing = read_json(manifest_path)
            existing_effects = Path(existing.get("paired_effects_csv", ""))
            reusable = (
                existing.get("status") == "completed"
                and existing.get("source_episode_csv_sha256")
                == merge_manifest.get("csv_sha256")
                and existing.get("training_freeze_sha256")
                == merge_manifest.get("training_freeze_sha256")
                and existing.get("analysis_code_audit") == analysis_code
                and existing_effects.is_file()
                and sha256_file(existing_effects)
                == existing.get("paired_effects_csv_sha256")
            )
            if reusable:
                return existing
        except (OSError, ValueError, TypeError):
            pass

    paired_effects_fn = paired_effects_fn or paired_effects
    analysis_manifest_fn = analysis_manifest_fn or statistical_analysis_manifest
    episode_row_type = episode_row_type or EpisodeMetrics
    contrasts = tuple(contrasts or PLANNED_CONTRASTS)
    primary_metrics = tuple(primary_metrics or PRIMARY_METRICS)
    rows = load_episode_metrics_csv(episode_csv, episode_row_type)
    effects = paired_effects_fn(rows, list(scenarios), contrasts)
    expected_effects = len(scenarios) * len(contrasts) * len(primary_metrics)
    if len(effects) != expected_effects:
        raise ValueError(
            f"paired analysis produced {len(effects)} rows, expected {expected_effects}"
        )

    def contrast_field(contrast: Any, name: str) -> Any:
        if isinstance(contrast, Mapping):
            return contrast[name]
        return getattr(contrast, name)

    scenario_positions = {name: index for index, name in enumerate(scenarios)}
    metric_positions = {name: index for index, name in enumerate(primary_metrics)}
    if len(scenario_positions) != len(scenarios):
        raise ValueError("statistical scenarios contain duplicates")
    if len(metric_positions) != len(primary_metrics):
        raise ValueError("primary metrics contain duplicates")
    contrast_contracts: dict[str, dict[str, Any]] = {}
    for contrast_index, contrast in enumerate(contrasts):
        name = str(contrast_field(contrast, "name"))
        if name in contrast_contracts:
            raise ValueError(f"planned contrast {name!r} is duplicated")
        contrast_contracts[name] = {
            "index": contrast_index,
            "contrast_family": str(contrast_field(contrast, "family")),
            "reference_variant": str(contrast_field(contrast, "reference")),
            "treatment_variant": str(contrast_field(contrast, "treatment")),
            "changed_flags": ";".join(contrast_field(contrast, "changed_flags")),
            "component_kind": str(contrast_field(contrast, "component_kind")),
        }

    expected_seed_ids = sorted(POLICY_SEEDS)
    expected_keys = {
        (scenario, contrast_name, metric)
        for scenario in scenarios
        for contrast_name in contrast_contracts
        for metric in primary_metrics
    }
    observed_keys: set[tuple[str, str, str]] = set()
    for index, effect in enumerate(effects):
        scenario = str(effect.get("scenario", ""))
        contrast_name = str(effect.get("contrast", ""))
        metric = str(effect.get("metric", ""))
        key = (scenario, contrast_name, metric)
        if key not in expected_keys:
            raise ValueError(f"paired effect row {index} has unexpected key {key!r}")
        if key in observed_keys:
            raise ValueError(f"paired effect row {index} duplicates key {key!r}")
        observed_keys.add(key)

        contrast_contract = contrast_contracts[contrast_name]
        for field in (
            "contrast_family",
            "reference_variant",
            "treatment_variant",
            "changed_flags",
            "component_kind",
        ):
            if str(effect.get(field, "")) != contrast_contract[field]:
                raise ValueError(
                    f"paired effect row {index} has incorrect {field}: "
                    f"{effect.get(field)!r} != {contrast_contract[field]!r}"
                )
        if str(effect.get("variant", "")) != contrast_contract["treatment_variant"]:
            raise ValueError(
                f"paired effect row {index} variant does not match treatment"
            )

        paired_policy_seeds = int(effect.get("paired_policy_seeds", -1))
        paired_workloads = int(effect.get("paired_workloads", -1))
        if paired_policy_seeds != len(POLICY_SEEDS):
            raise ValueError(
                f"paired effect row {index} has {paired_policy_seeds} policy seeds, "
                f"expected {len(POLICY_SEEDS)}"
            )
        if paired_workloads != len(TEST_WORKLOAD_SEEDS):
            raise ValueError(
                f"paired effect row {index} has {paired_workloads} workloads, "
                f"expected {len(TEST_WORKLOAD_SEEDS)}"
            )
        paired_cells = int(effect.get("paired_episode_cells", -1))
        expected_cells = len(POLICY_SEEDS) * len(TEST_WORKLOAD_SEEDS)
        if paired_cells != expected_cells:
            raise ValueError(
                f"paired effect row {index} has {paired_cells} episode pairs, "
                f"expected {expected_cells}"
            )
        if effect.get("policy_seed_pairing") != "matched_policy_seed":
            raise ValueError(
                f"paired effect row {index} is not matched by policy seed"
            )

        expected_rng_seed = (
            18000
            + 1000 * scenario_positions[scenario]
            + 100 * contrast_contract["index"]
            + metric_positions[metric]
        )
        if int(effect.get("bootstrap_rng_seed", -1)) != expected_rng_seed:
            raise ValueError(
                f"paired effect row {index} has incorrect bootstrap RNG seed"
            )
        try:
            seed_ids = json.loads(str(effect["policy_seed_ids_json"]))
            seed_differences = json.loads(
                str(effect["policy_seed_mean_differences_json"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"paired effect row {index} has malformed seed-level JSON"
            ) from error
        if (
            not isinstance(seed_ids, list)
            or any(type(seed) is not int for seed in seed_ids)
            or seed_ids != expected_seed_ids
        ):
            raise ValueError(
                f"paired effect row {index} does not contain the frozen "
                f"{len(POLICY_SEEDS)} policy seed IDs"
            )
        if not isinstance(seed_differences, list) or len(seed_differences) != len(
            POLICY_SEEDS
        ):
            raise ValueError(
                f"paired effect row {index} does not contain "
                f"{len(POLICY_SEEDS)} seed-level differences"
            )
        try:
            finite_differences = [float(value) for value in seed_differences]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"paired effect row {index} has malformed seed-level differences"
            ) from error
        if not all(math.isfinite(value) for value in finite_differences):
            raise ValueError(
                f"paired effect row {index} has non-finite seed-level differences"
            )
        reported_mean = float(effect.get("treatment_minus_reference", math.nan))
        if not math.isfinite(reported_mean) or not math.isclose(
            reported_mean,
            sum(finite_differences) / len(finite_differences),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"paired effect row {index} seed-level differences do not "
                "reproduce the reported mean"
            )

        permutations = int(effect.get("sign_flip_permutations", -1))
        tail_ties = int(effect.get("sign_flip_tail_tie_permutations", -1))
        zero_count = int(effect.get("sign_flip_zero_difference_count", -1))
        if permutations != 1 << len(POLICY_SEEDS):
            raise ValueError(
                f"paired effect row {index} has an incorrect exact-test universe"
            )
        if not 0 <= tail_ties <= permutations or not 0 <= zero_count <= len(
            POLICY_SEEDS
        ):
            raise ValueError(
                f"paired effect row {index} has invalid exact-test audit counts"
            )
        if effect.get("sign_flip_tail_ties_included") is not True:
            raise ValueError(
                f"paired effect row {index} does not include exact tail ties"
            )

        wilcoxon_counts = (
            int(effect.get("seed_level_wilcoxon_zero_count", -1)),
            int(effect.get("seed_level_wilcoxon_absolute_tie_group_count", -1)),
            int(effect.get("seed_level_wilcoxon_absolute_tied_value_count", -1)),
        )
        if (
            effect.get("seed_level_wilcoxon_requested_method") != "auto"
            or effect.get("seed_level_wilcoxon_zero_method") != "wilcox"
            or effect.get("seed_level_wilcoxon_method")
            != "scipy_wilcoxon_two_sided_on_workload_averaged_policy_seeds"
            or wilcoxon_counts[0] != zero_count
            or not 0 <= wilcoxon_counts[1] <= len(POLICY_SEEDS)
            or not 0 <= wilcoxon_counts[2] <= len(POLICY_SEEDS)
        ):
            raise ValueError(
                f"paired effect row {index} has invalid Wilcoxon audit metadata"
            )

        if "benjamini_hochberg_p" in effect:
            raise ValueError(
                f"paired effect row {index} uses deprecated multiplicity alias"
            )
        for p_value_field in (
            "raw_p_value",
            "confirmatory_holm_within_metric_p",
            "within_metric_bh_sensitivity_p",
            "global_holm_sensitivity_p",
            "global_bh_exploratory_p",
        ):
            p_value = float(effect.get(p_value_field, math.nan))
            if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                raise ValueError(
                    f"paired effect row {index} has invalid {p_value_field}"
                )
        if int(effect.get("within_metric_family_size", -1)) != (
            len(scenarios) * len(contrasts)
        ) or int(effect.get("global_family_size", -1)) != expected_effects:
            raise ValueError(
                f"paired effect row {index} has incorrect multiplicity family sizes"
            )
        expected_role = (
            "primary_confirmatory_family"
            if metric == "delivery_ratio"
            else "secondary_metric_family"
        )
        if effect.get("multiplicity_role") != expected_role:
            raise ValueError(
                f"paired effect row {index} has incorrect multiplicity role"
            )

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        raise ValueError(f"paired analysis is missing planned rows: {missing[:5]!r}")

    effects_path = output / "paired_ablation_effects.csv"
    atomic_write_csv(effects_path, effects)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "created_at_utc": utc_now(),
        "matrix_name": MATRIX_NAME,
        "design": "component_removals_plus_adjacent_lifetime_ladder",
        "protocol_spec_sha256": merge_manifest.get("spec_sha256"),
        "training_freeze_sha256": merge_manifest.get("training_freeze_sha256"),
        "confirmatory_statistics_requires_complete_matrix": True,
        "planned_contrasts": planned_contrasts_manifest(),
        "source_episode_csv": str(episode_csv.resolve()),
        "source_episode_csv_sha256": merge_manifest["csv_sha256"],
        "source_episode_rows": len(rows),
        "paired_effects_csv": str(effects_path.resolve()),
        "paired_effects_csv_sha256": sha256_file(effects_path),
        "paired_effect_rows": len(effects),
        "expected_paired_effect_rows": expected_effects,
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(TEST_WORKLOAD_SEEDS),
        "statistical_analysis": analysis_manifest_fn(),
        "analysis_code_audit": analysis_code,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _write_plan_csv(output: Path, jobs: Sequence[MatrixJob]) -> None:
    rows = [job.as_dict() for job in jobs]
    atomic_write_csv(
        output / "matrix_plan.csv",
        rows,
        ("index", "job_id", "scenario", "variant", "policy_seed"),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 9 x 5 x 12, 50k-step ablation matrix."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/ablation-50k-v2"),
    )
    parser.add_argument("--cleanmarl", type=Path, default=Path("F:/cleanmarl"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--lock-stale-hours", type=float, default=48.0)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=POLICY_SEEDS)
    parser.add_argument(
        "--job-index",
        "--index",
        dest="job_indices",
        action="append",
        type=int,
        help="Select a global zero-based matrix index; repeat as needed.",
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--merge-only", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if args.max_retries != 1:
        parser.error(
            "formal protocol fixes --max-retries=1 (two total attempts per fingerprint)"
        )
    if args.lock_stale_hours <= 0:
        parser.error("--lock-stale-hours must be positive")
    return args


def _phase_name(args: argparse.Namespace) -> str:
    if args.train_only:
        return "train-only"
    if args.evaluate_only:
        return "evaluate-only"
    if args.audit_only:
        return "audit-only"
    if args.merge_only:
        return "merge-only"
    return "train-and-evaluate"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = load_runtime_api()
    config = validate_full_budget_contract(
        runtime.mode_config("full"),
        runtime.runtime_scenarios,
        runtime.runtime_policy_seeds,
    )
    jobs = build_matrix()
    selected_jobs = select_jobs(
        jobs,
        scenarios=args.scenarios,
        variants=args.variants,
        policy_seeds=args.seeds,
        indices=args.job_indices,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not selected_jobs:
        raise ConfigurationError("job filters selected no matrix cells")
    if args.list_jobs:
        for job in selected_jobs:
            print(f"{job.index:03d}\t{job.job_id}")
        return 0

    args.output = args.output.resolve()
    args.project = args.project.resolve()
    args.cleanmarl = args.cleanmarl.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    validate_formal_environment()
    validate_variant_contract()
    stale_lock_seconds = args.lock_stale_hours * 3600.0
    historical_audit_job = MatrixJob(-2, "matrix", "historical_seed_audit", 0)
    with job_lock(
        args.output,
        historical_audit_job,
        stale_lock_seconds,
        wait_seconds=900.0,
    ):
        historical_seed_audit = ensure_historical_workload_seed_audit(
            args.output,
            args.project.parent / "experiments",
        )
    spec = matrix_spec(config, jobs)
    ensure_matrix_spec(args.output, spec)
    _write_plan_csv(args.output, jobs)

    device_metadata = ensure_device(args.device)
    code_audit = collect_code_audit(args.project, args.cleanmarl)
    code_sha256 = _code_sha_map(code_audit)
    invocation_id = uuid.uuid4().hex
    invocation_path = args.output / "invocations" / f"{invocation_id}.json"
    invocation = {
        "schema_version": SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "phase": _phase_name(args),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "selection": {
            "job_count": len(selected_jobs),
            "job_indices": [job.index for job in selected_jobs],
            "scenarios": args.scenarios,
            "variants": args.variants,
            "policy_seeds": args.seeds,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
        "execution": {
            "device": args.device,
            "max_parallel": args.max_parallel,
            "max_retries": args.max_retries,
            "lock_stale_hours": args.lock_stale_hours,
        },
        "environment_audit": collect_environment_audit(args.project, device_metadata),
        "code_audit": code_audit,
        "spec_sha256": spec["spec_sha256"],
        "historical_seed_audit": {
            "path": str(
                (args.output / HISTORICAL_WORKLOAD_SEED_AUDIT).resolve()
            ),
            "sha256": sha256_file(
                args.output / HISTORICAL_WORKLOAD_SEED_AUDIT
            ),
            "content_sha256": historical_seed_audit["content_sha256"],
        },
    }
    atomic_write_json(invocation_path, invocation)

    outcomes: list[PhaseOutcome] = []
    exit_code = 0
    finalization_job = MatrixJob(-1, "matrix", "finalization", 0)
    try:
        if not (args.evaluate_only or args.audit_only or args.merge_only):
            outcomes.extend(
                _run_parallel(
                    selected_jobs,
                    lambda job: run_training_job(
                        job,
                        output=args.output,
                        cleanmarl=args.cleanmarl,
                        project=args.project,
                        device=args.device,
                        config=config,
                        spec_sha256=spec["spec_sha256"],
                        code_sha256=code_sha256,
                        invocation_id=invocation_id,
                        max_retries=args.max_retries,
                        stale_lock_seconds=stale_lock_seconds,
                        train_one=runtime.train_one,
                    ),
                    args.max_parallel,
                )
            )

        training_freeze = None
        with job_lock(
            args.output,
            finalization_job,
            stale_lock_seconds,
            wait_seconds=900.0,
        ):
            pre_evaluation_fingerprints = expected_phase_fingerprints(
                args.output,
                jobs,
                spec["spec_sha256"],
                config,
                code_sha256,
            )
            pre_evaluation_audit = audit_matrix(
                args.output,
                jobs,
                spec["spec_sha256"],
                pre_evaluation_fingerprints,
            )
            atomic_write_json(
                args.output / "matrix_audit.json", pre_evaluation_audit
            )
            if pre_evaluation_audit["training_complete"]:
                training_freeze = ensure_training_freeze(
                    args.output,
                    jobs,
                    spec_sha256=spec["spec_sha256"],
                    config=config,
                    code_sha256=code_sha256,
                )

        if not (args.train_only or args.audit_only or args.merge_only):
            if training_freeze is None:
                raise ConfigurationError(
                    "test evaluation is locked until all 540 training jobs are "
                    "current and training_freeze_manifest.json is written; run "
                    "training shards with --train-only first"
                )
            outcomes.extend(
                _run_parallel(
                    selected_jobs,
                    lambda job: run_evaluation_job(
                        job,
                        output=args.output,
                        cleanmarl=args.cleanmarl,
                        project=args.project,
                        device=args.device,
                        config=config,
                        spec_sha256=spec["spec_sha256"],
                        code_sha256=code_sha256,
                        invocation_id=invocation_id,
                        max_retries=args.max_retries,
                        stale_lock_seconds=stale_lock_seconds,
                        load_checkpoint_policy=runtime.load_checkpoint_policy,
                        evaluate_policy=runtime.evaluate_policy,
                        training_freeze=training_freeze,
                    ),
                    args.max_parallel,
                )
            )

        with job_lock(
            args.output,
            finalization_job,
            stale_lock_seconds,
            wait_seconds=900.0,
        ):
            # Fingerprints are computed after acquiring the global finalization
            # lock so simultaneous shards cannot publish an older partial merge
            # after a newer complete one.
            fingerprints = expected_phase_fingerprints(
                args.output,
                jobs,
                spec["spec_sha256"],
                config,
                code_sha256,
                training_freeze=training_freeze,
            )
            merge_manifest = None
            if not args.train_only and not args.audit_only:
                merge_manifest = merge_evaluation_shards(
                    args.output,
                    jobs,
                    spec["spec_sha256"],
                    fingerprints,
                    training_freeze_sha256=(
                        training_freeze["content_sha256"]
                        if training_freeze is not None
                        else None
                    ),
                )
            audit = audit_matrix(
                args.output,
                jobs,
                spec["spec_sha256"],
                fingerprints,
                training_freeze_sha256=(
                    training_freeze["content_sha256"]
                    if training_freeze is not None
                    else None
                ),
            )
            atomic_write_json(args.output / "matrix_audit.json", audit)
            statistics_manifest = None
            if merge_manifest is not None:
                statistics_manifest = generate_final_statistics(
                    args.output,
                    merge_manifest,
                    code_audit=code_audit,
                )

        failures = [
            outcome for outcome in outcomes if outcome.status in {"failed", "locked"}
        ]
        if failures or (args.require_complete and not audit["matrix_complete"]):
            exit_code = 1
        published_artifacts = {
            "matrix_spec": _artifact_metadata(args.output / "matrix_spec.json"),
            "matrix_audit": _artifact_metadata(args.output / "matrix_audit.json"),
            "historical_workload_seed_audit": _artifact_metadata(
                args.output / HISTORICAL_WORKLOAD_SEED_AUDIT
            ),
        }
        freeze_path = args.output / TRAINING_FREEZE_MANIFEST
        if freeze_path.is_file():
            published_artifacts["training_freeze_manifest"] = _artifact_metadata(
                freeze_path
            )
        for name, path in (
            ("episode_metrics_manifest", args.output / "episode_metrics_manifest.json"),
            ("episode_metrics", args.output / "episode_metrics.csv"),
            (
                "statistical_analysis_manifest",
                args.output / "statistical_analysis_manifest.json",
            ),
        ):
            if path.is_file():
                published_artifacts[name] = _artifact_metadata(path)
        paired_effects_path = args.output / "paired_ablation_effects.csv"
        if (
            statistics_manifest is not None
            and statistics_manifest.get("status") == "completed"
            and paired_effects_path.is_file()
        ):
            published_artifacts["paired_ablation_effects"] = _artifact_metadata(
                paired_effects_path
            )
        invocation.update(
            {
                "status": "completed" if exit_code == 0 else "completed_with_errors",
                "finished_at_utc": utc_now(),
                "outcomes": [asdict(outcome) for outcome in outcomes],
                "outcome_counts": {
                    status: sum(outcome.status == status for outcome in outcomes)
                    for status in sorted({outcome.status for outcome in outcomes})
                },
                "matrix_audit_path": str((args.output / "matrix_audit.json").resolve()),
                "protocol_spec_sha256": spec["spec_sha256"],
                "training_freeze_content_sha256": (
                    training_freeze.get("content_sha256")
                    if training_freeze is not None
                    else None
                ),
                "matrix_complete": audit["matrix_complete"],
                "merge_manifest": merge_manifest,
                "statistics_manifest": statistics_manifest,
                "published_artifacts": published_artifacts,
            }
        )
        print(
            "matrix audit: "
            f"training {audit['counts']['training']['completed']}/{len(jobs)}, "
            f"evaluation {audit['counts']['evaluation']['completed']}/{len(jobs)}, "
            f"rows {audit['validated_evaluation_rows']}/"
            f"{audit['expected_evaluation_rows']}",
            flush=True,
        )
    except BaseException as error:
        invocation.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "finished_at_utc": utc_now(),
                "fatal_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    ),
                },
            }
        )
        atomic_write_json(invocation_path, invocation)
        raise
    atomic_write_json(invocation_path, invocation)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
