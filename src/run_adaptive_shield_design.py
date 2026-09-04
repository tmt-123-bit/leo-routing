"""Select one adaptive post-hoc shield on the exposed v7 design panel."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ablation_matrix_runner import atomic_write_csv, atomic_write_json, sha256_file
from adaptive_hysteresis_policy import load_adaptive_hysteresis_policy
from leo_multiagent_env import (
    BASE_CANDIDATE_FEATURE_NAMES,
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
    ROUTE_SWITCH_FEATURE_INDEX,
    candidate_feature_schema,
)
from mappo_evaluation import EpisodeMetrics, evaluate_policy
import run_hysteresis_screen as source_runner
import run_proposed_shield_preflight as v7


SCREEN_NAME = "ADAPTIVE-SHIELD-DESIGN-v8"
SCHEMA_VERSION = 1
SCENARIOS = v7.SCENARIOS
POLICY_SEEDS = v7.POLICY_SEEDS
DESIGN_WORKLOAD_SEEDS = v7.VALIDATION_WORKLOAD_SEEDS
STAY_BONUSES = (0.60, 0.80, 1.00)
URGENCY_RELIEFS = (0.50, 0.75, 1.00)
CLASS_2_RELIEFS = (0.75, 1.00)
DESIGN_GRID = tuple(
    {
        "candidate_index": index,
        "stay_bonus": float(stay_bonus),
        "urgency_relief": float(urgency_relief),
        "class_2_relief": float(class_2_relief),
    }
    for index, (stay_bonus, urgency_relief, class_2_relief) in enumerate(
        itertools.product(STAY_BONUSES, URGENCY_RELIEFS, CLASS_2_RELIEFS)
    )
)
CELL_GATES = dict(v7.CELL_GATES)
RAW_SWITCH_HEADROOM_BUFFER = 2.0
AVOIDABLE_MICRO_IMPROVEMENT_BUFFER = 0.005
SELECTION_RULE_ID = (
    "max_worst_delivery_then_class2_then_raw_switch_then_avoidable_"
    "then_min_beta_max_reliefs_index_v1"
)
EXPECTED_FEATURE_SCHEMA = candidate_feature_schema()
EXPECTED_FEATURE_NAMES = tuple(BASE_CANDIDATE_FEATURE_NAMES)
REFERENCE_POLICIES = {
    "proposed": v7.POLICY_NAMES["proposed"],
    "raw_context": v7.POLICY_NAMES["raw_context"],
}
BASELINE_FINGERPRINT_FILES = {
    "evaluation": "mappo_evaluation.py",
    "design": "mappo_design.py",
    "wrapper": "cleanmarl_leo_multiagent_wrapper.py",
    "multiagent_environment": "leo_multiagent_env.py",
    "base_environment": "leo_marl_env.py",
    "variants": "variant_definitions.py",
    "statistics": "hierarchical_statistics.py",
    "artifact_helpers": "ablation_matrix_runner.py",
}
HISTORICAL_SOURCE_FINGERPRINT_FIELDS = frozenset(
    {
        "evaluation",
        "design",
        "wrapper",
        "multiagent_environment",
        "variants",
    }
)
AUDIT_RUNTIME_FINGERPRINT_FILES = {
    "runner": "run_proposed_shield_preflight.py",
    "source_auditor": "run_hysteresis_screen.py",
}
SOURCE_RUNTIME_EQUIVALENCE_RELATIVE_PATH = (
    Path("experiments")
    / "source-runtime-equivalence-v8"
    / "source_runtime_equivalence.json"
)
SOURCE_RUNTIME_ARCHIVE_RELATIVE_PATH = (
    Path("experiments")
    / "archive"
    / "source-snapshots"
    / "actor-score-hysteresis-c3d05253666b.zip"
)
EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_SHA256 = (
    "494fb276939e8e74bc1af38e9425f32347ad43c0719e6e3738877fd5130802a9"
)
EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_FILE_SHA256 = (
    "ffe1d0bdb53714eb10c2ffca7feb381f3ac62b321ebfec8635d551972118a955"
)
EXPECTED_SOURCE_RUNTIME_ARCHIVE_SHA256 = (
    "4c5a418258979960d77d2be6228a5ad69823ae39c196a101b5192afceedacfd7"
)
EXPECTED_V8_SPEC_SHA256 = (
    "04351803c9b71222d9df5ce52cdb38392de892f3f1b08cda56987677326df567"
)
EXPECTED_V8_SPEC_FILE_SHA256 = (
    "d17dde3147ba83ca8bc2f15f43c43ee2ef9a827b4c734c4df622278ff5bc881b"
)
EXPECTED_V8_SELECTION_SHA256 = (
    "72dc71a1a178262c993a629ebb7d67df996dc718d4dd338f28b9c4be4227f905"
)
EXPECTED_V8_SELECTION_FILE_SHA256 = (
    "93e3cd0432c6ba8cc940aa6f9c52d5e102e36f1bb327f154447fa8de348be8b9"
)
HISTORICAL_V8_RUNNER_FINGERPRINTS = {
    "runner": "440068283d9b81b6ef7ff142613c61472d2f8cc9b3422dcf2b345338780c6872",
}


def _token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def candidate_id(candidate: Mapping[str, Any]) -> str:
    return "_".join(
        (
            f"b{_token(float(candidate['stay_bonus']))}",
            f"u{_token(float(candidate['urgency_relief']))}",
            f"c{_token(float(candidate['class_2_relief']))}",
        )
    )


def candidate_policy_name(candidate: Mapping[str, Any]) -> str:
    return f"mappo_proposed_adaptive_shield_{candidate_id(candidate)}_design_v8"


@dataclass(frozen=True)
class DesignJob:
    index: int
    scenario: str
    policy_seed: int
    candidate_index: int
    stay_bonus: float
    urgency_relief: float
    class_2_relief: float

    @property
    def phase(self) -> str:
        return "design"

    @property
    def source_variant(self) -> str:
        return "proposed"

    @property
    def source_job_id(self) -> str:
        return f"{self.scenario}/proposed/seed_{self.policy_seed}"

    @property
    def policy_name(self) -> str:
        return candidate_policy_name(self.as_parameters())

    @property
    def job_id(self) -> str:
        return (
            f"design/{self.scenario}/{candidate_id(self.as_parameters())}/"
            f"seed_{self.policy_seed}"
        )

    @property
    def slug(self) -> str:
        return "__".join(
            (
                self.scenario,
                candidate_id(self.as_parameters()),
                f"seed_{self.policy_seed}",
            )
        )

    def as_parameters(self) -> dict[str, float]:
        return {
            "stay_bonus": self.stay_bonus,
            "urgency_relief": self.urgency_relief,
            "class_2_relief": self.class_2_relief,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "phase": self.phase,
            "source_variant": self.source_variant,
            "source_job_id": self.source_job_id,
            "policy_name": self.policy_name,
            "job_id": self.job_id,
        }


def build_design_jobs() -> list[DesignJob]:
    jobs: list[DesignJob] = []
    for candidate in DESIGN_GRID:
        for scenario in SCENARIOS:
            for policy_seed in POLICY_SEEDS:
                jobs.append(
                    DesignJob(
                        index=len(jobs),
                        scenario=scenario,
                        policy_seed=policy_seed,
                        candidate_index=int(candidate["candidate_index"]),
                        stay_bonus=float(candidate["stay_bonus"]),
                        urgency_relief=float(candidate["urgency_relief"]),
                        class_2_relief=float(candidate["class_2_relief"]),
                    )
                )
    return jobs


def _self_hash(record: Mapping[str, Any], field: str) -> str:
    body = dict(record)
    observed = body.pop(field, None)
    expected = source_runner.sha256_json(body)
    if observed != expected:
        raise ValueError(f"invalid {field}: {observed!r} != {expected}")
    return expected


def _runtime_code_fingerprint(project: Path, protocol: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "protocol": protocol,
        "adaptive_policy": project / "adaptive_hysteresis_policy.py",
        "raw_hysteresis_policy": project / "hysteresis_policy.py",
        **{
            name: project / filename
            for name, filename in BASELINE_FINGERPRINT_FILES.items()
        },
        "source_auditor": project / "run_hysteresis_screen.py",
        "v7_runner": project / "run_proposed_shield_preflight.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"design fingerprint files missing: {missing}")
    modules = {
        "adaptive_hysteresis_policy": "adaptive_hysteresis_policy.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "run_hysteresis_screen": "run_hysteresis_screen.py",
        "run_proposed_shield_preflight": "run_proposed_shield_preflight.py",
    }
    for module_name, filename in modules.items():
        actual = Path(inspect.getfile(importlib.import_module(module_name))).resolve()
        expected = (project / filename).resolve()
        if actual != expected:
            raise RuntimeError(f"runtime module path mismatch: {module_name}")
    return {name: sha256_file(path) for name, path in files.items()}


def _audit_historical_v7_source_fingerprint(
    project: Path,
    source: Mapping[str, Any],
    name: str,
    filename: str,
    expected_sha256: Any,
) -> dict[str, Any]:
    if name not in HISTORICAL_SOURCE_FINGERPRINT_FIELDS:
        raise ValueError(f"unsupported v7 historical source field: {name}")
    if not isinstance(expected_sha256, str):
        raise ValueError(f"v7 spec lacks its {name} fingerprint")
    project_root = project.resolve().parent
    evidence_path = (
        project_root / SOURCE_RUNTIME_EQUIVALENCE_RELATIVE_PATH
    ).resolve()
    if (
        not evidence_path.is_file()
        or sha256_file(evidence_path)
        != EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_FILE_SHA256
    ):
        raise ValueError("v7 source equivalence evidence changed")
    evidence = source_runner._load_json(evidence_path)
    if (
        _self_hash(evidence, "source_runtime_equivalence_sha256")
        != EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_SHA256
    ):
        raise ValueError("unexpected v7 source equivalence evidence")
    if evidence.get("new_evaluation_panel_accessed") is not False:
        raise ValueError("v7 source equivalence evidence opened a new panel")
    if evidence.get("training_jobs") != 0 or evidence.get("evaluation_jobs") != 0:
        raise ValueError("v7 source equivalence evidence ran jobs")
    decision = evidence.get("decision")
    required_decisions = (
        "archive_contract_pass",
        "source_spec_contract_pass",
        "ast_contract_pass",
        "dynamic_contract_pass",
        "snapshot_current_exact_match",
        "all_pass",
    )
    if not isinstance(decision, Mapping) or any(
        decision.get(name) is not True for name in required_decisions
    ):
        raise ValueError("v7 source equivalence evidence did not pass")
    source_spec = evidence.get("frozen_specs", {}).get("source_screen")
    if (
        not isinstance(source_spec, Mapping)
        or source_spec.get("spec_sha256") != source.get("spec_sha256")
    ):
        raise ValueError("v7 source equivalence source mismatch")
    historical = evidence.get("current_runtime_files", {}).get(filename)
    expected_path = (project / filename).resolve()
    historical_length = (
        historical.get("length") if isinstance(historical, Mapping) else None
    )
    if (
        not isinstance(historical, Mapping)
        or Path(str(historical.get("path", ""))).resolve() != expected_path
        or Path(str(historical.get("path", ""))).name != filename
        or isinstance(historical_length, bool)
        or not isinstance(historical_length, int)
        or historical_length <= 0
        or historical.get("sha256") != expected_sha256
    ):
        raise ValueError(f"v7 historical {name} fingerprint mismatch")
    archive = evidence.get("archive")
    archive_path = (
        project_root / SOURCE_RUNTIME_ARCHIVE_RELATIVE_PATH
    ).resolve()
    if (
        not isinstance(archive, Mapping)
        or Path(str(archive.get("path", ""))).resolve() != archive_path
        or archive.get("sha256") != EXPECTED_SOURCE_RUNTIME_ARCHIVE_SHA256
        or not archive_path.is_file()
        or sha256_file(archive_path) != EXPECTED_SOURCE_RUNTIME_ARCHIVE_SHA256
    ):
        raise ValueError("v7 source snapshot changed")
    return {
        "equivalence_path": str(evidence_path),
        "equivalence_sha256": EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_SHA256,
        "archive_path": str(archive_path),
        "archive_sha256": EXPECTED_SOURCE_RUNTIME_ARCHIVE_SHA256,
        "field": name,
        "filename": filename,
        "historical_length": historical_length,
        "historical_sha256": expected_sha256,
    }


def _audit_historical_runtime_fingerprint(
    project: Path,
    source: Mapping[str, Any],
    historical: Any,
    current: Mapping[str, str],
    historical_runner_fingerprints: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(historical, Mapping) or set(historical) != set(current):
        raise ValueError("historical runtime fingerprint field set changed")
    provenance: dict[str, dict[str, Any]] = {}
    for name, observed in current.items():
        expected = historical.get(name)
        if observed == expected:
            continue
        if name in HISTORICAL_SOURCE_FINGERPRINT_FIELDS:
            filename = BASELINE_FINGERPRINT_FILES[name]
            provenance[name] = _audit_historical_v7_source_fingerprint(
                project,
                source,
                name,
                filename,
                expected,
            )
            continue
        pinned_runner_sha256 = historical_runner_fingerprints.get(name)
        if pinned_runner_sha256 is not None and expected == pinned_runner_sha256:
            provenance[name] = {
                "field": name,
                "historical_sha256": pinned_runner_sha256,
                "provenance": "exact_frozen_design_spec_fingerprint",
            }
            continue
        raise ValueError(f"historical runtime code drift is not covered: {name}")
    return provenance


def audit_v7_baseline(
    v7_output: Path,
    source: Mapping[str, Any],
    project: Path,
) -> dict[str, Any]:
    spec_path = v7_output / "proposed_shield_preflight_spec.json"
    decision_path = v7_output / "validation_decision.json"
    episode_path = v7_output / "validation_episode_metrics.csv"
    spec = source_runner._load_json(spec_path)
    decision = source_runner._load_json(decision_path)
    _self_hash(spec, "spec_sha256")
    _self_hash(decision, "validation_freeze_sha256")
    if spec.get("screen_name") != v7.SCREEN_NAME:
        raise ValueError("v7 spec screen mismatch")
    if decision.get("screen_name") != v7.SCREEN_NAME:
        raise ValueError("v7 decision screen mismatch")
    if decision.get("spec_sha256") != spec["spec_sha256"]:
        raise ValueError("v7 decision spec mismatch")
    if decision.get("decision") != "validation_fail":
        raise ValueError("design requires the recorded failed v7 validation")
    if decision.get("test_allowed") is not False:
        raise ValueError("v7 decision unexpectedly allows its test panel")
    if decision.get("workload_seeds") != list(DESIGN_WORKLOAD_SEEDS):
        raise ValueError("v7 exposed workload panel mismatch")
    if (
        decision.get("source_training_freeze_sha256")
        != source["training_freeze_sha256"]
    ):
        raise ValueError("v7 source freeze mismatch")
    expected_fingerprint = spec.get("code_fingerprint")
    if not isinstance(expected_fingerprint, Mapping):
        raise ValueError("v7 spec lacks its code fingerprint")
    for name, filename in AUDIT_RUNTIME_FINGERPRINT_FILES.items():
        if sha256_file(project / filename) != expected_fingerprint.get(name):
            raise ValueError(f"v7 baseline audit runtime changed: {name}")
    historical_source_provenance: dict[str, dict[str, Any]] = {}
    for name, filename in BASELINE_FINGERPRINT_FILES.items():
        observed = sha256_file(project / filename)
        expected = expected_fingerprint.get(name)
        if observed == expected:
            continue
        if name in HISTORICAL_SOURCE_FINGERPRINT_FIELDS:
            historical_source_provenance[name] = (
                _audit_historical_v7_source_fingerprint(
                    project,
                    source,
                    name,
                    filename,
                    expected,
                )
            )
            continue
        raise ValueError(f"v7 baseline reuse blocked by code drift: {name}")
    artifact = decision.get("artifacts", {}).get("episode_metrics")
    if not isinstance(artifact, Mapping):
        raise ValueError("v7 decision lacks the episode artifact")
    if Path(str(artifact.get("path", ""))).resolve() != episode_path.resolve():
        raise ValueError("v7 episode artifact path mismatch")
    if artifact.get("sha256") != sha256_file(episode_path):
        raise ValueError("v7 episode artifact hash mismatch")
    rows = source_runner._episode_rows_from_csv(episode_path)
    v7.validate_switch_rows(rows)
    jobs = v7.build_jobs("validation")
    shard_rows = v7._load_phase_shards(
        v7_output,
        jobs,
        spec["spec_sha256"],
        source["checkpoints"],
        DESIGN_WORKLOAD_SEEDS,
        None,
    )
    if rows != shard_rows:
        raise ValueError("v7 combined episode rows do not replay from shards")
    if decision.get("evaluation_shards") != v7._shard_index(v7_output, jobs):
        raise ValueError("v7 evaluation shard index does not replay")
    replayed_gates = v7.compute_cell_gates(rows, DESIGN_WORKLOAD_SEEDS)
    if decision.get("cell_gates") != replayed_gates:
        raise ValueError("v7 gate decision does not replay")
    artifact_paths = v7._phase_artifact_paths(v7_output, "validation")
    for name in ("episode_metrics", "aggregate_metrics", "cell_gates"):
        bound = decision.get("artifacts", {}).get(name)
        expected_path = artifact_paths[name]
        if not isinstance(bound, Mapping):
            raise ValueError(f"v7 decision lacks the {name} artifact")
        if Path(str(bound.get("path", ""))).resolve() != expected_path.resolve():
            raise ValueError(f"v7 {name} artifact path mismatch")
        if bound.get("sha256") != sha256_file(expected_path):
            raise ValueError(f"v7 {name} artifact hash mismatch")
    references = [row for row in rows if row.policy in set(REFERENCE_POLICIES.values())]
    expected_keys = {
        (scenario, policy, policy_seed, workload_seed)
        for scenario in SCENARIOS
        for policy in REFERENCE_POLICIES.values()
        for policy_seed in POLICY_SEEDS
        for workload_seed in DESIGN_WORKLOAD_SEEDS
    }
    observed_keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in references
    }
    if len(references) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError("v7 proposed/raw reference grid is incomplete or duplicated")
    return {
        "spec_sha256": spec["spec_sha256"],
        "validation_freeze_sha256": decision["validation_freeze_sha256"],
        "episode_metrics_path": str(episode_path.resolve()),
        "episode_metrics_sha256": artifact["sha256"],
        "reference_row_count": len(references),
        "references": references,
        "historical_source_provenance": historical_source_provenance,
    }


def build_spec(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = args.project.parent / "docs" / "ADAPTIVE_SHIELD_DESIGN_V8.md"
    fingerprint = _runtime_code_fingerprint(args.project, protocol)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exposed_panel_engineering_design",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "training_jobs": 0,
        "design_workload_seeds": list(DESIGN_WORKLOAD_SEEDS),
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "candidate_grid": [dict(candidate) for candidate in DESIGN_GRID],
        "candidate_count": len(DESIGN_GRID),
        "cell_gates": dict(CELL_GATES),
        "design_buffers": {
            "raw_context_switch_headroom_min": RAW_SWITCH_HEADROOM_BUFFER,
            "avoidable_micro_improvement_min": AVOIDABLE_MICRO_IMPROVEMENT_BUFFER,
            "delivery_extra_buffer": 0.0,
            "class_2_extra_buffer": 0.0,
        },
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_key_fields": [
            "worst_delivery_slack",
            "worst_class_2_slack",
            "worst_raw_switch_slack",
            "worst_avoidable_slack",
            "negative_stay_bonus",
            "urgency_relief",
            "class_2_relief",
            "negative_candidate_index",
        ],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "v7_baseline": {
            key: baseline[key]
            for key in (
                "spec_sha256",
                "validation_freeze_sha256",
                "episode_metrics_path",
                "episode_metrics_sha256",
                "reference_row_count",
            )
        },
        "feature_contract": {
            "candidate_feature_schema_id": EXPECTED_FEATURE_SCHEMA["schema_id"],
            "candidate_feature_schema_sha256": EXPECTED_FEATURE_SCHEMA["sha256"],
            "candidate_feature_names": list(EXPECTED_FEATURE_NAMES),
            "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
            "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
            "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
        },
        "runtime_code_fingerprint": fingerprint,
        "runtime_code_fingerprint_sha256": source_runner.sha256_json(fingerprint),
        "paths": {
            "project": str(args.project.resolve()),
            "source": str(args.source.resolve()),
            "v7_output": str(args.v7_output.resolve()),
            "output": str(args.output.resolve()),
            "protocol": str(protocol.resolve()),
        },
        "device": args.device,
    }
    body["design_spec_sha256"] = source_runner.sha256_json(body)
    return body


def _evaluation_paths(output: Path, job: DesignJob) -> tuple[Path, Path]:
    directory = output / "evaluation_shards" / "design"
    return directory / f"{job.slug}.csv", directory / f"{job.slug}.json"


def _expected_shard_metadata(
    job: DesignJob,
    spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "design_spec_sha256": spec["design_spec_sha256"],
        "runtime_code_fingerprint_sha256": spec[
            "runtime_code_fingerprint_sha256"
        ],
        "source_training_freeze_sha256": spec["source_training_freeze_sha256"],
        "v7_validation_freeze_sha256": spec["v7_baseline"][
            "validation_freeze_sha256"
        ],
        "job": job.as_dict(),
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "workload_seeds": list(DESIGN_WORKLOAD_SEEDS),
        "parameters": job.as_parameters(),
        "policy_schema": {
            "candidate_feature_dim": 26,
            "action_size": 7,
            "obs_size": 182,
            "n_agents": 24,
            "variant": "proposed",
            "candidate_feature_schema_id": EXPECTED_FEATURE_SCHEMA["schema_id"],
            "candidate_feature_schema_sha256": EXPECTED_FEATURE_SCHEMA["sha256"],
            "candidate_feature_names": list(EXPECTED_FEATURE_NAMES),
        },
    }


def evaluate_job(
    args: argparse.Namespace,
    job: DesignJob,
    spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> list[EpisodeMetrics]:
    csv_path, metadata_path = _evaluation_paths(args.output, job)
    expected = _expected_shard_metadata(job, spec, checkpoint)
    if csv_path.exists() or metadata_path.exists():
        if not csv_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"partial design shard exists: {job.job_id}")
        metadata = source_runner._load_json(metadata_path)
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(f"design shard metadata mismatch: {job.job_id}/{field}")
        if metadata.get("csv_sha256") != sha256_file(csv_path):
            raise ValueError(f"design shard hash mismatch: {job.job_id}")
        if metadata.get("row_count") != len(DESIGN_WORKLOAD_SEEDS):
            raise ValueError(f"design shard row count mismatch: {job.job_id}")
        if not isinstance(metadata.get("policy_diagnostics"), Mapping):
            raise ValueError(f"design shard diagnostics missing: {job.job_id}")
        rows = source_runner.validate_evaluation_shard(
            csv_path, job, DESIGN_WORKLOAD_SEEDS
        )
        v7.validate_switch_rows(rows)
        return rows
    checkpoint_path = Path(str(checkpoint["checkpoint_path"]))
    if sha256_file(checkpoint_path) != checkpoint["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash changed: {job.source_job_id}")
    policy, _ = load_adaptive_hysteresis_policy(
        checkpoint_path,
        **job.as_parameters(),
        device=args.device,
    )
    if policy.checkpoint_schema != expected["policy_schema"]:
        raise ValueError(f"loaded adaptive policy schema mismatch: {job.job_id}")
    rows = evaluate_policy(
        job.scenario,
        job.policy_name,
        policy,
        job.policy_seed,
        DESIGN_WORKLOAD_SEEDS,
        variant="proposed",
    )
    v7.validate_switch_rows(rows)
    atomic_write_csv(csv_path, (asdict(row) for row in rows))
    validated = source_runner.validate_evaluation_shard(
        csv_path, job, DESIGN_WORKLOAD_SEEDS
    )
    metadata = {
        **expected,
        "csv_sha256": sha256_file(csv_path),
        "row_count": len(validated),
        "policy_diagnostics": policy.diagnostics(),
    }
    atomic_write_json(metadata_path, metadata)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return validated


def evaluate_jobs(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    jobs = build_design_jobs()
    for completed, job in enumerate(jobs, start=1):
        rows.extend(evaluate_job(args, job, spec, checkpoints[job.source_job_id]))
        print(f"[{completed}/{len(jobs)}] completed {job.job_id}", flush=True)
    expected = len(jobs) * len(DESIGN_WORKLOAD_SEEDS)
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError("design candidate grid is incomplete or duplicated")
    return rows


def _cell_rows(
    rows: Sequence[EpisodeMetrics],
    scenario: str,
    policy_seed: int,
    policy: str,
) -> list[EpisodeMetrics]:
    selected = [
        row
        for row in rows
        if row.scenario == scenario
        and row.policy_seed == policy_seed
        and row.policy == policy
    ]
    if len(selected) != len(DESIGN_WORKLOAD_SEEDS):
        raise ValueError(f"cell grid mismatch: {scenario}/{policy_seed}/{policy}")
    if {row.workload_seed for row in selected} != set(DESIGN_WORKLOAD_SEEDS):
        raise ValueError("cell workload grid mismatch")
    return selected


def _mean(rows: Sequence[EpisodeMetrics], field: str) -> float:
    return sum(float(getattr(row, field)) for row in rows) / len(rows)


def _micro(rows: Sequence[EpisodeMetrics]) -> tuple[int, int, float]:
    avoidable = sum(row.avoidable_routing_switches for row in rows)
    opportunities = sum(row.switch_opportunities for row in rows)
    if opportunities <= 0:
        raise ValueError("avoidable micro-rate denominator must be positive")
    return avoidable, opportunities, avoidable / opportunities


def compute_candidate_cell_gates(
    rows: Sequence[EpisodeMetrics],
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for candidate in DESIGN_GRID:
        policy = candidate_policy_name(candidate)
        for scenario in SCENARIOS:
            for policy_seed in POLICY_SEEDS:
                proposed = _cell_rows(
                    rows, scenario, policy_seed, REFERENCE_POLICIES["proposed"]
                )
                raw = _cell_rows(
                    rows, scenario, policy_seed, REFERENCE_POLICIES["raw_context"]
                )
                shield = _cell_rows(rows, scenario, policy_seed, policy)
                proposed_avoidable, proposed_opportunities, proposed_micro = _micro(
                    proposed
                )
                shield_avoidable, shield_opportunities, shield_micro = _micro(shield)
                delivery_delta = _mean(shield, "delivery_ratio") - _mean(
                    proposed, "delivery_ratio"
                )
                class_2_delta = _mean(shield, "class_2_delivery_ratio") - _mean(
                    proposed, "class_2_delivery_ratio"
                )
                proposed_switch = _mean(proposed, "routing_switches")
                raw_switch = _mean(raw, "routing_switches")
                shield_switch = _mean(shield, "routing_switches")
                delivery_slack = delivery_delta - float(
                    CELL_GATES["delivery_difference_vs_proposed_min"]
                )
                class_2_slack = class_2_delta - float(
                    CELL_GATES["class_2_delivery_difference_vs_proposed_min"]
                )
                avoidable_slack = proposed_micro - shield_micro
                proposed_switch_slack = proposed_switch - shield_switch
                raw_switch_slack = raw_switch - shield_switch
                five_gate_pass = (
                    delivery_slack >= 0.0
                    and class_2_slack >= 0.0
                    and avoidable_slack > 0.0
                    and proposed_switch_slack >= 0.0
                    and raw_switch_slack >= 0.0
                )
                buffer_pass = (
                    raw_switch_slack >= RAW_SWITCH_HEADROOM_BUFFER
                    and avoidable_slack >= AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
                )
                gates.append(
                    {
                        "candidate_index": int(candidate["candidate_index"]),
                        "candidate_id": candidate_id(candidate),
                        "stay_bonus": float(candidate["stay_bonus"]),
                        "urgency_relief": float(candidate["urgency_relief"]),
                        "class_2_relief": float(candidate["class_2_relief"]),
                        "scenario": scenario,
                        "policy_seed": policy_seed,
                        "workload_count": len(DESIGN_WORKLOAD_SEEDS),
                        "delivery_difference_vs_proposed": delivery_delta,
                        "class_2_delivery_difference_vs_proposed": class_2_delta,
                        "proposed_avoidable_switches_total": proposed_avoidable,
                        "proposed_switch_opportunities_total": proposed_opportunities,
                        "proposed_avoidable_switch_micro_rate": proposed_micro,
                        "shield_avoidable_switches_total": shield_avoidable,
                        "shield_switch_opportunities_total": shield_opportunities,
                        "shield_avoidable_switch_micro_rate": shield_micro,
                        "proposed_routing_switch_mean": proposed_switch,
                        "raw_context_routing_switch_mean": raw_switch,
                        "shield_routing_switch_mean": shield_switch,
                        "delivery_slack": delivery_slack,
                        "class_2_slack": class_2_slack,
                        "avoidable_slack": avoidable_slack,
                        "proposed_switch_slack": proposed_switch_slack,
                        "raw_switch_slack": raw_switch_slack,
                        "delivery_gate_pass": delivery_slack >= 0.0,
                        "class_2_gate_pass": class_2_slack >= 0.0,
                        "avoidable_gate_pass": avoidable_slack > 0.0,
                        "switch_vs_proposed_gate_pass": proposed_switch_slack >= 0.0,
                        "switch_vs_raw_gate_pass": raw_switch_slack >= 0.0,
                        "five_gates_pass": five_gate_pass,
                        "raw_switch_buffer_pass": (
                            raw_switch_slack >= RAW_SWITCH_HEADROOM_BUFFER
                        ),
                        "avoidable_buffer_pass": (
                            avoidable_slack >= AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
                        ),
                        "buffer_gate_pass": buffer_pass,
                        "cell_eligible": five_gate_pass and buffer_pass,
                    }
                )
    if len(gates) != len(DESIGN_GRID) * len(SCENARIOS) * len(POLICY_SEEDS):
        raise RuntimeError("design gate grid has the wrong size")
    return gates


def select_candidate(
    gates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    results: list[dict[str, Any]] = []
    for candidate in DESIGN_GRID:
        index = int(candidate["candidate_index"])
        cells = [dict(row) for row in gates if int(row["candidate_index"]) == index]
        if len(cells) != len(SCENARIOS) * len(POLICY_SEEDS):
            raise ValueError(f"candidate {index} does not have eight design cells")
        worst_delivery = min(float(row["delivery_slack"]) for row in cells)
        worst_class_2 = min(float(row["class_2_slack"]) for row in cells)
        worst_raw_switch = min(float(row["raw_switch_slack"]) for row in cells)
        worst_avoidable = min(float(row["avoidable_slack"]) for row in cells)
        eligible = all(bool(row["cell_eligible"]) for row in cells)
        key = [
            worst_delivery,
            worst_class_2,
            worst_raw_switch,
            worst_avoidable,
            -float(candidate["stay_bonus"]),
            float(candidate["urgency_relief"]),
            float(candidate["class_2_relief"]),
            -index,
        ]
        results.append(
            {
                **dict(candidate),
                "candidate_id": candidate_id(candidate),
                "policy_name": candidate_policy_name(candidate),
                "five_gates_all_cells_pass": all(
                    bool(row["five_gates_pass"]) for row in cells
                ),
                "buffers_all_cells_pass": all(
                    bool(row["buffer_gate_pass"]) for row in cells
                ),
                "eligible": eligible,
                "worst_delivery_slack": worst_delivery,
                "worst_class_2_slack": worst_class_2,
                "worst_raw_switch_slack": worst_raw_switch,
                "worst_avoidable_slack": worst_avoidable,
                "selection_key": key,
                "cells": cells,
            }
        )
    eligible = [result for result in results if result["eligible"]]
    selected = max(eligible, key=lambda result: tuple(result["selection_key"])) if eligible else None
    if selected is not None:
        tied = [
            result
            for result in eligible
            if result["selection_key"] == selected["selection_key"]
        ]
        if len(tied) != 1:
            raise RuntimeError("design selection argmax is not unique")
    return results, selected


def _artifact(path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def write_design_artifacts(
    output: Path,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    rows: Sequence[EpisodeMetrics],
) -> dict[str, Any]:
    episode_path = output / "design_episode_metrics.csv"
    aggregate_path = output / "design_aggregate_metrics.csv"
    gates_path = output / "design_cell_gates.csv"
    selection_path = output / "design_selection.json"
    gates = compute_candidate_cell_gates(rows)
    results, selected = select_candidate(gates)
    aggregate = source_runner.aggregate_rows(rows, rng_base=73000)
    atomic_write_csv(episode_path, (asdict(row) for row in rows))
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(gates_path, gates)
    selected_parameters = (
        {
            "stay_bonus": selected["stay_bonus"],
            "urgency_relief": selected["urgency_relief"],
            "class_2_relief": selected["class_2_relief"],
        }
        if selected is not None
        else None
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exposed_panel_engineering_design",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "design_spec_sha256": spec["design_spec_sha256"],
        "runtime_code_fingerprint_sha256": spec[
            "runtime_code_fingerprint_sha256"
        ],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "v7_validation_freeze_sha256": spec["v7_baseline"][
            "validation_freeze_sha256"
        ],
        "selection_rule_id": SELECTION_RULE_ID,
        "candidate_grid": [dict(candidate) for candidate in DESIGN_GRID],
        "candidate_results": results,
        "eligible_candidate_indices": [
            int(result["candidate_index"]) for result in results if result["eligible"]
        ],
        "selected_candidate_index": (
            int(selected["candidate_index"]) if selected is not None else None
        ),
        "selected_parameters": selected_parameters,
        "all_cells_pass": selected is not None,
        "selection_status": "selected" if selected is not None else "no_eligible_candidate",
        "fresh_validation_allowed": selected is not None,
        "test_allowed": False,
        "artifacts": {
            "design_spec": _artifact(output / "adaptive_shield_design_spec.json", 1),
            "episode_metrics": _artifact(episode_path, len(rows)),
            "aggregate_metrics": _artifact(aggregate_path, len(aggregate)),
            "cell_gates": _artifact(gates_path, len(gates)),
        },
    }
    body["design_selection_sha256"] = source_runner.sha256_json(body)
    source_runner.ensure_immutable_json(selection_path, body)
    return body


def _validate_artifact(
    artifact: Mapping[str, Any], expected_path: Path
) -> None:
    path = Path(str(artifact.get("path", ""))).resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"design artifact path mismatch: {expected_path.name}")
    if artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"design artifact hash mismatch: {expected_path.name}")
    declared_rows = artifact.get("row_count")
    if type(declared_rows) is not int or declared_rows <= 0:
        raise ValueError(f"design artifact row count invalid: {expected_path.name}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            observed_rows = sum(1 for _ in csv.DictReader(handle))
        if observed_rows != declared_rows:
            raise ValueError(f"design artifact row count mismatch: {expected_path.name}")
    elif declared_rows != 1:
        raise ValueError(f"design JSON artifact row count mismatch: {expected_path.name}")


def _csv_rows_match(path: Path, expected: Sequence[Mapping[str, Any]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        observed = list(csv.DictReader(handle))
    rendered = [{key: str(value) for key, value in row.items()} for row in expected]
    if observed != rendered:
        raise ValueError(f"design CSV does not replay: {path.name}")


def load_design_selection(
    path: Path,
    *,
    allow_historical_runtime: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    if (
        allow_historical_runtime
        and sha256_file(path) != EXPECTED_V8_SELECTION_FILE_SHA256
    ):
        raise ValueError("frozen v8 design selection changed")
    selection = source_runner._load_json(path)
    selection_sha256 = _self_hash(selection, "design_selection_sha256")
    if allow_historical_runtime and selection_sha256 != EXPECTED_V8_SELECTION_SHA256:
        raise ValueError("unexpected v8 design selection")
    if selection.get("screen_name") != SCREEN_NAME:
        raise ValueError("design selection screen mismatch")
    if selection.get("selection_rule_id") != SELECTION_RULE_ID:
        raise ValueError("design selection rule mismatch")
    for field in ("confirmatory", "paper_claim_allowed", "promotion_decision_allowed"):
        if selection.get(field) is not False:
            raise ValueError(f"design selection must set {field}=false")
    if selection.get("candidate_grid") != [dict(candidate) for candidate in DESIGN_GRID]:
        raise ValueError("design selection candidate grid mismatch")
    output = path.parent
    artifacts = selection.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("design selection artifact mapping missing")
    expected_paths = {
        "design_spec": output / "adaptive_shield_design_spec.json",
        "episode_metrics": output / "design_episode_metrics.csv",
        "aggregate_metrics": output / "design_aggregate_metrics.csv",
        "cell_gates": output / "design_cell_gates.csv",
    }
    if set(artifacts) != set(expected_paths):
        raise ValueError("design selection artifact set mismatch")
    for name, expected_path in expected_paths.items():
        _validate_artifact(artifacts[name], expected_path)
    spec_path = expected_paths["design_spec"]
    if (
        allow_historical_runtime
        and sha256_file(spec_path) != EXPECTED_V8_SPEC_FILE_SHA256
    ):
        raise ValueError("frozen v8 design specification changed")
    spec = source_runner._load_json(spec_path)
    spec_sha256 = _self_hash(spec, "design_spec_sha256")
    if allow_historical_runtime and spec_sha256 != EXPECTED_V8_SPEC_SHA256:
        raise ValueError("unexpected v8 design specification")
    if selection.get("design_spec_sha256") != spec["design_spec_sha256"]:
        raise ValueError("design selection spec mismatch")
    project = Path(str(spec["paths"]["project"])).resolve()
    protocol = Path(str(spec["paths"]["protocol"])).resolve()
    source = source_runner.audit_source_screen(Path(str(spec["paths"]["source"])))
    fingerprint = _runtime_code_fingerprint(project, protocol)
    if allow_historical_runtime:
        _audit_historical_runtime_fingerprint(
            project,
            source,
            spec.get("runtime_code_fingerprint"),
            fingerprint,
            HISTORICAL_V8_RUNNER_FINGERPRINTS,
        )
    elif fingerprint != spec.get("runtime_code_fingerprint"):
        raise ValueError("design runtime code fingerprint changed")
    baseline = audit_v7_baseline(
        Path(str(spec["paths"]["v7_output"])), source, project
    )
    if selection.get("source_training_freeze_sha256") != source[
        "training_freeze_sha256"
    ]:
        raise ValueError("design selection source freeze mismatch")
    if selection.get("v7_validation_freeze_sha256") != baseline[
        "validation_freeze_sha256"
    ]:
        raise ValueError("design selection v7 freeze mismatch")
    rows = source_runner._episode_rows_from_csv(expected_paths["episode_metrics"])
    expected_count = (
        len(baseline["references"])
        + len(build_design_jobs()) * len(DESIGN_WORKLOAD_SEEDS)
    )
    if len(rows) != expected_count:
        raise ValueError("design episode artifact row count mismatch")
    replayed_references = [
        row for row in rows if row.policy in set(REFERENCE_POLICIES.values())
    ]
    if replayed_references != baseline["references"]:
        raise ValueError("design v7 reference rows do not replay exactly")
    gates = compute_candidate_cell_gates(rows)
    _csv_rows_match(expected_paths["cell_gates"], gates)
    results, selected = select_candidate(gates)
    if selection.get("candidate_results") != results:
        raise ValueError("design candidate results do not replay")
    eligible_indices = [
        int(result["candidate_index"]) for result in results if result["eligible"]
    ]
    if selection.get("eligible_candidate_indices") != eligible_indices:
        raise ValueError("design eligible set does not replay")
    selected_index = int(selected["candidate_index"]) if selected is not None else None
    if selection.get("selected_candidate_index") != selected_index:
        raise ValueError("design unique argmax does not replay")
    expected_parameters = (
        {
            "stay_bonus": selected["stay_bonus"],
            "urgency_relief": selected["urgency_relief"],
            "class_2_relief": selected["class_2_relief"],
        }
        if selected is not None
        else None
    )
    if selection.get("selected_parameters") != expected_parameters:
        raise ValueError("design selected parameters do not replay")
    if selection.get("all_cells_pass") is not (selected is not None):
        raise ValueError("design selection status does not replay")
    expected_status = "selected" if selected is not None else "no_eligible_candidate"
    if selection.get("selection_status") != expected_status:
        raise ValueError("design selection status label does not replay")
    if selection.get("fresh_validation_allowed") is not (selected is not None):
        raise ValueError("design validation permission does not replay")
    if selection.get("test_allowed") is not False:
        raise ValueError("design selection must never allow test evaluation")
    return selection


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.source, args.v7_output, args.project):
        if not path.is_dir():
            raise FileNotFoundError(path)
    root = args.project.parent.resolve()
    broad = {
        root,
        args.project.resolve(),
        (root / "experiments").resolve(),
        args.source.resolve(),
        args.v7_output.resolve(),
    }
    if args.output.resolve() in broad:
        raise RuntimeError("output must be a dedicated design directory")
    if len(DESIGN_GRID) != 18 or len(build_design_jobs()) != 144:
        raise RuntimeError("frozen design grid changed")
    if EXPECTED_FEATURE_SCHEMA.get("schema_id") != (
        "leo_multi_candidate_features_v1_dim_26"
    ):
        raise RuntimeError("candidate feature schema changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Select an adaptive shield on the exposed v7 panel."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "experiments" / "archive" / "congestion-context-screen-20k-v1",
    )
    parser.add_argument(
        "--v7-output",
        type=Path,
        default=root / "experiments" / "proposed-shield-preflight-v7",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "adaptive-shield-design-v8",
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in ("source", "v7_output", "output", "project"):
        setattr(args, field, getattr(args, field).resolve())
    validate_environment(args)
    source = source_runner.audit_source_screen(args.source)
    baseline = audit_v7_baseline(args.v7_output, source, args.project)
    spec = build_spec(args, source, baseline)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "screen_name": SCREEN_NAME,
                    "design_spec_sha256": spec["design_spec_sha256"],
                    "candidate_count": len(DESIGN_GRID),
                    "design_jobs": len(build_design_jobs()),
                    "candidate_rows": len(build_design_jobs())
                    * len(DESIGN_WORKLOAD_SEEDS),
                    "reference_rows": baseline["reference_row_count"],
                    "workloads": [
                        DESIGN_WORKLOAD_SEEDS[0],
                        DESIGN_WORKLOAD_SEEDS[-1],
                    ],
                    "selection_rule_id": SELECTION_RULE_ID,
                    "dry_run_writes_output": False,
                },
                indent=2,
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    source_runner.ensure_immutable_json(
        args.output / "adaptive_shield_design_spec.json", spec
    )
    atomic_write_csv(
        args.output / "design_plan.csv",
        (job.as_dict() for job in build_design_jobs()),
    )
    with source_runner.invocation_lock(args.output):
        candidate_rows = evaluate_jobs(args, spec, source["checkpoints"])
        combined = [*baseline["references"], *candidate_rows]
        selection = write_design_artifacts(
            args.output, spec, source, combined
        )
        load_design_selection(args.output / "design_selection.json")
    print(json.dumps(selection, indent=2))
    return 0 if selection["all_cells_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
