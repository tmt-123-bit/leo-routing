"""Run the frozen adaptive proposed-checkpoint shield preflight (v8).

The runner never trains a policy.  It evaluates one parameter triple selected
by the separately frozen adaptive-shield design artifact.  Validation and test
remain separate, hash-bound phases and the test panel is fail-closed behind a
fully replayed passing validation freeze.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import uuid

import numpy as np
import torch

from ablation_matrix_runner import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from adaptive_hysteresis_policy import load_adaptive_hysteresis_policy
from hysteresis_policy import SWITCH_FEATURE_INDEX, load_hysteresis_policy
from leo_multiagent_env import (
    BASE_CANDIDATE_FEATURE_NAMES,
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
    candidate_feature_schema,
)
from mappo_evaluation import EpisodeMetrics, evaluate_policy
import run_adaptive_shield_design as design
import run_hysteresis_screen as source_runner
import run_proposed_shield_preflight as v7
import run_source_runtime_equivalence_audit as runtime_equivalence


SCREEN_NAME = "ADAPTIVE-SHIELD-PREFLIGHT-v8"
SCHEMA_VERSION = 1
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072, 1047581915, 1245825580)
VALIDATION_WORKLOAD_SEEDS = tuple(range(71001, 71011))
TEST_WORKLOAD_SEEDS = tuple(range(72001, 72026))
ARMS = ("proposed", "raw_context", "shield")
POLICY_NAMES = {
    "proposed": "mappo_proposed_source_v8",
    "raw_context": "mappo_context_raw_source_v8",
    "shield": "mappo_proposed_adaptive_shield_v8",
}
CELL_GATES = dict(v7.CELL_GATES)
DESIGN_SCREEN_NAME = "ADAPTIVE-SHIELD-DESIGN-v8"
DESIGN_SELECTION_FILENAME = "design_selection.json"
SOURCE_RUNTIME_EQUIVALENCE_FILENAME = "source_runtime_equivalence.json"
EXPECTED_FEATURE_SCHEMA = candidate_feature_schema()
EXPECTED_FEATURE_NAMES = tuple(BASE_CANDIDATE_FEATURE_NAMES)
EXPECTED_FEATURE_DIM = 26
EXPECTED_FEATURE_INDEXES = {
    "route_switch": SWITCH_FEATURE_INDEX,
    "route_urgency": ROUTE_URGENCY_FEATURE_INDEX,
    "route_class_2": ROUTE_CLASS_2_FEATURE_INDEX,
}
KNOWN_EXPOSED_OR_RESERVED_PANELS = (
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
    (61001, 61004),
    (70001, 70025),
)
SOURCE_RUNTIME_FILES = {
    "design": "mappo_design.py",
    "evaluation": "mappo_evaluation.py",
    "wrapper": "cleanmarl_leo_multiagent_wrapper.py",
    "environment": "leo_multiagent_env.py",
    "base_environment": "leo_marl_env.py",
    "variant_definitions": "variant_definitions.py",
    "screen_statistics": "hierarchical_statistics.py",
    "screen_artifact_helpers": "ablation_matrix_runner.py",
}


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    phase: str
    scenario: str
    arm: str
    policy_seed: int

    @property
    def source_variant(self) -> str:
        if self.arm in {"proposed", "shield"}:
            return "proposed"
        if self.arm == "raw_context":
            return "with_congestion_context"
        raise ValueError(f"unknown arm: {self.arm}")

    @property
    def policy_name(self) -> str:
        return POLICY_NAMES[self.arm]

    @property
    def source_job_id(self) -> str:
        return f"{self.scenario}/{self.source_variant}/seed_{self.policy_seed}"

    @property
    def job_id(self) -> str:
        return f"{self.phase}/{self.scenario}/{self.arm}/seed_{self.policy_seed}"

    @property
    def slug(self) -> str:
        return "__".join((self.scenario, self.arm, f"seed_{self.policy_seed}"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "phase": self.phase,
            "scenario": self.scenario,
            "arm": self.arm,
            "policy_seed": self.policy_seed,
            "policy_name": self.policy_name,
            "source_variant": self.source_variant,
            "source_job_id": self.source_job_id,
            "job_id": self.job_id,
            "controller": "adaptive_shield" if self.arm == "shield" else "raw",
        }


def build_jobs(phase: str) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    if phase not in {"validation", "test"}:
        raise ValueError(f"unsupported phase: {phase}")
    for scenario in SCENARIOS:
        for arm in ARMS:
            for policy_seed in POLICY_SEEDS:
                jobs.append(EvaluationJob(len(jobs), phase, scenario, arm, policy_seed))
    return jobs


def _workloads_for_phase(phase: str) -> tuple[int, ...]:
    if phase == "validation":
        return VALIDATION_WORKLOAD_SEEDS
    if phase == "test":
        return TEST_WORKLOAD_SEEDS
    raise ValueError(f"unsupported phase: {phase}")


def _self_hash(record: Mapping[str, Any], field: str) -> str:
    body = dict(record)
    observed = body.pop(field, None)
    expected = source_runner.sha256_json(body)
    if observed != expected:
        raise ValueError(
            f"invalid {field}: observed={observed!r}, expected={expected}"
        )
    return expected


def _selected_parameters(selection: Mapping[str, Any]) -> dict[str, float]:
    raw = selection.get("selected_parameters")
    required = {"stay_bonus", "urgency_relief", "class_2_relief"}
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("design selection has an invalid selected_parameters object")
    parameters: dict[str, float] = {}
    for field in sorted(required):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"selected parameter {field} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"selected parameter {field} must be finite")
        parameters[field] = numeric
    if not 0.0 <= parameters["stay_bonus"] <= 1.0:
        raise ValueError("selected stay_bonus must be in [0, 1]")
    for field in ("urgency_relief", "class_2_relief"):
        if not 0.0 <= parameters[field] <= 1.0:
            raise ValueError(f"selected {field} must be in [0, 1]")
    return parameters


def load_design_selection(
    path: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    selection = design.load_design_selection(path.resolve())
    if not isinstance(selection, Mapping):
        raise TypeError("design selection loader must return a mapping")
    selection = dict(selection)
    _self_hash(selection, "design_selection_sha256")
    if selection.get("screen_name") != DESIGN_SCREEN_NAME:
        raise ValueError("design selection screen mismatch")
    if selection.get("paper_claim_allowed") is not False:
        raise ValueError("design selection must prohibit paper claims")
    if selection.get("promotion_decision_allowed") is not False:
        raise ValueError("design selection must prohibit promotion")
    if selection.get("all_cells_pass") is not True:
        raise RuntimeError("design selection did not pass all design cells")
    if (
        selection.get("source_training_freeze_sha256")
        != source["training_freeze_sha256"]
    ):
        raise ValueError("design selection source freeze mismatch")
    for field in (
        "design_spec_sha256",
        "v7_validation_freeze_sha256",
        "selection_rule_id",
        "design_selection_sha256",
    ):
        value = selection.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"design selection lacks {field}")
    _selected_parameters(selection)
    return selection


def _runtime_code_fingerprint(project: Path, protocol_path: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "protocol": protocol_path,
        "adaptive_policy": project / "adaptive_hysteresis_policy.py",
        "adaptive_design": project / "run_adaptive_shield_design.py",
        "raw_hysteresis_policy": project / "hysteresis_policy.py",
        **{
            name: project / filename
            for name, filename in SOURCE_RUNTIME_FILES.items()
        },
        "source_auditor": project / "run_hysteresis_screen.py",
        "source_runtime_equivalence_auditor": (
            project / "run_source_runtime_equivalence_audit.py"
        ),
        "v7_runner": project / "run_proposed_shield_preflight.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime fingerprint files missing: {missing}")
    modules = {
        "adaptive_hysteresis_policy": "adaptive_hysteresis_policy.py",
        "run_adaptive_shield_design": "run_adaptive_shield_design.py",
        "hysteresis_policy": "hysteresis_policy.py",
        "mappo_design": "mappo_design.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "cleanmarl_leo_multiagent_wrapper": "cleanmarl_leo_multiagent_wrapper.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "leo_marl_env": "leo_marl_env.py",
        "variant_definitions": "variant_definitions.py",
        "hierarchical_statistics": "hierarchical_statistics.py",
        "ablation_matrix_runner": "ablation_matrix_runner.py",
        "run_hysteresis_screen": "run_hysteresis_screen.py",
        "run_source_runtime_equivalence_audit": (
            "run_source_runtime_equivalence_audit.py"
        ),
        "run_proposed_shield_preflight": "run_proposed_shield_preflight.py",
    }
    for module_name, filename in modules.items():
        actual = Path(inspect.getfile(importlib.import_module(module_name))).resolve()
        expected = (project / filename).resolve()
        if actual != expected:
            raise RuntimeError(
                f"runtime module path mismatch for {module_name}: "
                f"{actual} != {expected}"
            )
    return {name: sha256_file(path) for name, path in files.items()}


def load_source_runtime_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project.parent.resolve()
    return runtime_equivalence.load_and_replay_equivalence_artifact(
        args.source_runtime_equivalence.resolve(),
        project_root=root,
        archive_path=(
            root
            / "experiments"
            / "archive"
            / "source-snapshots"
            / "actor-score-hysteresis-c3d05253666b.zip"
        ),
        source_spec_path=args.source.resolve() / "screen_spec.json",
        hysteresis_spec_path=(
            root
            / "experiments"
            / "archive"
            / "congestion-hysteresis-screen-v1"
            / "hysteresis_spec.json"
        ),
    )


def _source_runtime_equivalence_record(
    path: Path,
    artifact: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    self_hash = runtime_equivalence.validate_equivalence_artifact(artifact)
    source_screen = artifact.get("frozen_specs", {}).get("source_screen", {})
    if source_screen.get("spec_sha256") != source.get("spec_sha256"):
        raise ValueError("source runtime equivalence source-screen mismatch")
    feature_contract = artifact.get("dynamic_contract", {}).get(
        "feature_contract"
    )
    expected_feature_contract = {
        "action_size": 7,
        "candidate_feature_dim": EXPECTED_FEATURE_DIM,
        "feature_indexes": {
            "route_class_2": EXPECTED_FEATURE_INDEXES["route_class_2"],
            "route_switch": EXPECTED_FEATURE_INDEXES["route_switch"],
            "route_urgency": EXPECTED_FEATURE_INDEXES["route_urgency"],
        },
        "feature_names": list(EXPECTED_FEATURE_NAMES),
        "schema_id": EXPECTED_FEATURE_SCHEMA["schema_id"],
        "schema_sha256": EXPECTED_FEATURE_SCHEMA["sha256"],
    }
    if feature_contract != expected_feature_contract:
        raise ValueError("source runtime equivalence feature contract mismatch")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path.resolve()),
        "source_runtime_equivalence_sha256": self_hash,
        "audit_name": artifact["audit_name"],
        "source_screen_spec_sha256": source_screen["spec_sha256"],
        "workload_seed": artifact["dynamic_contract"]["workload_seed"],
        "workload_seed_provenance": artifact["dynamic_contract"][
            "workload_seed_provenance"
        ],
        "snapshot_current_exact_match": artifact["decision"][
            "snapshot_current_exact_match"
        ],
        "all_pass": artifact["decision"]["all_pass"],
    }


def build_spec(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_runtime_equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = args.project.parent / "docs" / "ADAPTIVE_SHIELD_PREFLIGHT_V8.md"
    code_fingerprint = _runtime_code_fingerprint(args.project, protocol)
    source_runtime_record = _source_runtime_equivalence_record(
        args.source_runtime_equivalence,
        source_runtime_equivalence,
        source,
    )
    code_sha = source_runner.sha256_json(code_fingerprint)
    parameters = _selected_parameters(selection)
    feature_contract = {
        "candidate_feature_dim": EXPECTED_FEATURE_DIM,
        "candidate_feature_schema_id": EXPECTED_FEATURE_SCHEMA["schema_id"],
        "candidate_feature_schema_sha256": EXPECTED_FEATURE_SCHEMA["sha256"],
        "candidate_feature_names": list(EXPECTED_FEATURE_NAMES),
        "feature_indexes": dict(EXPECTED_FEATURE_INDEXES),
    }
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "training_jobs": 0,
        "device": args.device,
        "method": {
            "type": "frozen_proposed_actor_adaptive_posthoc_shield",
            "checkpoint_and_actor_weights": "frozen_no_training",
            "selected_parameters": parameters,
            "parameter_source": "hash_verified_replayed_design_selection_only",
            "cli_parameter_override_allowed": False,
        },
        "design_selection": {
            "path": str(args.selection.resolve()),
            "file_sha256": sha256_file(args.selection),
            "design_selection_sha256": selection["design_selection_sha256"],
            "design_spec_sha256": selection["design_spec_sha256"],
            "v7_validation_freeze_sha256": selection[
                "v7_validation_freeze_sha256"
            ],
            "selection_rule_id": selection["selection_rule_id"],
            "selected_parameters": parameters,
        },
        "feature_contract": feature_contract,
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "arms": list(ARMS),
        "policy_names": dict(POLICY_NAMES),
        "validation_workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "known_exposed_or_reserved_panels": [
            list(panel) for panel in KNOWN_EXPOSED_OR_RESERVED_PANELS
        ],
        "fresh_panel_audit": {
            "validation_panel_prior_workload_hits": 0,
            "test_panel_prior_workload_hits": 0,
            "v7_test_panel_70001_70025_reserved_not_reused": True,
        },
        "cell_gates": dict(CELL_GATES),
        "avoidable_switch_estimand": (
            "sum(avoidable_routing_switches)/sum(switch_opportunities)_"
            "within_each_scenario_by_policy_seed_cell"
        ),
        "gate_scope": "all_2_scenarios_by_4_policy_seed_cells_must_pass",
        "expected_validation_jobs": len(build_jobs("validation")),
        "expected_validation_rows": len(build_jobs("validation"))
        * len(VALIDATION_WORKLOAD_SEEDS),
        "expected_test_jobs": len(build_jobs("test")),
        "expected_test_rows": len(build_jobs("test"))
        * len(TEST_WORKLOAD_SEEDS),
        "test_isolation": {
            "test_requires_replayed_passing_validation_freeze": True,
            "test_guard_runs_before_any_output_write": True,
            "test_shards_bind_validation_freeze_sha256": True,
        },
        "source": dict(source),
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "source_runtime_equivalence": source_runtime_record,
        "code_fingerprint": code_fingerprint,
        "code_fingerprint_sha256": code_sha,
        "paths": {
            "project": str(args.project.resolve()),
            "source": str(args.source.resolve()),
            "selection": str(args.selection.resolve()),
            "source_runtime_equivalence": str(
                args.source_runtime_equivalence.resolve()
            ),
            "output": str(args.output.resolve()),
            "protocol": str(protocol.resolve()),
        },
    }
    body["spec_sha256"] = source_runner.sha256_json(body)
    return body


def _evaluation_paths(output: Path, job: EvaluationJob) -> tuple[Path, Path]:
    directory = output / "evaluation_shards" / job.phase
    return directory / f"{job.slug}.csv", directory / f"{job.slug}.json"


def _policy_schema(
    job: EvaluationJob,
    checkpoint: Mapping[str, Any],
    parameters: Mapping[str, float],
) -> dict[str, Any]:
    feature_dim = int(checkpoint["candidate_feature_dim"])
    common = {
        "candidate_feature_dim": feature_dim,
        "action_size": 7,
        "obs_size": 7 * feature_dim,
        "n_agents": 24,
        "variant": job.source_variant,
        "controller": "adaptive_shield" if job.arm == "shield" else "raw",
    }
    if job.source_variant == "proposed":
        common.update(
            candidate_feature_schema_id=EXPECTED_FEATURE_SCHEMA["schema_id"],
            candidate_feature_schema_sha256=EXPECTED_FEATURE_SCHEMA["sha256"],
            candidate_feature_names=list(EXPECTED_FEATURE_NAMES),
            feature_indexes=dict(EXPECTED_FEATURE_INDEXES),
        )
    if job.arm == "shield":
        common["selected_parameters"] = dict(parameters)
    else:
        common["stay_bonus"] = 0.0
    return common


def _expected_shard_metadata(
    job: EvaluationJob,
    spec: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    workload_seeds: Sequence[int],
    validation_freeze_sha256: str | None,
) -> dict[str, Any]:
    parameters = _selected_parameters(selection)
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec["spec_sha256"],
        "code_fingerprint_sha256": spec["code_fingerprint_sha256"],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "source_training_freeze_sha256": spec[
            "source_training_freeze_sha256"
        ],
        "design_selection_sha256": selection["design_selection_sha256"],
        "validation_freeze_sha256": validation_freeze_sha256,
        "job": job.as_dict(),
        "checkpoint_path": source_checkpoint["checkpoint_path"],
        "checkpoint_sha256": source_checkpoint["checkpoint_sha256"],
        "policy_schema": _policy_schema(job, source_checkpoint, parameters),
        "workload_seeds": list(workload_seeds),
    }


def _load_valid_shard(
    output: Path,
    job: EvaluationJob,
    expected: Mapping[str, Any],
    workload_seeds: Sequence[int],
) -> list[EpisodeMetrics]:
    csv_path, metadata_path = _evaluation_paths(output, job)
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing evaluation shard for {job.job_id}")
    metadata = source_runner._load_json(metadata_path)
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"evaluation shard metadata mismatch: {job.job_id}/{field}")
    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(f"evaluation shard hash mismatch: {job.job_id}")
    if metadata.get("row_count") != len(workload_seeds):
        raise ValueError(f"evaluation shard row count mismatch: {job.job_id}")
    if not isinstance(metadata.get("policy_diagnostics"), Mapping):
        raise ValueError(f"evaluation shard diagnostics missing: {job.job_id}")
    rows = source_runner.validate_evaluation_shard(csv_path, job, workload_seeds)
    v7.validate_switch_rows(rows)
    return rows


def evaluate_job(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    workload_seeds: Sequence[int],
    selection: Mapping[str, Any],
    *,
    validation_freeze_sha256: str | None = None,
) -> list[EpisodeMetrics]:
    expected = _expected_shard_metadata(
        job,
        spec,
        selection,
        source_checkpoint,
        workload_seeds,
        validation_freeze_sha256,
    )
    csv_path, metadata_path = _evaluation_paths(args.output, job)
    if csv_path.exists() or metadata_path.exists():
        if not csv_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"partial evaluation shard exists: {job.job_id}")
        return _load_valid_shard(args.output, job, expected, workload_seeds)

    checkpoint_path = Path(str(source_checkpoint["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != source_checkpoint["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash changed before {job.job_id}")
    parameters = _selected_parameters(selection)
    if job.arm == "shield":
        policy, _ = load_adaptive_hysteresis_policy(
            checkpoint_path,
            **parameters,
            device=args.device,
        )
    else:
        policy, _ = load_hysteresis_policy(
            checkpoint_path,
            stay_bonus=0.0,
            device=args.device,
        )
    observed_schema = {
        **policy.checkpoint_schema,
        "controller": "adaptive_shield" if job.arm == "shield" else "raw",
    }
    if job.source_variant == "proposed" and job.arm != "shield":
        observed_schema.update(
            candidate_feature_schema_id=EXPECTED_FEATURE_SCHEMA["schema_id"],
            candidate_feature_schema_sha256=EXPECTED_FEATURE_SCHEMA["sha256"],
            candidate_feature_names=list(EXPECTED_FEATURE_NAMES),
            feature_indexes=dict(EXPECTED_FEATURE_INDEXES),
        )
    if job.arm == "shield":
        observed_schema.update(
            feature_indexes=dict(EXPECTED_FEATURE_INDEXES),
            selected_parameters={
                "stay_bonus": policy.stay_bonus,
                "urgency_relief": policy.urgency_relief,
                "class_2_relief": policy.class_2_relief,
            },
        )
    else:
        observed_schema["stay_bonus"] = policy.stay_bonus
    if observed_schema != expected["policy_schema"]:
        raise ValueError(
            f"loaded policy schema mismatch for {job.job_id}: "
            f"{observed_schema} != {expected['policy_schema']}"
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
    v7.validate_switch_rows(rows)
    atomic_write_csv(csv_path, (asdict(row) for row in rows))
    validated = source_runner.validate_evaluation_shard(
        csv_path, job, workload_seeds
    )
    v7.validate_switch_rows(validated)
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
    jobs: Sequence[EvaluationJob],
    spec: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    workload_seeds: Sequence[int],
    selection: Mapping[str, Any],
    *,
    validation_freeze_sha256: str | None = None,
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for completed, job in enumerate(jobs, start=1):
        rows.extend(
            evaluate_job(
                args,
                job,
                spec,
                checkpoints[job.source_job_id],
                workload_seeds,
                selection,
                validation_freeze_sha256=validation_freeze_sha256,
            )
        )
        print(f"[{completed}/{len(jobs)}] completed {job.job_id}", flush=True)
    expected_rows = len(jobs) * len(workload_seeds)
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in rows
    }
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise RuntimeError("evaluation row grid is incomplete or duplicated")
    v7.validate_switch_rows(rows)
    return rows


@contextmanager
def _configured_v7_gates() -> Iterator[None]:
    overrides = {
        "SCENARIOS": SCENARIOS,
        "POLICY_SEEDS": POLICY_SEEDS,
        "POLICY_NAMES": POLICY_NAMES,
        "CELL_GATES": CELL_GATES,
    }
    saved = {field: getattr(v7, field) for field in overrides}
    try:
        for field, value in overrides.items():
            setattr(v7, field, value)
        yield
    finally:
        for field, value in saved.items():
            setattr(v7, field, value)


def compute_cell_gates(
    rows: Sequence[EpisodeMetrics],
    workload_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    with _configured_v7_gates():
        gates = v7.compute_cell_gates(rows, workload_seeds)
    if len(gates) != 8:
        raise RuntimeError("adaptive preflight gate grid must contain eight cells")
    return gates


def _phase_artifact_paths(output: Path, phase: str) -> dict[str, Path]:
    return {
        "episode_metrics": output / f"{phase}_episode_metrics.csv",
        "aggregate_metrics": output / f"{phase}_aggregate_metrics.csv",
        "cell_gates": output / f"{phase}_cell_gates.csv",
        "decision": output / f"{phase}_decision.json",
    }


def _shard_index(output: Path, jobs: Sequence[EvaluationJob]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for job in jobs:
        csv_path, metadata_path = _evaluation_paths(output, job)
        metadata = source_runner._load_json(metadata_path)
        if metadata.get("csv_sha256") != sha256_file(csv_path):
            raise ValueError(f"shard index hash mismatch: {job.job_id}")
        index.append(
            {
                "job_id": job.job_id,
                "csv_path": str(csv_path.resolve()),
                "csv_sha256": sha256_file(csv_path),
                "csv_row_count": int(metadata["row_count"]),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "metadata_row_count": 1,
            }
        )
    return index


def write_phase_artifacts(
    output: Path,
    phase: str,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    rows: Sequence[EpisodeMetrics],
    *,
    validation_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    workloads = _workloads_for_phase(phase)
    gates = compute_cell_gates(rows, workloads)
    aggregate = source_runner.aggregate_rows(
        rows, rng_base=71000 if phase == "validation" else 72000
    )
    paths = _phase_artifact_paths(output, phase)
    atomic_write_csv(paths["episode_metrics"], (asdict(row) for row in rows))
    atomic_write_csv(paths["aggregate_metrics"], aggregate)
    atomic_write_csv(paths["cell_gates"], gates)
    artifacts = {
        "episode_metrics": {
            "path": str(paths["episode_metrics"].resolve()),
            "sha256": sha256_file(paths["episode_metrics"]),
            "row_count": len(rows),
        },
        "aggregate_metrics": {
            "path": str(paths["aggregate_metrics"].resolve()),
            "sha256": sha256_file(paths["aggregate_metrics"]),
            "row_count": len(aggregate),
        },
        "cell_gates": {
            "path": str(paths["cell_gates"].resolve()),
            "sha256": sha256_file(paths["cell_gates"]),
            "row_count": len(gates),
        },
    }
    passed = all(bool(gate["all_gates_pass"]) for gate in gates)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "phase": phase,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "code_fingerprint_sha256": spec["code_fingerprint_sha256"],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "design_selection_sha256": selection["design_selection_sha256"],
        "selected_parameters": _selected_parameters(selection),
        "input_validation_freeze_sha256": validation_freeze_sha256,
        "workload_seeds": list(workloads),
        "cell_gates": gates,
        "all_cells_pass": passed,
        "decision": f"{phase}_{'pass' if passed else 'fail'}",
        "test_allowed": passed if phase == "validation" else None,
        "evaluation_shards": _shard_index(output, build_jobs(phase)),
        "artifacts": artifacts,
    }
    hash_field = (
        "validation_freeze_sha256"
        if phase == "validation"
        else "test_decision_sha256"
    )
    body[hash_field] = source_runner.sha256_json(body)
    source_runner.ensure_immutable_json(paths["decision"], body)
    return body


def _validate_bound_artifact(
    artifact: Mapping[str, Any],
    expected_path: Path,
    expected_rows: int,
) -> None:
    path = Path(str(artifact.get("path", ""))).resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"bound artifact path mismatch: {expected_path}")
    if artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"bound artifact hash mismatch: {path}")
    if artifact.get("row_count") != expected_rows:
        raise ValueError(f"bound artifact row count mismatch: {path}")


def _load_phase_shards(
    output: Path,
    jobs: Sequence[EvaluationJob],
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    workloads: Sequence[int],
    validation_freeze_sha256: str | None,
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for job in jobs:
        expected = _expected_shard_metadata(
            job,
            spec,
            selection,
            source["checkpoints"][job.source_job_id],
            workloads,
            validation_freeze_sha256,
        )
        rows.extend(_load_valid_shard(output, job, expected, workloads))
    return rows


def load_validation_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    path = args.output / "validation_decision.json"
    freeze = source_runner._load_json(path)
    _self_hash(freeze, "validation_freeze_sha256")
    required = {
        "screen_name": SCREEN_NAME,
        "phase": "validation",
        "spec_sha256": spec["spec_sha256"],
        "code_fingerprint_sha256": spec["code_fingerprint_sha256"],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "design_selection_sha256": selection["design_selection_sha256"],
        "selected_parameters": _selected_parameters(selection),
        "workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
        "all_cells_pass": True,
        "test_allowed": True,
    }
    for field, value in required.items():
        if freeze.get(field) != value:
            raise ValueError(f"validation freeze mismatch: {field}")
    jobs = build_jobs("validation")
    shard_index = _shard_index(args.output, jobs)
    if freeze.get("evaluation_shards") != shard_index:
        raise ValueError("validation shard index does not replay")
    rows = _load_phase_shards(
        args.output,
        jobs,
        spec,
        source,
        selection,
        VALIDATION_WORKLOAD_SEEDS,
        None,
    )
    paths = _phase_artifact_paths(args.output, "validation")
    aggregate = source_runner.aggregate_rows(rows, rng_base=71000)
    gates = compute_cell_gates(rows, VALIDATION_WORKLOAD_SEEDS)
    _validate_bound_artifact(
        freeze["artifacts"]["episode_metrics"], paths["episode_metrics"], len(rows)
    )
    _validate_bound_artifact(
        freeze["artifacts"]["aggregate_metrics"],
        paths["aggregate_metrics"],
        len(aggregate),
    )
    _validate_bound_artifact(
        freeze["artifacts"]["cell_gates"], paths["cell_gates"], len(gates)
    )
    combined = source_runner._episode_rows_from_csv(paths["episode_metrics"])
    if combined != rows:
        raise ValueError("validation combined episode artifact does not replay")
    if freeze.get("cell_gates") != gates:
        raise ValueError("validation gate decision does not replay")
    if not all(bool(gate["all_gates_pass"]) for gate in gates):
        raise RuntimeError("replayed validation gates fail")
    return freeze


def _artifact_record(path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def write_manifest(
    output: Path,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "spec": _artifact_record(output / "adaptive_shield_preflight_spec.json", 1),
        "validation_plan": _artifact_record(output / "validation_plan.csv", 24),
        "test_plan": _artifact_record(output / "test_plan.csv", 24),
        "validation_episode_metrics": _artifact_record(
            output / "validation_episode_metrics.csv", 240
        ),
        "validation_aggregate_metrics": _artifact_record(
            output / "validation_aggregate_metrics.csv", 24
        ),
        "validation_cell_gates": _artifact_record(
            output / "validation_cell_gates.csv", 8
        ),
        "validation_decision": _artifact_record(
            output / "validation_decision.json", 1
        ),
        "test_episode_metrics": _artifact_record(
            output / "test_episode_metrics.csv", 600
        ),
        "test_aggregate_metrics": _artifact_record(
            output / "test_aggregate_metrics.csv", 24
        ),
        "test_cell_gates": _artifact_record(output / "test_cell_gates.csv", 8),
        "test_decision": _artifact_record(output / "test_decision.json", 1),
    }
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "code_fingerprint_sha256": spec["code_fingerprint_sha256"],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "design_selection_sha256": selection["design_selection_sha256"],
        "validation_freeze_sha256": validation["validation_freeze_sha256"],
        "test_decision_sha256": test["test_decision_sha256"],
        "selected_parameters": _selected_parameters(selection),
        "validation_passed": validation["all_cells_pass"],
        "test_passed": test["all_cells_pass"],
        "training_jobs": 0,
        "validation_jobs": 24,
        "validation_rows": 240,
        "test_jobs": 24,
        "test_rows": 600,
        "validation_shards": _shard_index(output, build_jobs("validation")),
        "test_shards": _shard_index(output, build_jobs("test")),
        "artifacts": artifacts,
    }
    body["manifest_sha256"] = source_runner.sha256_json(body)
    source_runner.ensure_immutable_json(
        output / "adaptive_shield_preflight_manifest.json", body
    )
    return body


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset for this preflight")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.source, args.project):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.selection.is_file():
        raise FileNotFoundError(args.selection)
    if not args.source_runtime_equivalence.is_file():
        raise FileNotFoundError(args.source_runtime_equivalence)
    root = args.project.parent.resolve()
    experiments = (root / "experiments").resolve()
    broad = {
        root,
        args.project.resolve(),
        experiments,
        (experiments / "archive").resolve(),
        args.source.resolve(),
        args.selection.parent.resolve(),
        args.source_runtime_equivalence.parent.resolve(),
    }
    if args.output.resolve() in broad:
        raise RuntimeError("output must be a dedicated experiment directory")
    if args.source.resolve() in args.output.resolve().parents:
        raise RuntimeError("output must not be inside the frozen source")
    if args.output.is_dir() and not (
        args.output / "adaptive_shield_preflight_spec.json"
    ).is_file():
        conflicts = [
            path.name
            for path in args.output.iterdir()
            if path.name.endswith("_spec.json") or path.name == "screen_spec.json"
        ]
        if conflicts:
            raise RuntimeError(f"output contains another experiment: {conflicts}")
    validation = set(VALIDATION_WORKLOAD_SEEDS)
    test = set(TEST_WORKLOAD_SEEDS)
    exposed = set().union(
        *(
            set(range(start, stop + 1))
            for start, stop in KNOWN_EXPOSED_OR_RESERVED_PANELS
        )
    )
    if validation & test:
        raise RuntimeError("validation and test panels overlap")
    if validation & exposed or test & exposed:
        raise RuntimeError("fresh panels overlap an exposed or reserved panel")
    if EXPECTED_FEATURE_DIM != len(EXPECTED_FEATURE_NAMES):
        raise RuntimeError("candidate feature dimension contract changed")
    if EXPECTED_FEATURE_SCHEMA.get("schema_id") != (
        "leo_multi_candidate_features_v1_dim_26"
    ):
        raise RuntimeError("candidate feature schema id changed")
    if EXPECTED_FEATURE_SCHEMA.get("sha256") != (
        "be660bb34d6d8579773b643f2824e6dac5069fe67cfd8a8d71a36b1b147f9f70"
    ):
        raise RuntimeError("candidate feature schema hash changed")
    if tuple(EXPECTED_FEATURE_INDEXES.values()) != (17, 20, 23):
        raise RuntimeError("adaptive feature index contract changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen adaptive proposed-checkpoint shield."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "experiments" / "archive" / "congestion-context-screen-20k-v1",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=root / "experiments" / "adaptive-shield-design-v8" / DESIGN_SELECTION_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "adaptive-shield-preflight-v8",
    )
    parser.add_argument(
        "--source-runtime-equivalence",
        type=Path,
        default=(
            root
            / "experiments"
            / "source-runtime-equivalence-v8"
            / SOURCE_RUNTIME_EQUIVALENCE_FILENAME
        ),
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _dry_run_summary(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "code_fingerprint_sha256": spec["code_fingerprint_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "design_selection_sha256": selection["design_selection_sha256"],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "selected_parameters": _selected_parameters(selection),
        "training_jobs": 0,
        "validation_jobs": 24,
        "validation_rows": 240,
        "test_jobs": 24,
        "test_rows": 600,
        "validation_workloads": [71001, 71010],
        "test_workloads": [72001, 72025],
        "requested_phase": args.phase,
        "output": str(args.output),
        "dry_run_writes_output": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in (
        "source",
        "selection",
        "source_runtime_equivalence",
        "output",
        "project",
    ):
        setattr(args, field, getattr(args, field).resolve())
    validate_environment(args)
    # This full replay uses only already exposed seed 60001 and must complete
    # before source/design replay, output creation, or any fresh evaluation.
    source_runtime_equivalence = load_source_runtime_equivalence(args)
    source = source_runner.audit_source_screen(args.source)
    selection = load_design_selection(args.selection, source)
    spec = build_spec(args, source, selection, source_runtime_equivalence)
    if args.dry_run:
        print(json.dumps(_dry_run_summary(args, spec, source, selection), indent=2))
        return 0

    preliminary_validation: dict[str, Any] | None = None
    if args.phase == "test":
        # This guard must finish before mkdir, plan, invocation, or shard writes.
        preliminary_validation = load_validation_freeze(
            args, spec, source, selection
        )

    args.output.mkdir(parents=True, exist_ok=True)
    source_runner.ensure_immutable_json(
        args.output / "adaptive_shield_preflight_spec.json", spec
    )
    jobs = build_jobs(args.phase)
    atomic_write_csv(
        args.output / f"{args.phase}_plan.csv", (job.as_dict() for job in jobs)
    )
    invocation_id = uuid.uuid4().hex
    invocation_path = args.output / "invocations" / f"{invocation_id}.json"
    invocation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "invocation_id": invocation_id,
        "spec_sha256": spec["spec_sha256"],
        "design_selection_sha256": selection["design_selection_sha256"],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "phase": args.phase,
        "device": args.device,
        "started_at_utc": utc_now(),
        "status": "running",
    }
    atomic_write_json(invocation_path, invocation)
    try:
        with source_runner.invocation_lock(args.output):
            validation = preliminary_validation
            if args.phase == "test":
                validation = load_validation_freeze(args, spec, source, selection)
            validation_sha = (
                validation["validation_freeze_sha256"]
                if validation is not None
                else None
            )
            rows = evaluate_jobs(
                args,
                jobs,
                spec,
                source["checkpoints"],
                _workloads_for_phase(args.phase),
                selection,
                validation_freeze_sha256=validation_sha,
            )
            decision = write_phase_artifacts(
                args.output,
                args.phase,
                spec,
                source,
                selection,
                rows,
                validation_freeze_sha256=validation_sha,
            )
            manifest = None
            if args.phase == "test":
                if validation is None:
                    raise RuntimeError("test phase lost its validation freeze")
                manifest = write_manifest(
                    args.output, spec, source, selection, validation, decision
                )
            passed = bool(decision["all_cells_pass"])
            decision_field = (
                "validation_freeze_sha256"
                if args.phase == "validation"
                else "test_decision_sha256"
            )
            invocation.update(
                status="completed" if passed else "completed_gate_failed",
                finished_at_utc=utc_now(),
                decision=decision["decision"],
                decision_sha256=decision[decision_field],
                manifest_sha256=(
                    manifest["manifest_sha256"] if manifest is not None else None
                ),
            )
            atomic_write_json(invocation_path, invocation)
            print(
                f"{args.phase} {'passed' if passed else 'failed'}: {args.output}",
                flush=True,
            )
            if passed:
                return 0
            return 2 if args.phase == "validation" else 3
    except BaseException as error:
        invocation.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            finished_at_utc=utc_now(),
            error=repr(error),
        )
        atomic_write_json(invocation_path, invocation)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
