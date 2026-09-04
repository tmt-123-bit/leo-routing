"""Design a calm-flow selective shield on the exposed v7 panel."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ablation_matrix_runner import atomic_write_csv, atomic_write_json, sha256_file
from calm_selective_hysteresis_policy import (
    load_calm_selective_hysteresis_policy,
)
from leo_multiagent_env import (
    BASE_CANDIDATE_FEATURE_NAMES,
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
    ROUTE_SWITCH_FEATURE_INDEX,
    candidate_feature_schema,
)
from mappo_evaluation import EpisodeMetrics, evaluate_policy
import run_adaptive_shield_design as v8
import run_hysteresis_screen as source_runner
import run_proposed_shield_preflight as v7


SCREEN_NAME = "CALM-SELECTIVE-SHIELD-DESIGN-v9"
SCHEMA_VERSION = 1
SCENARIOS = v7.SCENARIOS
POLICY_SEEDS = v7.POLICY_SEEDS
DESIGN_WORKLOAD_SEEDS = v7.VALIDATION_WORKLOAD_SEEDS
FIXED_STAY_BONUS = 1.0
FIXED_CLASS_2_RELIEF = 0.75
CALM_AGE_CUTOFF = 0.5
CANDIDATE_PARAMETER_PAIRS = (
    (0.500, 0.000),
    (0.625, 0.000),
    (0.750, 0.000),
    (0.625, 0.250),
    (0.625, 0.500),
    (0.625, 0.750),
    (0.750, 0.250),
    (0.750, 0.500),
    (0.750, 0.750),
)
V8_CONTROL_CANDIDATE_MAP = {0: 12, 2: 14}
DESIGN_GRID = tuple(
    {
        "candidate_index": index,
        "stay_bonus": FIXED_STAY_BONUS,
        "urgency_relief": urgency_relief,
        "class_2_relief": FIXED_CLASS_2_RELIEF,
        "calm_bonus": calm_bonus,
    }
    for index, (urgency_relief, calm_bonus) in enumerate(
        CANDIDATE_PARAMETER_PAIRS
    )
)
CELL_GATES = dict(v7.CELL_GATES)
RAW_SWITCH_HEADROOM_BUFFER = 2.0
AVOIDABLE_MICRO_IMPROVEMENT_BUFFER = 0.005
DELIVERY_SLACK_BUFFER = 1.0 / 1920.0
SELECTION_RULE_ID = (
    "max_worst_delivery_then_class2_then_raw_switch_then_avoidable_"
    "then_min_calm_bonus_max_urgency_index_v1"
)
EXPECTED_FEATURE_SCHEMA = candidate_feature_schema()
EXPECTED_FEATURE_NAMES = tuple(BASE_CANDIDATE_FEATURE_NAMES)
REFERENCE_POLICIES = dict(v8.REFERENCE_POLICIES)
BASELINE_FINGERPRINT_FILES = dict(v8.BASELINE_FINGERPRINT_FILES)
EXPECTED_V9_SELECTION_SHA256 = (
    "b58aaf3ad17c07cf10dd6f2d03a6f12d5917cbb24ad604a18999d21db9520c8d"
)
EXPECTED_V9_SELECTION_FILE_SHA256 = (
    "72c9ef6ba1d955ac5c6ee467b55cfec8a5cce657b6d0c89f9d88cdc6ac6616fc"
)
EXPECTED_V9_SPEC_SHA256 = (
    "2ccef8703e597da5daef2f996c2bac9acf1d7cc25871cd7d040e0f6a4acb3a36"
)
EXPECTED_V9_SPEC_FILE_SHA256 = (
    "fdd6e33b206778beb045af1a5277663a5d1160c22bf2ed31252b97f420a80802"
)
HISTORICAL_V9_RUNNER_FINGERPRINTS = {
    "runner": "d66ec43e5d9905ad2a9b45805dcf960659a0aa9ec9ea29fd9546f6bd71da550c",
    "v8_runner": "440068283d9b81b6ef7ff142613c61472d2f8cc9b3422dcf2b345338780c6872",
}


def _token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def candidate_id(candidate: Mapping[str, Any]) -> str:
    return "_".join(
        (
            f"b{_token(float(candidate['stay_bonus']))}",
            f"u{_token(float(candidate['urgency_relief']))}",
            f"c{_token(float(candidate['class_2_relief']))}",
            f"k{_token(float(candidate['calm_bonus']))}",
        )
    )


def candidate_policy_name(candidate: Mapping[str, Any]) -> str:
    return f"mappo_proposed_calm_selective_{candidate_id(candidate)}_design_v9"


@dataclass(frozen=True)
class DesignJob:
    index: int
    scenario: str
    policy_seed: int
    candidate_index: int
    stay_bonus: float
    urgency_relief: float
    class_2_relief: float
    calm_bonus: float

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
            "calm_bonus": self.calm_bonus,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "phase": self.phase,
            "source_variant": self.source_variant,
            "source_job_id": self.source_job_id,
            "policy_name": self.policy_name,
            "job_id": self.job_id,
            "execution_source": (
                "v8_row_reuse"
                if self.candidate_index in V8_CONTROL_CANDIDATE_MAP
                else "new_v9_evaluation"
            ),
            "v8_source_candidate_index": V8_CONTROL_CANDIDATE_MAP.get(
                self.candidate_index
            ),
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
                        calm_bonus=float(candidate["calm_bonus"]),
                    )
                )
    return jobs


def build_new_evaluation_jobs() -> list[DesignJob]:
    return [
        job
        for job in build_design_jobs()
        if job.candidate_index not in V8_CONTROL_CANDIDATE_MAP
    ]


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
        "calm_selective_policy": project / "calm_selective_hysteresis_policy.py",
        "adaptive_policy": project / "adaptive_hysteresis_policy.py",
        "raw_hysteresis_policy": project / "hysteresis_policy.py",
        **{
            name: project / filename
            for name, filename in BASELINE_FINGERPRINT_FILES.items()
        },
        "source_auditor": project / "run_hysteresis_screen.py",
        "v7_runner": project / "run_proposed_shield_preflight.py",
        "v8_runner": project / "run_adaptive_shield_design.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v9 fingerprint files missing: {missing}")
    modules = {
        "calm_selective_hysteresis_policy": "calm_selective_hysteresis_policy.py",
        "adaptive_hysteresis_policy": "adaptive_hysteresis_policy.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "run_adaptive_shield_design": "run_adaptive_shield_design.py",
        "run_hysteresis_screen": "run_hysteresis_screen.py",
        "run_proposed_shield_preflight": "run_proposed_shield_preflight.py",
    }
    for module_name, filename in modules.items():
        actual = Path(inspect.getfile(importlib.import_module(module_name))).resolve()
        expected = (project / filename).resolve()
        if actual != expected:
            raise RuntimeError(f"runtime module path mismatch: {module_name}")
    return {name: sha256_file(path) for name, path in files.items()}


def audit_v7_baseline(
    v7_output: Path,
    source: Mapping[str, Any],
    project: Path,
) -> dict[str, Any]:
    return v8.audit_v7_baseline(v7_output, source, project)


def audit_v8_failure(v8_output: Path) -> dict[str, Any]:
    selection_path = v8_output / "design_selection.json"
    selection = v8.load_design_selection(
        selection_path,
        allow_historical_runtime=True,
    )
    if selection.get("selection_status") != "no_eligible_candidate":
        raise ValueError("v9 requires the recorded failed v8 design selection")
    if selection.get("eligible_candidate_indices") != []:
        raise ValueError("v8 failure unexpectedly contains an eligible candidate")
    if selection.get("selected_candidate_index") is not None:
        raise ValueError("v8 failure unexpectedly selected a candidate")
    if selection.get("selected_parameters") is not None:
        raise ValueError("v8 failure unexpectedly selected parameters")
    if selection.get("all_cells_pass") is not False:
        raise ValueError("v8 failure all-cells status changed")
    if selection.get("fresh_validation_allowed") is not False:
        raise ValueError("v8 failure unexpectedly allows fresh validation")
    if selection.get("test_allowed") is not False:
        raise ValueError("v8 failure unexpectedly allows test evaluation")
    return {
        "screen_name": selection["screen_name"],
        "selection_status": selection["selection_status"],
        "design_spec_sha256": selection["design_spec_sha256"],
        "design_selection_sha256": selection["design_selection_sha256"],
        "selection_file_path": str(selection_path.resolve()),
        "selection_file_sha256": sha256_file(selection_path),
        "v7_validation_freeze_sha256": selection[
            "v7_validation_freeze_sha256"
        ],
    }


def audit_v8_control_reuse(
    v8_output: Path,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[EpisodeMetrics]]:
    failure = audit_v8_failure(v8_output)
    spec_path = v8_output / "adaptive_shield_design_spec.json"
    episode_path = v8_output / "design_episode_metrics.csv"
    spec = source_runner._load_json(spec_path)
    v8._self_hash(spec, "design_spec_sha256")
    if spec["design_spec_sha256"] != failure["design_spec_sha256"]:
        raise ValueError("v8 control source spec does not match failed selection")
    combined_rows = source_runner._episode_rows_from_csv(episode_path)
    reused_rows: list[EpisodeMetrics] = []
    provenance: list[dict[str, Any]] = []
    v8_jobs = v8.build_design_jobs()
    for v9_index, v8_index in V8_CONTROL_CANDIDATE_MAP.items():
        target = DESIGN_GRID[v9_index]
        source_candidate = v8.DESIGN_GRID[v8_index]
        if (
            float(target["stay_bonus"])
            != float(source_candidate["stay_bonus"])
            or float(target["urgency_relief"])
            != float(source_candidate["urgency_relief"])
            or float(target["class_2_relief"])
            != float(source_candidate["class_2_relief"])
            or float(target["calm_bonus"]) != 0.0
        ):
            raise RuntimeError("v8 control parameter mapping changed")
        source_policy = v8.candidate_policy_name(source_candidate)
        target_policy = candidate_policy_name(target)
        jobs = [job for job in v8_jobs if job.candidate_index == v8_index]
        if len(jobs) != len(SCENARIOS) * len(POLICY_SEEDS):
            raise ValueError(f"v8 control candidate {v8_index} shard grid mismatch")
        for job in jobs:
            csv_path, metadata_path = v8._evaluation_paths(v8_output, job)
            if not csv_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f"v8 control reuse shard is missing: {job.job_id}"
                )
            checkpoint = source["checkpoints"][job.source_job_id]
            expected = v8._expected_shard_metadata(job, spec, checkpoint)
            metadata = source_runner._load_json(metadata_path)
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ValueError(
                        f"v8 control shard metadata mismatch: {job.job_id}/{field}"
                    )
            if metadata.get("csv_sha256") != sha256_file(csv_path):
                raise ValueError(f"v8 control shard hash mismatch: {job.job_id}")
            if metadata.get("row_count") != len(DESIGN_WORKLOAD_SEEDS):
                raise ValueError(
                    f"v8 control shard row count mismatch: {job.job_id}"
                )
            if not isinstance(metadata.get("policy_diagnostics"), Mapping):
                raise ValueError(
                    f"v8 control shard diagnostics missing: {job.job_id}"
                )
            shard_rows = source_runner.validate_evaluation_shard(
                csv_path, job, DESIGN_WORKLOAD_SEEDS
            )
            v7.validate_switch_rows(shard_rows)
            combined_cell = [
                row
                for row in combined_rows
                if row.scenario == job.scenario
                and row.policy == source_policy
                and row.policy_seed == job.policy_seed
            ]
            if shard_rows != combined_cell:
                raise ValueError(
                    f"v8 control shard does not match combined rows: {job.job_id}"
                )
            reused_rows.extend(
                replace(row, policy=target_policy) for row in shard_rows
            )
            provenance.append(
                {
                    "v9_candidate_index": v9_index,
                    "v8_candidate_index": v8_index,
                    "scenario": job.scenario,
                    "policy_seed": job.policy_seed,
                    "source_policy_name": source_policy,
                    "target_policy_name": target_policy,
                    "source_csv_path": str(csv_path.resolve()),
                    "source_csv_sha256": sha256_file(csv_path),
                    "source_metadata_path": str(metadata_path.resolve()),
                    "source_metadata_sha256": sha256_file(metadata_path),
                    "row_count": len(shard_rows),
                }
            )
    expected_rows = (
        len(V8_CONTROL_CANDIDATE_MAP)
        * len(SCENARIOS)
        * len(POLICY_SEEDS)
        * len(DESIGN_WORKLOAD_SEEDS)
    )
    if len(reused_rows) != expected_rows:
        raise ValueError("v8 control reuse row grid is incomplete")
    record: dict[str, Any] = {
        "v8_design_selection_sha256": failure["design_selection_sha256"],
        "control_candidate_map": {
            str(key): value for key, value in V8_CONTROL_CANDIDATE_MAP.items()
        },
        "source_episode_metrics_path": str(episode_path.resolve()),
        "source_episode_metrics_sha256": sha256_file(episode_path),
        "logical_control_jobs": len(provenance),
        "reused_row_count": len(reused_rows),
        "source_shards": provenance,
    }
    record["reuse_sha256"] = source_runner.sha256_json(record)
    return record, reused_rows


def build_spec(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    baseline: Mapping[str, Any],
    v8_failure: Mapping[str, Any],
    v8_control_reuse: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = args.project.parent / "docs" / "CALM_SELECTIVE_SHIELD_DESIGN_V9.md"
    fingerprint = _runtime_code_fingerprint(args.project, protocol)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exposed_panel_engineering_design",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "fresh_panels_allowed": False,
        "training_jobs": 0,
        "design_workload_seeds": list(DESIGN_WORKLOAD_SEEDS),
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "candidate_grid": [dict(candidate) for candidate in DESIGN_GRID],
        "candidate_count": len(DESIGN_GRID),
        "fixed_parameters": {
            "stay_bonus": FIXED_STAY_BONUS,
            "class_2_relief": FIXED_CLASS_2_RELIEF,
            "calm_age_cutoff": CALM_AGE_CUTOFF,
        },
        "formula": (
            "beta_eff=(1-u*a)*(1-0.75*z)+k*(1-z)*max(0,1-2*a); "
            "a=clip(deadline_normalized_waiting_age,0,1); z=traffic_class_2"
        ),
        "cell_gates": dict(CELL_GATES),
        "design_buffers": {
            "raw_context_switch_headroom_min": RAW_SWITCH_HEADROOM_BUFFER,
            "avoidable_micro_improvement_min": (
                AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
            ),
            "delivery_slack_min": DELIVERY_SLACK_BUFFER,
            "delivery_slack_basis": "one_medium_load_packet_over_1920",
            "class_2_extra_buffer": 0.0,
        },
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_key_fields": [
            "worst_delivery_slack",
            "worst_class_2_slack",
            "worst_raw_switch_slack",
            "worst_avoidable_slack",
            "negative_calm_bonus",
            "urgency_relief",
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
        "v8_failure_selection": dict(v8_failure),
        "v8_control_reuse": dict(v8_control_reuse),
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
            "v8_output": str(args.v8_output.resolve()),
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
        "v8_design_selection_sha256": spec["v8_failure_selection"][
            "design_selection_sha256"
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
    if job.candidate_index in V8_CONTROL_CANDIDATE_MAP:
        raise RuntimeError("v8 control candidates must be reused, not evaluated")
    csv_path, metadata_path = _evaluation_paths(args.output, job)
    expected = _expected_shard_metadata(job, spec, checkpoint)
    if csv_path.exists() or metadata_path.exists():
        if not csv_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"partial design shard exists: {job.job_id}")
        metadata = source_runner._load_json(metadata_path)
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"v9 shard metadata mismatch: {job.job_id}/{field}"
                )
        if metadata.get("csv_sha256") != sha256_file(csv_path):
            raise ValueError(f"v9 shard hash mismatch: {job.job_id}")
        if metadata.get("row_count") != len(DESIGN_WORKLOAD_SEEDS):
            raise ValueError(f"v9 shard row count mismatch: {job.job_id}")
        if not isinstance(metadata.get("policy_diagnostics"), Mapping):
            raise ValueError(f"v9 shard diagnostics missing: {job.job_id}")
        rows = source_runner.validate_evaluation_shard(
            csv_path, job, DESIGN_WORKLOAD_SEEDS
        )
        v7.validate_switch_rows(rows)
        return rows
    checkpoint_path = Path(str(checkpoint["checkpoint_path"]))
    if sha256_file(checkpoint_path) != checkpoint["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash changed: {job.source_job_id}")
    policy, _ = load_calm_selective_hysteresis_policy(
        checkpoint_path,
        **job.as_parameters(),
        device=args.device,
    )
    if policy.checkpoint_schema != expected["policy_schema"]:
        raise ValueError(f"loaded v9 policy schema mismatch: {job.job_id}")
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
    jobs = build_new_evaluation_jobs()
    for completed, job in enumerate(jobs, start=1):
        rows.extend(evaluate_job(args, job, spec, checkpoints[job.source_job_id]))
        print(f"[{completed}/{len(jobs)}] completed {job.job_id}", flush=True)
    expected = len(jobs) * len(DESIGN_WORKLOAD_SEEDS)
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError("v9 new-evaluation grid is incomplete or duplicated")
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
        shield_policy = candidate_policy_name(candidate)
        for scenario in SCENARIOS:
            for policy_seed in POLICY_SEEDS:
                proposed_rows = _cell_rows(
                    rows,
                    scenario,
                    policy_seed,
                    REFERENCE_POLICIES["proposed"],
                )
                raw_rows = _cell_rows(
                    rows,
                    scenario,
                    policy_seed,
                    REFERENCE_POLICIES["raw_context"],
                )
                shield_rows = _cell_rows(
                    rows, scenario, policy_seed, shield_policy
                )
                delivery_delta = _mean(
                    shield_rows, "delivery_ratio"
                ) - _mean(proposed_rows, "delivery_ratio")
                class_2_delta = _mean(
                    shield_rows, "class_2_delivery_ratio"
                ) - _mean(proposed_rows, "class_2_delivery_ratio")
                proposed_avoidable, proposed_opportunities, proposed_micro = (
                    _micro(proposed_rows)
                )
                shield_avoidable, shield_opportunities, shield_micro = _micro(
                    shield_rows
                )
                proposed_switch = _mean(proposed_rows, "routing_switches")
                raw_switch = _mean(raw_rows, "routing_switches")
                shield_switch = _mean(shield_rows, "routing_switches")
                delivery_slack = delivery_delta - float(
                    CELL_GATES["delivery_difference_vs_proposed_min"]
                )
                class_2_slack = class_2_delta - float(
                    CELL_GATES[
                        "class_2_delivery_difference_vs_proposed_min"
                    ]
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
                delivery_buffer_pass = (
                    delivery_slack >= DELIVERY_SLACK_BUFFER
                )
                buffer_pass = (
                    raw_switch_slack >= RAW_SWITCH_HEADROOM_BUFFER
                    and avoidable_slack >= AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
                    and delivery_buffer_pass
                )
                gates.append(
                    {
                        "candidate_index": int(candidate["candidate_index"]),
                        "candidate_id": candidate_id(candidate),
                        "stay_bonus": float(candidate["stay_bonus"]),
                        "urgency_relief": float(candidate["urgency_relief"]),
                        "class_2_relief": float(candidate["class_2_relief"]),
                        "calm_bonus": float(candidate["calm_bonus"]),
                        "scenario": scenario,
                        "policy_seed": policy_seed,
                        "workload_count": len(DESIGN_WORKLOAD_SEEDS),
                        "delivery_difference_vs_proposed": delivery_delta,
                        "class_2_delivery_difference_vs_proposed": class_2_delta,
                        "proposed_avoidable_switches_total": proposed_avoidable,
                        "proposed_switch_opportunities_total": (
                            proposed_opportunities
                        ),
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
                        "switch_vs_proposed_gate_pass": (
                            proposed_switch_slack >= 0.0
                        ),
                        "switch_vs_raw_gate_pass": raw_switch_slack >= 0.0,
                        "five_gates_pass": five_gate_pass,
                        "raw_switch_buffer_pass": (
                            raw_switch_slack >= RAW_SWITCH_HEADROOM_BUFFER
                        ),
                        "avoidable_buffer_pass": (
                            avoidable_slack
                            >= AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
                        ),
                        "delivery_buffer_pass": delivery_buffer_pass,
                        "buffer_gate_pass": buffer_pass,
                        "cell_eligible": five_gate_pass and buffer_pass,
                    }
                )
    expected = len(DESIGN_GRID) * len(SCENARIOS) * len(POLICY_SEEDS)
    if len(gates) != expected:
        raise RuntimeError("v9 gate grid has the wrong size")
    return gates


def select_candidate(
    gates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    results: list[dict[str, Any]] = []
    for candidate in DESIGN_GRID:
        index = int(candidate["candidate_index"])
        cells = [
            dict(row)
            for row in gates
            if int(row["candidate_index"]) == index
        ]
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
            -float(candidate["calm_bonus"]),
            float(candidate["urgency_relief"]),
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
                "delivery_buffer_all_cells_pass": all(
                    bool(row["delivery_buffer_pass"]) for row in cells
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
    selected = (
        max(eligible, key=lambda result: tuple(result["selection_key"]))
        if eligible
        else None
    )
    if selected is not None:
        tied = [
            result
            for result in eligible
            if result["selection_key"] == selected["selection_key"]
        ]
        if len(tied) != 1:
            raise RuntimeError("v9 selection argmax is not unique")
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
    aggregate = source_runner.aggregate_rows(rows, rng_base=79000)
    atomic_write_csv(episode_path, (asdict(row) for row in rows))
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(gates_path, gates)
    selected_parameters = (
        {
            "stay_bonus": selected["stay_bonus"],
            "urgency_relief": selected["urgency_relief"],
            "class_2_relief": selected["class_2_relief"],
            "calm_bonus": selected["calm_bonus"],
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
        "v8_design_selection_sha256": spec["v8_failure_selection"][
            "design_selection_sha256"
        ],
        "selection_rule_id": SELECTION_RULE_ID,
        "candidate_grid": [dict(candidate) for candidate in DESIGN_GRID],
        "candidate_results": results,
        "eligible_candidate_indices": [
            int(result["candidate_index"])
            for result in results
            if result["eligible"]
        ],
        "selected_candidate_index": (
            int(selected["candidate_index"]) if selected is not None else None
        ),
        "selected_parameters": selected_parameters,
        "all_cells_pass": selected is not None,
        "selection_status": (
            "selected_exposed_design_candidate"
            if selected is not None
            else "no_eligible_candidate"
        ),
        "fresh_validation_allowed": False,
        "fresh_test_allowed": False,
        "test_allowed": False,
        "artifacts": {
            "design_spec": _artifact(
                output / "calm_selective_shield_design_spec.json", 1
            ),
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
        raise ValueError(f"v9 artifact path mismatch: {expected_path.name}")
    if artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"v9 artifact hash mismatch: {expected_path.name}")
    declared_rows = artifact.get("row_count")
    if type(declared_rows) is not int or declared_rows <= 0:
        raise ValueError(f"v9 artifact row count invalid: {expected_path.name}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            observed_rows = sum(1 for _ in csv.DictReader(handle))
        if observed_rows != declared_rows:
            raise ValueError(
                f"v9 artifact row count mismatch: {expected_path.name}"
            )
    elif declared_rows != 1:
        raise ValueError(f"v9 JSON artifact row count mismatch: {expected_path.name}")


def _csv_rows_match(path: Path, expected: Sequence[Mapping[str, Any]]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        observed = list(csv.DictReader(handle))
    rendered = [{key: str(value) for key, value in row.items()} for row in expected]
    if observed != rendered:
        raise ValueError(f"v9 CSV does not replay: {path.name}")


def load_design_selection(
    path: Path,
    *,
    allow_historical_runtime: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    if (
        allow_historical_runtime
        and sha256_file(path) != EXPECTED_V9_SELECTION_FILE_SHA256
    ):
        raise ValueError("frozen v9 design selection changed")
    selection = source_runner._load_json(path)
    selection_sha256 = _self_hash(selection, "design_selection_sha256")
    if allow_historical_runtime and selection_sha256 != EXPECTED_V9_SELECTION_SHA256:
        raise ValueError("unexpected v9 design selection")
    if selection.get("screen_name") != SCREEN_NAME:
        raise ValueError("v9 selection screen mismatch")
    if selection.get("selection_rule_id") != SELECTION_RULE_ID:
        raise ValueError("v9 selection rule mismatch")
    for field in (
        "confirmatory",
        "paper_claim_allowed",
        "promotion_decision_allowed",
        "fresh_validation_allowed",
        "fresh_test_allowed",
        "test_allowed",
    ):
        if selection.get(field) is not False:
            raise ValueError(f"v9 selection must set {field}=false")
    if selection.get("candidate_grid") != [
        dict(candidate) for candidate in DESIGN_GRID
    ]:
        raise ValueError("v9 selection candidate grid mismatch")
    output = path.parent
    artifacts = selection.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("v9 selection artifact mapping missing")
    expected_paths = {
        "design_spec": output / "calm_selective_shield_design_spec.json",
        "episode_metrics": output / "design_episode_metrics.csv",
        "aggregate_metrics": output / "design_aggregate_metrics.csv",
        "cell_gates": output / "design_cell_gates.csv",
    }
    if set(artifacts) != set(expected_paths):
        raise ValueError("v9 selection artifact set mismatch")
    for name, expected_path in expected_paths.items():
        _validate_artifact(artifacts[name], expected_path)
    spec_path = expected_paths["design_spec"]
    if (
        allow_historical_runtime
        and sha256_file(spec_path) != EXPECTED_V9_SPEC_FILE_SHA256
    ):
        raise ValueError("frozen v9 design specification changed")
    spec = source_runner._load_json(spec_path)
    spec_sha256 = _self_hash(spec, "design_spec_sha256")
    if allow_historical_runtime and spec_sha256 != EXPECTED_V9_SPEC_SHA256:
        raise ValueError("unexpected v9 design specification")
    if selection.get("design_spec_sha256") != spec["design_spec_sha256"]:
        raise ValueError("v9 selection spec mismatch")
    project = Path(str(spec["paths"]["project"])).resolve()
    protocol = Path(str(spec["paths"]["protocol"])).resolve()
    fingerprint = _runtime_code_fingerprint(project, protocol)
    if allow_historical_runtime:
        source = source_runner.audit_source_screen(
            Path(str(spec["paths"]["source"]))
        )
        v8._audit_historical_runtime_fingerprint(
            project,
            source,
            spec.get("runtime_code_fingerprint"),
            fingerprint,
            HISTORICAL_V9_RUNNER_FINGERPRINTS,
        )
    elif fingerprint != spec.get("runtime_code_fingerprint"):
        raise ValueError("v9 runtime code fingerprint changed")
    else:
        source = source_runner.audit_source_screen(
            Path(str(spec["paths"]["source"]))
        )
    baseline = audit_v7_baseline(
        Path(str(spec["paths"]["v7_output"])), source, project
    )
    v8_failure = audit_v8_failure(Path(str(spec["paths"]["v8_output"])))
    v8_control_reuse, reused_control_rows = audit_v8_control_reuse(
        Path(str(spec["paths"]["v8_output"])), source
    )
    if selection.get("source_training_freeze_sha256") != source[
        "training_freeze_sha256"
    ]:
        raise ValueError("v9 source freeze mismatch")
    if selection.get("v7_validation_freeze_sha256") != baseline[
        "validation_freeze_sha256"
    ]:
        raise ValueError("v9 v7 freeze mismatch")
    if spec.get("v8_failure_selection") != v8_failure:
        raise ValueError("v9 spec v8 failure binding changed")
    if spec.get("v8_control_reuse") != v8_control_reuse:
        raise ValueError("v9 spec v8 control-reuse binding changed")
    if selection.get("v8_design_selection_sha256") != v8_failure[
        "design_selection_sha256"
    ]:
        raise ValueError("v9 selection v8 failure binding changed")
    rows = source_runner._episode_rows_from_csv(expected_paths["episode_metrics"])
    expected_count = (
        len(baseline["references"])
        + len(build_design_jobs()) * len(DESIGN_WORKLOAD_SEEDS)
    )
    if len(rows) != expected_count:
        raise ValueError("v9 episode artifact row count mismatch")
    replayed_references = [
        row for row in rows if row.policy in set(REFERENCE_POLICIES.values())
    ]
    if replayed_references != baseline["references"]:
        raise ValueError("v9 v7 reference rows do not replay exactly")
    control_policies = {
        candidate_policy_name(DESIGN_GRID[index])
        for index in V8_CONTROL_CANDIDATE_MAP
    }
    replayed_controls = [row for row in rows if row.policy in control_policies]
    if replayed_controls != reused_control_rows:
        raise ValueError("v9 reused v8 control rows do not replay exactly")
    gates = compute_candidate_cell_gates(rows)
    _csv_rows_match(expected_paths["cell_gates"], gates)
    results, selected = select_candidate(gates)
    if selection.get("candidate_results") != results:
        raise ValueError("v9 candidate results do not replay")
    eligible_indices = [
        int(result["candidate_index"])
        for result in results
        if result["eligible"]
    ]
    if selection.get("eligible_candidate_indices") != eligible_indices:
        raise ValueError("v9 eligible set does not replay")
    selected_index = int(selected["candidate_index"]) if selected else None
    if selection.get("selected_candidate_index") != selected_index:
        raise ValueError("v9 unique argmax does not replay")
    expected_parameters = (
        {
            key: selected[key]
            for key in (
                "stay_bonus",
                "urgency_relief",
                "class_2_relief",
                "calm_bonus",
            )
        }
        if selected is not None
        else None
    )
    if selection.get("selected_parameters") != expected_parameters:
        raise ValueError("v9 selected parameters do not replay")
    if selection.get("all_cells_pass") is not (selected is not None):
        raise ValueError("v9 all-cells status does not replay")
    expected_status = (
        "selected_exposed_design_candidate"
        if selected is not None
        else "no_eligible_candidate"
    )
    if selection.get("selection_status") != expected_status:
        raise ValueError("v9 selection status label does not replay")
    return selection


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.source, args.v7_output, args.v8_output, args.project):
        if not path.is_dir():
            raise FileNotFoundError(path)
    root = args.project.parent.resolve()
    broad = {
        root,
        args.project.resolve(),
        (root / "experiments").resolve(),
        args.source.resolve(),
        args.v7_output.resolve(),
        args.v8_output.resolve(),
    }
    if args.output.resolve() in broad:
        raise RuntimeError("output must be a dedicated v9 design directory")
    if (
        len(DESIGN_GRID) != 9
        or len(build_design_jobs()) != 72
        or len(build_new_evaluation_jobs()) != 56
    ):
        raise RuntimeError("frozen v9 design grid changed")
    if DESIGN_GRID[0] != {
        "candidate_index": 0,
        "stay_bonus": 1.0,
        "urgency_relief": 0.5,
        "class_2_relief": 0.75,
        "calm_bonus": 0.0,
    }:
        raise RuntimeError("v9 anchor candidate changed")
    if EXPECTED_FEATURE_SCHEMA.get("schema_id") != (
        "leo_multi_candidate_features_v1_dim_26"
    ):
        raise RuntimeError("candidate feature schema changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Design a calm-selective shield on the exposed v7 panel."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root
        / "experiments"
        / "archive"
        / "congestion-context-screen-20k-v1",
    )
    parser.add_argument(
        "--v7-output",
        type=Path,
        default=root / "experiments" / "proposed-shield-preflight-v7",
    )
    parser.add_argument(
        "--v8-output",
        type=Path,
        default=root / "experiments" / "adaptive-shield-design-v8",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "calm-selective-shield-design-v9",
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in ("source", "v7_output", "v8_output", "output", "project"):
        setattr(args, field, getattr(args, field).resolve())
    validate_environment(args)
    source = source_runner.audit_source_screen(args.source)
    baseline = audit_v7_baseline(args.v7_output, source, args.project)
    v8_failure = audit_v8_failure(args.v8_output)
    v8_control_reuse, reused_control_rows = audit_v8_control_reuse(
        args.v8_output, source
    )
    if v8_failure["v7_validation_freeze_sha256"] != baseline[
        "validation_freeze_sha256"
    ]:
        raise ValueError("v8 failure and v7 reference freeze do not match")
    spec = build_spec(
        args, source, baseline, v8_failure, v8_control_reuse
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "screen_name": SCREEN_NAME,
                    "design_spec_sha256": spec["design_spec_sha256"],
                    "v8_design_selection_sha256": v8_failure[
                        "design_selection_sha256"
                    ],
                    "candidate_count": len(DESIGN_GRID),
                    "logical_design_jobs": len(build_design_jobs()),
                    "new_evaluation_jobs": len(build_new_evaluation_jobs()),
                    "new_candidate_rows": len(build_new_evaluation_jobs())
                    * len(DESIGN_WORKLOAD_SEEDS),
                    "reused_control_rows": len(reused_control_rows),
                    "total_candidate_rows": len(build_design_jobs())
                    * len(DESIGN_WORKLOAD_SEEDS),
                    "reference_rows": baseline["reference_row_count"],
                    "workloads": [
                        DESIGN_WORKLOAD_SEEDS[0],
                        DESIGN_WORKLOAD_SEEDS[-1],
                    ],
                    "selection_rule_id": SELECTION_RULE_ID,
                    "fresh_panels_allowed": False,
                    "dry_run_writes_output": False,
                },
                indent=2,
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    source_runner.ensure_immutable_json(
        args.output / "calm_selective_shield_design_spec.json", spec
    )
    atomic_write_csv(
        args.output / "design_plan.csv",
        (job.as_dict() for job in build_design_jobs()),
    )
    with source_runner.invocation_lock(args.output):
        candidate_rows = evaluate_jobs(args, spec, source["checkpoints"])
        combined = [
            *baseline["references"],
            *reused_control_rows,
            *candidate_rows,
        ]
        selection = write_design_artifacts(args.output, spec, source, combined)
        load_design_selection(args.output / "design_selection.json")
    print(json.dumps(selection, indent=2))
    return 0 if selection["all_cells_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
