"""Design the frozen age-band shield on the exposed v7 workload panel."""

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
from age_band_hysteresis_policy import (
    ADDON_STAY_BONUS,
    BASE_CLASS_2_RELIEF,
    BASE_STAY_BONUS,
    BASE_URGENCY_RELIEF,
    load_age_band_hysteresis_policy,
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
import run_margin_capped_shield_design as v10
import run_proposed_shield_preflight as v7
import run_source_runtime_equivalence_audit as runtime_equivalence


SCREEN_NAME = "AGE-BAND-SHIELD-DESIGN-v11"
SCHEMA_VERSION = 1
SCENARIOS = v7.SCENARIOS
POLICY_SEEDS = v7.POLICY_SEEDS
DESIGN_WORKLOAD_SEEDS = v7.VALIDATION_WORKLOAD_SEEDS
FIXED_STAY_BONUS = 1.0
FIXED_URGENCY_RELIEF = 0.625
FIXED_CLASS_2_RELIEF = 0.75
FIXED_CALM_BONUS = 0.75
AGE_BAND_MIXES = (0.0, 1.0)
V10_REUSE_CANDIDATE_INDEX = 0
V9_REUSE_CANDIDATE_INDEX = 5
DESIGN_GRID = tuple(
    {
        "candidate_index": index,
        "stay_bonus": FIXED_STAY_BONUS,
        "urgency_relief": FIXED_URGENCY_RELIEF,
        "class_2_relief": FIXED_CLASS_2_RELIEF,
        "calm_bonus": FIXED_CALM_BONUS,
        "age_band_mix": age_band_mix,
    }
    for index, age_band_mix in enumerate(AGE_BAND_MIXES)
)
CELL_GATES = dict(v7.CELL_GATES)
RAW_SWITCH_HEADROOM_BUFFER = v10.RAW_SWITCH_HEADROOM_BUFFER
AVOIDABLE_MICRO_IMPROVEMENT_BUFFER = v10.AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
DELIVERY_SLACK_BUFFER = v10.DELIVERY_SLACK_BUFFER
SELECTION_RULE_ID = (
    "max_worst_delivery_buffer_then_class2_then_raw_buffer_then_avoidable_"
    "buffer_then_min_age_band_mix_then_min_candidate_index_v1"
)
EXPECTED_FEATURE_SCHEMA = candidate_feature_schema()
EXPECTED_FEATURE_NAMES = tuple(BASE_CANDIDATE_FEATURE_NAMES)
REFERENCE_POLICIES = dict(v10.REFERENCE_POLICIES)
BASELINE_FINGERPRINT_FILES = dict(v10.BASELINE_FINGERPRINT_FILES)
SOURCE_RUNTIME_EQUIVALENCE_FILENAME = "source_runtime_equivalence.json"

EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_SHA256 = (
    "494fb276939e8e74bc1af38e9425f32347ad43c0719e6e3738877fd5130802a9"
)
EXPECTED_V10_SELECTION_SHA256 = (
    "1f51e4a0bdfab0381e224403193a38783b1e54b6b550a3f6450d8754a05765f2"
)
EXPECTED_V10_SELECTION_FILE_SHA256 = (
    "0302875cc4cc34dd3c6001acfb3542fc8fe75df448ca18846f97a46538ad1c43"
)
EXPECTED_V10_SPEC_SHA256 = (
    "fca6738ec4303dd0da4d87d28620871e736e352c866cfa683bda7d7902262906"
)
EXPECTED_V10_SPEC_FILE_SHA256 = (
    "6a4ceb2c48dd46bff61da3fdbe9003cd47be564c1f503e357b464917e2e7eabe"
)
EXPECTED_AGE_BAND_POLICY_SHA256 = (
    "7c54287824a87db3274ba63f7947426484c3b156a4833c3767a1908a9cdc36c9"
)
EXPECTED_PROTOCOL_SHA256 = (
    "9d229818c56b7964e0795d65a4eb7256a28db26baef974530bdb6c4749683a98"
)
EXPECTED_V11_SELECTION_SHA256 = (
    "133f2ab88b5fc5291b29b92f1cb61ebf9512e71450a9b82a431a916c1d6814c5"
)
EXPECTED_V11_SELECTION_FILE_SHA256 = (
    "ac338b5967b985c2efafa4d728c1350d54e45393308213267e3af663323ed19c"
)
EXPECTED_V11_SPEC_SHA256 = (
    "14cede5336b7b2ed3b63275fa85a95e6448bb979d251cee7486d016471e4cb22"
)
EXPECTED_V11_SPEC_FILE_SHA256 = (
    "b033cfb03c68774579dc346551a041d2a3b79515eb80513263192ccd88508c86"
)
HISTORICAL_V11_RUNNER_FINGERPRINTS = {
    "runner": "313e7c62d1974d167f72053dc1e6626336da42b52835081f43e30b7a619c10fb",
    "v8_runner": "440068283d9b81b6ef7ff142613c61472d2f8cc9b3422dcf2b345338780c6872",
    "v9_runner": "d66ec43e5d9905ad2a9b45805dcf960659a0aa9ec9ea29fd9546f6bd71da550c",
    "v10_runner": "22785166023882035728679219b9304af2108d509de220ac9852bc90e5959835",
}

CLOSED_PERMISSION_FIELDS = (
    "paper_claim_allowed",
    "promotion_decision_allowed",
    "fresh_validation_allowed",
    "fresh_test_allowed",
    "test_allowed",
)


def _token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def candidate_id(candidate: Mapping[str, Any]) -> str:
    return "_".join(
        (
            f"b{_token(float(candidate['stay_bonus']))}",
            f"u{_token(float(candidate['urgency_relief']))}",
            f"c{_token(float(candidate['class_2_relief']))}",
            f"k{_token(float(candidate['calm_bonus']))}",
            f"m{_token(float(candidate['age_band_mix']))}",
        )
    )


def candidate_policy_name(candidate: Mapping[str, Any]) -> str:
    return f"mappo_proposed_age_band_{candidate_id(candidate)}_design_v11"


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
    age_band_mix: float

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
            "age_band_mix": self.age_band_mix,
        }

    def as_dict(self) -> dict[str, Any]:
        reused = self.candidate_index == 0
        return {
            **asdict(self),
            "phase": self.phase,
            "source_variant": self.source_variant,
            "source_job_id": self.source_job_id,
            "policy_name": self.policy_name,
            "job_id": self.job_id,
            "execution_source": (
                "v9_candidate_5_row_reuse" if reused else "new_v11_evaluation"
            ),
            "v10_source_candidate_index": (
                V10_REUSE_CANDIDATE_INDEX if reused else None
            ),
            "v9_source_candidate_index": (
                V9_REUSE_CANDIDATE_INDEX if reused else None
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
                        age_band_mix=float(candidate["age_band_mix"]),
                    )
                )
    return jobs


def build_new_evaluation_jobs() -> list[DesignJob]:
    return [job for job in build_design_jobs() if job.candidate_index == 1]


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
        "age_band_policy": project / "age_band_hysteresis_policy.py",
        "adaptive_policy": project / "adaptive_hysteresis_policy.py",
        "raw_hysteresis_policy": project / "hysteresis_policy.py",
        **{
            name: project / filename
            for name, filename in BASELINE_FINGERPRINT_FILES.items()
        },
        "source_auditor": project / "run_hysteresis_screen.py",
        "source_runtime_auditor": (
            project / "run_source_runtime_equivalence_audit.py"
        ),
        "v7_runner": project / "run_proposed_shield_preflight.py",
        "v8_runner": project / "run_adaptive_shield_design.py",
        "v9_runner": project / "run_calm_selective_shield_design.py",
        "v10_runner": project / "run_margin_capped_shield_design.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v11 fingerprint files missing: {missing}")
    modules = {
        "age_band_hysteresis_policy": "age_band_hysteresis_policy.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "run_hysteresis_screen": "run_hysteresis_screen.py",
        "run_source_runtime_equivalence_audit": (
            "run_source_runtime_equivalence_audit.py"
        ),
        "run_proposed_shield_preflight": "run_proposed_shield_preflight.py",
        "run_margin_capped_shield_design": (
            "run_margin_capped_shield_design.py"
        ),
    }
    for module_name, filename in modules.items():
        actual = Path(inspect.getfile(importlib.import_module(module_name))).resolve()
        expected = (project / filename).resolve()
        if actual != expected:
            raise RuntimeError(f"runtime module path mismatch: {module_name}")
    fingerprint = {name: sha256_file(path) for name, path in files.items()}
    if fingerprint["age_band_policy"] != EXPECTED_AGE_BAND_POLICY_SHA256:
        raise ValueError("frozen v11 age-band policy changed")
    if fingerprint["protocol"] != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen v11 protocol changed")
    return fingerprint


def load_source_runtime_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    return v10.load_source_runtime_equivalence(args)


def _source_runtime_equivalence_record(
    path: Path,
    artifact: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    record = v10._source_runtime_equivalence_record(path, artifact, source)
    if (
        record["source_runtime_equivalence_sha256"]
        != EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_SHA256
    ):
        raise ValueError("unexpected source/runtime equivalence artifact")
    return record


def audit_v7_baseline(
    v7_output: Path,
    source: Mapping[str, Any],
    project: Path,
) -> dict[str, Any]:
    return v10.audit_v7_baseline(v7_output, source, project)


def audit_v10_failure_and_control(
    v10_output: Path,
    source: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, Any], list[EpisodeMetrics]]:
    v10_output = v10_output.resolve()
    selection_path = v10_output / "design_selection.json"
    spec_path = v10_output / "margin_capped_shield_design_spec.json"
    if (
        not selection_path.is_file()
        or sha256_file(selection_path) != EXPECTED_V10_SELECTION_FILE_SHA256
    ):
        raise ValueError("frozen v10 design selection changed")
    if (
        not spec_path.is_file()
        or sha256_file(spec_path) != EXPECTED_V10_SPEC_FILE_SHA256
    ):
        raise ValueError("frozen v10 design specification changed")

    # This loader replays source/runtime, v7, v8, v9, every v10 shard, the
    # combined rows, all derived CSVs, and the complete selection body.
    selection = v10.load_design_selection(
        selection_path,
        allow_historical_runtime=True,
    )
    if selection.get("design_selection_sha256") != EXPECTED_V10_SELECTION_SHA256:
        raise ValueError("unexpected v10 selection self-hash")
    if selection.get("selection_status") != "no_eligible_candidate":
        raise ValueError("v11 requires the recorded failed v10 selection")
    if selection.get("eligible_candidate_indices") != []:
        raise ValueError("v10 failure unexpectedly contains eligible candidates")
    if selection.get("selected_candidate_index") is not None:
        raise ValueError("v10 failure unexpectedly selected a candidate")
    if selection.get("selected_parameters") is not None:
        raise ValueError("v10 failure unexpectedly selected parameters")
    if selection.get("all_cells_pass") is not False:
        raise ValueError("v10 failure must set all_cells_pass=false")
    for field in CLOSED_PERMISSION_FIELDS:
        if selection.get(field) is not False:
            raise ValueError(f"v10 failure must set {field}=false")

    spec = source_runner._load_json(spec_path)
    if _self_hash(spec, "design_spec_sha256") != EXPECTED_V10_SPEC_SHA256:
        raise ValueError("unexpected v10 spec self-hash")
    if selection.get("design_spec_sha256") != EXPECTED_V10_SPEC_SHA256:
        raise ValueError("v10 selection/spec binding changed")
    if selection.get("source_training_freeze_sha256") != source.get(
        "training_freeze_sha256"
    ):
        raise ValueError("v10 source-training binding changed")
    if selection.get("v7_validation_freeze_sha256") != baseline.get(
        "validation_freeze_sha256"
    ):
        raise ValueError("v10 v7 binding changed")

    episode_path = v10_output / "design_episode_metrics.csv"
    combined_rows = source_runner._episode_rows_from_csv(episode_path)
    artifact = selection.get("artifacts", {}).get("episode_metrics", {})
    if (
        Path(str(artifact.get("path", ""))).resolve() != episode_path
        or artifact.get("sha256") != sha256_file(episode_path)
        or artifact.get("row_count") != len(combined_rows)
    ):
        raise ValueError("v10 combined episode artifact binding changed")
    source_candidate = v10.DESIGN_GRID[V10_REUSE_CANDIDATE_INDEX]
    source_policy = v10.candidate_policy_name(source_candidate)
    control_rows = [row for row in combined_rows if row.policy == source_policy]
    expected_count = len(SCENARIOS) * len(POLICY_SEEDS) * len(
        DESIGN_WORKLOAD_SEEDS
    )
    expected_keys = [
        (scenario, policy_seed, workload_seed)
        for scenario in SCENARIOS
        for policy_seed in POLICY_SEEDS
        for workload_seed in DESIGN_WORKLOAD_SEEDS
    ]
    observed_keys = [
        (row.scenario, row.policy_seed, row.workload_seed) for row in control_rows
    ]
    if len(control_rows) != expected_count or observed_keys != expected_keys:
        raise ValueError("v10 candidate-0 control rows are incomplete or reordered")

    record: dict[str, Any] = {
        "screen_name": selection["screen_name"],
        "selection_status": selection["selection_status"],
        "selection_path": str(selection_path),
        "selection_file_sha256": EXPECTED_V10_SELECTION_FILE_SHA256,
        "design_selection_sha256": EXPECTED_V10_SELECTION_SHA256,
        "spec_path": str(spec_path),
        "spec_file_sha256": EXPECTED_V10_SPEC_FILE_SHA256,
        "design_spec_sha256": EXPECTED_V10_SPEC_SHA256,
        "runtime_code_fingerprint_sha256": selection[
            "runtime_code_fingerprint_sha256"
        ],
        "source_training_freeze_sha256": selection[
            "source_training_freeze_sha256"
        ],
        "v7_validation_freeze_sha256": selection[
            "v7_validation_freeze_sha256"
        ],
        "v9_design_selection_sha256": selection[
            "v9_design_selection_sha256"
        ],
        "v9_failure_audit_sha256": selection["v9_failure_audit_sha256"],
        "combined_episode_path": str(episode_path),
        "combined_episode_sha256": artifact["sha256"],
        "combined_row_count": len(combined_rows),
        "combined_rows_reconstructed_from_sources_by_full_loader": True,
        "control_rows_extracted_from_combined_artifact": True,
        "reuse_v10_candidate_index": V10_REUSE_CANDIDATE_INDEX,
        "reuse_v9_candidate_index": V9_REUSE_CANDIDATE_INDEX,
        "reuse_candidate_id": v10.candidate_id(source_candidate),
        "reuse_policy_name": source_policy,
        "reuse_row_count": len(control_rows),
        "all_cells_pass": False,
        **{field: False for field in CLOSED_PERMISSION_FIELDS},
    }
    record["v10_failure_audit_sha256"] = source_runner.sha256_json(record)
    return record, control_rows


def reuse_v10_control_rows(
    source_rows: Sequence[EpisodeMetrics],
) -> list[EpisodeMetrics]:
    source_candidate = v10.DESIGN_GRID[V10_REUSE_CANDIDATE_INDEX]
    target = DESIGN_GRID[0]
    for field in (
        "stay_bonus",
        "urgency_relief",
        "class_2_relief",
        "calm_bonus",
    ):
        if float(target[field]) != float(source_candidate[field]):
            raise RuntimeError(f"v10 control mapping changed: {field}")
    if float(source_candidate["margin_cap"]) != 0.75:
        raise RuntimeError("v10 candidate-0 is no longer the v9-equivalent control")
    if float(target["age_band_mix"]) != 0.0:
        raise RuntimeError("v11 control must use age_band_mix=0")
    if (
        BASE_STAY_BONUS,
        BASE_URGENCY_RELIEF,
        BASE_CLASS_2_RELIEF,
        ADDON_STAY_BONUS,
    ) != (
        FIXED_STAY_BONUS,
        FIXED_URGENCY_RELIEF,
        FIXED_CLASS_2_RELIEF,
        FIXED_CALM_BONUS,
    ):
        raise RuntimeError("v11 policy constants no longer match v9 candidate 5")
    source_policy = v10.candidate_policy_name(source_candidate)
    if any(row.policy != source_policy for row in source_rows):
        raise ValueError("v10 reuse rows contain another policy")
    target_policy = candidate_policy_name(target)
    return [replace(row, policy=target_policy) for row in source_rows]


def build_spec(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    source_runtime: Mapping[str, Any],
    baseline: Mapping[str, Any],
    v10_audit: Mapping[str, Any],
    *,
    runtime_code_fingerprint: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    protocol = args.project.parent / "docs" / "AGE_BAND_SHIELD_DESIGN_V11.md"
    fingerprint = (
        dict(runtime_code_fingerprint)
        if runtime_code_fingerprint is not None
        else _runtime_code_fingerprint(args.project, protocol)
    )
    source_runtime_record = _source_runtime_equivalence_record(
        args.source_runtime_equivalence,
        source_runtime,
        source,
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exposed_panel_engineering_design",
        "confirmatory": False,
        **{field: False for field in CLOSED_PERMISSION_FIELDS},
        "final_exposed_design_attempt": True,
        "further_exposed_design_allowed": False,
        "training_jobs": 0,
        "design_workload_seeds": list(DESIGN_WORKLOAD_SEEDS),
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "candidate_grid": [dict(candidate) for candidate in DESIGN_GRID],
        "candidate_count": len(DESIGN_GRID),
        "logical_design_jobs": len(build_design_jobs()),
        "reused_logical_jobs": (
            len(build_design_jobs()) - len(build_new_evaluation_jobs())
        ),
        "reused_episode_rows": 80,
        "new_evaluation_jobs": len(build_new_evaluation_jobs()),
        "new_evaluation_episode_rows": 80,
        "cell_gates": dict(CELL_GATES),
        "design_buffers": {
            "raw_context_switch_headroom_min": RAW_SWITCH_HEADROOM_BUFFER,
            "avoidable_micro_improvement_min": (
                AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
            ),
            "delivery_gate_slack_min": DELIVERY_SLACK_BUFFER,
            "class_2_extra_buffer": 0.0,
        },
        "selection_rule_id": SELECTION_RULE_ID,
        "selection_key_fields": [
            "worst_delivery_buffer_slack",
            "worst_class_2_slack",
            "worst_raw_switch_buffer_slack",
            "worst_avoidable_buffer_slack",
            "negative_age_band_mix",
            "negative_candidate_index",
        ],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "source_runtime_equivalence": source_runtime_record,
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
        "v10_failure_audit": dict(v10_audit),
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
            "source_runtime_equivalence": str(
                args.source_runtime_equivalence.resolve()
            ),
            "v7_output": str(args.v7_output.resolve()),
            "v10_output": str(args.v10_output.resolve()),
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
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "source_training_freeze_sha256": spec["source_training_freeze_sha256"],
        "v7_validation_freeze_sha256": spec["v7_baseline"][
            "validation_freeze_sha256"
        ],
        "v10_design_selection_sha256": spec["v10_failure_audit"][
            "design_selection_sha256"
        ],
        "v10_design_spec_sha256": spec["v10_failure_audit"][
            "design_spec_sha256"
        ],
        "v10_failure_audit_sha256": spec["v10_failure_audit"][
            "v10_failure_audit_sha256"
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
        **{field: False for field in CLOSED_PERMISSION_FIELDS},
    }


def _load_valid_shard(
    output: Path,
    job: DesignJob,
    expected: Mapping[str, Any],
) -> list[EpisodeMetrics]:
    csv_path, metadata_path = _evaluation_paths(output, job)
    if not csv_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing v11 shard: {job.job_id}")
    metadata = source_runner._load_json(metadata_path)
    required_fields = set(expected) | {
        "csv_sha256",
        "row_count",
        "policy_diagnostics",
    }
    if set(metadata) != required_fields:
        raise ValueError(f"v11 shard metadata field set mismatch: {job.job_id}")
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"v11 shard metadata mismatch: {job.job_id}/{field}")
    if metadata.get("csv_sha256") != sha256_file(csv_path):
        raise ValueError(f"v11 shard hash mismatch: {job.job_id}")
    if metadata.get("row_count") != len(DESIGN_WORKLOAD_SEEDS):
        raise ValueError(f"v11 shard row count mismatch: {job.job_id}")
    if not isinstance(metadata.get("policy_diagnostics"), Mapping):
        raise ValueError(f"v11 shard diagnostics missing: {job.job_id}")
    rows = source_runner.validate_evaluation_shard(
        csv_path, job, DESIGN_WORKLOAD_SEEDS
    )
    v7.validate_switch_rows(rows)
    return rows


def evaluate_job(
    args: argparse.Namespace,
    job: DesignJob,
    spec: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> list[EpisodeMetrics]:
    if job.candidate_index != 1:
        raise RuntimeError("v10/v9 control candidate must be reused, not evaluated")
    expected = _expected_shard_metadata(job, spec, checkpoint)
    csv_path, metadata_path = _evaluation_paths(args.output, job)
    if csv_path.exists() or metadata_path.exists():
        if not csv_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"partial v11 shard exists: {job.job_id}")
        return _load_valid_shard(args.output, job, expected)

    checkpoint_path = Path(str(checkpoint["checkpoint_path"]))
    if sha256_file(checkpoint_path) != checkpoint["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash changed: {job.source_job_id}")
    policy, _ = load_age_band_hysteresis_policy(
        checkpoint_path,
        age_band_mix=job.age_band_mix,
        device=args.device,
    )
    if policy.checkpoint_schema != expected["policy_schema"]:
        raise ValueError(f"loaded v11 policy schema mismatch: {job.job_id}")
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
    expected_count = len(jobs) * len(DESIGN_WORKLOAD_SEEDS)
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in rows
    }
    if len(rows) != expected_count or len(keys) != expected_count:
        raise RuntimeError("v11 new-evaluation grid is incomplete or duplicated")
    return rows


def _shard_index(
    output: Path,
    jobs: Sequence[DesignJob],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for job in jobs:
        csv_path, metadata_path = _evaluation_paths(output, job)
        metadata = source_runner._load_json(metadata_path)
        result.append(
            {
                "job_id": job.job_id,
                "csv_path": str(csv_path.resolve()),
                "csv_sha256": sha256_file(csv_path),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "row_count": metadata["row_count"],
            }
        )
    return result


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
        raise ValueError(
            f"cell row count mismatch: {scenario}/{policy_seed}/{policy}"
        )
    if [row.workload_seed for row in selected] != list(DESIGN_WORKLOAD_SEEDS):
        raise ValueError(
            f"cell workload order mismatch: {scenario}/{policy_seed}/{policy}"
        )
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
                delivery_buffer_slack = delivery_slack - DELIVERY_SLACK_BUFFER
                raw_switch_buffer_slack = (
                    raw_switch_slack - RAW_SWITCH_HEADROOM_BUFFER
                )
                avoidable_buffer_slack = (
                    avoidable_slack - AVOIDABLE_MICRO_IMPROVEMENT_BUFFER
                )
                five_gate_pass = (
                    delivery_slack >= 0.0
                    and class_2_slack >= 0.0
                    and avoidable_slack > 0.0
                    and proposed_switch_slack >= 0.0
                    and raw_switch_slack >= 0.0
                )
                buffer_pass = (
                    delivery_buffer_slack >= 0.0
                    and raw_switch_buffer_slack >= 0.0
                    and avoidable_buffer_slack >= 0.0
                )
                gates.append(
                    {
                        **dict(candidate),
                        "candidate_id": candidate_id(candidate),
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
                        "delivery_buffer_slack": delivery_buffer_slack,
                        "raw_switch_buffer_slack": raw_switch_buffer_slack,
                        "avoidable_buffer_slack": avoidable_buffer_slack,
                        "delivery_gate_pass": delivery_slack >= 0.0,
                        "class_2_gate_pass": class_2_slack >= 0.0,
                        "avoidable_gate_pass": avoidable_slack > 0.0,
                        "switch_vs_proposed_gate_pass": (
                            proposed_switch_slack >= 0.0
                        ),
                        "switch_vs_raw_gate_pass": raw_switch_slack >= 0.0,
                        "five_gates_pass": five_gate_pass,
                        "delivery_buffer_pass": delivery_buffer_slack >= 0.0,
                        "raw_switch_buffer_pass": (
                            raw_switch_buffer_slack >= 0.0
                        ),
                        "avoidable_buffer_pass": avoidable_buffer_slack >= 0.0,
                        "buffer_gate_pass": buffer_pass,
                        "cell_eligible": five_gate_pass and buffer_pass,
                    }
                )
    expected = len(DESIGN_GRID) * len(SCENARIOS) * len(POLICY_SEEDS)
    if len(gates) != expected:
        raise RuntimeError("v11 gate grid has the wrong size")
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
            raise ValueError(f"candidate {index} does not have eight cells")
        worst_delivery = min(
            float(row["delivery_buffer_slack"]) for row in cells
        )
        worst_class_2 = min(float(row["class_2_slack"]) for row in cells)
        worst_raw = min(
            float(row["raw_switch_buffer_slack"]) for row in cells
        )
        worst_avoidable = min(
            float(row["avoidable_buffer_slack"]) for row in cells
        )
        eligible = all(bool(row["cell_eligible"]) for row in cells)
        key = [
            worst_delivery,
            worst_class_2,
            worst_raw,
            worst_avoidable,
            -float(candidate["age_band_mix"]),
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
                "worst_delivery_buffer_slack": worst_delivery,
                "worst_class_2_slack": worst_class_2,
                "worst_raw_switch_buffer_slack": worst_raw,
                "worst_avoidable_buffer_slack": worst_avoidable,
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
            raise RuntimeError("v11 selection argmax is not unique")
    return results, selected


def _artifact(path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def _validate_artifact(
    artifact: Mapping[str, Any],
    expected_path: Path,
    expected_rows: int,
) -> None:
    path = Path(str(artifact.get("path", ""))).resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise ValueError(f"v11 artifact path mismatch: {expected_path.name}")
    if artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"v11 artifact hash mismatch: {expected_path.name}")
    if artifact.get("row_count") != expected_rows:
        raise ValueError(f"v11 artifact row count mismatch: {expected_path.name}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            observed_rows = sum(1 for _ in csv.DictReader(handle))
        if observed_rows != expected_rows:
            raise ValueError(
                f"v11 CSV physical row count mismatch: {expected_path.name}"
            )


def _csv_rows_match(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        observed = list(csv.DictReader(handle))
    rendered = [
        {
            key: "" if value is None else str(value)
            for key, value in row.items()
        }
        for row in expected
    ]
    if observed != rendered:
        raise ValueError(f"v11 CSV does not replay: {path.name}")


def _selected_parameters(selected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if selected is None:
        return None
    return {
        key: selected[key]
        for key in (
            "stay_bonus",
            "urgency_relief",
            "class_2_relief",
            "calm_bonus",
            "age_band_mix",
        )
    }


def _build_selection_body(
    output: Path,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    rows: Sequence[EpisodeMetrics],
    aggregate: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan_path = output / "design_plan.csv"
    episode_path = output / "design_episode_metrics.csv"
    aggregate_path = output / "design_aggregate_metrics.csv"
    gates_path = output / "design_cell_gates.csv"
    results, selected = select_candidate(gates)
    new_shards = _shard_index(output, build_new_evaluation_jobs())
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "exposed_panel_engineering_design",
        "confirmatory": False,
        **{field: False for field in CLOSED_PERMISSION_FIELDS},
        "final_exposed_design_attempt": True,
        "further_exposed_design_allowed": False,
        "design_spec_sha256": spec["design_spec_sha256"],
        "runtime_code_fingerprint_sha256": spec[
            "runtime_code_fingerprint_sha256"
        ],
        "source_runtime_equivalence_sha256": spec[
            "source_runtime_equivalence"
        ]["source_runtime_equivalence_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "v7_validation_freeze_sha256": spec["v7_baseline"][
            "validation_freeze_sha256"
        ],
        "v10_design_selection_sha256": spec["v10_failure_audit"][
            "design_selection_sha256"
        ],
        "v10_design_spec_sha256": spec["v10_failure_audit"][
            "design_spec_sha256"
        ],
        "v10_failure_audit_sha256": spec["v10_failure_audit"][
            "v10_failure_audit_sha256"
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
        "selected_parameters": _selected_parameters(selected),
        "all_cells_pass": selected is not None,
        "selection_status": (
            "selected_exposed_design_candidate"
            if selected is not None
            else "no_eligible_candidate"
        ),
        "training_jobs": 0,
        "logical_design_jobs": len(build_design_jobs()),
        "reused_logical_jobs": 8,
        "reused_episode_rows": 80,
        "new_evaluation_jobs": len(build_new_evaluation_jobs()),
        "new_evaluation_episode_rows": 80,
        "new_evaluation_shards": new_shards,
        "artifacts": {
            "design_spec": _artifact(
                output / "age_band_shield_design_spec.json", 1
            ),
            "design_plan": _artifact(plan_path, len(build_design_jobs())),
            "episode_metrics": _artifact(episode_path, len(rows)),
            "aggregate_metrics": _artifact(aggregate_path, len(aggregate)),
            "cell_gates": _artifact(gates_path, len(gates)),
        },
    }
    body["design_selection_sha256"] = source_runner.sha256_json(body)
    return body


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
    aggregate = source_runner.aggregate_rows(rows, rng_base=82000)
    atomic_write_csv(episode_path, (asdict(row) for row in rows))
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(gates_path, gates)
    body = _build_selection_body(output, spec, source, rows, aggregate, gates)
    source_runner.ensure_immutable_json(selection_path, body)
    return body


def load_design_selection(
    path: Path,
    *,
    allow_historical_runtime: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    if (
        allow_historical_runtime
        and sha256_file(path) != EXPECTED_V11_SELECTION_FILE_SHA256
    ):
        raise ValueError("frozen v11 design selection changed")
    selection = source_runner._load_json(path)
    selection_sha256 = _self_hash(selection, "design_selection_sha256")
    if (
        allow_historical_runtime
        and selection_sha256 != EXPECTED_V11_SELECTION_SHA256
    ):
        raise ValueError("unexpected v11 design selection")
    if selection.get("screen_name") != SCREEN_NAME:
        raise ValueError("v11 selection screen mismatch")
    if selection.get("selection_rule_id") != SELECTION_RULE_ID:
        raise ValueError("v11 selection rule mismatch")
    if selection.get("confirmatory") is not False:
        raise ValueError("v11 selection must set confirmatory=false")
    for field in CLOSED_PERMISSION_FIELDS:
        if selection.get(field) is not False:
            raise ValueError(f"v11 selection must set {field}=false")
    if selection.get("final_exposed_design_attempt") is not True:
        raise ValueError("v11 selection must be the final exposed design attempt")
    if selection.get("further_exposed_design_allowed") is not False:
        raise ValueError("v11 selection cannot allow another exposed design")
    if selection.get("candidate_grid") != [
        dict(candidate) for candidate in DESIGN_GRID
    ]:
        raise ValueError("v11 selection candidate grid mismatch")

    output = path.parent
    expected_paths = {
        "design_spec": output / "age_band_shield_design_spec.json",
        "design_plan": output / "design_plan.csv",
        "episode_metrics": output / "design_episode_metrics.csv",
        "aggregate_metrics": output / "design_aggregate_metrics.csv",
        "cell_gates": output / "design_cell_gates.csv",
    }
    artifacts = selection.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_paths):
        raise ValueError("v11 selection artifact mapping mismatch")

    spec_path = expected_paths["design_spec"]
    if (
        allow_historical_runtime
        and sha256_file(spec_path) != EXPECTED_V11_SPEC_FILE_SHA256
    ):
        raise ValueError("frozen v11 design specification changed")
    spec = source_runner._load_json(spec_path)
    spec_sha256 = _self_hash(spec, "design_spec_sha256")
    if allow_historical_runtime and spec_sha256 != EXPECTED_V11_SPEC_SHA256:
        raise ValueError("unexpected v11 design specification")
    if selection.get("design_spec_sha256") != spec["design_spec_sha256"]:
        raise ValueError("v11 selection spec mismatch")
    project = Path(str(spec["paths"]["project"])).resolve()
    protocol = Path(str(spec["paths"]["protocol"])).resolve()
    fingerprint = _runtime_code_fingerprint(project, protocol)
    args = argparse.Namespace(
        project=project,
        source=Path(str(spec["paths"]["source"])).resolve(),
        source_runtime_equivalence=Path(
            str(spec["paths"]["source_runtime_equivalence"])
        ).resolve(),
        v7_output=Path(str(spec["paths"]["v7_output"])).resolve(),
        v10_output=Path(str(spec["paths"]["v10_output"])).resolve(),
        output=output,
        device=spec.get("device"),
    )
    # Preserve the mandatory ordering here as well as in main.
    source = source_runner.audit_source_screen(args.source)
    if allow_historical_runtime:
        v8._audit_historical_runtime_fingerprint(
            project,
            source,
            spec.get("runtime_code_fingerprint"),
            fingerprint,
            HISTORICAL_V11_RUNNER_FINGERPRINTS,
        )
        source_runtime = source_runner._load_json(
            args.source_runtime_equivalence
        )
    else:
        if fingerprint != spec.get("runtime_code_fingerprint"):
            raise ValueError("v11 runtime code fingerprint changed")
        source_runtime = load_source_runtime_equivalence(args)
    source_runtime_record = _source_runtime_equivalence_record(
        args.source_runtime_equivalence,
        source_runtime,
        source,
    )
    if spec.get("source_runtime_equivalence") != source_runtime_record:
        raise ValueError("v11 source/runtime equivalence binding changed")
    baseline = audit_v7_baseline(args.v7_output, source, project)
    expected_baseline = {
        key: baseline[key]
        for key in (
            "spec_sha256",
            "validation_freeze_sha256",
            "episode_metrics_path",
            "episode_metrics_sha256",
            "reference_row_count",
        )
    }
    if spec.get("v7_baseline") != expected_baseline:
        raise ValueError("v11 v7 baseline binding changed")
    v10_audit, source_control_rows = audit_v10_failure_and_control(
        args.v10_output, source, baseline
    )
    if spec.get("v10_failure_audit") != v10_audit:
        raise ValueError("v11 v10 failure audit binding changed")
    expected_spec = build_spec(
        args,
        source,
        source_runtime,
        baseline,
        v10_audit,
        runtime_code_fingerprint=(
            spec["runtime_code_fingerprint"]
            if allow_historical_runtime
            else None
        ),
    )
    if spec != expected_spec:
        raise ValueError("v11 design specification does not replay exactly")

    expected_bindings = {
        "design_spec_sha256": spec["design_spec_sha256"],
        "runtime_code_fingerprint_sha256": spec[
            "runtime_code_fingerprint_sha256"
        ],
        "source_runtime_equivalence_sha256": source_runtime_record[
            "source_runtime_equivalence_sha256"
        ],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "v7_validation_freeze_sha256": baseline["validation_freeze_sha256"],
        "v10_design_selection_sha256": v10_audit[
            "design_selection_sha256"
        ],
        "v10_design_spec_sha256": v10_audit["design_spec_sha256"],
        "v10_failure_audit_sha256": v10_audit["v10_failure_audit_sha256"],
    }
    for field, expected_value in expected_bindings.items():
        if selection.get(field) != expected_value:
            raise ValueError(f"v11 selection binding changed: {field}")

    reused_rows = reuse_v10_control_rows(source_control_rows)
    new_rows: list[EpisodeMetrics] = []
    for job in build_new_evaluation_jobs():
        checkpoint = source["checkpoints"][job.source_job_id]
        expected = _expected_shard_metadata(job, spec, checkpoint)
        new_rows.extend(_load_valid_shard(output, job, expected))
    expected_shards = _shard_index(output, build_new_evaluation_jobs())
    if selection.get("new_evaluation_shards") != expected_shards:
        raise ValueError("v11 new shard index does not replay")
    if len(new_rows) != 80:
        raise ValueError("v11 new episode count does not replay")

    rows = [*baseline["references"], *reused_rows, *new_rows]
    observed_rows = source_runner._episode_rows_from_csv(
        expected_paths["episode_metrics"]
    )
    if observed_rows != rows:
        raise ValueError("v11 combined rows do not replay from sources")
    expected_plan = [job.as_dict() for job in build_design_jobs()]
    expected_aggregate = source_runner.aggregate_rows(rows, rng_base=82000)
    gates = compute_candidate_cell_gates(rows)
    _csv_rows_match(expected_paths["design_plan"], expected_plan)
    _csv_rows_match(expected_paths["aggregate_metrics"], expected_aggregate)
    _csv_rows_match(expected_paths["cell_gates"], gates)
    results, selected = select_candidate(gates)
    if selection.get("candidate_results") != results:
        raise ValueError("v11 candidate results do not replay")
    eligible_indices = [
        int(result["candidate_index"])
        for result in results
        if result["eligible"]
    ]
    if selection.get("eligible_candidate_indices") != eligible_indices:
        raise ValueError("v11 eligible set does not replay")
    selected_index = int(selected["candidate_index"]) if selected else None
    if selection.get("selected_candidate_index") != selected_index:
        raise ValueError("v11 selected candidate does not replay")
    if selection.get("selected_parameters") != _selected_parameters(selected):
        raise ValueError("v11 selected parameters do not replay")
    if selection.get("all_cells_pass") is not (selected is not None):
        raise ValueError("v11 all-cells status does not replay")
    expected_status = (
        "selected_exposed_design_candidate"
        if selected is not None
        else "no_eligible_candidate"
    )
    if selection.get("selection_status") != expected_status:
        raise ValueError("v11 selection status does not replay")
    for field, expected_value in {
        "training_jobs": 0,
        "logical_design_jobs": 16,
        "reused_logical_jobs": 8,
        "reused_episode_rows": 80,
        "new_evaluation_jobs": 8,
        "new_evaluation_episode_rows": 80,
    }.items():
        if selection.get(field) != expected_value:
            raise ValueError(f"v11 selection accounting changed: {field}")

    expected_row_counts = {
        "design_spec": 1,
        "design_plan": len(build_design_jobs()),
        "episode_metrics": len(rows),
        "aggregate_metrics": len(expected_aggregate),
        "cell_gates": len(gates),
    }
    for name, expected_path in expected_paths.items():
        _validate_artifact(artifacts[name], expected_path, expected_row_counts[name])
    expected_selection = _build_selection_body(
        output, spec, source, rows, expected_aggregate, gates
    )
    if selection != expected_selection:
        raise ValueError("v11 design selection does not replay exactly")
    return selection


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for directory in (
        args.source,
        args.v7_output,
        args.v10_output,
        args.project,
    ):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not args.source_runtime_equivalence.is_file():
        raise FileNotFoundError(args.source_runtime_equivalence)
    root = args.project.parent.resolve()
    broad = {
        root,
        args.project.resolve(),
        (root / "experiments").resolve(),
        (root / "experiments" / "archive").resolve(),
        args.source.resolve(),
        args.v7_output.resolve(),
        args.v10_output.resolve(),
        args.source_runtime_equivalence.parent.resolve(),
    }
    if args.output.resolve() in broad:
        raise RuntimeError("output must be a dedicated v11 design directory")
    if args.output.is_dir() and not (
        args.output / "age_band_shield_design_spec.json"
    ).is_file():
        if any(args.output.iterdir()):
            raise RuntimeError("output contains an unrecognized experiment")
    if (
        AGE_BAND_MIXES != (0.0, 1.0)
        or len(DESIGN_GRID) != 2
        or len(build_design_jobs()) != 16
        or len(build_new_evaluation_jobs()) != 8
    ):
        raise RuntimeError("frozen v11 design grid changed")
    if DESIGN_WORKLOAD_SEEDS != tuple(range(60001, 60011)):
        raise RuntimeError("v11 must use only the exposed v7 design panel")
    if any(job.candidate_index != 1 for job in build_new_evaluation_jobs()):
        raise RuntimeError("v11 may evaluate only candidate 1")
    if CELL_GATES != v7.CELL_GATES:
        raise RuntimeError("v7 five-gate contract changed")
    if (
        RAW_SWITCH_HEADROOM_BUFFER != 2.0
        or AVOIDABLE_MICRO_IMPROVEMENT_BUFFER != 0.005
        or DELIVERY_SLACK_BUFFER != 1.0 / 1920.0
    ):
        raise RuntimeError("v9 three-buffer contract changed")
    if EXPECTED_FEATURE_SCHEMA.get("schema_id") != (
        "leo_multi_candidate_features_v1_dim_26"
    ):
        raise RuntimeError("candidate feature schema changed")
    if (
        ROUTE_SWITCH_FEATURE_INDEX,
        ROUTE_URGENCY_FEATURE_INDEX,
        ROUTE_CLASS_2_FEATURE_INDEX,
    ) != (17, 20, 23):
        raise RuntimeError("candidate feature indexes changed")
    protocol = args.project.parent / "docs" / "AGE_BAND_SHIELD_DESIGN_V11.md"
    if sha256_file(args.project / "age_band_hysteresis_policy.py") != (
        EXPECTED_AGE_BAND_POLICY_SHA256
    ):
        raise RuntimeError("frozen v11 age-band policy changed")
    if sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen v11 protocol changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Design the age-band shield on the exposed panel."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            root
            / "experiments"
            / "archive"
            / "congestion-context-screen-20k-v1"
        ),
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
    parser.add_argument(
        "--v7-output",
        type=Path,
        default=root / "experiments" / "proposed-shield-preflight-v7",
    )
    parser.add_argument(
        "--v10-output",
        type=Path,
        default=root / "experiments" / "margin-capped-shield-design-v10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "age-band-shield-design-v11",
    )
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in (
        "source",
        "source_runtime_equivalence",
        "v7_output",
        "v10_output",
        "output",
        "project",
    ):
        setattr(args, field, getattr(args, field).resolve())
    validate_environment(args)

    # This ordering is deliberate: equivalence replay must fail before the
    # source screen, v7 baseline, v10 artifacts, or any v11 output is touched.
    source_runtime = load_source_runtime_equivalence(args)
    source = source_runner.audit_source_screen(args.source)
    baseline = audit_v7_baseline(args.v7_output, source, args.project)
    v10_audit, source_control_rows = audit_v10_failure_and_control(
        args.v10_output, source, baseline
    )
    reused_rows = reuse_v10_control_rows(source_control_rows)
    spec = build_spec(args, source, source_runtime, baseline, v10_audit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "screen_name": SCREEN_NAME,
                    "design_spec_sha256": spec["design_spec_sha256"],
                    "source_runtime_equivalence_sha256": spec[
                        "source_runtime_equivalence"
                    ]["source_runtime_equivalence_sha256"],
                    "v10_design_selection_sha256": v10_audit[
                        "design_selection_sha256"
                    ],
                    "v10_full_selection_replayed": True,
                    "v10_combined_rows_reconstructed_from_sources": v10_audit[
                        "combined_rows_reconstructed_from_sources_by_full_loader"
                    ],
                    "candidate_count": len(DESIGN_GRID),
                    "logical_design_jobs": len(build_design_jobs()),
                    "new_evaluation_jobs": len(build_new_evaluation_jobs()),
                    "new_candidate_rows": 80,
                    "reused_control_rows": len(reused_rows),
                    "reference_rows": baseline["reference_row_count"],
                    "workloads": [
                        DESIGN_WORKLOAD_SEEDS[0],
                        DESIGN_WORKLOAD_SEEDS[-1],
                    ],
                    **{field: False for field in CLOSED_PERMISSION_FIELDS},
                    "dry_run_writes_output": False,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0

    selection_path = args.output / "design_selection.json"
    # A completed reentry is a pure replay: do not create a directory, lock,
    # shard, plan, or derived artifact before validating the selection.
    if selection_path.is_file():
        selection = load_design_selection(selection_path)
    else:
        if selection_path.exists():
            raise ValueError("v11 design selection is not a file")
        args.output.mkdir(parents=True, exist_ok=True)
        with source_runner.invocation_lock(args.output):
            if selection_path.exists():
                if not selection_path.is_file():
                    raise ValueError("v11 design selection is not a file")
                selection = load_design_selection(selection_path)
            else:
                source_runner.ensure_immutable_json(
                    args.output / "age_band_shield_design_spec.json", spec
                )
                atomic_write_csv(
                    args.output / "design_plan.csv",
                    (job.as_dict() for job in build_design_jobs()),
                )
                new_rows = evaluate_jobs(args, spec, source["checkpoints"])
                combined = [*baseline["references"], *reused_rows, *new_rows]
                selection = write_design_artifacts(
                    args.output, spec, source, combined
                )
                load_design_selection(selection_path)
    print(json.dumps(selection, indent=2))
    return 0 if selection["all_cells_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
