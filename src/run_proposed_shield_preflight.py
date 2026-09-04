"""Frozen proposed-checkpoint route-shield preflight (v7).

This runner does not train a policy.  It evaluates a fixed actor-score shield on
the already frozen proposed checkpoints, with validation and test kept as two
separate, hash-bound phases.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch

from ablation_matrix_runner import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from hysteresis_policy import SWITCH_FEATURE_INDEX, load_hysteresis_policy
from mappo_evaluation import EpisodeMetrics, evaluate_policy
import run_hysteresis_screen as v1


SCREEN_NAME = "PROPOSED-SHIELD-PREFLIGHT-v7"
SCHEMA_VERSION = 1
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072, 1047581915, 1245825580)
VALIDATION_WORKLOAD_SEEDS = tuple(range(60001, 60011))
TEST_WORKLOAD_SEEDS = tuple(range(70001, 70026))
STAY_BONUS = 0.40
ARMS = ("proposed", "raw_context", "shield")
POLICY_NAMES = {
    "proposed": "mappo_proposed_source",
    "raw_context": "mappo_context_raw_source",
    "shield": "mappo_proposed_shield_beta_0p40_v7",
}
CELL_GATES = {
    "delivery_difference_vs_proposed_min": -0.003,
    "class_2_delivery_difference_vs_proposed_min": -0.010,
    "avoidable_switch_micro_rate_rule": "shield_strictly_less_than_proposed",
    "routing_switch_mean_vs_proposed_rule": (
        "shield_less_than_or_equal_to_proposed"
    ),
    "routing_switch_mean_vs_raw_context_rule": (
        "shield_less_than_or_equal_to_raw_context"
    ),
}
KNOWN_EXPOSED_PANELS = (
    (9001, 9200),
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
    (61001, 61004),
)


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
            "stay_bonus": self.stay_bonus,
            "policy_name": self.policy_name,
            "source_variant": self.source_variant,
            "source_job_id": self.source_job_id,
            "job_id": self.job_id,
        }


def build_jobs(phase: str) -> list[EvaluationJob]:
    if phase not in {"validation", "test"}:
        raise ValueError(f"unsupported phase: {phase}")
    jobs: list[EvaluationJob] = []
    for scenario in SCENARIOS:
        for arm in ARMS:
            for policy_seed in POLICY_SEEDS:
                jobs.append(
                    EvaluationJob(
                        index=len(jobs),
                        phase=phase,
                        scenario=scenario,
                        arm=arm,
                        policy_seed=policy_seed,
                        stay_bonus=STAY_BONUS if arm == "shield" else 0.0,
                    )
                )
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
    expected = v1.sha256_json(body)
    if observed != expected:
        raise ValueError(
            f"invalid {field}: observed={observed!r}, expected={expected}"
        )
    return expected


def _runtime_code_fingerprint(project: Path, protocol_path: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "protocol": protocol_path,
        "hysteresis_policy": project / "hysteresis_policy.py",
        "evaluation": project / "mappo_evaluation.py",
        "design": project / "mappo_design.py",
        "wrapper": project / "cleanmarl_leo_multiagent_wrapper.py",
        "multiagent_environment": project / "leo_multiagent_env.py",
        "base_environment": project / "leo_marl_env.py",
        "variants": project / "variant_definitions.py",
        "statistics": project / "hierarchical_statistics.py",
        "artifact_helpers": project / "ablation_matrix_runner.py",
        "source_auditor": project / "run_hysteresis_screen.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime fingerprint files missing: {missing}")

    module_files = {
        "hysteresis_policy": "hysteresis_policy.py",
        "mappo_evaluation": "mappo_evaluation.py",
        "mappo_design": "mappo_design.py",
        "cleanmarl_leo_multiagent_wrapper": "cleanmarl_leo_multiagent_wrapper.py",
        "leo_multiagent_env": "leo_multiagent_env.py",
        "leo_marl_env": "leo_marl_env.py",
        "variant_definitions": "variant_definitions.py",
        "hierarchical_statistics": "hierarchical_statistics.py",
        "ablation_matrix_runner": "ablation_matrix_runner.py",
        "run_hysteresis_screen": "run_hysteresis_screen.py",
    }
    for module_name, filename in module_files.items():
        actual = Path(
            inspect.getfile(importlib.import_module(module_name))
        ).resolve()
        expected = (project / filename).resolve()
        if actual != expected:
            raise RuntimeError(
                f"runtime module path mismatch for {module_name}: "
                f"{actual} != {expected}"
            )
    return {name: sha256_file(path) for name, path in files.items()}


def build_spec(
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    protocol_path = args.project.parent / "docs" / "PROPOSED_SHIELD_PREFLIGHT_V7.md"
    code_fingerprint = _runtime_code_fingerprint(args.project, protocol_path)
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
            "type": "frozen_actor_score_route_shield",
            "base_checkpoint_variant": "proposed",
            "stay_bonus": STAY_BONUS,
            "switch_rule": (
                "retain_cached_feasible_route_unless_raw_switch_logit_advantage_"
                "is_at_least_0.40"
            ),
            "first_use_and_forced_switch_behavior": "unchanged",
            "checkpoint_and_actor_weights": "frozen_no_training",
        },
        "beta_provenance": {
            "status": "fixed_before_v7_validation",
            "basis": (
                "mechanism_motivated_heuristic_after_unarchived_manual_probe_on_"
                "already_exposed_51001_51025"
            ),
            "immutable_beta_0p40_selection_artifact_available": False,
            "probe_panel_was_already_exposed": True,
            "validation_used_for_tuning": False,
            "test_used_for_tuning": False,
        },
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "arms": list(ARMS),
        "policy_names": dict(POLICY_NAMES),
        "validation_workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "known_exposed_panels": [list(panel) for panel in KNOWN_EXPOSED_PANELS],
        "fresh_panel_audit": {
            "scope": "structured_seed_and_workload_fields_in_local_routing_sources_and_artifacts",
            "validation_panel_prior_workload_hits": 0,
            "test_panel_prior_workload_hits": 0,
            "rejected_candidate": {
                "panel": [61001, 61025],
                "reason": "61001..61004_used_by_v2_smoke_train_seed_start",
            },
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
            "validation_phase_evaluates_test_panel": False,
            "test_requires_replayed_passing_validation_freeze": True,
            "test_shards_bind_validation_freeze_sha256": True,
        },
        "source": dict(source),
        "code_fingerprint": code_fingerprint,
        "paths": {
            "project": str(args.project.resolve()),
            "source": str(args.source.resolve()),
            "output": str(args.output.resolve()),
            "protocol": str(protocol_path.resolve()),
        },
    }
    body["spec_sha256"] = v1.sha256_json(body)
    return body


def _evaluation_paths(output: Path, job: EvaluationJob) -> tuple[Path, Path]:
    directory = output / "evaluation_shards" / job.phase
    return directory / f"{job.slug}.csv", directory / f"{job.slug}.json"


def _expected_shard_metadata(
    job: EvaluationJob,
    spec_sha256: str,
    source_checkpoint: Mapping[str, Any],
    workload_seeds: Sequence[int],
    validation_freeze_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "spec_sha256": spec_sha256,
        "validation_freeze_sha256": validation_freeze_sha256,
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


def validate_switch_rows(rows: Sequence[EpisodeMetrics]) -> None:
    for row in rows:
        counts = (
            row.avoidable_routing_switches,
            row.forced_routing_switches,
            row.switch_opportunities,
            row.routing_switches,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("switch diagnostics must be non-negative integers")
        if row.avoidable_routing_switches > row.switch_opportunities:
            raise ValueError("avoidable switches exceed switch opportunities")
        if row.routing_switches != (
            row.avoidable_routing_switches + row.forced_routing_switches
        ):
            raise ValueError("route-switch accounting is inconsistent")
        expected_rate = row.avoidable_routing_switches / max(
            1, row.switch_opportunities
        )
        if not math.isclose(
            row.avoidable_switch_rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("episode avoidable switch rate is inconsistent")


def _load_valid_shard(
    output: Path,
    job: EvaluationJob,
    expected_metadata: Mapping[str, Any],
    workload_seeds: Sequence[int],
) -> list[EpisodeMetrics]:
    shard_path, metadata_path = _evaluation_paths(output, job)
    if not shard_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing evaluation shard for {job.job_id}")
    metadata = v1._load_json(metadata_path)
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"evaluation shard metadata mismatch: {job.job_id}/{key}")
    if metadata.get("csv_sha256") != sha256_file(shard_path):
        raise ValueError(f"evaluation shard hash mismatch: {job.job_id}")
    if metadata.get("row_count") != len(workload_seeds):
        raise ValueError(f"evaluation shard row count mismatch: {job.job_id}")
    if not isinstance(metadata.get("policy_diagnostics"), dict):
        raise ValueError(f"evaluation shard diagnostics missing: {job.job_id}")
    rows = v1.validate_evaluation_shard(shard_path, job, workload_seeds)
    validate_switch_rows(rows)
    return rows


def evaluate_job(
    args: argparse.Namespace,
    job: EvaluationJob,
    spec_sha256: str,
    source_checkpoint: Mapping[str, Any],
    workload_seeds: Sequence[int],
    *,
    validation_freeze_sha256: str | None = None,
) -> list[EpisodeMetrics]:
    expected_metadata = _expected_shard_metadata(
        job,
        spec_sha256,
        source_checkpoint,
        workload_seeds,
        validation_freeze_sha256,
    )
    try:
        return _load_valid_shard(
            args.output, job, expected_metadata, workload_seeds
        )
    except (FileNotFoundError, ValueError):
        pass

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
    observed_schema = {
        **policy.checkpoint_schema,
        "switch_feature_index": SWITCH_FEATURE_INDEX,
        "stay_bonus": policy.stay_bonus,
    }
    if observed_schema != expected_metadata["policy_schema"]:
        raise ValueError(
            f"loaded policy schema mismatch for {job.job_id}: "
            f"{observed_schema} != {expected_metadata['policy_schema']}"
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
    validate_switch_rows(rows)
    shard_path, metadata_path = _evaluation_paths(args.output, job)
    atomic_write_csv(shard_path, (asdict(row) for row in rows))
    validated_rows = v1.validate_evaluation_shard(
        shard_path, job, workload_seeds
    )
    validate_switch_rows(validated_rows)
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
    validation_freeze_sha256: str | None = None,
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
                validation_freeze_sha256=validation_freeze_sha256,
            )
        )
        print(f"[{completed}/{len(jobs)}] completed {job.job_id}", flush=True)
    expected_rows = len(jobs) * len(workload_seeds)
    if len(rows) != expected_rows:
        raise RuntimeError(f"evaluation has {len(rows)} rows, expected {expected_rows}")
    keys = {
        (row.scenario, row.policy, row.policy_seed, row.workload_seed)
        for row in rows
    }
    if len(keys) != expected_rows:
        raise RuntimeError("evaluation contains duplicate episode cells")
    validate_switch_rows(rows)
    return rows


def _cell_rows(
    rows: Sequence[EpisodeMetrics],
    scenario: str,
    policy_seed: int,
    arm: str,
    workload_seeds: Sequence[int],
) -> list[EpisodeMetrics]:
    selected = [
        row
        for row in rows
        if row.scenario == scenario
        and row.policy_seed == policy_seed
        and row.policy == POLICY_NAMES[arm]
    ]
    expected = {int(seed) for seed in workload_seeds}
    if len(selected) != len(expected):
        raise ValueError(f"incomplete gate cell: {scenario}/{policy_seed}/{arm}")
    if {row.workload_seed for row in selected} != expected:
        raise ValueError(f"workload mismatch: {scenario}/{policy_seed}/{arm}")
    return selected


def _mean(rows: Sequence[EpisodeMetrics], field: str) -> float:
    return float(np.mean([float(getattr(row, field)) for row in rows]))


def _micro_rate(rows: Sequence[EpisodeMetrics]) -> tuple[int, int, float]:
    avoidable = sum(row.avoidable_routing_switches for row in rows)
    opportunities = sum(row.switch_opportunities for row in rows)
    if opportunities <= 0:
        raise ValueError("micro avoidable-switch denominator must be positive")
    if avoidable > opportunities:
        raise ValueError("micro avoidable switches exceed opportunities")
    return avoidable, opportunities, avoidable / opportunities


def compute_cell_gates(
    rows: Sequence[EpisodeMetrics],
    workload_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    validate_switch_rows(rows)
    gates: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for policy_seed in POLICY_SEEDS:
            proposed = _cell_rows(
                rows, scenario, policy_seed, "proposed", workload_seeds
            )
            raw = _cell_rows(
                rows, scenario, policy_seed, "raw_context", workload_seeds
            )
            shield = _cell_rows(
                rows, scenario, policy_seed, "shield", workload_seeds
            )
            proposed_avoidable, proposed_opportunities, proposed_micro = (
                _micro_rate(proposed)
            )
            shield_avoidable, shield_opportunities, shield_micro = _micro_rate(
                shield
            )
            delivery_delta = _mean(shield, "delivery_ratio") - _mean(
                proposed, "delivery_ratio"
            )
            class_2_delta = _mean(
                shield, "class_2_delivery_ratio"
            ) - _mean(proposed, "class_2_delivery_ratio")
            shield_switch_mean = _mean(shield, "routing_switches")
            proposed_switch_mean = _mean(proposed, "routing_switches")
            raw_switch_mean = _mean(raw, "routing_switches")
            delivery_pass = (
                delivery_delta
                >= CELL_GATES["delivery_difference_vs_proposed_min"]
            )
            class_2_pass = (
                class_2_delta
                >= CELL_GATES[
                    "class_2_delivery_difference_vs_proposed_min"
                ]
            )
            avoidable_pass = shield_micro < proposed_micro
            switch_vs_proposed_pass = (
                shield_switch_mean <= proposed_switch_mean
            )
            switch_vs_raw_pass = shield_switch_mean <= raw_switch_mean
            gates.append(
                {
                    "scenario": scenario,
                    "policy_seed": policy_seed,
                    "workload_count": len(workload_seeds),
                    "shield_delivery_difference_vs_proposed": delivery_delta,
                    "shield_class_2_delivery_difference_vs_proposed": class_2_delta,
                    "proposed_avoidable_switches_total": proposed_avoidable,
                    "proposed_switch_opportunities_total": proposed_opportunities,
                    "proposed_avoidable_switch_micro_rate": proposed_micro,
                    "shield_avoidable_switches_total": shield_avoidable,
                    "shield_switch_opportunities_total": shield_opportunities,
                    "shield_avoidable_switch_micro_rate": shield_micro,
                    "proposed_routing_switch_mean": proposed_switch_mean,
                    "raw_context_routing_switch_mean": raw_switch_mean,
                    "shield_routing_switch_mean": shield_switch_mean,
                    "proposed_forced_switches_total": sum(
                        row.forced_routing_switches for row in proposed
                    ),
                    "shield_forced_switches_total": sum(
                        row.forced_routing_switches for row in shield
                    ),
                    "delivery_gate_pass": delivery_pass,
                    "class_2_delivery_gate_pass": class_2_pass,
                    "avoidable_switch_micro_gate_pass": avoidable_pass,
                    "routing_switch_vs_proposed_gate_pass": (
                        switch_vs_proposed_pass
                    ),
                    "routing_switch_vs_raw_context_gate_pass": switch_vs_raw_pass,
                    "all_gates_pass": all(
                        (
                            delivery_pass,
                            class_2_pass,
                            avoidable_pass,
                            switch_vs_proposed_pass,
                            switch_vs_raw_pass,
                        )
                    ),
                }
            )
    if len(gates) != len(SCENARIOS) * len(POLICY_SEEDS):
        raise RuntimeError("gate grid is incomplete")
    return gates


def _phase_artifact_paths(output: Path, phase: str) -> dict[str, Path]:
    return {
        "episode_metrics": output / f"{phase}_episode_metrics.csv",
        "aggregate_metrics": output / f"{phase}_aggregate_metrics.csv",
        "cell_gates": output / f"{phase}_cell_gates.csv",
        "decision": output / f"{phase}_decision.json",
    }


def write_phase_artifacts(
    output: Path,
    phase: str,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    rows: Sequence[EpisodeMetrics],
    *,
    validation_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    workload_seeds = _workloads_for_phase(phase)
    gates = compute_cell_gates(rows, workload_seeds)
    aggregate = v1.aggregate_rows(
        rows, rng_base=60000 if phase == "validation" else 61000
    )
    paths = _phase_artifact_paths(output, phase)
    atomic_write_csv(paths["episode_metrics"], (asdict(row) for row in rows))
    atomic_write_csv(paths["aggregate_metrics"], aggregate)
    atomic_write_csv(paths["cell_gates"], gates)
    artifacts = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
        if name != "decision"
    }
    artifacts["episode_metrics"]["row_count"] = len(rows)
    artifacts["aggregate_metrics"]["row_count"] = len(aggregate)
    artifacts["cell_gates"]["row_count"] = len(gates)
    passed = all(bool(row["all_gates_pass"]) for row in gates)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "phase": phase,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "input_validation_freeze_sha256": validation_freeze_sha256,
        "workload_seeds": list(workload_seeds),
        "cell_gates": gates,
        "all_cells_pass": passed,
        "decision": f"{phase}_{'pass' if passed else 'fail'}",
        "test_allowed": passed if phase == "validation" else None,
        "evaluation_shards": _shard_index(output, build_jobs(phase)),
        "artifacts": artifacts,
    }
    hash_field = (
        "validation_freeze_sha256" if phase == "validation" else "test_decision_sha256"
    )
    body[hash_field] = v1.sha256_json(body)
    v1.ensure_immutable_json(paths["decision"], body)
    return body


def _load_phase_shards(
    output: Path,
    jobs: Sequence[EvaluationJob],
    spec_sha256: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
    workload_seeds: Sequence[int],
    validation_freeze_sha256: str | None,
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for job in jobs:
        checkpoint = checkpoints[job.source_job_id]
        expected = _expected_shard_metadata(
            job,
            spec_sha256,
            checkpoint,
            workload_seeds,
            validation_freeze_sha256,
        )
        rows.extend(_load_valid_shard(output, job, expected, workload_seeds))
    return rows


def _validate_bound_artifact(
    artifact: Mapping[str, Any],
    expected_path: Path,
) -> None:
    path = Path(str(artifact.get("path", ""))).resolve()
    if path != expected_path.resolve():
        raise ValueError(f"validation artifact path mismatch: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"validation artifact hash mismatch: {path}")


def load_validation_freeze(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    path = args.output / "validation_decision.json"
    freeze = v1._load_json(path)
    _self_hash(freeze, "validation_freeze_sha256")
    if freeze.get("screen_name") != SCREEN_NAME:
        raise ValueError("validation freeze screen mismatch")
    if freeze.get("phase") != "validation":
        raise ValueError("validation freeze phase mismatch")
    if freeze.get("spec_sha256") != spec["spec_sha256"]:
        raise ValueError("validation freeze spec mismatch")
    if (
        freeze.get("source_training_freeze_sha256")
        != source["training_freeze_sha256"]
    ):
        raise ValueError("validation freeze source mismatch")
    if freeze.get("all_cells_pass") is not True:
        raise RuntimeError("validation did not pass every frozen cell gate")
    if freeze.get("test_allowed") is not True:
        raise RuntimeError("validation freeze does not allow test evaluation")

    observed_shard_index = _shard_index(
        args.output, build_jobs("validation")
    )
    if freeze.get("evaluation_shards") != observed_shard_index:
        raise ValueError("validation shard index does not replay")

    artifact_paths = _phase_artifact_paths(args.output, "validation")
    for name in ("episode_metrics", "aggregate_metrics", "cell_gates"):
        _validate_bound_artifact(freeze["artifacts"][name], artifact_paths[name])

    jobs = build_jobs("validation")
    rows = _load_phase_shards(
        args.output,
        jobs,
        spec["spec_sha256"],
        source["checkpoints"],
        VALIDATION_WORKLOAD_SEEDS,
        None,
    )
    combined_rows = v1._episode_rows_from_csv(artifact_paths["episode_metrics"])
    if combined_rows != rows:
        raise ValueError("validation combined episode artifact does not replay")
    replayed_gates = compute_cell_gates(rows, VALIDATION_WORKLOAD_SEEDS)
    if freeze.get("cell_gates") != replayed_gates:
        raise ValueError("validation gate decision does not replay")
    if not all(bool(row["all_gates_pass"]) for row in replayed_gates):
        raise RuntimeError("replayed validation gates fail")
    return freeze


def _shard_index(
    output: Path,
    jobs: Sequence[EvaluationJob],
) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for job in jobs:
        csv_path, metadata_path = _evaluation_paths(output, job)
        index.append(
            {
                "job_id": job.job_id,
                "csv_path": str(csv_path.resolve()),
                "csv_sha256": sha256_file(csv_path),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
    return index


def write_manifest(
    output: Path,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    validation_freeze: Mapping[str, Any],
    test_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validation_path = output / "validation_decision.json"
    test_path = output / "test_decision.json"
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "validation_freeze_sha256": validation_freeze[
            "validation_freeze_sha256"
        ],
        "validation_freeze_file_sha256": sha256_file(validation_path),
        "test_decision_sha256": test_decision["test_decision_sha256"],
        "test_decision_file_sha256": sha256_file(test_path),
        "validation_passed": validation_freeze["all_cells_pass"],
        "test_passed": test_decision["all_cells_pass"],
        "training_jobs": 0,
        "validation_jobs": len(build_jobs("validation")),
        "validation_rows": len(build_jobs("validation"))
        * len(VALIDATION_WORKLOAD_SEEDS),
        "test_jobs": len(build_jobs("test")),
        "test_rows": len(build_jobs("test")) * len(TEST_WORKLOAD_SEEDS),
        "validation_shards": _shard_index(
            output, build_jobs("validation")
        ),
        "test_shards": _shard_index(output, build_jobs("test")),
    }
    body["manifest_sha256"] = v1.sha256_json(body)
    v1.ensure_immutable_json(
        output / "proposed_shield_preflight_manifest.json", body
    )
    return body


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("LEO_REWARD_OVERRIDES", "").strip():
        raise RuntimeError("LEO_REWARD_OVERRIDES must be unset for this preflight")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    repository_root = args.project.parent.resolve()
    experiments_root = (repository_root / "experiments").resolve()
    broad_outputs = {
        repository_root,
        args.project.resolve(),
        experiments_root,
        (experiments_root / "archive").resolve(),
        args.source.resolve(),
    }
    if args.output.resolve() in broad_outputs:
        raise RuntimeError("output must be a dedicated experiment directory")
    if args.source.resolve() in args.output.resolve().parents:
        raise RuntimeError("output must not be inside the frozen source")
    if args.output.resolve() in args.source.resolve().parents:
        raise RuntimeError("source must not be inside the output")
    if args.output.is_dir() and not (
        args.output / "proposed_shield_preflight_spec.json"
    ).is_file():
        conflicting = [
            path.name
            for path in args.output.iterdir()
            if path.name.endswith("_spec.json") or path.name == "screen_spec.json"
        ]
        if conflicting:
            raise RuntimeError(f"output contains another experiment: {conflicting}")
    validation = set(VALIDATION_WORKLOAD_SEEDS)
    test = set(TEST_WORKLOAD_SEEDS)
    exposed = set().union(
        *(set(range(start, stop + 1)) for start, stop in KNOWN_EXPOSED_PANELS)
    )
    if validation & test:
        raise RuntimeError("validation and test panels overlap")
    if validation & exposed or test & exposed:
        raise RuntimeError("fresh panels overlap a known exposed panel")
    if STAY_BONUS != 0.40:
        raise RuntimeError("the v7 stay bonus contract changed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed proposed-checkpoint route shield."
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
        default=repository_root / "experiments" / "proposed-shield-preflight-v7",
    )
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--phase", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _dry_run_summary(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "screen_name": SCREEN_NAME,
        "inferential_status": spec["inferential_status"],
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "source_training_freeze_sha256": source["training_freeze_sha256"],
        "training_jobs": 0,
        "validation_jobs": len(build_jobs("validation")),
        "validation_rows": len(build_jobs("validation"))
        * len(VALIDATION_WORKLOAD_SEEDS),
        "test_jobs": len(build_jobs("test")),
        "test_rows": len(build_jobs("test")) * len(TEST_WORKLOAD_SEEDS),
        "validation_workloads": [
            VALIDATION_WORKLOAD_SEEDS[0],
            VALIDATION_WORKLOAD_SEEDS[-1],
        ],
        "test_workloads": [TEST_WORKLOAD_SEEDS[0], TEST_WORKLOAD_SEEDS[-1]],
        "policy_seeds": list(POLICY_SEEDS),
        "stay_bonus": STAY_BONUS,
        "requested_phase": args.phase,
        "output": str(args.output),
        "dry_run_writes_output": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in ("source", "output", "project"):
        setattr(args, field, getattr(args, field).resolve())
    validate_environment(args)
    source = v1.audit_source_screen(args.source)
    spec = build_spec(args, source)
    if args.dry_run:
        print(json.dumps(_dry_run_summary(args, spec, source), indent=2))
        return 0

    preliminary_validation_freeze: dict[str, Any] | None = None
    if args.phase == "test":
        preliminary_validation_freeze = load_validation_freeze(
            args, spec, source
        )

    args.output.mkdir(parents=True, exist_ok=True)
    v1.ensure_immutable_json(
        args.output / "proposed_shield_preflight_spec.json", spec
    )
    jobs = build_jobs(args.phase)
    atomic_write_csv(
        args.output / f"{args.phase}_plan.csv",
        (job.as_dict() for job in jobs),
    )
    invocation_id = uuid.uuid4().hex
    invocation_path = args.output / "invocations" / f"{invocation_id}.json"
    invocation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "invocation_id": invocation_id,
        "spec_sha256": spec["spec_sha256"],
        "phase": args.phase,
        "device": args.device,
        "started_at_utc": utc_now(),
        "status": "running",
    }
    atomic_write_json(invocation_path, invocation)
    try:
        with v1.invocation_lock(args.output):
            validation_freeze = preliminary_validation_freeze
            if args.phase == "test":
                validation_freeze = load_validation_freeze(args, spec, source)
            validation_freeze_sha256 = (
                validation_freeze["validation_freeze_sha256"]
                if validation_freeze is not None
                else None
            )
            workloads = _workloads_for_phase(args.phase)
            rows = evaluate_jobs(
                args,
                jobs,
                spec["spec_sha256"],
                source["checkpoints"],
                workloads,
                validation_freeze_sha256=validation_freeze_sha256,
            )
            decision = write_phase_artifacts(
                args.output,
                args.phase,
                spec,
                source,
                rows,
                validation_freeze_sha256=validation_freeze_sha256,
            )
            manifest = None
            if args.phase == "test":
                if validation_freeze is None:
                    raise RuntimeError("test phase lost its validation freeze")
                manifest = write_manifest(
                    args.output,
                    spec,
                    source,
                    validation_freeze,
                    decision,
                )
            passed = bool(decision["all_cells_pass"])
            invocation.update(
                status="completed" if passed else "completed_gate_failed",
                finished_at_utc=utc_now(),
                decision=decision["decision"],
                decision_sha256=decision[
                    "validation_freeze_sha256"
                    if args.phase == "validation"
                    else "test_decision_sha256"
                ],
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
