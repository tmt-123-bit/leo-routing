"""Run the frozen diagnostic preflight for switch-regularized MAPPO."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import uuid

import torch

from ablation_matrix_runner import atomic_write_csv, atomic_write_json, sha256_file, utc_now
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from leo_multiagent_env import (
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_SWITCH_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
)
from mappo_design import (
    route_hysteresis_residual_dtype_cap,
    select_leo_validation_record,
)
import run_integrated_hysteresis_screen as v1
from run_exp004_mappo import code_fingerprint
from run_hysteresis_screen import invocation_lock


SCREEN_NAME = "SWITCH-REGULARIZED-PREFLIGHT-v3"
SCHEMA_VERSION = 3
EXPERIMENT_RUNNER_PATH = Path(__file__).resolve()
PREFLIGHT_ORCHESTRATION_PATH = Path(__file__).resolve()
PROTOCOL_FILENAME = "SWITCH_REGULARIZED_PREFLIGHT_V3.md"
SUMMARY_TITLE = "Switch-Regularized MAPPO Preflight v3"
DEFAULT_OUTPUT_DIRECTORY_NAME = "switch-regularized-preflight-6k-v3"
PARSER_DESCRIPTION = (
    "Run four 6k-step schema-v3 diagnostic training jobs and a fresh "
    "holdout evaluation."
)
ENVIRONMENT_VARIANT = "with_congestion_context"
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072)
TRAIN_WORKLOAD_SEEDS = tuple(range(9001, 9201))
VALIDATION_WORKLOAD_SEEDS = tuple(range(43001, 43011))
TEST_WORKLOAD_SEEDS = tuple(range(44001, 44026))

INTEGRATED_BETA = 0.20
URGENCY_RELIEF = 0.50
CLASS_2_RELIEF = 0.0
AVOIDABLE_SWITCH_PROBABILITY_COEF = 0.10
VALIDATION_SELECTION_MODE = "avoidable_stability_constrained"
VALIDATION_DELIVERY_TOLERANCE = 0.003
VALIDATION_CLASS_2_TOLERANCE = 0.010
CANDIDATE_FEATURE_SCHEMA_ID = "leo_multi_candidate_features_v1_dim_28"
CANDIDATE_FEATURE_SCHEMA_SHA256 = (
    "c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503"
)
PAIRED_ANALYSIS = "diagnostic_switch_regularized_preflight_v3"
POLICY_NAMES = {
    **v1.POLICY_NAMES,
    v1.ARM_INTEGRATED: (
        "mappo_context_switch_regularized_v3_beta_0p20_coef_0p10"
    ),
}

CONFIG = {
    "scenarios": list(SCENARIOS),
    "timesteps": 6000,
    "validation_episodes": len(VALIDATION_WORKLOAD_SEEDS),
    "test_episodes": len(TEST_WORKLOAD_SEEDS),
    "eval_every_rollouts": 5,
    "save_every_steps": 1500,
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
    "--route-hysteresis-mode",
    "decoupled_adaptive",
    "--route-urgency-feature-index",
    str(ROUTE_URGENCY_FEATURE_INDEX),
    "--route-class-2-feature-index",
    str(ROUTE_CLASS_2_FEATURE_INDEX),
    "--route-hysteresis-urgency-relief",
    str(URGENCY_RELIEF),
    "--route-hysteresis-class-2-relief",
    str(CLASS_2_RELIEF),
    "--avoidable-switch-probability-coef",
    str(AVOIDABLE_SWITCH_PROBABILITY_COEF),
    "--validation-selection-mode",
    VALIDATION_SELECTION_MODE,
    "--validation-delivery-tolerance",
    str(VALIDATION_DELIVERY_TOLERANCE),
    "--validation-class-2-tolerance",
    str(VALIDATION_CLASS_2_TOLERANCE),
)

EXPECTED_ACTOR_SPEC = {
    "schema_version": 3,
    "type": "shared_candidate_actor",
    "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
    "route_hysteresis_beta": INTEGRATED_BETA,
    "route_hysteresis_mode": "decoupled_adaptive",
    "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
    "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
    "route_hysteresis_urgency_relief": URGENCY_RELIEF,
    "route_hysteresis_class_2_relief": CLASS_2_RELIEF,
    "avoidable_switch_probability_coef": AVOIDABLE_SWITCH_PROBABILITY_COEF,
    "candidate_feature_schema_id": CANDIDATE_FEATURE_SCHEMA_ID,
    "candidate_feature_schema_sha256": CANDIDATE_FEATURE_SCHEMA_SHA256,
}
EXPECTED_SWITCH_REGULARIZER_SPEC: dict[str, Any] | None = None
EXPECTED_EXTRA_RUN_CONFIG: dict[str, Any] = {}
SWITCH_REGULARIZER_FREEZE_RATIONALE = (
    "coefficient 0.10 fixed before the fresh validation and test panels; "
    "targets the avoidable-switch probability identified by the v2 diagnostic"
)

EXPECTED_SELECTION_SPEC = {
    "schema_version": 3,
    "mode": VALIDATION_SELECTION_MODE,
    "delivery_tolerance": VALIDATION_DELIVERY_TOLERANCE,
    "class_2_tolerance": VALIDATION_CLASS_2_TOLERANCE,
    "source": "validation_only",
    "test_panel_consulted": False,
    "validation_seed_start": VALIDATION_WORKLOAD_SEEDS[0],
    "validation_episodes": len(VALIDATION_WORKLOAD_SEEDS),
    "metric_aggregation": {
        "delivery_ratio": "macro_mean_over_validation_episodes",
        "class_2_delivery_ratio": "macro_mean_over_validation_episodes",
        "routing_switches": "mean_of_episode_total_switch_counts",
        "avoidable_switch_rate": "micro_ratio_over_validation_switch_opportunities",
    },
    "selection_timing": "after_all_validation_candidates",
}

SELECTED_VALIDATION_METRIC_FIELDS = (
    "environment_steps",
    "episodes",
    "seed_start",
    "delivery_ratio",
    "mean_reward",
    "drop_rate",
    "average_delay_slots",
    "routing_switches",
    "routing_switches_total",
    "avoidable_routing_switches",
    "avoidable_routing_switches_total",
    "forced_routing_switches",
    "forced_routing_switches_total",
    "switch_opportunities",
    "avoidable_switch_rate",
    "class_2_delivery_ratio",
)

KNOWN_EXPOSED_PANELS = (
    (32001, 32020),
    (33001, 33050),
    (34001, 34020),
    (35001, 35050),
    (37001, 37050),
    (41001, 41010),
    (42001, 42025),
)

METRICS = (
    *v1.METRICS,
    "avoidable_routing_switches",
    "forced_routing_switches",
    "switch_opportunities",
    "avoidable_switch_rate",
)
LOWER_IS_BETTER_COSTS = (
    *v1.LOWER_IS_BETTER_COSTS,
    "avoidable_routing_switches",
    "forced_routing_switches",
    "avoidable_switch_rate",
)

_V1_OVERRIDES = {
    "SCREEN_NAME": SCREEN_NAME,
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "SCENARIOS": SCENARIOS,
    "POLICY_SEEDS": POLICY_SEEDS,
    "TRAIN_WORKLOAD_SEEDS": TRAIN_WORKLOAD_SEEDS,
    "VALIDATION_WORKLOAD_SEEDS": VALIDATION_WORKLOAD_SEEDS,
    "TEST_WORKLOAD_SEEDS": TEST_WORKLOAD_SEEDS,
    "INTEGRATED_BETA": INTEGRATED_BETA,
    "CONFIG": CONFIG,
    "TRAINER_OVERRIDES": TRAINER_OVERRIDES,
    "EXPECTED_ACTOR_SPEC": EXPECTED_ACTOR_SPEC,
    "EXPECTED_SWITCH_REGULARIZER_SPEC": EXPECTED_SWITCH_REGULARIZER_SPEC,
    "POLICY_NAMES": POLICY_NAMES,
    "METRICS": METRICS,
    "LOWER_IS_BETTER_COSTS": LOWER_IS_BETTER_COSTS,
}


@contextmanager
def configured_v1() -> Iterator[None]:
    """Temporarily bind v1's generic orchestration helpers to this frozen design."""

    saved = {name: getattr(v1, name) for name in _V1_OVERRIDES}
    saved_audit = v1.audit_integrated_checkpoint
    try:
        for name, value in _V1_OVERRIDES.items():
            setattr(v1, name, value)
        v1.audit_integrated_checkpoint = audit_switch_regularized_checkpoint
        yield
    finally:
        for name, value in saved.items():
            setattr(v1, name, value)
        v1.audit_integrated_checkpoint = saved_audit


def build_training_jobs() -> list[v1.TrainingJob]:
    with configured_v1():
        return v1.build_training_jobs()


def build_evaluation_jobs() -> list[v1.EvaluationJob]:
    with configured_v1():
        return v1.build_evaluation_jobs()


def training_job_record(job: v1.TrainingJob) -> dict[str, Any]:
    record = {
        **job.as_dict(),
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "validation_selection": dict(EXPECTED_SELECTION_SPEC),
        "timesteps": CONFIG["timesteps"],
        "validation_workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
    }
    if EXPECTED_SWITCH_REGULARIZER_SPEC is not None:
        record["switch_regularizer_spec"] = dict(
            EXPECTED_SWITCH_REGULARIZER_SPEC
        )
    return record


def evaluation_job_record(job: v1.EvaluationJob) -> dict[str, Any]:
    record = job.as_dict()
    record["policy_name"] = POLICY_NAMES[job.arm]
    return record


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(run_directory: Path, manifest: Mapping[str, Any], field: str) -> Path:
    value = manifest.get(field)
    if not value:
        raise ValueError(f"run manifest lacks {field}")
    path = Path(str(value))
    return (path if path.is_absolute() else run_directory / path.name).resolve()


def _expected_run_config(job: v1.TrainingJob, args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "env_type": "leo_multi",
        "env_name": job.scenario,
        "leo_variant": ENVIRONMENT_VARIANT,
        "route_hysteresis_beta": INTEGRATED_BETA,
        "route_hysteresis_mode": "decoupled_adaptive",
        "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
        "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
        "route_hysteresis_urgency_relief": URGENCY_RELIEF,
        "route_hysteresis_class_2_relief": CLASS_2_RELIEF,
        "avoidable_switch_probability_coef": AVOIDABLE_SWITCH_PROBABILITY_COEF,
        "validation_selection_mode": VALIDATION_SELECTION_MODE,
        "validation_delivery_tolerance": VALIDATION_DELIVERY_TOLERANCE,
        "validation_class_2_tolerance": VALIDATION_CLASS_2_TOLERANCE,
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
    if EXPECTED_SWITCH_REGULARIZER_SPEC is not None:
        expected.update(
            avoidable_switch_regularization_mode=(
                EXPECTED_SWITCH_REGULARIZER_SPEC["mode"]
            ),
            avoidable_switch_logit_margin=(
                EXPECTED_SWITCH_REGULARIZER_SPEC["logit_margin"]
            ),
        )
    expected.update(EXPECTED_EXTRA_RUN_CONFIG)
    return expected


def _audit_candidate_actor_learned_state(
    checkpoint: Mapping[str, Any],
) -> dict[str, float] | None:
    spec = checkpoint.get("candidate_actor_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("checkpoint candidate actor spec is invalid")
    schema_version = int(spec.get("schema_version", -1))
    learned_state = checkpoint.get("candidate_actor_learned_state")
    actor_state = checkpoint.get("actor")
    parameter_names_by_schema = {
        4: ("route_hysteresis_residual_bias",),
        5: (
            "route_hysteresis_residual_calm_bias",
            "route_hysteresis_residual_urgent_bias",
        ),
    }
    known_parameter_keys = {
        f"shared_candidate_actor.{name}"
        for names in parameter_names_by_schema.values()
        for name in names
    }
    if schema_version < 4:
        if learned_state is not None or (
            isinstance(actor_state, Mapping)
            and known_parameter_keys.intersection(actor_state)
        ):
            raise ValueError("pre-v4 checkpoint contains route residual state")
        return None
    parameter_names = parameter_names_by_schema.get(schema_version)
    if parameter_names is None:
        raise ValueError("unsupported residual checkpoint schema")
    if not isinstance(actor_state, Mapping):
        raise ValueError("residual checkpoint actor state is invalid")
    if not isinstance(learned_state, Mapping) or set(learned_state) != set(
        parameter_names
    ):
        raise ValueError("residual checkpoint learned actor state mismatch")
    unexpected_parameters = known_parameter_keys.intersection(actor_state).difference(
        {f"shared_candidate_actor.{name}" for name in parameter_names}
    )
    if unexpected_parameters:
        raise ValueError("residual checkpoint mixes parameterizations")
    cap = float(spec.get("route_hysteresis_residual_cap", math.nan))
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("learned route residual cap is invalid")
    audited = {}
    for parameter_name in parameter_names:
        raw_value = learned_state[parameter_name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError("learned route residual must be numeric")
        value = float(raw_value)
        tensor = actor_state.get(f"shared_candidate_actor.{parameter_name}")
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
            raise ValueError("learned route residual tensor is invalid")
        dtype_cap = route_hysteresis_residual_dtype_cap(cap, dtype=tensor.dtype)
        if (
            not math.isfinite(value)
            or not 0.0 <= value <= dtype_cap
            or not bool(torch.isfinite(tensor).all())
            or float(tensor.detach().cpu().item()) != value
        ):
            raise ValueError("learned route residual state is invalid or inconsistent")
        audited[parameter_name] = value
    return audited


def _audit_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    job: v1.TrainingJob,
    expected_step: int,
) -> None:
    if checkpoint.get("candidate_actor_spec") != EXPECTED_ACTOR_SPEC:
        raise ValueError(f"candidate actor spec mismatch for {job.job_id}")
    if (
        EXPECTED_SWITCH_REGULARIZER_SPEC is not None
        and checkpoint.get("switch_regularizer_spec")
        != EXPECTED_SWITCH_REGULARIZER_SPEC
    ):
        raise ValueError(f"switch regularizer spec mismatch for {job.job_id}")
    if checkpoint.get("validation_selection_spec") != EXPECTED_SELECTION_SPEC:
        raise ValueError(f"validation selection spec mismatch for {job.job_id}")
    actor_args = checkpoint.get("args")
    if not isinstance(actor_args, dict):
        raise ValueError(f"checkpoint actor args are invalid for {job.job_id}")
    expected = _expected_run_config(job, argparse.Namespace(device=actor_args.get("device")))
    mismatches = {
        key: {"observed": actor_args.get(key), "expected": value}
        for key, value in expected.items()
        if actor_args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint run configuration mismatch: {mismatches}")
    schema = {
        "candidate_feature_dim": 28,
        "action_size": 7,
        "n_agents": 24,
        "obs_size": 196,
    }
    for field, expected_value in schema.items():
        if int(checkpoint.get(field, -1)) != expected_value:
            raise ValueError(f"checkpoint {field} mismatch for {job.job_id}")
    critic_spec = checkpoint.get("critic_spec") or {}
    if int(critic_spec.get("node_feature_dim", -1)) != 26:
        raise ValueError(f"checkpoint critic schema mismatch for {job.job_id}")
    if int(checkpoint.get("step", -1)) != expected_step:
        raise ValueError(f"checkpoint step mismatch for {job.job_id}")
    _audit_candidate_actor_learned_state(checkpoint)


def _audit_validation_record_aggregation(record: Mapping[str, Any]) -> None:
    episodes = int(record.get("episodes", -1))
    if episodes <= 0:
        raise ValueError("validation episode count must be positive")

    totals_and_means = (
        ("routing_switches_total", "routing_switches"),
        ("avoidable_routing_switches_total", "avoidable_routing_switches"),
        ("forced_routing_switches_total", "forced_routing_switches"),
    )
    totals: dict[str, int] = {}
    for total_field, mean_field in totals_and_means:
        raw_total = record.get(total_field)
        if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
            raise ValueError(f"validation {total_field} must be a non-negative integer")
        totals[total_field] = raw_total
        observed_mean = float(record.get(mean_field, math.nan))
        expected_mean = raw_total / episodes
        if not math.isfinite(observed_mean) or not math.isclose(
            observed_mean, expected_mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"validation {mean_field} aggregation mismatch")

    opportunities = record.get("switch_opportunities")
    if (
        isinstance(opportunities, bool)
        or not isinstance(opportunities, int)
        or opportunities < 0
    ):
        raise ValueError("validation switch_opportunities must be a non-negative integer")
    avoidable_total = totals["avoidable_routing_switches_total"]
    if avoidable_total > opportunities:
        raise ValueError("validation avoidable switches exceed switch opportunities")
    expected_rate = avoidable_total / max(1, opportunities)
    observed_rate = float(record.get("avoidable_switch_rate", math.nan))
    if not math.isfinite(observed_rate) or not math.isclose(
        observed_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("validation avoidable switch rate aggregation mismatch")


def audit_switch_regularized_checkpoint(
    checkpoint_path: Path,
    job: v1.TrainingJob,
    args: argparse.Namespace,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file() or not v1._path_is_within(checkpoint_path, args.output):
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

    run_config = _load_json(paths["run_config"])
    expected_config = _expected_run_config(job, args)
    mismatches = {
        key: {"observed": run_config.get(key), "expected": value}
        for key, value in expected_config.items()
        if run_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint run configuration mismatch: {mismatches}")

    run_manifest = _load_json(paths["run_manifest"])
    if run_manifest.get("candidate_actor_spec") != EXPECTED_ACTOR_SPEC:
        raise ValueError("run manifest candidate actor spec mismatch")
    if (
        EXPECTED_SWITCH_REGULARIZER_SPEC is not None
        and run_manifest.get("switch_regularizer_spec")
        != EXPECTED_SWITCH_REGULARIZER_SPEC
    ):
        raise ValueError("run manifest switch regularizer spec mismatch")
    if run_manifest.get("validation_selection_spec") != EXPECTED_SELECTION_SPEC:
        raise ValueError("run manifest validation selection spec mismatch")
    if _manifest_path(run_directory, run_manifest, "validation_best_checkpoint") != checkpoint_path:
        raise ValueError("run manifest does not select the audited checkpoint")
    if _manifest_path(run_directory, run_manifest, "final_checkpoint") != paths["final_checkpoint"].resolve():
        raise ValueError("run manifest final checkpoint path mismatch")

    expected_fingerprint = {
        "code": code_fingerprint(args),
        "scenario": job.scenario,
        "policy_seed": job.policy_seed,
        "experiment_variant": ENVIRONMENT_VARIANT,
        "environment_variant": ENVIRONMENT_VARIANT,
        "variant_definition": v1.resolve_variant(ENVIRONMENT_VARIANT).as_dict(),
        "config": dict(CONFIG),
        "trainer_overrides": list(TRAINER_OVERRIDES),
    }
    if _load_json(paths["job_fingerprint"]) != expected_fingerprint:
        raise ValueError(f"per-job training fingerprint mismatch for {job.job_id}")

    actual_steps = int(run_manifest.get("environment_steps", -1))
    step_span = CONFIG["batch_size"] * 30
    if not CONFIG["timesteps"] <= actual_steps < CONFIG["timesteps"] + step_span:
        raise ValueError(f"actual training steps out of range: {actual_steps}")

    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    final = torch.load(paths["final_checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(selected, dict) or not isinstance(final, dict):
        raise ValueError("checkpoint payload is not a mapping")
    selected_step = int(selected.get("step", -1))
    _audit_checkpoint_payload(selected, job, selected_step)
    _audit_checkpoint_payload(final, job, actual_steps)
    if selected.get("args") != run_config or final.get("args") != run_config:
        raise ValueError("checkpoint args differ from run config")
    selected_learned_state = _audit_candidate_actor_learned_state(selected)
    final_learned_state = _audit_candidate_actor_learned_state(final)
    actor_schema_version = int(EXPECTED_ACTOR_SPEC.get("schema_version", -1))
    residual_parameter_names = {
        4: ("route_hysteresis_residual_bias",),
        5: (
            "route_hysteresis_residual_calm_bias",
            "route_hysteresis_residual_urgent_bias",
        ),
    }.get(actor_schema_version, ())
    if residual_parameter_names:
        if (
            run_manifest.get("selected_candidate_actor_learned_state")
            != selected_learned_state
            or run_manifest.get("final_candidate_actor_learned_state")
            != final_learned_state
        ):
            raise ValueError("run manifest learned actor state mismatch")

    records = v1._read_training_metrics(paths["training_metrics"])
    if residual_parameter_names:
        training_updates = [
            record
            for record in records
            if record.get("record_type") == "training_update"
        ]
        if not training_updates:
            raise ValueError("residual training update records are missing")
        finite_fields = (
            "actor_loss",
            "actor_primary_loss",
            "avoidable_switch_regularization_penalty",
            "avoidable_switch_regularization_loss",
            "actor_primary_gradient_norm",
            "avoidable_switch_regularization_gradient_norm",
        )
        residual_states = []
        cap = float(EXPECTED_ACTOR_SPEC["route_hysteresis_residual_cap"])
        final_actor_state = final["actor"]
        dtype_caps = {
            name: route_hysteresis_residual_dtype_cap(
                cap,
                dtype=final_actor_state[
                    f"shared_candidate_actor.{name}"
                ].dtype,
            )
            for name in residual_parameter_names
        }
        for record in training_updates:
            values = [float(record.get(field, math.nan)) for field in finite_fields]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("residual training loss record is non-finite")
            if not math.isclose(
                float(record["actor_loss"]),
                float(record["actor_primary_loss"])
                + float(record["avoidable_switch_regularization_loss"]),
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError("residual actor loss identity mismatch")
            state = record.get("candidate_actor_learned_state")
            if not isinstance(state, Mapping) or set(state) != set(
                residual_parameter_names
            ):
                raise ValueError("training record lacks learned actor state")
            normalized_state = {name: float(state[name]) for name in state}
            if any(
                not math.isfinite(value)
                or not 0.0 <= value <= dtype_caps[name]
                for name, value in normalized_state.items()
            ):
                raise ValueError("training residual is outside its cap")
            residual_states.append(normalized_state)
            if actor_schema_version == 5:
                cap_hits = record.get("route_hysteresis_residual_cap_hit")
                if not isinstance(cap_hits, Mapping) or set(cap_hits) != set(
                    residual_parameter_names
                ) or any(type(value) is not bool for value in cap_hits.values()):
                    raise ValueError("training residual cap-hit audit is invalid")
                if record.get("avoidable_switch_eligibility_stage") != "pre_contention":
                    raise ValueError("schema-v5 eligibility stage mismatch")
                active_count = int(record.get("rollout_active_policy_decisions", -1))
                eligible_count = int(
                    record.get("rollout_avoidable_switch_eligible_decisions", -1)
                )
                if active_count <= 0 or not 0 <= eligible_count <= active_count:
                    raise ValueError("schema-v5 rollout micro counts are invalid")
                primary_signed = record.get(
                    "route_hysteresis_residual_primary_signed_gradient"
                )
                regularizer_signed = record.get(
                    "route_hysteresis_residual_regularizer_signed_gradient"
                )
                for gradients, label in (
                    (primary_signed, "primary"),
                    (regularizer_signed, "regularizer"),
                ):
                    if not isinstance(gradients, Mapping) or set(gradients) != set(
                        residual_parameter_names
                    ) or not all(
                        math.isfinite(float(value)) for value in gradients.values()
                    ):
                        raise ValueError(f"schema-v5 {label} gradient audit is invalid")
                if any(float(value) != 0.0 for value in primary_signed.values()):
                    raise ValueError("schema-v5 PPO gradient reached residual endpoints")
        if not any(
            float(record["avoidable_switch_regularization_gradient_norm"]) > 0.0
            for record in training_updates
        ):
            raise ValueError("regularizer never reached its residual")
        if residual_states[-1] != final_learned_state:
            raise ValueError("final learned actor state differs from training log")
    validation = [record for record in records if record.get("record_type") == "validation"]
    selections = [record for record in records if record.get("record_type") == "validation_selection"]
    if not validation or len(selections) != 1:
        raise ValueError("validation candidate or selection record is missing")
    if int(run_manifest.get("validation_candidate_count", -1)) != len(validation):
        raise ValueError("validation candidate count mismatch")
    if any(
        int(record.get("episodes", -1)) != len(VALIDATION_WORKLOAD_SEEDS)
        or int(record.get("seed_start", -1)) != VALIDATION_WORKLOAD_SEEDS[0]
        for record in validation
    ):
        raise ValueError("validation workload panel mismatch")
    for record in validation:
        _audit_validation_record_aggregation(record)

    selection = selections[0]
    if selection.get("selection_spec") != EXPECTED_SELECTION_SPEC:
        raise ValueError("training metrics validation selection spec mismatch")
    if int(selection.get("selected_candidate_step", -1)) != selected_step:
        raise ValueError("selected candidate step mismatch")
    if selection.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("selected checkpoint hash mismatch")
    if run_manifest.get("selected_validation_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("run manifest selected checkpoint hash mismatch")

    selected_metrics = run_manifest.get("selected_validation_metrics")
    if not isinstance(selected_metrics, dict) or int(selected_metrics.get("environment_steps", -1)) != selected_step:
        raise ValueError("run manifest selected validation metrics mismatch")
    missing_selected_fields = set(SELECTED_VALIDATION_METRIC_FIELDS).difference(
        selected_metrics
    )
    if missing_selected_fields:
        raise ValueError(
            "run manifest selected validation metrics lack fields: "
            f"{sorted(missing_selected_fields)}"
        )
    if selection.get("selected_metrics") != selected_metrics:
        raise ValueError("selection record metrics differ from run manifest")
    selected_candidates = [
        record for record in validation if int(record.get("environment_steps", -1)) == selected_step
    ]
    if len(selected_candidates) != 1:
        raise ValueError("selected validation candidate is not unique")
    for field in SELECTED_VALIDATION_METRIC_FIELDS:
        if not math.isclose(
            float(selected_metrics[field]),
            float(selected_candidates[0][field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"selected validation {field} mismatch")
    expected_selected = select_leo_validation_record(
        validation,
        mode=VALIDATION_SELECTION_MODE,
        delivery_tolerance=VALIDATION_DELIVERY_TOLERANCE,
        class_2_tolerance=VALIDATION_CLASS_2_TOLERANCE,
    )
    if int(expected_selected["environment_steps"]) != selected_step:
        raise ValueError("selected checkpoint violates frozen validation rule")
    selected_candidate_path = Path(
        str(selected_candidates[0]["candidate_checkpoint"])
    ).resolve()
    if Path(str(selection.get("selected_candidate_checkpoint", ""))).resolve() != selected_candidate_path:
        raise ValueError("selection record candidate checkpoint path mismatch")
    if Path(str(selection.get("validation_best_checkpoint", ""))).resolve() != checkpoint_path:
        raise ValueError("selection record validation-best checkpoint path mismatch")
    candidate_steps: set[int] = set()
    for record in validation:
        candidate_path = Path(str(record.get("candidate_checkpoint", ""))).resolve()
        if not candidate_path.is_file() or candidate_path.parent != run_directory:
            raise FileNotFoundError(f"validation candidate checkpoint is invalid: {candidate_path}")
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if not isinstance(candidate, dict):
            raise ValueError("validation candidate checkpoint is not a mapping")
        candidate_step = int(record["environment_steps"])
        if candidate_step in candidate_steps:
            raise ValueError("duplicate validation candidate step")
        candidate_steps.add(candidate_step)
        _audit_checkpoint_payload(candidate, job, candidate_step)
        if candidate.get("args") != run_config:
            raise ValueError("validation candidate args differ from run config")
        candidate_learned_state = _audit_candidate_actor_learned_state(candidate)
        if residual_parameter_names and (
            record.get("candidate_actor_learned_state")
            != candidate_learned_state
        ):
            raise ValueError("validation record learned actor state mismatch")

    artifact_paths = {"selected_checkpoint": checkpoint_path, **paths}
    audit = {
        "job_id": job.job_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_step": selected_step,
        "actual_environment_steps": actual_steps,
        "best_validation_score": [float(value) for value in run_manifest["best_validation_score"]],
        "selected_validation_metrics": dict(selected_metrics),
        "validation_selection_spec": dict(EXPECTED_SELECTION_SPEC),
        "candidate_feature_dim": 28,
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "switch_regularizer_spec": (
            dict(EXPECTED_SWITCH_REGULARIZER_SPEC)
            if EXPECTED_SWITCH_REGULARIZER_SPEC is not None
            else None
        ),
        "critic_spec": dict(selected["critic_spec"]),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
    }
    if selected_learned_state is not None:
        audit["selected_candidate_actor_learned_state"] = selected_learned_state
        audit["final_candidate_actor_learned_state"] = final_learned_state
    return audit


def screen_code_fingerprint(args: argparse.Namespace) -> dict[str, str]:
    repository_root = args.project.parent.resolve()
    external_design = (args.cleanmarl / "cleanmarl" / "mappo_design.py").resolve()
    repository_design = (args.project / "mappo_design.py").resolve()
    if sha256_file(external_design) != sha256_file(repository_design):
        raise RuntimeError(
            "external CleanMARL design module differs from the repository snapshot"
        )
    files = {
        "preflight_runner": EXPERIMENT_RUNNER_PATH,
        "preflight_orchestration_dependency": PREFLIGHT_ORCHESTRATION_PATH,
        "preflight_protocol": repository_root
        / "docs"
        / PROTOCOL_FILENAME,
        "v1_orchestration_dependency": args.project / "run_integrated_hysteresis_screen.py",
        "external_design": external_design,
    }
    return {
        **code_fingerprint(args),
        **{name: sha256_file(path) for name, path in files.items()},
    }


def build_screen_spec(args: argparse.Namespace, source: Mapping[str, Any]) -> dict[str, Any]:
    training_jobs = build_training_jobs()
    evaluation_jobs = build_evaluation_jobs()
    body = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "design": "short_mechanism_preflight_with_frozen_references",
        "environment_variant": ENVIRONMENT_VARIANT,
        "integrated_policy_contract": dict(EXPECTED_ACTOR_SPEC),
        "urgency_relief_freeze_rationale": (
            "mechanism-level pre-test choice; maximum urgency retains beta=0.10; "
            "not selected using validation or test outcomes"
        ),
        "class_2_relief_freeze_rationale": (
            "fixed at zero before the preflight to avoid post-hoc tuning after v1"
        ),
        "switch_regularizer_freeze_rationale": (
            SWITCH_REGULARIZER_FREEZE_RATIONALE
        ),
        "validation_selection": dict(EXPECTED_SELECTION_SPEC),
        "config": dict(CONFIG),
        "scenarios": list(SCENARIOS),
        "policy_seeds": list(POLICY_SEEDS),
        "train_workload_seeds": list(TRAIN_WORKLOAD_SEEDS),
        "validation_workload_seeds": list(VALIDATION_WORKLOAD_SEEDS),
        "validation_panel_status": "fresh_frozen_selection_only",
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "test_panel_status": "fresh_frozen_diagnostic_holdout",
        "known_exposed_panels": [list(panel) for panel in KNOWN_EXPOSED_PANELS],
        "trainer_overrides": list(TRAINER_OVERRIDES),
        "evaluation_arms": list(v1.EVALUATION_ARMS),
        "policy_names": dict(POLICY_NAMES),
        "metrics": list(v1.METRICS),
        "decision_rule": "diagnostic_only_no_promotion_or_paper_claim",
        "expected_training_jobs": len(training_jobs),
        "expected_evaluation_jobs": len(evaluation_jobs),
        "expected_evaluation_rows": len(evaluation_jobs) * len(TEST_WORKLOAD_SEEDS),
        "training_jobs": [training_job_record(job) for job in training_jobs],
        "evaluation_jobs": [evaluation_job_record(job) for job in evaluation_jobs],
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
                    / PROTOCOL_FILENAME
                ).resolve()
            ),
        },
    }
    if EXPECTED_SWITCH_REGULARIZER_SPEC is not None:
        body["switch_regularizer_contract"] = dict(
            EXPECTED_SWITCH_REGULARIZER_SPEC
        )
    body["spec_sha256"] = v1.sha256_json(body)
    return body


def raw_exploratory_statistics_manifest() -> dict[str, Any]:
    manifest = v1.statistical_analysis_manifest()
    manifest["inferential_status"] = "raw_exploratory_only"
    manifest["hypothesis_test"] = {
        "method": "exact_policy_seed_sign_flip_on_workload_means",
        "two_sided": True,
        "tail_ties_included": True,
        "episode_rows_treated_as_independent": False,
        "reported_field": "raw_p_value",
        "status": "exploratory_diagnostic_only",
    }
    manifest["multiple_comparisons"] = {
        "adjustments_applied": [],
        "holm_applied": False,
        "benjamini_hochberg_applied": False,
        "reported_p_values": "raw_two_sided_exact_sign_flip_p_values_only",
        "claim_limit": "descriptive diagnostic evidence; no confirmatory inference",
    }
    return manifest


def _is_disallowed_p_value_field(field: str) -> bool:
    normalized = field.lower()
    if normalized == "raw_p_value":
        return False
    return (
        "p_value" in normalized
        or normalized.endswith("_p")
        or "holm" in normalized
        or "benjamini" in normalized
        or "adjusted" in normalized
        or "multiplicity" in normalized
        or normalized.startswith("seed_level_wilcoxon")
    )


def validate_raw_only_paired_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("paired effect rows must not be empty")
    for index, row in enumerate(rows):
        if row.get("analysis") != PAIRED_ANALYSIS:
            raise ValueError(f"paired row {index} has the wrong analysis label")
        if row.get("inferential_status") != "raw_exploratory_only":
            raise ValueError(f"paired row {index} has the wrong inferential status")
        if row.get("confirmatory") is not False:
            raise ValueError(f"paired row {index} must be non-confirmatory")
        if "raw_p_value" not in row:
            raise ValueError(f"paired row {index} lacks raw_p_value")
        disallowed = sorted(
            field for field in row if _is_disallowed_p_value_field(field)
        )
        if disallowed:
            raise ValueError(
                f"paired row {index} contains non-raw p-value fields: {disallowed}"
            )


def prepare_paired_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        normalized = {
            field: value
            for field, value in row.items()
            if not _is_disallowed_p_value_field(field)
        }
        normalized.update(
            analysis=PAIRED_ANALYSIS,
            inferential_status="raw_exploratory_only",
            confirmatory=False,
        )
        prepared.append(normalized)
    validate_raw_only_paired_rows(prepared)
    return prepared


def _diagnostic_decision(paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for scenario in SCENARIOS:
        delivery = v1._effect_row(paired, "integrated_minus_proposed", scenario, "delivery_ratio")
        switches = v1._effect_row(paired, "integrated_minus_proposed", scenario, "routing_switches")
        class_2 = v1._effect_row(paired, "integrated_minus_proposed", scenario, "class_2_delivery_ratio")
        avoidable_rate = v1._effect_row(
            paired,
            "integrated_minus_proposed",
            scenario,
            "avoidable_switch_rate",
        )
        switches_vs_raw = v1._effect_row(
            paired,
            "integrated_minus_raw_context",
            scenario,
            "routing_switches",
        )
        observations[scenario] = {
            "delivery_difference_vs_proposed": float(delivery["mean_difference"]),
            "routing_switch_difference_vs_proposed": float(switches["mean_difference"]),
            "class_2_delivery_difference_vs_proposed": float(class_2["mean_difference"]),
            "avoidable_switch_rate_difference_vs_proposed": float(
                avoidable_rate["mean_difference"]
            ),
            "routing_switch_difference_vs_raw_context": float(
                switches_vs_raw["mean_difference"]
            ),
            "positive_delivery_policy_seed_fraction": float(delivery["positive_policy_seed_fraction"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "decision": "diagnostic_only_no_promotion_decision",
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "policy_seed_count": len(POLICY_SEEDS),
        "observations": observations,
    }


def _write_summary(path: Path, paired: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]) -> None:
    lines = [
        f"# {SUMMARY_TITLE}",
        "",
        "This is a diagnostic exploratory preflight. It cannot support a paper claim or a promotion decision.",
        "",
        "The CSV reports raw two-sided exact sign-flip p-values only. No Holm or Benjamini-Hochberg adjustment was applied.",
        "",
        "| Scenario | Endpoint | Difference vs proposed | Raw p |",
        "|---|---|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for metric in (
            "delivery_ratio",
            "routing_switches",
            "avoidable_switch_rate",
            "class_2_delivery_ratio",
        ):
            row = v1._effect_row(paired, "integrated_minus_proposed", scenario, metric)
            lines.append(
                f"| {scenario} | {metric} | {float(row['mean_difference']):+.6f} | {float(row['raw_p_value']):.4f} |"
            )
    lines.extend(["", f"Decision label: `{decision['decision']}`.", ""])
    v1._atomic_write_text(path, "\n".join(lines))


def write_final_artifacts(
    output: Path,
    spec: Mapping[str, Any],
    training_freeze: Mapping[str, Any],
    jobs: Sequence[v1.EvaluationJob],
    rows: Sequence[v1.EpisodeMetrics],
) -> dict[str, Any]:
    episode_path = output / "preflight_episode_metrics.csv"
    aggregate_path = output / "preflight_aggregate_metrics.csv"
    paired_path = output / "preflight_paired_effects.csv"
    decision_path = output / "preflight_decision.json"
    summary_path = output / "SWITCH_REGULARIZED_PREFLIGHT_SUMMARY.md"
    episode_rows = [asdict(row) for row in rows]
    aggregate = v1.aggregate_rows(rows)
    paired = prepare_paired_rows(v1.paired_rows(rows))
    decision = _diagnostic_decision(paired)
    atomic_write_csv(episode_path, episode_rows)
    atomic_write_csv(aggregate_path, aggregate)
    atomic_write_csv(paired_path, paired)
    atomic_write_json(decision_path, decision)
    _write_summary(summary_path, paired, decision)
    artifact_paths = {
        "episode_metrics": episode_path,
        "aggregate_metrics": aggregate_path,
        "paired_effects": paired_path,
        "decision": decision_path,
        "summary": summary_path,
    }
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in artifact_paths.items()
    }
    artifacts["episode_metrics"]["row_count"] = len(episode_rows)
    artifacts["aggregate_metrics"]["row_count"] = len(aggregate)
    artifacts["paired_effects"]["row_count"] = len(paired)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "confirmatory": False,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "training_freeze_sha256": training_freeze["freeze_sha256"],
        "source_training_freeze_sha256": training_freeze["source_training_freeze_sha256"],
        "training_jobs": len(build_training_jobs()),
        "evaluation_jobs": len(jobs),
        "evaluation_rows": len(rows),
        "policy_seed_count": len(POLICY_SEEDS),
        "workload_seed_count": len(TEST_WORKLOAD_SEEDS),
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "switch_regularizer_spec": (
            dict(EXPECTED_SWITCH_REGULARIZER_SPEC)
            if EXPECTED_SWITCH_REGULARIZER_SPEC is not None
            else None
        ),
        "validation_selection": dict(EXPECTED_SELECTION_SPEC),
        "statistical_analysis": raw_exploratory_statistics_manifest(),
        "decision": decision,
        "evaluation_shards": v1._evaluation_artifact_index(output, jobs),
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = v1.sha256_json(manifest)
    v1.ensure_immutable_json(
        output / "switch_regularized_preflight_manifest.json", manifest
    )
    return manifest


def validate_environment(args: argparse.Namespace) -> None:
    v1.validate_environment(args)
    if args.output.is_dir() and not (
        args.output / "switch_regularized_preflight_spec.json"
    ).is_file():
        conflicting = [
            name
            for name in (
                "integrated_hysteresis_spec.json",
                "switch_regularized_preflight_manifest.json",
            )
            if (args.output / name).exists()
        ]
        if conflicting:
            raise RuntimeError(f"output contains another experiment: {conflicting}")
    if (ROUTE_SWITCH_FEATURE_INDEX, ROUTE_URGENCY_FEATURE_INDEX, ROUTE_CLASS_2_FEATURE_INDEX) != (17, 20, 23):
        raise RuntimeError("adaptive route feature contract changed")
    validation = set(VALIDATION_WORKLOAD_SEEDS)
    test = set(TEST_WORKLOAD_SEEDS)
    exposed = set().union(*(set(range(start, stop + 1)) for start, stop in KNOWN_EXPOSED_PANELS))
    if validation & test:
        raise RuntimeError("validation and test panels overlap")
    if validation & exposed or test & exposed:
        raise RuntimeError("fresh preflight panels overlap exposed panels")
    if set(TRAIN_WORKLOAD_SEEDS) & (validation | test):
        raise RuntimeError("training workloads overlap held-out panels")
    wrapper = CleanMARLLeoMultiAgentWrapper(
        "medium_load", variant=ENVIRONMENT_VARIANT
    )
    try:
        observed_schema = wrapper.get_candidate_feature_schema()
    finally:
        wrapper.close()
    observed_contract = {
        "candidate_feature_schema_id": observed_schema.get("schema_id"),
        "candidate_feature_schema_sha256": observed_schema.get("sha256"),
    }
    expected_contract = {
        "candidate_feature_schema_id": CANDIDATE_FEATURE_SCHEMA_ID,
        "candidate_feature_schema_sha256": CANDIDATE_FEATURE_SCHEMA_SHA256,
    }
    if observed_contract != expected_contract:
        raise RuntimeError(
            "with_congestion_context candidate feature schema changed: "
            f"{observed_contract} != {expected_contract}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=PARSER_DESCRIPTION
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=repository_root / "experiments" / "archive" / "congestion-context-screen-20k-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "experiments" / DEFAULT_OUTPUT_DIRECTORY_NAME,
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
        (training_job_record(job) for job in build_training_jobs()),
    )
    atomic_write_csv(
        output / "evaluation_plan.csv",
        (evaluation_job_record(job) for job in build_evaluation_jobs()),
    )


def _dry_run_summary(args: argparse.Namespace, spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "screen_name": SCREEN_NAME,
        "inferential_status": "diagnostic_exploratory_preflight",
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "spec_sha256": spec["spec_sha256"],
        "training_jobs": len(build_training_jobs()),
        "evaluation_jobs": len(build_evaluation_jobs()),
        "evaluation_rows": len(build_evaluation_jobs()) * len(TEST_WORKLOAD_SEEDS),
        "candidate_actor_spec": dict(EXPECTED_ACTOR_SPEC),
        "switch_regularizer_spec": (
            dict(EXPECTED_SWITCH_REGULARIZER_SPEC)
            if EXPECTED_SWITCH_REGULARIZER_SPEC is not None
            else None
        ),
        "validation_selection": dict(EXPECTED_SELECTION_SPEC),
        "validation_workloads": [VALIDATION_WORKLOAD_SEEDS[0], VALIDATION_WORKLOAD_SEEDS[-1]],
        "test_workloads": [TEST_WORKLOAD_SEEDS[0], TEST_WORKLOAD_SEEDS[-1]],
        "output": str(args.output),
        "dry_run_writes_output": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for field in ("source", "output", "cleanmarl", "project"):
        setattr(args, field, getattr(args, field).resolve())
    with configured_v1():
        validate_environment(args)
        source = v1.audit_source_screen(args.source)
        spec = build_screen_spec(args, source)
        if args.dry_run:
            print(json.dumps(_dry_run_summary(args, spec), indent=2))
            return 0

        args.output.mkdir(parents=True, exist_ok=True)
        v1.ensure_immutable_json(
            args.output / "switch_regularized_preflight_spec.json", spec
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
            "mode": "train_only" if args.train_only else "evaluate_only" if args.evaluate_only else "train_and_evaluate",
            "device": args.device,
            "max_parallel": args.max_parallel,
        }
        atomic_write_json(invocation_path, invocation)
        try:
            with invocation_lock(args.output):
                training_jobs = v1.build_training_jobs()
                checkpoints = (
                    v1.resolve_integrated_checkpoints(args, training_jobs, spec["spec_sha256"])
                    if args.evaluate_only
                    else v1.train_all(args, training_jobs, spec["spec_sha256"])
                )
                training_freeze = v1.ensure_training_freeze(
                    args.output,
                    spec["spec_sha256"],
                    str(source["training_freeze_sha256"]),
                    checkpoints,
                )
                if args.train_only:
                    invocation.update(
                        status="completed",
                        finished_at_utc=utc_now(),
                        training_freeze_sha256=training_freeze["freeze_sha256"],
                    )
                    atomic_write_json(invocation_path, invocation)
                    print(f"preflight training freeze complete: {args.output}", flush=True)
                    return 0
                evaluation_jobs = v1.build_evaluation_jobs()
                rows = v1.evaluate_all(
                    args,
                    evaluation_jobs,
                    spec["spec_sha256"],
                    training_freeze,
                    source,
                    checkpoints,
                )
                manifest = write_final_artifacts(
                    args.output, spec, training_freeze, evaluation_jobs, rows
                )
                invocation.update(
                    status="completed",
                    finished_at_utc=utc_now(),
                    decision=manifest["decision"]["decision"],
                    manifest_sha256=manifest["manifest_sha256"],
                )
                atomic_write_json(invocation_path, invocation)
                print(f"diagnostic preflight complete: {args.output}", flush=True)
                return 0
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
