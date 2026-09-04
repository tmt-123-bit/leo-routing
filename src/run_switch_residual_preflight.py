"""Run the frozen v5 residual-isolated MAPPO diagnostic preflight."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from leo_multiagent_env import (
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_SWITCH_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
)
import run_integrated_hysteresis_screen as v1
import run_switch_regularized_preflight as base


SCREEN_NAME = "SWITCH-RESIDUAL-PREFLIGHT-v5"
SCHEMA_VERSION = 5
ENVIRONMENT_VARIANT = "with_congestion_context"
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072)
TRAIN_WORKLOAD_SEEDS = tuple(range(9001, 9201))
VALIDATION_WORKLOAD_SEEDS = tuple(range(48001, 48011))
TEST_WORKLOAD_SEEDS = tuple(range(49001, 49026))

INTEGRATED_BETA = 0.20
URGENCY_RELIEF = 0.50
CLASS_2_RELIEF = 0.0
AVOIDABLE_SWITCH_PROBABILITY_COEF = 0.01
AVOIDABLE_SWITCH_LOGIT_MARGIN = 0.01
ROUTE_HYSTERESIS_RESIDUAL_INIT = 0.0
ROUTE_HYSTERESIS_RESIDUAL_CAP = 0.20
VALIDATION_SELECTION_MODE = "avoidable_stability_constrained"
VALIDATION_DELIVERY_TOLERANCE = 0.003
VALIDATION_CLASS_2_TOLERANCE = 0.010
CANDIDATE_FEATURE_SCHEMA_ID = "leo_multi_candidate_features_v1_dim_28"
CANDIDATE_FEATURE_SCHEMA_SHA256 = (
    "c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503"
)
PAIRED_ANALYSIS = "diagnostic_switch_residual_preflight_v5"
POLICY_NAMES = {
    **base.POLICY_NAMES,
    v1.ARM_INTEGRATED: (
        "mappo_context_switch_residual_v5_beta_0p20_coef_0p01_margin_0p01"
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
    "--route-hysteresis-residual-init",
    str(ROUTE_HYSTERESIS_RESIDUAL_INIT),
    "--route-hysteresis-residual-cap",
    str(ROUTE_HYSTERESIS_RESIDUAL_CAP),
    "--avoidable-switch-probability-coef",
    str(AVOIDABLE_SWITCH_PROBABILITY_COEF),
    "--avoidable-switch-regularization-mode",
    "isolated_greedy_logit_margin",
    "--avoidable-switch-logit-margin",
    str(AVOIDABLE_SWITCH_LOGIT_MARGIN),
    "--log-loss-component-gradients",
    "--validation-selection-mode",
    VALIDATION_SELECTION_MODE,
    "--validation-delivery-tolerance",
    str(VALIDATION_DELIVERY_TOLERANCE),
    "--validation-class-2-tolerance",
    str(VALIDATION_CLASS_2_TOLERANCE),
)

EXPECTED_ACTOR_SPEC = {
    "schema_version": 4,
    "type": "shared_candidate_actor",
    "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
    "route_hysteresis_beta": INTEGRATED_BETA,
    "route_hysteresis_mode": "decoupled_adaptive",
    "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
    "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
    "route_hysteresis_urgency_relief": URGENCY_RELIEF,
    "route_hysteresis_class_2_relief": CLASS_2_RELIEF,
    "route_hysteresis_residual_parameterization": "projected_nonnegative_scalar",
    "route_hysteresis_residual_init": ROUTE_HYSTERESIS_RESIDUAL_INIT,
    "route_hysteresis_residual_cap": ROUTE_HYSTERESIS_RESIDUAL_CAP,
    "avoidable_switch_probability_coef": AVOIDABLE_SWITCH_PROBABILITY_COEF,
    "candidate_feature_schema_id": CANDIDATE_FEATURE_SCHEMA_ID,
    "candidate_feature_schema_sha256": CANDIDATE_FEATURE_SCHEMA_SHA256,
}

EXPECTED_SWITCH_REGULARIZER_SPEC = {
    "schema_version": 2,
    "mode": "isolated_greedy_logit_margin",
    "coefficient": AVOIDABLE_SWITCH_PROBABILITY_COEF,
    "logit_margin": AVOIDABLE_SWITCH_LOGIT_MARGIN,
    "reduction": "conditional_mean_over_eligible_active_decisions",
    "no_op_action_index": 0,
    "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
    "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
    "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
    "route_hysteresis_beta": INTEGRATED_BETA,
    "route_hysteresis_urgency_relief": URGENCY_RELIEF,
    "route_hysteresis_class_2_relief": CLASS_2_RELIEF,
    "relief_weighting": "actual_policy_logits_no_extra_weighting",
    "logit_source": "actual_policy_logits",
    "gradient_scope": "cached_route_residual_only",
}

EXPECTED_SELECTION_SPEC = {
    **base.EXPECTED_SELECTION_SPEC,
    "schema_version": 4,
    "validation_seed_start": VALIDATION_WORKLOAD_SEEDS[0],
    "validation_episodes": len(VALIDATION_WORKLOAD_SEEDS),
}

KNOWN_EXPOSED_PANELS = (
    *base.KNOWN_EXPOSED_PANELS,
    (43001, 43010),
    (44001, 44025),
    (45001, 45002),
    (46001, 46010),
    (47001, 47025),
)
METRICS = base.METRICS
LOWER_IS_BETTER_COSTS = base.LOWER_IS_BETTER_COSTS
SWITCH_REGULARIZER_FREEZE_RATIONALE = (
    "v4 increased avoidable switching because its counterfactual full-hysteresis "
    "hinge did not target the deployed adaptive policy logits; v5 freezes the same "
    "coefficient and margin but isolates the regularizer gradient to a bounded "
    "cached-route residual, without using v5 validation or test outcomes"
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

_BASE_OVERRIDES = {
    "SCREEN_NAME": SCREEN_NAME,
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "EXPERIMENT_RUNNER_PATH": Path(__file__).resolve(),
    "PROTOCOL_FILENAME": "SWITCH_RESIDUAL_PREFLIGHT_V5.md",
    "SUMMARY_TITLE": "Switch-Residual MAPPO Preflight v5",
    "DEFAULT_OUTPUT_DIRECTORY_NAME": "switch-residual-preflight-6k-v5",
    "PARSER_DESCRIPTION": (
        "Run the frozen 6k-step residual-isolated MAPPO diagnostic preflight."
    ),
    "ENVIRONMENT_VARIANT": ENVIRONMENT_VARIANT,
    "SCENARIOS": SCENARIOS,
    "POLICY_SEEDS": POLICY_SEEDS,
    "TRAIN_WORKLOAD_SEEDS": TRAIN_WORKLOAD_SEEDS,
    "VALIDATION_WORKLOAD_SEEDS": VALIDATION_WORKLOAD_SEEDS,
    "TEST_WORKLOAD_SEEDS": TEST_WORKLOAD_SEEDS,
    "INTEGRATED_BETA": INTEGRATED_BETA,
    "URGENCY_RELIEF": URGENCY_RELIEF,
    "CLASS_2_RELIEF": CLASS_2_RELIEF,
    "AVOIDABLE_SWITCH_PROBABILITY_COEF": AVOIDABLE_SWITCH_PROBABILITY_COEF,
    "VALIDATION_SELECTION_MODE": VALIDATION_SELECTION_MODE,
    "VALIDATION_DELIVERY_TOLERANCE": VALIDATION_DELIVERY_TOLERANCE,
    "VALIDATION_CLASS_2_TOLERANCE": VALIDATION_CLASS_2_TOLERANCE,
    "CANDIDATE_FEATURE_SCHEMA_ID": CANDIDATE_FEATURE_SCHEMA_ID,
    "CANDIDATE_FEATURE_SCHEMA_SHA256": CANDIDATE_FEATURE_SCHEMA_SHA256,
    "PAIRED_ANALYSIS": PAIRED_ANALYSIS,
    "POLICY_NAMES": POLICY_NAMES,
    "CONFIG": CONFIG,
    "TRAINER_OVERRIDES": TRAINER_OVERRIDES,
    "EXPECTED_ACTOR_SPEC": EXPECTED_ACTOR_SPEC,
    "EXPECTED_SWITCH_REGULARIZER_SPEC": EXPECTED_SWITCH_REGULARIZER_SPEC,
    "EXPECTED_EXTRA_RUN_CONFIG": {
        "log_loss_component_gradients": True,
        "route_hysteresis_residual_init": ROUTE_HYSTERESIS_RESIDUAL_INIT,
        "route_hysteresis_residual_cap": ROUTE_HYSTERESIS_RESIDUAL_CAP,
    },
    "SWITCH_REGULARIZER_FREEZE_RATIONALE": SWITCH_REGULARIZER_FREEZE_RATIONALE,
    "EXPECTED_SELECTION_SPEC": EXPECTED_SELECTION_SPEC,
    "KNOWN_EXPOSED_PANELS": KNOWN_EXPOSED_PANELS,
    "METRICS": METRICS,
    "LOWER_IS_BETTER_COSTS": LOWER_IS_BETTER_COSTS,
    "_V1_OVERRIDES": _V1_OVERRIDES,
}


@contextmanager
def configured_v5() -> Iterator[None]:
    saved = {name: getattr(base, name) for name in _BASE_OVERRIDES}
    try:
        for name, value in _BASE_OVERRIDES.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def build_training_jobs() -> list[v1.TrainingJob]:
    with configured_v5():
        return base.build_training_jobs()


def build_evaluation_jobs() -> list[v1.EvaluationJob]:
    with configured_v5():
        return base.build_evaluation_jobs()


def training_job_record(job: v1.TrainingJob) -> dict[str, Any]:
    with configured_v5():
        return base.training_job_record(job)


def build_screen_spec(
    args: argparse.Namespace, source: Mapping[str, Any]
) -> dict[str, Any]:
    with configured_v5():
        return base.build_screen_spec(args, source)


def _expected_run_config(
    job: v1.TrainingJob, args: argparse.Namespace
) -> dict[str, Any]:
    with configured_v5():
        return base._expected_run_config(job, args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    with configured_v5():
        return base.parse_args(argv)


def _dry_run_summary(
    args: argparse.Namespace, spec: Mapping[str, Any]
) -> dict[str, Any]:
    with configured_v5():
        return base._dry_run_summary(args, spec)


def main(argv: Sequence[str] | None = None) -> int:
    with configured_v5():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
