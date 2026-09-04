"""Run the frozen v6 urgency-controller MAPPO diagnostic preflight."""

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


SCREEN_NAME = "SWITCH-CONTROLLER-PREFLIGHT-v6"
SCHEMA_VERSION = 6
ENVIRONMENT_VARIANT = "with_avoidable_switch_reward"
SCENARIOS = ("medium_load", "hotspot_high_load")
POLICY_SEEDS = (1710210210, 2078783072)
TRAIN_WORKLOAD_SEEDS = tuple(range(9001, 9201))
VALIDATION_WORKLOAD_SEEDS = tuple(range(50001, 50011))
TEST_WORKLOAD_SEEDS = tuple(range(51001, 51026))

INTEGRATED_BETA = 0.20
URGENCY_RELIEF = 0.50
CLASS_2_RELIEF = 0.0
AVOIDABLE_SWITCH_PROBABILITY_COEF = 0.01
AVOIDABLE_SWITCH_LOGIT_MARGIN = 0.01
ROUTE_HYSTERESIS_RESIDUAL_INIT = 0.0
ROUTE_HYSTERESIS_RESIDUAL_CAP = 0.20
ROUTE_HYSTERESIS_RESIDUAL_PARAMETERIZATION = "urgency_linear"
AVOIDABLE_SWITCH_REDUCTION = "rollout_micro_mean"
VALIDATION_SELECTION_MODE = "avoidable_stability_constrained"
VALIDATION_DELIVERY_TOLERANCE = 0.003
VALIDATION_CLASS_2_TOLERANCE = 0.010
CANDIDATE_FEATURE_SCHEMA_ID = "leo_multi_candidate_features_v1_dim_28"
CANDIDATE_FEATURE_SCHEMA_SHA256 = (
    "c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503"
)
PAIRED_ANALYSIS = "diagnostic_switch_controller_preflight_v6"
POLICY_NAMES = {
    **base.POLICY_NAMES,
    v1.ARM_INTEGRATED: (
        "mappo_avoidable_reward_switch_controller_v6_"
        "beta_0p20_coef_0p01_margin_0p01"
    ),
}
SOURCE_CONTEXT_VARIANT = "with_congestion_context"


class ControllerTrainingJob(v1.TrainingJob):
    """Capture the v6 treatment variant independently of v1 module state."""

    @property
    def variant(self) -> str:
        return ENVIRONMENT_VARIANT


class ControllerEvaluationJob(v1.EvaluationJob):
    """Keep frozen references native while replaying the v6 treatment."""

    @property
    def variant(self) -> str:
        if self.arm == v1.ARM_PROPOSED:
            return "proposed"
        if self.arm == v1.ARM_INTEGRATED:
            return ENVIRONMENT_VARIANT
        return SOURCE_CONTEXT_VARIANT

    @property
    def policy_name(self) -> str:
        return POLICY_NAMES[self.arm]

    @property
    def source_job_id(self) -> str:
        if self.arm == v1.ARM_INTEGRATED:
            return (
                f"{self.scenario}/integrated_beta_0p20/"
                f"seed_{self.policy_seed}"
            )
        source_variant = (
            "proposed"
            if self.arm == v1.ARM_PROPOSED
            else SOURCE_CONTEXT_VARIANT
        )
        return f"{self.scenario}/{source_variant}/seed_{self.policy_seed}"

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
    "--route-hysteresis-residual-parameterization",
    ROUTE_HYSTERESIS_RESIDUAL_PARAMETERIZATION,
    "--avoidable-switch-probability-coef",
    str(AVOIDABLE_SWITCH_PROBABILITY_COEF),
    "--avoidable-switch-regularization-mode",
    "isolated_greedy_logit_margin",
    "--avoidable-switch-logit-margin",
    str(AVOIDABLE_SWITCH_LOGIT_MARGIN),
    "--avoidable-switch-reduction",
    AVOIDABLE_SWITCH_REDUCTION,
    "--log-loss-component-gradients",
    "--validation-selection-mode",
    VALIDATION_SELECTION_MODE,
    "--validation-delivery-tolerance",
    str(VALIDATION_DELIVERY_TOLERANCE),
    "--validation-class-2-tolerance",
    str(VALIDATION_CLASS_2_TOLERANCE),
)

EXPECTED_ACTOR_SPEC = {
    "schema_version": 5,
    "type": "shared_candidate_actor",
    "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
    "route_hysteresis_beta": INTEGRATED_BETA,
    "route_hysteresis_mode": "decoupled_adaptive",
    "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
    "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
    "route_hysteresis_urgency_relief": URGENCY_RELIEF,
    "route_hysteresis_class_2_relief": CLASS_2_RELIEF,
    "route_hysteresis_residual_parameterization": (
        "projected_nonnegative_urgency_linear_endpoints"
    ),
    "route_hysteresis_residual_init": ROUTE_HYSTERESIS_RESIDUAL_INIT,
    "route_hysteresis_residual_cap": ROUTE_HYSTERESIS_RESIDUAL_CAP,
    "avoidable_switch_probability_coef": AVOIDABLE_SWITCH_PROBABILITY_COEF,
    "candidate_feature_schema_id": CANDIDATE_FEATURE_SCHEMA_ID,
    "candidate_feature_schema_sha256": CANDIDATE_FEATURE_SCHEMA_SHA256,
}

EXPECTED_SWITCH_REGULARIZER_SPEC = {
    "schema_version": 3,
    "mode": "isolated_greedy_logit_margin",
    "coefficient": AVOIDABLE_SWITCH_PROBABILITY_COEF,
    "logit_margin": AVOIDABLE_SWITCH_LOGIT_MARGIN,
    "reduction": (
        "rollout_micro_mean_over_eligible_pre_contention_decisions"
    ),
    "no_op_action_index": 0,
    "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
    "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
    "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
    "route_hysteresis_beta": INTEGRATED_BETA,
    "route_hysteresis_urgency_relief": URGENCY_RELIEF,
    "route_hysteresis_class_2_relief": CLASS_2_RELIEF,
    "relief_weighting": "actual_policy_logits_no_extra_weighting",
    "logit_source": "actual_policy_logits",
    "gradient_scope": "cached_route_residual_endpoints_only",
    "eligibility_stage": "pre_contention",
    "minibatch_weighting": "eligible_sum_scaled_to_rollout_micro_mean",
}

EXPECTED_SELECTION_SPEC = {
    **base.EXPECTED_SELECTION_SPEC,
    "schema_version": 5,
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
    (48001, 48010),
    (49001, 49025),
)
METRICS = base.METRICS
LOWER_IS_BETTER_COSTS = base.LOWER_IS_BETTER_COSTS
SWITCH_REGULARIZER_FREEZE_RATIONALE = (
    "v5 showed that a single residual and legacy minibatch-conditional weighting "
    "lacked sufficient state expression and made regularizer weighting depend on "
    "the minibatch partition, while reward penalized forced reroutes; "
    "v6 freezes an avoidable-only switch reward, two urgency-linear projected "
    "endpoints, and rollout-micro pre-contention weighting before consulting any "
    "v6 validation or test outcome"
)

_V1_OVERRIDES = {
    "SCREEN_NAME": SCREEN_NAME,
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "ENVIRONMENT_VARIANT": ENVIRONMENT_VARIANT,
    "TrainingJob": ControllerTrainingJob,
    "EvaluationJob": ControllerEvaluationJob,
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
    "PROTOCOL_FILENAME": "SWITCH_CONTROLLER_PREFLIGHT_V6.md",
    "SUMMARY_TITLE": "Switch-Controller MAPPO Preflight v6",
    "DEFAULT_OUTPUT_DIRECTORY_NAME": "switch-controller-preflight-6k-v6",
    "PARSER_DESCRIPTION": (
        "Run the frozen 6k-step urgency-controller MAPPO diagnostic preflight."
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
        "route_hysteresis_residual_parameterization": (
            ROUTE_HYSTERESIS_RESIDUAL_PARAMETERIZATION
        ),
        "avoidable_switch_reduction": AVOIDABLE_SWITCH_REDUCTION,
    },
    "SWITCH_REGULARIZER_FREEZE_RATIONALE": SWITCH_REGULARIZER_FREEZE_RATIONALE,
    "EXPECTED_SELECTION_SPEC": EXPECTED_SELECTION_SPEC,
    "KNOWN_EXPOSED_PANELS": KNOWN_EXPOSED_PANELS,
    "METRICS": METRICS,
    "LOWER_IS_BETTER_COSTS": LOWER_IS_BETTER_COSTS,
    "_V1_OVERRIDES": _V1_OVERRIDES,
}


@contextmanager
def configured_v6() -> Iterator[None]:
    saved = {name: getattr(base, name) for name in _BASE_OVERRIDES}
    try:
        for name, value in _BASE_OVERRIDES.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def build_training_jobs() -> list[v1.TrainingJob]:
    with configured_v6():
        return base.build_training_jobs()


def build_evaluation_jobs() -> list[v1.EvaluationJob]:
    with configured_v6():
        return base.build_evaluation_jobs()


def training_job_record(job: v1.TrainingJob) -> dict[str, Any]:
    with configured_v6():
        return base.training_job_record(job)


def build_screen_spec(
    args: argparse.Namespace, source: Mapping[str, Any]
) -> dict[str, Any]:
    with configured_v6():
        return base.build_screen_spec(args, source)


def _expected_run_config(
    job: v1.TrainingJob, args: argparse.Namespace
) -> dict[str, Any]:
    with configured_v6():
        return base._expected_run_config(job, args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    with configured_v6():
        return base.parse_args(argv)


def _dry_run_summary(
    args: argparse.Namespace, spec: Mapping[str, Any]
) -> dict[str, Any]:
    with configured_v6():
        return base._dry_run_summary(args, spec)


def validate_job_bindings() -> None:
    training_jobs = build_training_jobs()
    if any(job.variant != ENVIRONMENT_VARIANT for job in training_jobs):
        raise RuntimeError("v6 training job environment variant drifted")

    for job in build_evaluation_jobs():
        if job.arm == v1.ARM_PROPOSED:
            expected_variant = "proposed"
            expected_source = (
                f"{job.scenario}/proposed/seed_{job.policy_seed}"
            )
        elif job.arm == v1.ARM_INTEGRATED:
            expected_variant = ENVIRONMENT_VARIANT
            expected_source = (
                f"{job.scenario}/integrated_beta_0p20/"
                f"seed_{job.policy_seed}"
            )
        else:
            expected_variant = SOURCE_CONTEXT_VARIANT
            expected_source = (
                f"{job.scenario}/{SOURCE_CONTEXT_VARIANT}/"
                f"seed_{job.policy_seed}"
            )
        if job.variant != expected_variant:
            raise RuntimeError(
                f"v6 evaluation variant drifted for {job.job_id}"
            )
        if job.source_job_id != expected_source:
            raise RuntimeError(
                f"v6 checkpoint source drifted for {job.job_id}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    with configured_v6():
        validate_job_bindings()
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
