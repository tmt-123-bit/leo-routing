"""Held-out evaluation helpers for the satellite-level MAPPO experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from pathlib import Path
import random
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from leo_multiagent_env import (
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_SWITCH_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
)
from mappo_design import (
    SharedCandidateActor,
    projected_lagrange_multiplier_update,
    route_hysteresis_residual_dtype_cap,
)
from variant_definitions import canonical_variant_name


Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _finite_nonnegative_beta(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite non-negative number")
    beta = float(value)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return beta


def _positive_checkpoint_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_unit_interval(value: object, field: str) -> float:
    numeric = _finite_nonnegative_beta(value, field)
    if numeric > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return numeric


def _checkpoint_candidate_actor_spec(
    checkpoint: dict,
    actor_args: dict,
) -> dict:
    def validate_legacy_actor_args() -> None:
        mode = actor_args.get("route_hysteresis_mode", "legacy_additive")
        if mode != "legacy_additive":
            raise ValueError(
                "legacy candidate actor spec disagrees with actor hysteresis mode"
            )
        for field, label in (
            ("route_hysteresis_urgency_relief", "actor urgency relief"),
            ("route_hysteresis_class_2_relief", "actor class-2 relief"),
        ):
            value = _finite_unit_interval(actor_args.get(field, 0.0), label)
            if value != 0.0:
                raise ValueError(
                    "legacy candidate actor spec cannot use adaptive relief"
                )
        regularizer = _finite_nonnegative_beta(
            actor_args.get("avoidable_switch_probability_coef", 0.0),
            "actor avoidable-switch probability coefficient",
        )
        if regularizer != 0.0:
            raise ValueError(
                "legacy candidate actor spec cannot use switch regularization"
            )
        for field in (
            "route_hysteresis_residual_init",
            "route_hysteresis_residual_cap",
        ):
            value = _finite_nonnegative_beta(
                actor_args.get(field, 0.0), f"legacy actor {field}"
            )
            if value != 0.0:
                raise ValueError("legacy candidate actor cannot use a route residual")
        parameterization = actor_args.get(
            "route_hysteresis_residual_parameterization", "scalar"
        )
        if parameterization != "scalar":
            raise ValueError(
                "legacy candidate actor cannot use a route residual controller"
            )

    candidate_actor_spec = checkpoint.get("candidate_actor_spec")
    if candidate_actor_spec is None:
        actor_beta = _finite_nonnegative_beta(
            actor_args.get("route_hysteresis_beta", 0.0),
            "checkpoint actor beta",
        )
        if actor_beta != 0.0:
            raise ValueError("positive-beta checkpoint has no candidate actor spec")
        validate_legacy_actor_args()
        return {
            "schema_version": 0,
            "type": "shared_candidate_actor",
            "route_switch_feature_index": None,
            "route_hysteresis_beta": 0.0,
        }

    common_required = {
        "schema_version",
        "type",
        "route_switch_feature_index",
        "route_hysteresis_beta",
    }
    if not isinstance(candidate_actor_spec, dict):
        raise ValueError("candidate actor spec schema mismatch")
    schema_version = candidate_actor_spec.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2, 3, 4, 5}:
        raise ValueError("unsupported candidate actor spec version")
    adaptive_required = {
        "route_hysteresis_mode",
        "route_urgency_feature_index",
        "route_class_2_feature_index",
        "route_hysteresis_urgency_relief",
        "route_hysteresis_class_2_relief",
        "candidate_feature_schema_id",
        "candidate_feature_schema_sha256",
    }
    required = common_required | (
        adaptive_required if schema_version in {2, 3, 4, 5} else set()
    )
    if schema_version in {3, 4, 5}:
        required.add("avoidable_switch_probability_coef")
    if schema_version in {4, 5}:
        required.update(
            {
                "route_hysteresis_residual_parameterization",
                "route_hysteresis_residual_init",
                "route_hysteresis_residual_cap",
            }
        )
    if set(candidate_actor_spec) != required:
        raise ValueError("candidate actor spec schema mismatch")
    if candidate_actor_spec["type"] != "shared_candidate_actor":
        raise ValueError("candidate actor type mismatch")
    if "route_hysteresis_beta" not in actor_args:
        raise ValueError("checkpoint actor args omit route_hysteresis_beta")

    actor_beta = _finite_nonnegative_beta(
        actor_args["route_hysteresis_beta"], "checkpoint actor beta"
    )
    spec_beta = _finite_nonnegative_beta(
        candidate_actor_spec["route_hysteresis_beta"],
        "candidate actor spec beta",
    )
    if actor_beta != spec_beta:
        raise ValueError("checkpoint beta disagrees with candidate actor spec")

    switch_feature_index = candidate_actor_spec["route_switch_feature_index"]
    if type(switch_feature_index) is not int:
        raise ValueError("candidate actor route-switch index must be an integer")
    if switch_feature_index != ROUTE_SWITCH_FEATURE_INDEX:
        raise ValueError("checkpoint route-switch feature contract mismatch")
    normalized = {
        **candidate_actor_spec,
        "route_switch_feature_index": switch_feature_index,
        "route_hysteresis_beta": spec_beta,
    }
    if schema_version == 1:
        validate_legacy_actor_args()
        return normalized

    if candidate_actor_spec["route_hysteresis_mode"] != "decoupled_adaptive":
        raise ValueError("unsupported adaptive route hysteresis mode")
    if actor_args.get("route_hysteresis_mode") != "decoupled_adaptive":
        raise ValueError("checkpoint route hysteresis mode disagrees with actor args")
    index_contracts = {
        "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
        "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
    }
    for field, expected in index_contracts.items():
        value = candidate_actor_spec[field]
        if type(value) is not int or value != expected:
            raise ValueError(f"checkpoint {field} contract mismatch")
        if actor_args.get(field) != value:
            raise ValueError(f"checkpoint {field} disagrees with actor args")
        normalized[field] = value
    relief_contracts = {
        "route_hysteresis_urgency_relief": "urgency relief",
        "route_hysteresis_class_2_relief": "class-2 relief",
    }
    for field, label in relief_contracts.items():
        spec_value = _finite_unit_interval(candidate_actor_spec[field], label)
        actor_value = _finite_unit_interval(actor_args.get(field), f"actor {label}")
        if actor_value != spec_value:
            raise ValueError(f"checkpoint {label} disagrees with actor args")
        normalized[field] = spec_value
    schema_id = candidate_actor_spec["candidate_feature_schema_id"]
    schema_sha256 = candidate_actor_spec["candidate_feature_schema_sha256"]
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError("checkpoint candidate feature schema id is invalid")
    if (
        not isinstance(schema_sha256, str)
        or len(schema_sha256) != 64
        or any(character not in "0123456789abcdef" for character in schema_sha256)
    ):
        raise ValueError("checkpoint candidate feature schema hash is invalid")
    normalized["candidate_feature_schema_id"] = schema_id
    normalized["candidate_feature_schema_sha256"] = schema_sha256
    actor_regularizer = _finite_nonnegative_beta(
        actor_args.get("avoidable_switch_probability_coef", 0.0),
        "actor avoidable-switch probability coefficient",
    )
    if schema_version == 2:
        if actor_regularizer != 0.0:
            raise ValueError(
                "schema-v2 checkpoint cannot use switch regularization"
            )
    else:
        spec_regularizer = _finite_nonnegative_beta(
            candidate_actor_spec["avoidable_switch_probability_coef"],
            "candidate avoidable-switch probability coefficient",
        )
        if spec_regularizer <= 0.0:
            raise ValueError(
                "regularized checkpoint requires positive switch regularization"
            )
        if actor_regularizer != spec_regularizer:
            raise ValueError(
                "checkpoint switch regularization disagrees with actor args"
            )
        normalized["avoidable_switch_probability_coef"] = spec_regularizer
    if schema_version < 4:
        for field in (
            "route_hysteresis_residual_init",
            "route_hysteresis_residual_cap",
        ):
            value = _finite_nonnegative_beta(
                actor_args.get(field, 0.0), f"actor {field}"
            )
            if value != 0.0:
                raise ValueError(
                    "pre-v4 candidate actor cannot use a route residual"
                )
        if actor_args.get(
            "route_hysteresis_residual_parameterization", "scalar"
        ) != "scalar":
            raise ValueError(
                "pre-v4 candidate actor cannot use a route residual controller"
            )
    else:
        parameterization = candidate_actor_spec[
            "route_hysteresis_residual_parameterization"
        ]
        expected_parameterization = {
            4: "projected_nonnegative_scalar",
            5: "projected_nonnegative_urgency_linear_endpoints",
        }[schema_version]
        if parameterization != expected_parameterization:
            raise ValueError("unsupported route hysteresis residual parameterization")
        actor_parameterization = actor_args.get(
            "route_hysteresis_residual_parameterization", "scalar"
        )
        expected_actor_parameterization = {
            4: "scalar",
            5: "urgency_linear",
        }[schema_version]
        if actor_parameterization != expected_actor_parameterization:
            raise ValueError(
                "checkpoint residual parameterization disagrees with actor args"
            )
        residual_init = _finite_nonnegative_beta(
            candidate_actor_spec["route_hysteresis_residual_init"],
            "candidate actor residual init",
        )
        residual_cap = _finite_nonnegative_beta(
            candidate_actor_spec["route_hysteresis_residual_cap"],
            "candidate actor residual cap",
        )
        if residual_cap <= 0.0 or residual_init > residual_cap:
            raise ValueError("candidate actor residual bounds are invalid")
        actor_residual_init = _finite_nonnegative_beta(
            actor_args.get("route_hysteresis_residual_init"),
            "actor residual init",
        )
        actor_residual_cap = _finite_nonnegative_beta(
            actor_args.get("route_hysteresis_residual_cap"),
            "actor residual cap",
        )
        if (
            actor_residual_init != residual_init
            or actor_residual_cap != residual_cap
        ):
            raise ValueError(
                "checkpoint residual bounds disagree with actor args"
            )
        normalized.update(
            route_hysteresis_residual_parameterization=parameterization,
            route_hysteresis_residual_init=residual_init,
            route_hysteresis_residual_cap=residual_cap,
        )
    return normalized


def _checkpoint_switch_regularizer_spec(
    checkpoint: dict,
    actor_args: dict,
    candidate_actor_spec: dict,
) -> dict | None:
    schema_version = int(candidate_actor_spec["schema_version"])
    raw_spec = checkpoint.get("switch_regularizer_spec")
    mode_present = "avoidable_switch_regularization_mode" in actor_args
    margin_present = "avoidable_switch_logit_margin" in actor_args

    if schema_version < 3:
        if raw_spec is not None:
            raise ValueError(
                "non-regularized checkpoint cannot contain a switch regularizer spec"
            )
        if mode_present and actor_args["avoidable_switch_regularization_mode"] != (
            "conditional_probability"
        ):
            raise ValueError("non-regularized checkpoint has a non-default switch mode")
        if margin_present and _finite_nonnegative_beta(
            actor_args["avoidable_switch_logit_margin"],
            "actor avoidable-switch logit margin",
        ) != 0.0:
            raise ValueError("non-regularized checkpoint has a non-default margin")
        return None

    coefficient = float(candidate_actor_spec["avoidable_switch_probability_coef"])
    if raw_spec is None:
        if schema_version >= 4 or mode_present or margin_present:
            raise ValueError(
                "new regularized checkpoint is missing switch_regularizer_spec"
            )
        return {
            "schema_version": 0,
            "mode": "conditional_probability",
            "coefficient": coefficient,
            "logit_margin": 0.0,
            "reduction": "conditional_mean_over_eligible_active_decisions",
            "no_op_action_index": 0,
            "route_switch_feature_index": int(
                candidate_actor_spec["route_switch_feature_index"]
            ),
            "route_urgency_feature_index": int(
                candidate_actor_spec["route_urgency_feature_index"]
            ),
            "route_class_2_feature_index": int(
                candidate_actor_spec["route_class_2_feature_index"]
            ),
            "route_hysteresis_beta": float(
                candidate_actor_spec["route_hysteresis_beta"]
            ),
            "route_hysteresis_urgency_relief": float(
                candidate_actor_spec["route_hysteresis_urgency_relief"]
            ),
            "route_hysteresis_class_2_relief": float(
                candidate_actor_spec["route_hysteresis_class_2_relief"]
            ),
            "relief_weighting": "none",
            "migration": "legacy_schema_v3_probability_without_explicit_spec",
        }

    if not mode_present or not margin_present:
        raise ValueError("regularized checkpoint actor args omit mode or margin")
    if not isinstance(raw_spec, dict):
        raise ValueError("switch regularizer spec must be a mapping")
    required = {
        "schema_version",
        "mode",
        "coefficient",
        "logit_margin",
        "reduction",
        "no_op_action_index",
        "route_switch_feature_index",
        "route_urgency_feature_index",
        "route_class_2_feature_index",
        "route_hysteresis_beta",
        "route_hysteresis_urgency_relief",
        "route_hysteresis_class_2_relief",
        "relief_weighting",
    }
    expected_spec_version = {4: 2, 5: 3}.get(schema_version, 1)
    if schema_version >= 4:
        required.update({"logit_source", "gradient_scope"})
    if schema_version == 5:
        required.update({"eligibility_stage", "minibatch_weighting"})
    if (
        set(raw_spec) != required
        or raw_spec.get("schema_version") != expected_spec_version
    ):
        raise ValueError("switch regularizer spec schema mismatch")

    mode = raw_spec["mode"]
    if mode not in {
        "conditional_probability",
        "greedy_logit_margin",
        "isolated_greedy_logit_margin",
    }:
        raise ValueError("unsupported checkpoint switch regularizer mode")
    if schema_version >= 4 and mode != "isolated_greedy_logit_margin":
        raise ValueError(
            f"schema-v{schema_version} actor requires isolated switch regularization"
        )
    if schema_version < 4 and mode == "isolated_greedy_logit_margin":
        raise ValueError("isolated switch regularization requires a schema-v4+ actor")
    if actor_args["avoidable_switch_regularization_mode"] != mode:
        raise ValueError("checkpoint switch mode disagrees with actor args")
    spec_coefficient = _finite_nonnegative_beta(
        raw_spec["coefficient"], "switch regularizer coefficient"
    )
    actor_coefficient = _finite_nonnegative_beta(
        actor_args.get("avoidable_switch_probability_coef"),
        "actor avoidable-switch probability coefficient",
    )
    if spec_coefficient <= 0.0 or spec_coefficient != coefficient:
        raise ValueError("switch regularizer coefficient disagrees with actor spec")
    if actor_coefficient != spec_coefficient:
        raise ValueError("switch regularizer coefficient disagrees with actor args")
    margin = _finite_nonnegative_beta(
        raw_spec["logit_margin"], "switch regularizer logit margin"
    )
    actor_margin = _finite_nonnegative_beta(
        actor_args["avoidable_switch_logit_margin"],
        "actor avoidable-switch logit margin",
    )
    if margin != actor_margin:
        raise ValueError("checkpoint switch margin disagrees with actor args")
    if mode == "conditional_probability" and margin != 0.0:
        raise ValueError("conditional-probability checkpoint cannot use a margin")
    if mode in {"greedy_logit_margin", "isolated_greedy_logit_margin"} and margin <= 0.0:
        raise ValueError("greedy-logit-margin checkpoint requires a positive margin")
    expected_reduction = (
        "rollout_micro_mean_over_eligible_pre_contention_decisions"
        if schema_version == 5
        else "conditional_mean_over_eligible_active_decisions"
    )
    if raw_spec["reduction"] != expected_reduction:
        raise ValueError("checkpoint switch regularizer reduction mismatch")
    if type(raw_spec["no_op_action_index"]) is not int or raw_spec[
        "no_op_action_index"
    ] != 0:
        raise ValueError("checkpoint switch regularizer NO_OP contract mismatch")

    index_fields = (
        "route_switch_feature_index",
        "route_urgency_feature_index",
        "route_class_2_feature_index",
    )
    indices = []
    for field in index_fields:
        value = raw_spec[field]
        if type(value) is not int or value != candidate_actor_spec[field]:
            raise ValueError(f"checkpoint switch regularizer {field} mismatch")
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError("checkpoint switch regularizer feature indices must be distinct")

    beta = _finite_nonnegative_beta(
        raw_spec["route_hysteresis_beta"], "switch regularizer beta"
    )
    if beta != float(candidate_actor_spec["route_hysteresis_beta"]):
        raise ValueError("checkpoint switch regularizer beta mismatch")
    for field, label in (
        ("route_hysteresis_urgency_relief", "urgency relief"),
        ("route_hysteresis_class_2_relief", "class-2 relief"),
    ):
        value = _finite_unit_interval(raw_spec[field], f"switch regularizer {label}")
        if value != float(candidate_actor_spec[field]) or value != float(
            actor_args[field]
        ):
            raise ValueError(f"checkpoint switch regularizer {label} mismatch")

    expected_weighting = {
        "conditional_probability": "none",
        "greedy_logit_margin": (
            "restore_full_hysteresis_before_hinge_then_multiply_adaptive_relief_scale"
        ),
        "isolated_greedy_logit_margin": (
            "actual_policy_logits_no_extra_weighting"
        ),
    }[mode]
    if raw_spec["relief_weighting"] != expected_weighting:
        raise ValueError("checkpoint switch regularizer relief weighting mismatch")
    if schema_version >= 4:
        if raw_spec["logit_source"] != "actual_policy_logits":
            raise ValueError("checkpoint switch regularizer logit source mismatch")
        expected_gradient_scope = (
            "cached_route_residual_endpoints_only"
            if schema_version == 5
            else "cached_route_residual_only"
        )
        if raw_spec["gradient_scope"] != expected_gradient_scope:
            raise ValueError("checkpoint switch regularizer gradient scope mismatch")
    if schema_version == 5:
        if actor_args.get("avoidable_switch_reduction") != "rollout_micro_mean":
            raise ValueError(
                "checkpoint switch regularizer reduction disagrees with actor args"
            )
        if raw_spec["eligibility_stage"] != "pre_contention":
            raise ValueError("checkpoint switch regularizer eligibility stage mismatch")
        if raw_spec["minibatch_weighting"] != (
            "eligible_sum_scaled_to_rollout_micro_mean"
        ):
            raise ValueError(
                "checkpoint switch regularizer minibatch weighting mismatch"
            )
    return {
        **raw_spec,
        "coefficient": spec_coefficient,
        "logit_margin": margin,
        "route_hysteresis_beta": beta,
        "route_hysteresis_urgency_relief": float(
            raw_spec["route_hysteresis_urgency_relief"]
        ),
        "route_hysteresis_class_2_relief": float(
            raw_spec["route_hysteresis_class_2_relief"]
        ),
    }


def _checkpoint_switch_constraint_spec(
    checkpoint: dict,
    actor_args: dict,
    candidate_actor_spec: dict,
) -> tuple[dict | None, dict | None]:
    enabled = actor_args.get("avoidable_switch_constraint_enabled", False)
    if type(enabled) is not bool:
        raise ValueError("checkpoint switch-constraint enabled flag must be boolean")
    raw_spec = checkpoint.get("switch_constraint_spec")
    raw_state = checkpoint.get("switch_constraint_state")
    if not enabled:
        if raw_spec is not None or raw_state is not None:
            raise ValueError("disabled checkpoint contains switch-constraint state")
        return None, None

    required_spec = {
        "schema_version",
        "type",
        "cost_definition",
        "decision_stage",
        "normalization",
        "forced_reroutes_counted",
        "first_route_counted",
        "zero_opportunity_update",
        "actor_surrogate",
        "dual_update_timing",
        "budget",
        "dual_learning_rate",
        "dual_initial",
        "dual_projection",
        "no_op_action_index",
        "route_switch_feature_index",
        "candidate_feature_schema_id",
        "candidate_feature_schema_sha256",
        "reward_objective",
        "leo_variant",
    }
    if not isinstance(raw_spec, dict) or set(raw_spec) != required_spec:
        raise ValueError("switch-constraint spec schema mismatch")
    if (
        type(raw_spec["schema_version"]) is not int
        or type(raw_spec["forced_reroutes_counted"]) is not bool
        or type(raw_spec["first_route_counted"]) is not bool
        or type(raw_spec["no_op_action_index"]) is not int
        or type(raw_spec["route_switch_feature_index"]) is not int
    ):
        raise ValueError("switch-constraint spec field type mismatch")
    constant_contract = {
        "schema_version": 1,
        "type": "projected_lagrangian_avoidable_switch_rate",
        "cost_definition": (
            "policy_selects_different_next_hop_while_cached_next_hop_and_"
            "at_least_one_alternative_are_feasible"
        ),
        "decision_stage": "pre_contention",
        "normalization": "rollout_micro_ratio_of_sums_over_opportunities",
        "forced_reroutes_counted": False,
        "first_route_counted": False,
        "zero_opportunity_update": "skip",
        "actor_surrogate": (
            "conditional_switch_probability_micro_mean_on_rollout_occupancy"
        ),
        "dual_update_timing": "once_after_each_completed_ppo_rollout",
        "no_op_action_index": 0,
        "reward_objective": "qos_only_without_switch_reward",
        "leo_variant": "qos_only",
    }
    for field, expected in constant_contract.items():
        if raw_spec[field] != expected:
            raise ValueError(f"switch-constraint {field} contract mismatch")
    if canonical_variant_name(actor_args.get("leo_variant", "")) != "qos_only":
        raise ValueError("switch-constraint checkpoint reward objective mismatch")
    if (
        candidate_actor_spec.get("schema_version") != 1
        or candidate_actor_spec.get("route_hysteresis_beta") != 0.0
        or actor_args.get("route_hysteresis_mode") != "legacy_additive"
        or actor_args.get("route_hysteresis_residual_init") != 0.0
        or actor_args.get("route_hysteresis_residual_cap") != 0.0
        or actor_args.get("route_hysteresis_residual_parameterization") != "scalar"
    ):
        raise ValueError("switch constraint cannot contain route hysteresis")
    if actor_args.get("avoidable_switch_probability_coef", 0.0) != 0.0:
        raise ValueError("switch constraint cannot contain fixed regularization")
    if (
        actor_args.get("avoidable_switch_regularization_mode")
        != "conditional_probability"
        or actor_args.get("avoidable_switch_logit_margin") != 0.0
        or actor_args.get("avoidable_switch_reduction") != "rollout_micro_mean"
    ):
        raise ValueError("switch-constraint checkpoint reduction mismatch")

    budget = _finite_unit_interval(raw_spec["budget"], "switch constraint budget")
    dual_learning_rate = _finite_nonnegative_beta(
        raw_spec["dual_learning_rate"], "switch constraint dual learning rate"
    )
    dual_initial = _finite_nonnegative_beta(
        raw_spec["dual_initial"], "switch constraint dual initial"
    )
    if dual_learning_rate <= 0.0:
        raise ValueError("switch constraint dual learning rate must be positive")
    projection = raw_spec["dual_projection"]
    if not isinstance(projection, list) or len(projection) != 2:
        raise ValueError("switch constraint dual projection is invalid")
    projection_min = _finite_nonnegative_beta(
        projection[0], "switch constraint projection minimum"
    )
    projection_max = _finite_nonnegative_beta(
        projection[1], "switch constraint projection maximum"
    )
    if projection_min != 0.0 or projection_max <= projection_min:
        raise ValueError("switch constraint dual projection is invalid")
    if not projection_min <= dual_initial <= projection_max:
        raise ValueError("switch constraint dual initial is outside its projection")
    numeric_args = {
        "avoidable_switch_budget": budget,
        "avoidable_switch_dual_learning_rate": dual_learning_rate,
        "avoidable_switch_dual_initial": dual_initial,
        "avoidable_switch_dual_max": projection_max,
    }
    for field, expected in numeric_args.items():
        actual = _finite_nonnegative_beta(actor_args.get(field), f"actor {field}")
        if actual != expected:
            raise ValueError(f"switch constraint disagrees with actor {field}")
    if raw_spec["route_switch_feature_index"] != candidate_actor_spec.get(
        "route_switch_feature_index"
    ):
        raise ValueError("switch constraint route-switch feature mismatch")
    schema_id = raw_spec["candidate_feature_schema_id"]
    schema_hash = raw_spec["candidate_feature_schema_sha256"]
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError("switch constraint candidate schema id is invalid")
    if (
        not isinstance(schema_hash, str)
        or len(schema_hash) != 64
        or any(character not in "0123456789abcdef" for character in schema_hash)
    ):
        raise ValueError("switch constraint candidate schema hash is invalid")

    required_state = {
        "schema_version",
        "multiplier",
        "update_count",
        "skipped_zero_opportunity_count",
        "cumulative_cost",
        "cumulative_opportunity",
        "last_rollout_cost",
        "last_rollout_opportunity",
        "last_rollout_rate",
        "last_rollout_violation",
    }
    if not isinstance(raw_state, dict) or set(raw_state) != required_state:
        raise ValueError("switch-constraint state schema mismatch")
    if type(raw_state["schema_version"]) is not int or raw_state[
        "schema_version"
    ] != 1:
        raise ValueError("unsupported switch-constraint state version")

    def nonnegative_int(field: str) -> int:
        value = raw_state[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"switch-constraint state {field} is invalid")
        return value

    multiplier = _finite_nonnegative_beta(
        raw_state["multiplier"], "switch constraint multiplier"
    )
    if not projection_min <= multiplier <= projection_max:
        raise ValueError("switch constraint multiplier is outside its projection")
    update_count = nonnegative_int("update_count")
    skipped_count = nonnegative_int("skipped_zero_opportunity_count")
    cumulative_cost = nonnegative_int("cumulative_cost")
    cumulative_opportunity = nonnegative_int("cumulative_opportunity")
    last_cost = nonnegative_int("last_rollout_cost")
    last_opportunity = nonnegative_int("last_rollout_opportunity")
    if cumulative_cost > cumulative_opportunity or last_cost > last_opportunity:
        raise ValueError("switch-constraint cost exceeds opportunities")
    if last_cost > cumulative_cost or last_opportunity > cumulative_opportunity:
        raise ValueError("switch-constraint last rollout exceeds cumulative totals")
    trainer_state = checkpoint.get("trainer_state")
    if (
        not isinstance(trainer_state, dict)
        or trainer_state.get("schema_version") != 1
        or type(trainer_state.get("update_round")) is not int
        or trainer_state["update_round"] < 0
        or update_count + skipped_count != trainer_state["update_round"]
    ):
        raise ValueError("switch-constraint state disagrees with trainer state")
    if (cumulative_opportunity == 0) != (update_count == 0):
        raise ValueError("switch-constraint state has an impossible update history")
    if cumulative_opportunity < update_count:
        raise ValueError("switch-constraint state has too few update opportunities")
    if update_count == 0 and multiplier != dual_initial:
        raise ValueError("switch-constraint state has an impossible multiplier")
    if update_count == 1:
        expected_multiplier = projected_lagrange_multiplier_update(
            dual_initial,
            cumulative_cost / cumulative_opportunity,
            budget,
            dual_learning_rate,
            minimum=projection_min,
            maximum=projection_max,
        )
        if multiplier != expected_multiplier:
            raise ValueError(
                "switch-constraint state disagrees with its first dual update"
            )
    last_rate = raw_state["last_rollout_rate"]
    last_violation = raw_state["last_rollout_violation"]
    if last_opportunity == 0:
        if last_rate is not None or last_violation is not None:
            raise ValueError("zero-opportunity constraint state has a rate")
    else:
        last_rate = _finite_unit_interval(last_rate, "last switch constraint rate")
        if isinstance(last_violation, bool) or not isinstance(
            last_violation, (int, float, np.integer, np.floating)
        ) or not np.isfinite(last_violation):
            raise ValueError("last switch constraint violation must be finite")
        expected_rate = last_cost / last_opportunity
        if last_rate != expected_rate or float(last_violation) != expected_rate - budget:
            raise ValueError("switch-constraint last rollout state is inconsistent")
    normalized_spec = {
        **raw_spec,
        "budget": budget,
        "dual_learning_rate": dual_learning_rate,
        "dual_initial": dual_initial,
        "dual_projection": [projection_min, projection_max],
    }
    normalized_state = {
        **raw_state,
        "multiplier": multiplier,
        "update_count": update_count,
        "skipped_zero_opportunity_count": skipped_count,
        "cumulative_cost": cumulative_cost,
        "cumulative_opportunity": cumulative_opportunity,
        "last_rollout_cost": last_cost,
        "last_rollout_opportunity": last_opportunity,
    }
    return normalized_spec, normalized_state


def _checkpoint_candidate_actor_learned_state(
    checkpoint: dict,
    candidate_actor_spec: dict,
    actor_state: dict,
) -> dict | None:
    schema_version = int(candidate_actor_spec["schema_version"])
    raw_state = checkpoint.get("candidate_actor_learned_state")
    scalar_parameter = "route_hysteresis_residual_bias"
    endpoint_parameters = (
        "route_hysteresis_residual_calm_bias",
        "route_hysteresis_residual_urgent_bias",
    )
    residual_parameters = {scalar_parameter, *endpoint_parameters}
    actor_residual_parameters = residual_parameters.intersection(actor_state)
    if schema_version < 4:
        if raw_state is not None or actor_residual_parameters:
            raise ValueError("pre-v4 checkpoint cannot contain route residual state")
        return None
    expected_parameters = (
        (scalar_parameter,) if schema_version == 4 else endpoint_parameters
    )
    if (
        not isinstance(raw_state, dict)
        or set(raw_state) != set(expected_parameters)
        or actor_residual_parameters != set(expected_parameters)
    ):
        raise ValueError(
            f"schema-v{schema_version} checkpoint learned actor state mismatch"
        )
    cap = float(candidate_actor_spec["route_hysteresis_residual_cap"])
    normalized = {}
    for parameter_name in expected_parameters:
        residual = _finite_nonnegative_beta(
            raw_state[parameter_name],
            f"checkpoint learned {parameter_name}",
        )
        state_tensor = actor_state.get(parameter_name)
        if (
            not isinstance(state_tensor, torch.Tensor)
            or state_tensor.ndim != 0
            or not torch.is_floating_point(state_tensor)
            or not torch.isfinite(state_tensor).all()
            or float(state_tensor.detach().cpu().item()) != residual
        ):
            raise ValueError(
                "checkpoint learned route residual disagrees with actor state_dict"
            )
        dtype_cap = route_hysteresis_residual_dtype_cap(
            cap,
            dtype=state_tensor.dtype,
        )
        if residual > dtype_cap:
            raise ValueError("checkpoint learned route residual exceeds its cap")
        normalized[parameter_name] = residual
    return normalized


@dataclass
class EpisodeMetrics:
    scenario: str
    policy: str
    policy_seed: int
    workload_seed: int
    generated: int
    delivered: int
    dropped: int
    backlog: int
    delivery_ratio: float
    drop_rate: float
    throughput_packets_per_slot: float
    average_delay_slots: float
    p95_delay_slots: float
    mean_queue_packets: float
    max_queue_packets: int
    routing_switches: int
    episode_reward: float
    global_delay_cost: float
    global_queue_cost: float
    global_load_imbalance: float
    global_switch_cost: float
    global_throughput_reward: float
    global_control_overhead_ratio: float
    global_drop_cost: float
    class_0_delivery_ratio: float
    class_1_delivery_ratio: float
    class_2_delivery_ratio: float
    avoidable_routing_switches: int = 0
    forced_routing_switches: int = 0
    switch_opportunities: int = 0
    avoidable_switch_rate: float = 0.0


@dataclass(kw_only=True)
class ConstraintEpisodeMetrics(EpisodeMetrics):
    decision_avoidable_switches: int
    decision_switch_opportunities: int
    decision_forced_switches: int
    decision_avoidable_switch_rate: float


def load_checkpoint_policy(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[Policy, Dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a mapping")
    feature_dim = _positive_checkpoint_int(
        checkpoint.get("candidate_feature_dim"), "candidate feature dimension"
    )
    action_size = _positive_checkpoint_int(
        checkpoint.get("action_size"), "action size"
    )
    actor_args = checkpoint["args"]
    if not isinstance(actor_args, dict):
        raise ValueError("checkpoint actor args must be a mapping")
    checkpoint_variant = actor_args.get("leo_variant")
    if not isinstance(checkpoint_variant, str) or not checkpoint_variant:
        raise ValueError("checkpoint actor args omit leo_variant")
    checkpoint_variant = canonical_variant_name(checkpoint_variant)
    candidate_actor_spec = _checkpoint_candidate_actor_spec(checkpoint, actor_args)
    switch_regularizer_spec = _checkpoint_switch_regularizer_spec(
        checkpoint, actor_args, candidate_actor_spec
    )
    switch_constraint_spec, switch_constraint_state = (
        _checkpoint_switch_constraint_spec(
            checkpoint,
            actor_args,
            candidate_actor_spec,
        )
    )
    checkpoint_n_agents = (
        _positive_checkpoint_int(
            checkpoint.get("n_agents"), "switch-constraint checkpoint agent count"
        )
        if switch_constraint_spec is not None
        else int(checkpoint.get("n_agents", 0))
    )
    switch_feature_index = candidate_actor_spec["route_switch_feature_index"]
    actor = SharedCandidateActor(
        candidate_feature_dim=feature_dim,
        hidden_dim=int(actor_args["actor_hidden_dim"]),
        num_layers=int(actor_args["actor_num_layers"]),
        route_switch_feature_index=switch_feature_index,
        route_hysteresis_beta=float(candidate_actor_spec["route_hysteresis_beta"]),
        route_hysteresis_mode=candidate_actor_spec.get(
            "route_hysteresis_mode", "legacy_additive"
        ),
        route_urgency_feature_index=candidate_actor_spec.get(
            "route_urgency_feature_index"
        ),
        route_class_2_feature_index=candidate_actor_spec.get(
            "route_class_2_feature_index"
        ),
        route_hysteresis_urgency_relief=float(
            candidate_actor_spec.get("route_hysteresis_urgency_relief", 0.0)
        ),
        route_hysteresis_class_2_relief=float(
            candidate_actor_spec.get("route_hysteresis_class_2_relief", 0.0)
        ),
        route_hysteresis_residual_init=float(
            candidate_actor_spec.get("route_hysteresis_residual_init", 0.0)
        ),
        route_hysteresis_residual_cap=float(
            candidate_actor_spec.get("route_hysteresis_residual_cap", 0.0)
        ),
        route_hysteresis_residual_parameterization={
            "projected_nonnegative_scalar": "scalar",
            "projected_nonnegative_urgency_linear_endpoints": (
                "urgency_linear"
            ),
        }.get(
            candidate_actor_spec.get(
                "route_hysteresis_residual_parameterization"
            ),
            "scalar",
        ),
    ).to(device)
    prefix = "shared_candidate_actor."
    actor_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["actor"].items()
        if key.startswith(prefix)
    }
    if not actor_state:
        raise ValueError("checkpoint does not contain a shared candidate actor")
    candidate_actor_learned_state = _checkpoint_candidate_actor_learned_state(
        checkpoint, candidate_actor_spec, actor_state
    )
    actor.load_state_dict(actor_state)
    actor.eval()

    def policy(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        expected_observation_shape = (
            observation.shape[0],
            action_size * feature_dim,
        )
        if observation.shape != expected_observation_shape:
            raise ValueError(
                f"observation shape {observation.shape} != "
                f"{expected_observation_shape}"
            )
        if mask.shape != (observation.shape[0], action_size):
            raise ValueError("action-mask shape does not match the checkpoint")
        candidates = torch.from_numpy(observation).float().to(device).reshape(
            observation.shape[0], action_size, feature_dim
        )
        action_mask = torch.from_numpy(mask).bool().to(device)
        with torch.no_grad():
            return actor(candidates, action_mask).argmax(dim=-1).cpu().numpy()

    checkpoint_schema = {
        "candidate_feature_dim": feature_dim,
        "action_size": action_size,
        "obs_size": int(checkpoint.get("obs_size", action_size * feature_dim)),
        "n_agents": checkpoint_n_agents,
        "variant": checkpoint_variant,
        "candidate_actor_spec": dict(candidate_actor_spec),
    }
    if (
        switch_regularizer_spec is not None
        and int(switch_regularizer_spec["schema_version"]) > 0
    ):
        checkpoint_schema["switch_regularizer_spec"] = dict(
            switch_regularizer_spec
        )
    if switch_constraint_spec is not None:
        checkpoint_schema["switch_constraint_spec"] = dict(
            switch_constraint_spec
        )
        checkpoint_schema["switch_constraint_state"] = dict(
            switch_constraint_state
        )
    setattr(policy, "checkpoint_schema", checkpoint_schema)
    setattr(
        policy,
        "switch_regularizer_spec",
        dict(switch_regularizer_spec)
        if switch_regularizer_spec is not None
        else None,
    )
    setattr(
        policy,
        "candidate_actor_learned_state",
        dict(candidate_actor_learned_state)
        if candidate_actor_learned_state is not None
        else None,
    )
    setattr(
        policy,
        "switch_constraint_spec",
        dict(switch_constraint_spec)
        if switch_constraint_spec is not None
        else None,
    )
    setattr(
        policy,
        "switch_constraint_state",
        dict(switch_constraint_state)
        if switch_constraint_state is not None
        else None,
    )
    return policy, checkpoint


def validate_policy_environment_schema(
    policy: Policy,
    wrapper: CleanMARLLeoMultiAgentWrapper,
) -> None:
    """Fail before an episode when a checkpoint and environment disagree."""

    expected = getattr(policy, "checkpoint_schema", None)
    if expected is None:
        return
    observed = {
        "candidate_feature_dim": wrapper.get_candidate_feature_dim(),
        "action_size": wrapper.get_action_size(),
        "obs_size": wrapper.get_obs_size(),
        "variant": wrapper.variant,
    }
    mismatches = [
        f"{key}={expected[key]} (checkpoint) != {observed[key]} (environment)"
        for key in observed
        if expected.get(key) not in (None, 0) and expected[key] != observed[key]
    ]
    if mismatches:
        raise ValueError(
            "checkpoint/environment schema mismatch: " + "; ".join(mismatches)
        )
    actor_spec = expected.get("candidate_actor_spec") or {}
    expected_switch_index = actor_spec.get("route_switch_feature_index")
    if (
        expected_switch_index is not None
        and int(expected_switch_index) != wrapper.get_route_switch_feature_index()
    ):
        raise ValueError("checkpoint/environment route-switch schema mismatch")
    if actor_spec.get("schema_version") in {2, 3, 4, 5}:
        adaptive_contracts = {
            "route_urgency_feature_index": "get_route_urgency_feature_index",
            "route_class_2_feature_index": "get_route_class_2_feature_index",
        }
        for field, getter_name in adaptive_contracts.items():
            getter = getattr(wrapper, getter_name, None)
            if getter is None or int(actor_spec[field]) != int(getter()):
                raise ValueError(
                    f"checkpoint/environment {field.replace('_', '-')} schema mismatch"
                )
        feature_schema_getter = getattr(
            wrapper, "get_candidate_feature_schema", None
        )
        if feature_schema_getter is None:
            raise ValueError("environment has no candidate feature schema")
        observed_feature_schema = feature_schema_getter()
        for field, observed_field in (
            ("candidate_feature_schema_id", "schema_id"),
            ("candidate_feature_schema_sha256", "sha256"),
        ):
            if actor_spec[field] != observed_feature_schema[observed_field]:
                raise ValueError(
                    "checkpoint/environment candidate feature schema mismatch"
                )
    constraint_spec = expected.get("switch_constraint_spec")
    if constraint_spec is not None:
        checkpoint_n_agents = expected.get("n_agents")
        environment_n_agents = getattr(wrapper, "n_agents", None)
        if (
            type(checkpoint_n_agents) is not int
            or checkpoint_n_agents <= 0
            or type(environment_n_agents) is not int
            or checkpoint_n_agents != environment_n_agents
        ):
            raise ValueError(
                "checkpoint/environment switch-constraint agent-count mismatch"
            )
        if expected.get("variant") != "qos_only" or observed["variant"] != "qos_only":
            raise ValueError(
                "switch-constraint checkpoint requires the qos_only environment"
            )
        variant_spec_getter = getattr(wrapper, "get_variant_spec", None)
        if variant_spec_getter is None:
            raise ValueError("environment has no auditable qos_only variant schema")
        observed_variant_spec = variant_spec_getter()
        if (
            not isinstance(observed_variant_spec, dict)
            or observed_variant_spec.get("name") != "qos_only"
            or observed_variant_spec.get("switch_reward") is not False
            or observed_variant_spec.get("packet_context") is not True
        ):
            raise ValueError("checkpoint/environment qos_only schema mismatch")
        if constraint_spec["route_switch_feature_index"] != (
            wrapper.get_route_switch_feature_index()
        ):
            raise ValueError(
                "checkpoint/environment switch-constraint route schema mismatch"
            )
        feature_schema_getter = getattr(
            wrapper, "get_candidate_feature_schema", None
        )
        if feature_schema_getter is None:
            raise ValueError("environment has no candidate feature schema")
        observed_feature_schema = feature_schema_getter()
        if not isinstance(observed_feature_schema, dict):
            raise ValueError("environment candidate feature schema is invalid")
        if (
            constraint_spec["candidate_feature_schema_id"]
            != observed_feature_schema.get("schema_id")
            or constraint_spec["candidate_feature_schema_sha256"]
            != observed_feature_schema.get("sha256")
        ):
            raise ValueError(
                "checkpoint/environment switch-constraint candidate feature schema mismatch"
            )


def heuristic_policy(name: str, seed: int = 0) -> Policy:
    rng = random.Random(seed)

    def choose(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_agents, flat_dim = observation.shape
        action_size = mask.shape[1]
        feature_dim = flat_dim // action_size
        candidates = observation.reshape(n_agents, action_size, feature_dim)
        actions = np.zeros(n_agents, dtype=np.int64)
        for agent in range(n_agents):
            feasible = np.flatnonzero(mask[agent] > 0.5)
            if len(feasible) == 0:
                actions[agent] = 0
                continue
            if len(feasible) == 1:
                actions[agent] = int(feasible[0])
                continue
            if name == "random":
                actions[agent] = int(rng.choice(feasible.tolist()))
                continue
            rows = candidates[agent, feasible]
            if name == "delay_only":
                score = rows[:, 2]
            elif name == "full_heuristic":
                lifetime_cost = 1.0 / np.maximum(rows[:, 6], 1e-6)
                score = (
                    1.5 * rows[:, 2]
                    + rows[:, 1]
                    + rows[:, 4]
                    + 2.0 * (1.0 - rows[:, 5])
                    + lifetime_cost
                    + 0.2 * rows[:, 17]
                    - 0.5 * rows[:, 8]
                )
            else:
                raise ValueError(f"unknown heuristic policy: {name}")
            actions[agent] = int(feasible[int(np.argmin(score))])
        return actions

    return choose


class GlobalDijkstraPolicy:
    """Global link-state baseline using current propagation delay weights."""

    def __init__(self):
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper

    def _distance(self, source: int, destination: int) -> float:
        assert self.wrapper is not None
        graph = self.wrapper.env.graph
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if node == destination:
                return distance
            if distance != distances.get(node):
                continue
            for (edge_source, neighbor), edge in graph.items():
                if edge_source != node or not edge.available:
                    continue
                candidate = distance + edge.delay_ms
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return float("inf")

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        for external_index, internal_sat in enumerate(
            self.wrapper.external_to_internal
        ):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[internal_sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            scores = []
            for action in feasible:
                if action == 0:
                    scores.append(float("inf"))
                    continue
                neighbor = obs["neighbor_ids"][int(action) - 1]
                edge = self.wrapper.env.graph[(internal_sat, neighbor)]
                scores.append(
                    edge.delay_ms + self._distance(neighbor, packet.dst)
                )
            actions[external_index] = int(feasible[int(np.argmin(scores))])
        return actions


class OspfEcmpPolicy:
    """OSPF/ECMP baseline — the realistic distributed incumbent.

    Each node forwards along the delay-weighted shortest path (SPF on its
    flooded link-state database), breaking ties across the equal-cost multi-path
    (ECMP) next-hop set by seeded random split. This is the canonical LEO/IP
    routing protocol and a required baseline: it tells the reviewer how much of
    MAPPO's gain is "learning" versus "having a sensible shortest-path policy at
    all". Uses full current link state (same information set as
    GlobalDijkstraPolicy), so it sits in the centralized-oracle column; the
    difference from Dijkstra is ECMP load splitting, which helps under
    multipath/hotspot traffic.
    """

    def __init__(self, seed: int = 0):
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None
        self.rng = random.Random(seed)
        self._cache_key: tuple = (-1, -1)
        self._cached_dist: Dict[int, float] = {}

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper
        self._cache_key = (-1, -1)
        self._cached_dist = {}

    def _dist_to_destination(self, dst: int) -> Dict[int, float]:
        """Forward distance dist(node -> dst) for every node, via one reverse
        Dijkstra from dst. Cached per (slot, dst): the topology is frozen within
        a slot, so all agents in the same slot sharing a destination reuse it."""
        assert self.wrapper is not None
        slot = self.wrapper.env.slot
        key = (slot, dst)
        if key != self._cache_key:
            self._cache_key = key
            self._cached_dist = self._reverse_dijkstra(dst)
        return self._cached_dist

    def _reverse_dijkstra(self, dst: int) -> Dict[int, float]:
        graph = self.wrapper.env.graph  # type: ignore[union-attr]
        dist: Dict[int, float] = {dst: 0.0}
        queue = [(0.0, dst)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != dist.get(node):
                continue
            for (source, neighbor), edge in graph.items():
                # reversed edge: neighbor -> source, so this relaxes source
                if neighbor != node or not edge.available:
                    continue
                candidate = distance + edge.delay_ms
                if candidate < dist.get(source, float("inf")):
                    dist[source] = candidate
                    heapq.heappush(queue, (candidate, source))
        return dist

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.wrapper.n_agents, dtype=np.int64)
        for external_index, internal_sat in enumerate(
            self.wrapper.external_to_internal
        ):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[internal_sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            dist_to_dst = self._dist_to_destination(packet.dst)
            best = float("inf")
            totals: Dict[int, float] = {}
            for action in feasible:
                if action == 0:  # NO_OP never helps a forwarder
                    totals[int(action)] = float("inf")
                    continue
                neighbor = obs["neighbor_ids"][int(action) - 1]
                edge = self.wrapper.env.graph[(internal_sat, neighbor)]
                total = edge.delay_ms + dist_to_dst.get(neighbor, float("inf"))
                totals[int(action)] = total
                if total < best:
                    best = total
            if best == float("inf"):
                actions[external_index] = 0
                continue
            # ECMP set: all equal-cost next-hops within a tight tolerance
            ecmp = [a for a, t in totals.items() if t <= best + 1e-6]
            actions[external_index] = int(self.rng.choice(ecmp))
        return actions


class QRoutingPolicy:
    """Tabular distributed Q-routing baseline indexed by node/destination/neighbor."""

    def __init__(self, n_nodes: int = 24, alpha: float = 0.3, epsilon: float = 0.1, seed: int = 0):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q = np.full((n_nodes + 1, n_nodes + 1, n_nodes + 1), 10.0, dtype=np.float32)
        self.wrapper: Optional[CleanMARLLeoMultiAgentWrapper] = None
        self.pending = []
        self.training = True

    def bind(self, wrapper: CleanMARLLeoMultiAgentWrapper) -> None:
        self.wrapper = wrapper
        self.pending = []

    def __call__(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.wrapper is not None
        actions = np.zeros(self.n_nodes, dtype=np.int64)
        self.pending = []
        for external_index, sat in enumerate(self.wrapper.external_to_internal):
            feasible = np.flatnonzero(mask[external_index] > 0.5)
            if len(feasible) <= 1:
                actions[external_index] = int(feasible[0]) if len(feasible) else 0
                continue
            obs = self.wrapper._obs[sat - 1]
            packet = self.wrapper.env.packets[obs["hol_packet_id"]]
            candidates = []
            for action in feasible:
                neighbor = obs["neighbor_ids"][int(action) - 1]
                feature = obs["candidate_features"][int(action) - 1]
                immediate = feature[2] + feature[1] + feature[4]
                candidates.append((int(action), neighbor, self.q[sat, packet.dst, neighbor] + immediate))
            if self.training and self.rng.random() < self.epsilon:
                action, neighbor, _ = self.rng.choice(candidates)
            else:
                action, neighbor, _ = min(candidates, key=lambda item: item[2])
            actions[external_index] = action
            self.pending.append((sat, packet.dst, neighbor))
        return actions

    def observe_transition(self, info: Dict) -> None:
        if not self.training or self.wrapper is None:
            return
        for sat, destination, neighbor in self.pending:
            packet_candidates = []
            for next_neighbor in self.wrapper.env.base._neighbors(neighbor):
                packet_candidates.append(self.q[neighbor, destination, next_neighbor])
            bootstrap = min(packet_candidates, default=0.0)
            edge = self.wrapper.env.graph.get((sat, neighbor))
            immediate = edge.delay_ms / self.wrapper.env.cfg.env.d_ref_ms if edge else 2.0
            immediate += len(self.wrapper.env.queues[neighbor]) / max(
                1, self.wrapper.env.cfg.max_queue_packets
            )
            target = immediate + bootstrap
            old = self.q[sat, destination, neighbor]
            self.q[sat, destination, neighbor] = (1.0 - self.alpha) * old + self.alpha * target

    def freeze(self) -> None:
        self.training = False
        self.epsilon = 0.0


def train_q_routing(
    scenario: str,
    seed: int,
    episodes: int = 200,
    workload_seed_start: int = 9001,
    workload_seed_count: int = 20,
    variant: str = "proposed",
) -> QRoutingPolicy:
    policy = QRoutingPolicy(seed=seed)
    for episode in range(episodes):
        workload_seed = workload_seed_start + episode % workload_seed_count
        wrapper = CleanMARLLeoMultiAgentWrapper(
            scenario=scenario, seed=workload_seed, variant=variant
        )
        observation, _ = wrapper.reset(seed=workload_seed)
        policy.bind(wrapper)
        terminated = truncated = False
        while not terminated and not truncated:
            actions = policy(observation, wrapper.get_avail_actions())
            observation, _, terminated, truncated, info = wrapper.step(actions)
            policy.observe_transition(info)
        wrapper.close()
    policy.freeze()
    return policy


def _decision_switch_counts(info: Dict, n_agents: int) -> tuple[int, int, int]:
    arrays = {}
    for field in (
        "decision_avoidable_switch_costs",
        "decision_switch_opportunities",
        "decision_forced_switches",
    ):
        if field not in info:
            raise ValueError(f"evaluation transition omits {field}")
        values = np.asarray(info[field])
        if values.shape != (n_agents,):
            raise ValueError(f"evaluation transition {field} shape mismatch")
        if not np.isin(values, (False, True, 0, 1)).all():
            raise ValueError(f"evaluation transition {field} must be binary")
        arrays[field] = values.astype(bool, copy=False)
    costs = arrays["decision_avoidable_switch_costs"]
    opportunities = arrays["decision_switch_opportunities"]
    forced = arrays["decision_forced_switches"]
    if np.any(costs & ~opportunities):
        raise ValueError(
            "evaluation avoidable-switch cost occurs without an opportunity"
        )
    if np.any(costs & forced) or np.any(opportunities & forced):
        raise ValueError(
            "evaluation forced reroute entered the avoidable-switch constraint"
        )
    return (
        int(np.count_nonzero(costs)),
        int(np.count_nonzero(opportunities)),
        int(np.count_nonzero(forced)),
    )


def _episode_switch_summary(info: Dict) -> tuple[int, int, int, int, float]:
    counts = {}
    for field in (
        "routing_switches",
        "avoidable_routing_switches",
        "forced_routing_switches",
        "switch_opportunities",
    ):
        if field not in info:
            raise ValueError(f"evaluation episode omits {field}")
        value = info[field]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"evaluation episode {field} must be an integer")
        if int(value) < 0:
            raise ValueError(f"evaluation episode {field} must be non-negative")
        counts[field] = int(value)
    if counts["routing_switches"] != (
        counts["avoidable_routing_switches"] + counts["forced_routing_switches"]
    ):
        raise ValueError("evaluation episode route-switch partition is inconsistent")
    if counts["avoidable_routing_switches"] > counts["switch_opportunities"]:
        raise ValueError("evaluation episode avoidable switches exceed opportunities")
    if "avoidable_switch_rate" not in info:
        raise ValueError("evaluation episode omits avoidable_switch_rate")
    rate = _finite_unit_interval(
        info["avoidable_switch_rate"], "evaluation episode avoidable-switch rate"
    )
    expected_rate = counts["avoidable_routing_switches"] / max(
        1, counts["switch_opportunities"]
    )
    if rate != expected_rate:
        raise ValueError("evaluation episode avoidable-switch rate is inconsistent")
    return (
        counts["routing_switches"],
        counts["avoidable_routing_switches"],
        counts["forced_routing_switches"],
        counts["switch_opportunities"],
        rate,
    )


def _evaluate_policy(
    scenario: str,
    policy_name: str,
    policy: Policy,
    policy_seed: int,
    workload_seeds: Iterable[int],
    wrapper_factory: Optional[Callable[[int], CleanMARLLeoMultiAgentWrapper]] = None,
    variant: str = "proposed",
    *,
    include_constraint_metrics: bool,
) -> list[EpisodeMetrics] | list[ConstraintEpisodeMetrics]:
    rows = []
    for workload_seed in workload_seeds:
        if wrapper_factory is None:
            wrapper = CleanMARLLeoMultiAgentWrapper(
                scenario=scenario, seed=int(workload_seed), variant=variant
            )
        else:
            wrapper = wrapper_factory(int(workload_seed))
        validate_policy_environment_schema(policy, wrapper)
        binder = getattr(policy, "bind", None)
        if binder is not None:
            binder(wrapper)
        observation, _ = wrapper.reset(seed=int(workload_seed))
        terminated = truncated = False
        episode_reward = 0.0
        queue_samples = []
        max_queue = 0
        info = {}
        component_samples = []
        decision_avoidable_switches = 0
        decision_switch_opportunities = 0
        decision_forced_switches = 0
        while not terminated and not truncated:
            queue_lengths = [len(queue) for queue in wrapper.env.queues.values()]
            queue_samples.extend(queue_lengths)
            max_queue = max(max_queue, max(queue_lengths, default=0))
            action = policy(observation, wrapper.get_avail_actions())
            observation, reward, terminated, truncated, info = wrapper.step(action)
            if include_constraint_metrics:
                (
                    step_avoidable_switches,
                    step_switch_opportunities,
                    step_forced_switches,
                ) = _decision_switch_counts(info, wrapper.n_agents)
                decision_avoidable_switches += step_avoidable_switches
                decision_switch_opportunities += step_switch_opportunities
                decision_forced_switches += step_forced_switches
            observer = getattr(policy, "observe_transition", None)
            if observer is not None:
                observer(info)
            episode_reward += reward
            component_samples.append(info["reward_components"])

        env = wrapper.env
        (
            routing_switches,
            avoidable_routing_switches,
            forced_routing_switches,
            switch_opportunities,
            avoidable_switch_rate,
        ) = _episode_switch_summary(info)
        delays = [
            env.delivery_slots[packet_id]
            - env.packets[packet_id].created_slot
            + 1
            for packet_id in env.delivered
        ]
        class_ratios = []
        for traffic_class in range(3):
            generated = sum(
                packet.traffic_class == traffic_class
                for packet in env.packets.values()
            )
            delivered = sum(
                env.packets[packet_id].traffic_class == traffic_class
                for packet_id in env.delivered
            )
            class_ratios.append(delivered / max(1, generated))
        slots = max(1, env.slot - 1)

        def component_mean(name: str) -> float:
            return float(
                np.mean([sample[name] for sample in component_samples])
            ) if component_samples else 0.0

        common_metrics = dict(
            scenario=scenario,
            policy=policy_name,
            policy_seed=int(policy_seed),
            workload_seed=int(workload_seed),
            generated=len(env.generated),
            delivered=len(env.delivered),
            dropped=len(env.dropped),
            backlog=len(env._backlog_ids()),
            delivery_ratio=len(env.delivered) / max(1, len(env.generated)),
            drop_rate=len(env.dropped) / max(1, len(env.generated)),
            throughput_packets_per_slot=len(env.delivered) / slots,
            average_delay_slots=float(np.mean(delays)) if delays else 0.0,
            p95_delay_slots=float(np.percentile(delays, 95)) if delays else 0.0,
            mean_queue_packets=(
                float(np.mean(queue_samples)) if queue_samples else 0.0
            ),
            max_queue_packets=max_queue,
            routing_switches=routing_switches,
            episode_reward=float(episode_reward),
            global_delay_cost=component_mean("delay_cost"),
            global_queue_cost=component_mean("queue_cost"),
            global_load_imbalance=component_mean("load_imbalance"),
            global_switch_cost=component_mean("switch_cost"),
            global_throughput_reward=component_mean("throughput_reward"),
            global_control_overhead_ratio=component_mean(
                "control_overhead_ratio"
            ),
            global_drop_cost=component_mean("drop_cost"),
            class_0_delivery_ratio=class_ratios[0],
            class_1_delivery_ratio=class_ratios[1],
            class_2_delivery_ratio=class_ratios[2],
            avoidable_routing_switches=avoidable_routing_switches,
            forced_routing_switches=forced_routing_switches,
            switch_opportunities=switch_opportunities,
            avoidable_switch_rate=avoidable_switch_rate,
        )
        if include_constraint_metrics:
            rows.append(
                ConstraintEpisodeMetrics(
                    **common_metrics,
                    decision_avoidable_switches=decision_avoidable_switches,
                    decision_switch_opportunities=decision_switch_opportunities,
                    decision_forced_switches=decision_forced_switches,
                    decision_avoidable_switch_rate=(
                        decision_avoidable_switches
                        / decision_switch_opportunities
                        if decision_switch_opportunities > 0
                        else 0.0
                    ),
                )
            )
        else:
            rows.append(EpisodeMetrics(**common_metrics))
        wrapper.close()
    return rows


def evaluate_policy(
    scenario: str,
    policy_name: str,
    policy: Policy,
    policy_seed: int,
    workload_seeds: Iterable[int],
    wrapper_factory: Optional[Callable[[int], CleanMARLLeoMultiAgentWrapper]] = None,
    variant: str = "proposed",
) -> list[EpisodeMetrics]:
    if getattr(policy, "switch_constraint_spec", None) is not None:
        raise ValueError(
            "constrained checkpoints require evaluate_policy_with_constraint_metrics"
        )
    return _evaluate_policy(
        scenario,
        policy_name,
        policy,
        policy_seed,
        workload_seeds,
        wrapper_factory,
        variant,
        include_constraint_metrics=False,
    )


def evaluate_policy_with_constraint_metrics(
    scenario: str,
    policy_name: str,
    policy: Policy,
    policy_seed: int,
    workload_seeds: Iterable[int],
    wrapper_factory: Optional[Callable[[int], CleanMARLLeoMultiAgentWrapper]] = None,
    variant: str = "qos_only",
) -> list[ConstraintEpisodeMetrics]:
    rows = _evaluate_policy(
        scenario,
        policy_name,
        policy,
        policy_seed,
        workload_seeds,
        wrapper_factory,
        variant,
        include_constraint_metrics=True,
    )
    if not all(isinstance(row, ConstraintEpisodeMetrics) for row in rows):
        raise RuntimeError("constraint evaluation produced a legacy metrics row")
    return rows


def metrics_as_dicts(rows: Iterable[EpisodeMetrics]) -> list[Dict]:
    return [asdict(row) for row in rows]
