from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from leo_multiagent_env import (
    ROUTE_SWITCH_FEATURE_INDEX,
    candidate_feature_schema,
)
from mappo_design import SharedCandidateActor
from mappo_evaluation import (
    ConstraintEpisodeMetrics,
    EpisodeMetrics,
    _decision_switch_counts,
    _episode_switch_summary,
    evaluate_policy,
    evaluate_policy_with_constraint_metrics,
    load_checkpoint_policy,
    validate_policy_environment_schema,
)


ACTION_SIZE = 4
HIDDEN_DIM = 8
N_AGENTS = 3
BUDGET = 0.12
DUAL_LEARNING_RATE = 0.05
DUAL_MAX = 5.0
FEATURE_SCHEMA = candidate_feature_schema()
FEATURE_DIM = len(FEATURE_SCHEMA["feature_names"])


def switch_constraint_spec() -> dict:
    return {
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
        "budget": BUDGET,
        "dual_learning_rate": DUAL_LEARNING_RATE,
        "dual_initial": 0.0,
        "dual_projection": [0.0, DUAL_MAX],
        "no_op_action_index": 0,
        "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
        "candidate_feature_schema_id": FEATURE_SCHEMA["schema_id"],
        "candidate_feature_schema_sha256": FEATURE_SCHEMA["sha256"],
        "reward_objective": "qos_only_without_switch_reward",
        "leo_variant": "qos_only",
    }


def switch_constraint_state() -> dict:
    return {
        "schema_version": 1,
        "multiplier": DUAL_LEARNING_RATE * (0.25 - BUDGET),
        "update_count": 1,
        "skipped_zero_opportunity_count": 0,
        "cumulative_cost": 1,
        "cumulative_opportunity": 4,
        "last_rollout_cost": 1,
        "last_rollout_opportunity": 4,
        "last_rollout_rate": 0.25,
        "last_rollout_violation": 0.25 - BUDGET,
    }


def constraint_checkpoint() -> dict:
    actor = SharedCandidateActor(
        candidate_feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        route_switch_feature_index=ROUTE_SWITCH_FEATURE_INDEX,
        route_hysteresis_beta=0.0,
    )
    return {
        "candidate_feature_dim": FEATURE_DIM,
        "action_size": ACTION_SIZE,
        "obs_size": ACTION_SIZE * FEATURE_DIM,
        "n_agents": N_AGENTS,
        "args": {
            "actor_hidden_dim": HIDDEN_DIM,
            "actor_num_layers": 1,
            "leo_variant": "qos_only",
            "route_hysteresis_beta": 0.0,
            "route_hysteresis_mode": "legacy_additive",
            "route_hysteresis_residual_init": 0.0,
            "route_hysteresis_residual_cap": 0.0,
            "route_hysteresis_residual_parameterization": "scalar",
            "avoidable_switch_constraint_enabled": True,
            "avoidable_switch_probability_coef": 0.0,
            "avoidable_switch_regularization_mode": "conditional_probability",
            "avoidable_switch_logit_margin": 0.0,
            "avoidable_switch_reduction": "rollout_micro_mean",
            "avoidable_switch_budget": BUDGET,
            "avoidable_switch_dual_learning_rate": DUAL_LEARNING_RATE,
            "avoidable_switch_dual_initial": 0.0,
            "avoidable_switch_dual_max": DUAL_MAX,
        },
        "actor": {
            f"shared_candidate_actor.{key}": value
            for key, value in actor.state_dict().items()
        },
        "candidate_actor_spec": {
            "schema_version": 1,
            "type": "shared_candidate_actor",
            "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
            "route_hysteresis_beta": 0.0,
        },
        "switch_regularizer_spec": None,
        "switch_constraint_spec": switch_constraint_spec(),
        "switch_constraint_state": switch_constraint_state(),
        "trainer_state": {
            "schema_version": 1,
            "update_round": 1,
        },
    }


def schema_wrapper(
    *,
    feature_schema: dict | None = None,
    variant: str = "qos_only",
    switch_reward: bool = False,
    packet_context: bool = True,
    n_agents: int = N_AGENTS,
):
    return SimpleNamespace(
        n_agents=n_agents,
        variant=variant,
        get_candidate_feature_dim=lambda: FEATURE_DIM,
        get_action_size=lambda: ACTION_SIZE,
        get_obs_size=lambda: ACTION_SIZE * FEATURE_DIM,
        get_route_switch_feature_index=lambda: ROUTE_SWITCH_FEATURE_INDEX,
        get_candidate_feature_schema=lambda: (
            FEATURE_SCHEMA if feature_schema is None else feature_schema
        ),
        get_variant_spec=lambda: {
            "name": variant,
            "switch_reward": switch_reward,
            "packet_context": packet_context,
        },
    )


class FakeEpisodeEnv:
    def __init__(self):
        self.queues = {1: [], 2: []}
        self.delivery_slots = {}
        self.packets = {}
        self.delivered = set()
        self.dropped = set()
        self.generated = set()
        self.slot = 3

    def _backlog_ids(self):
        return set()


class FakeEvaluationWrapper:
    def __init__(self, transitions: list[dict]):
        self.n_agents = 2
        self.env = FakeEpisodeEnv()
        self.transitions = transitions
        self.index = 0
        self.closed = False

    def reset(self, seed=None):
        self.index = 0
        return np.zeros((self.n_agents, 4), dtype=np.float32), {}

    def get_avail_actions(self):
        return np.ones((self.n_agents, 2), dtype=np.float32)

    def step(self, actions):
        info = self.transitions[self.index]
        self.index += 1
        terminated = self.index == len(self.transitions)
        return (
            np.zeros((self.n_agents, 4), dtype=np.float32),
            0.0,
            terminated,
            False,
            info,
        )

    def close(self):
        self.closed = True


def transition_info(costs, opportunities, forced) -> dict:
    return {
        "decision_avoidable_switch_costs": costs,
        "decision_switch_opportunities": opportunities,
        "decision_forced_switches": forced,
        "reward_components": {
            "delay_cost": 0.0,
            "queue_cost": 0.0,
            "load_imbalance": 0.0,
            "switch_cost": 0.0,
            "throughput_reward": 0.0,
            "control_overhead_ratio": 0.0,
            "drop_cost": 0.0,
        },
        "routing_switches": 0,
        "avoidable_routing_switches": 0,
        "forced_routing_switches": 0,
        "switch_opportunities": 0,
        "avoidable_switch_rate": 0.0,
    }


class CheckpointConstraintSchemaTests(unittest.TestCase):
    @staticmethod
    def _save(directory: str, payload: dict) -> Path:
        path = Path(directory) / "checkpoint.pt"
        torch.save(payload, path)
        return path

    def test_constraint_spec_and_state_are_loaded_and_exposed(self):
        checkpoint = constraint_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(self._save(directory, checkpoint))

        self.assertEqual(policy.switch_constraint_spec, switch_constraint_spec())
        self.assertEqual(policy.switch_constraint_state, switch_constraint_state())
        self.assertEqual(
            policy.checkpoint_schema["switch_constraint_spec"],
            switch_constraint_spec(),
        )
        self.assertEqual(
            policy.checkpoint_schema["switch_constraint_state"],
            switch_constraint_state(),
        )
        validate_policy_environment_schema(policy, schema_wrapper())

    def test_constraint_checkpoint_metadata_is_fail_closed(self):
        cases = {}
        missing_spec = constraint_checkpoint()
        missing_spec.pop("switch_constraint_spec")
        cases["missing spec"] = missing_spec
        missing_state = constraint_checkpoint()
        missing_state.pop("switch_constraint_state")
        cases["missing state"] = missing_state
        disabled_injection = constraint_checkpoint()
        disabled_injection["args"]["avoidable_switch_constraint_enabled"] = False
        cases["disabled injection"] = disabled_injection
        inconsistent_state = constraint_checkpoint()
        inconsistent_state["switch_constraint_state"]["last_rollout_rate"] = 0.5
        cases["inconsistent state"] = inconsistent_state
        inconsistent_multiplier = constraint_checkpoint()
        inconsistent_multiplier["switch_constraint_state"]["multiplier"] = 0.05
        cases["inconsistent first multiplier"] = inconsistent_multiplier
        impossible_history = constraint_checkpoint()
        impossible_history["switch_constraint_state"].update(
            multiplier=1.0,
            update_count=0,
            cumulative_cost=0,
            cumulative_opportunity=0,
            last_rollout_cost=0,
            last_rollout_opportunity=0,
            last_rollout_rate=None,
            last_rollout_violation=None,
        )
        impossible_history["trainer_state"]["update_round"] = 0
        cases["impossible history"] = impossible_history
        invalid_agent_count = constraint_checkpoint()
        invalid_agent_count["n_agents"] = str(N_AGENTS)
        cases["invalid agent count"] = invalid_agent_count

        for label, checkpoint in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "switch-constraint"):
                    load_checkpoint_policy(self._save(directory, checkpoint))

    def test_constraint_environment_feature_schema_is_checked(self):
        checkpoint = constraint_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(self._save(directory, checkpoint))
        mismatched = dict(FEATURE_SCHEMA, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "candidate feature schema mismatch"):
            validate_policy_environment_schema(
                policy,
                schema_wrapper(feature_schema=mismatched),
            )

    def test_constraint_environment_requires_auditable_qos_only_reward(self):
        checkpoint = constraint_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(self._save(directory, checkpoint))
        with self.assertRaisesRegex(ValueError, "qos_only schema mismatch"):
            validate_policy_environment_schema(
                policy,
                schema_wrapper(switch_reward=True),
            )

    def test_constraint_environment_requires_matching_agent_count(self):
        checkpoint = constraint_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(self._save(directory, checkpoint))
        with self.assertRaisesRegex(ValueError, "agent-count mismatch"):
            validate_policy_environment_schema(
                policy,
                schema_wrapper(n_agents=N_AGENTS + 1),
            )

    def test_constrained_policy_requires_the_extended_evaluator(self):
        checkpoint = constraint_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(self._save(directory, checkpoint))
        with self.assertRaisesRegex(ValueError, "constraint_metrics"):
            evaluate_policy(
                scenario="medium_load",
                policy_name="constraint",
                policy=policy,
                policy_seed=123,
                workload_seeds=[],
            )


class DecisionMetricTests(unittest.TestCase):
    def test_evaluate_policy_accumulates_pre_contention_decisions(self):
        wrapper = FakeEvaluationWrapper(
            [
                transition_info([1, 0], [1, 1], [0, 0]),
                transition_info([0, 1], [0, 1], [1, 0]),
            ]
        )
        rows = evaluate_policy_with_constraint_metrics(
            scenario="medium_load",
            policy_name="test_policy",
            policy=lambda observation, mask: np.zeros(2, dtype=np.int64),
            policy_seed=123,
            workload_seeds=[1],
            wrapper_factory=lambda seed: wrapper,
        )

        self.assertTrue(wrapper.closed)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].avoidable_routing_switches, 0)
        self.assertEqual(rows[0].decision_avoidable_switches, 2)
        self.assertEqual(rows[0].decision_switch_opportunities, 3)
        self.assertEqual(rows[0].decision_forced_switches, 1)
        self.assertAlmostEqual(rows[0].decision_avoidable_switch_rate, 2 / 3)

    def test_evaluate_policy_marks_zero_opportunity_with_its_denominator(self):
        wrapper = FakeEvaluationWrapper(
            [transition_info([0, 0], [0, 0], [1, 0])]
        )
        row = evaluate_policy_with_constraint_metrics(
            scenario="medium_load",
            policy_name="test_policy",
            policy=lambda observation, mask: np.zeros(2, dtype=np.int64),
            policy_seed=123,
            workload_seeds=[1],
            wrapper_factory=lambda seed: wrapper,
        )[0]

        self.assertEqual(row.decision_avoidable_switches, 0)
        self.assertEqual(row.decision_switch_opportunities, 0)
        self.assertEqual(row.decision_forced_switches, 1)
        self.assertEqual(row.decision_avoidable_switch_rate, 0.0)

    def test_legacy_evaluator_keeps_the_frozen_metrics_schema(self):
        legacy_fields = tuple(field.name for field in fields(EpisodeMetrics))
        constraint_fields = tuple(
            field.name for field in fields(ConstraintEpisodeMetrics)
        )
        decision_fields = (
            "decision_avoidable_switches",
            "decision_switch_opportunities",
            "decision_forced_switches",
            "decision_avoidable_switch_rate",
        )
        self.assertEqual(
            constraint_fields[: len(legacy_fields)],
            legacy_fields,
        )
        self.assertTrue(set(decision_fields).isdisjoint(legacy_fields))
        self.assertEqual(constraint_fields[-len(decision_fields) :], decision_fields)

        wrapper = FakeEvaluationWrapper(
            [transition_info([0, 0], [0, 0], [0, 0])]
        )
        row = evaluate_policy(
            scenario="medium_load",
            policy_name="legacy",
            policy=lambda observation, mask: np.zeros(2, dtype=np.int64),
            policy_seed=123,
            workload_seeds=[1],
            wrapper_factory=lambda seed: wrapper,
        )[0]
        self.assertIs(type(row), EpisodeMetrics)
        for field in decision_fields:
            self.assertFalse(hasattr(row, field))

    def test_decision_accounting_rejects_forced_or_unavailable_costs(self):
        invalid = (
            {
                "decision_avoidable_switch_costs": [1, 0],
                "decision_switch_opportunities": [0, 0],
                "decision_forced_switches": [0, 0],
            },
            {
                "decision_avoidable_switch_costs": [0, 0],
                "decision_switch_opportunities": [1, 0],
                "decision_forced_switches": [1, 0],
            },
        )
        for info in invalid:
            with self.subTest(info=info), self.assertRaises(ValueError):
                _decision_switch_counts(info, 2)

    def test_episode_switch_summary_is_fail_closed(self):
        valid = transition_info([0, 0], [0, 0], [0, 0])
        invalid = []
        missing = dict(valid)
        missing.pop("routing_switches")
        invalid.append(missing)
        invalid.append(dict(valid, routing_switches=1))
        invalid.append(
            dict(
                valid,
                routing_switches=1,
                avoidable_routing_switches=1,
                switch_opportunities=2,
                avoidable_switch_rate=0.25,
            )
        )
        for info in invalid:
            with self.subTest(info=info), self.assertRaises(ValueError):
                _episode_switch_summary(info)


if __name__ == "__main__":
    unittest.main()
