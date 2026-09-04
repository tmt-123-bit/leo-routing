"""Focused contracts for decision-level avoidable-switch constraints."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from cleanmarl_mappo_leo import (
    rollout_micro_minibatch_mean,
    validate_resume_total_timesteps,
)
from leo_marl_env import EnvConfig, SCENARIOS
from leo_multiagent_env import MultiAgentConfig, SynchronousLeoMultiAgentEnv
from mappo_design import (
    avoidable_switch_decisions,
    avoidable_switch_probabilities,
    projected_lagrange_multiplier_update,
    select_leo_validation_record,
)
from variant_definitions import PROPOSED, QOS_ONLY, resolve_variant


class DecisionConstraintMathTests(unittest.TestCase):
    def test_decision_indicators_cover_stay_switch_first_forced_and_noop(self):
        candidates = torch.zeros(5, 3, 26)
        candidates[..., 17] = torch.tensor(
            [
                [0.0, 0.0, 1.0],  # stay while an alternative is feasible
                [0.0, 0.0, 1.0],  # avoidable switch
                [0.0, 0.0, 0.0],  # first route: no cached next hop
                [0.0, 1.0, 1.0],  # forced: cached next hop is infeasible
                [0.0, 0.0, 0.0],  # inactive agent's NO_OP
            ]
        )
        masks = torch.tensor(
            [
                [False, True, True],
                [False, True, True],
                [False, True, True],
                [False, True, True],
                [True, False, False],
            ]
        )
        actions = torch.tensor([1, 2, 1, 1, 0], dtype=torch.long)

        costs, opportunities = avoidable_switch_decisions(
            candidates,
            masks,
            actions,
            route_switch_feature_index=17,
        )

        self.assertEqual(costs.tolist(), [False, True, False, False, False])
        self.assertEqual(
            opportunities.tolist(), [True, True, False, False, False]
        )

    def test_positive_multiplier_gradient_reduces_switch_probability(self):
        logits = torch.nn.Parameter(torch.zeros(1, 3))
        candidates = torch.zeros(1, 3, 26)
        candidates[..., 17] = torch.tensor([[0.0, 0.0, 1.0]])
        masks = torch.tensor([[False, True, True]])
        optimizer = torch.optim.SGD([logits], lr=0.5)

        before, eligible = avoidable_switch_probabilities(
            logits, candidates, masks, 17
        )
        loss = 2.0 * before[eligible].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after, _ = avoidable_switch_probabilities(logits, candidates, masks, 17)

        self.assertLess(float(after.item()), float(before.item()))

    def test_projected_dual_update_handles_direction_boundary_and_caps(self):
        kwargs = {
            "budget": 0.12,
            "learning_rate": 0.05,
            "maximum": 5.0,
        }
        self.assertAlmostEqual(
            projected_lagrange_multiplier_update(0.1, 0.20, **kwargs),
            0.104,
        )
        self.assertAlmostEqual(
            projected_lagrange_multiplier_update(0.1, 0.04, **kwargs),
            0.096,
        )
        self.assertEqual(
            projected_lagrange_multiplier_update(0.1, 0.12, **kwargs),
            0.1,
        )
        self.assertEqual(
            projected_lagrange_multiplier_update(0.0, 0.0, **kwargs),
            0.0,
        )
        self.assertEqual(
            projected_lagrange_multiplier_update(4.99, 1.0, **kwargs),
            5.0,
        )

    def test_rollout_micro_mean_is_minibatch_partition_invariant(self):
        values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        def reduced(partitions):
            count = len(partitions)
            losses = [
                rollout_micro_minibatch_mean(
                    values[indexes],
                    rollout_denominator=len(values),
                    active_minibatch_count=count,
                )
                for indexes in partitions
            ]
            return torch.stack(losses).mean()

        one = reduced([torch.arange(5)])
        uneven = reduced(
            [torch.tensor([0, 4]), torch.tensor([1]), torch.tensor([2, 3])]
        )
        self.assertTrue(torch.allclose(one, values.mean()))
        self.assertTrue(torch.allclose(uneven, values.mean()))


class ExactResumeHorizonTests(unittest.TestCase):
    def test_disabled_lr_decay_allows_strict_horizon_extension(self):
        validate_resume_total_timesteps(
            {"total_timesteps": 1, "lr_decay": False},
            {"total_timesteps": 50, "lr_decay": False},
            saved_step=30,
        )

    def test_unchanged_horizon_is_valid_with_lr_decay(self):
        validate_resume_total_timesteps(
            {"total_timesteps": 50, "lr_decay": True},
            {"total_timesteps": 50, "lr_decay": True},
            saved_step=30,
        )

    def test_lr_decay_rejects_horizon_change(self):
        with self.assertRaisesRegex(ValueError, "learning-rate decay"):
            validate_resume_total_timesteps(
                {"total_timesteps": 1, "lr_decay": True},
                {"total_timesteps": 50, "lr_decay": True},
                saved_step=30,
            )

    def test_horizon_cannot_shrink(self):
        with self.assertRaisesRegex(ValueError, "only be extended"):
            validate_resume_total_timesteps(
                {"total_timesteps": 100, "lr_decay": False},
                {"total_timesteps": 50, "lr_decay": False},
                saved_step=30,
            )

    def test_extended_horizon_must_exceed_checkpoint_step(self):
        with self.assertRaisesRegex(ValueError, "exceed the saved step"):
            validate_resume_total_timesteps(
                {"total_timesteps": 1, "lr_decay": False},
                {"total_timesteps": 30, "lr_decay": False},
                saved_step=30,
            )


class BudgetSelectionTests(unittest.TestCase):
    @staticmethod
    def record(step: int, delivery: float, rate: float) -> dict:
        return {
            "environment_steps": step,
            "delivery_ratio": delivery,
            "class_2_delivery_ratio": 0.8,
            "decision_avoidable_switch_rate": rate,
            "mean_reward": 1.0,
            "drop_rate": 0.1,
            "average_delay_slots": 2.0,
        }

    def test_selection_prefers_qos_within_feasible_set(self):
        lowest_rate = self.record(100, 0.80, 0.05)
        best_feasible_qos = self.record(200, 0.90, 0.12)
        infeasible = self.record(300, 0.99, 0.121)
        selected = select_leo_validation_record(
            [lowest_rate, infeasible, best_feasible_qos],
            mode="avoidable_switch_budget_constrained",
            switch_budget=0.12,
        )
        self.assertIs(selected, best_feasible_qos)

    def test_selection_uses_minimum_violation_when_none_is_feasible(self):
        lower_violation = self.record(100, 0.70, 0.13)
        higher_qos = self.record(200, 0.99, 0.14)
        selected = select_leo_validation_record(
            [higher_qos, lower_violation],
            mode="avoidable_switch_budget_constrained",
            switch_budget=0.12,
        )
        self.assertIs(selected, lower_violation)

    def test_selection_rejects_nonfinite_tolerance_and_invalid_rate(self):
        with self.assertRaisesRegex(ValueError, "delivery tolerance"):
            select_leo_validation_record(
                [self.record(100, 0.9, 0.1)],
                mode="avoidable_switch_budget_constrained",
                switch_budget=0.12,
                delivery_tolerance=float("inf"),
            )
        with self.assertRaisesRegex(ValueError, "must be in"):
            select_leo_validation_record(
                [self.record(100, 0.9, 1.01)],
                mode="avoidable_switch_budget_constrained",
                switch_budget=0.12,
            )


class EnvironmentDecisionLedgerTests(unittest.TestCase):
    @staticmethod
    def make_env(variant: str = "proposed") -> SynchronousLeoMultiAgentEnv:
        seed = 3301
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=seed, scenario=SCENARIOS["medium_load"]),
            initial_packets=1,
            exogenous_packets_per_slot=0,
            seed=seed,
            variant=variant,
        )
        return SynchronousLeoMultiAgentEnv(cfg)

    def prepare_switch(self, variant: str = "proposed"):
        env = self.make_env(variant)
        observations, _ = env.reset(seed=3301, initial_pairs=[(1, 12, 0)])
        feasible = [
            action
            for action in range(1, len(observations[0]["action_mask"]))
            if observations[0]["action_mask"][action]
        ]
        self.assertGreaterEqual(len(feasible), 2)
        selected_action, cached_action = feasible[:2]
        packet = env.packets[1]
        cached_hop = observations[0]["neighbor_ids"][cached_action - 1]
        env.route_cache[(1, packet.dst, packet.traffic_class)] = cached_hop
        actions = [0] * env.n_agents
        actions[0] = selected_action
        return env, actions

    def test_precontention_ledger_includes_blocked_switch_proposal(self):
        env, actions = self.prepare_switch()
        env._resolve_link_capacity = lambda proposals: ([], list(proposals))

        _, _, _, _, info = env.step(actions)

        self.assertTrue(info["decision_avoidable_switch_costs"][0])
        self.assertTrue(info["decision_switch_opportunities"][0])
        self.assertFalse(info["decision_forced_switches"][0])
        self.assertEqual(info["avoidable_routing_switches"], 0)
        self.assertEqual(info["switch_opportunities"], 0)
        self.assertEqual(info["blocked"], [1])

    def test_forced_reroute_has_zero_constraint_cost(self):
        env = self.make_env()
        observations, _ = env.reset(seed=3301, initial_pairs=[(1, 12, 0)])
        selected_action = next(
            action
            for action in range(1, len(observations[0]["action_mask"]))
            if observations[0]["action_mask"][action]
        )
        packet = env.packets[1]
        env.route_cache[(1, packet.dst, packet.traffic_class)] = env.n_agents + 1
        actions = [0] * env.n_agents
        actions[0] = selected_action

        _, _, _, _, info = env.step(actions)

        self.assertFalse(info["decision_avoidable_switch_costs"][0])
        self.assertFalse(info["decision_switch_opportunities"][0])
        self.assertTrue(info["decision_forced_switches"][0])

    def test_qos_only_removes_switch_reward_but_preserves_accounting(self):
        proposed_env, proposed_actions = self.prepare_switch("proposed")
        qos_env, qos_actions = self.prepare_switch("qos_only")

        _, _, _, _, proposed = proposed_env.step(proposed_actions)
        _, _, _, _, qos = qos_env.step(qos_actions)

        self.assertIs(resolve_variant("proposed"), PROPOSED)
        self.assertIs(resolve_variant("qos_only"), QOS_ONLY)
        self.assertEqual(proposed["routing_switches"], qos["routing_switches"])
        self.assertEqual(
            proposed["decision_avoidable_switch_costs"],
            qos["decision_avoidable_switch_costs"],
        )
        self.assertEqual(proposed["reward_components"]["switch_cost"], 1.0)
        self.assertEqual(qos["reward_components"]["switch_cost"], 0.0)
        self.assertAlmostEqual(
            qos["local_rewards"][0] - proposed["local_rewards"][0],
            qos_env.cfg.env.w_switch,
        )
        self.assertAlmostEqual(
            qos["global_reward"] - proposed["global_reward"],
            qos_env.cfg.global_switch_weight,
        )

    def test_wrapper_reorders_all_decision_ledgers(self):
        permutation = list(range(24, 0, -1))
        wrapper = CleanMARLLeoMultiAgentWrapper(
            "medium_load",
            seed=3301,
            agent_permutation=permutation,
            variant="qos_only",
        )
        wrapper.reset(seed=3301, initial_pairs=[(1, 12, 0)])
        observation = wrapper._obs[0]
        feasible = [
            action
            for action in range(1, len(observation["action_mask"]))
            if observation["action_mask"][action]
        ]
        self.assertGreaterEqual(len(feasible), 2)
        selected_action, cached_action = feasible[:2]
        packet = wrapper.env.packets[1]
        cached_hop = observation["neighbor_ids"][cached_action - 1]
        wrapper.env.route_cache[(1, packet.dst, packet.traffic_class)] = cached_hop
        external_actions = np.zeros(wrapper.n_agents, dtype=np.int64)
        external_actions[wrapper.external_to_internal.index(1)] = selected_action

        _, _, _, _, info = wrapper.step(external_actions)

        for field, getter in (
            (
                "decision_avoidable_switch_costs",
                wrapper.get_last_avoidable_switch_costs,
            ),
            (
                "decision_switch_opportunities",
                wrapper.get_last_switch_opportunities,
            ),
            ("decision_forced_switches", wrapper.get_last_forced_switches),
        ):
            expected = np.asarray(
                [info[field][sat - 1] for sat in permutation], dtype=np.bool_
            )
            np.testing.assert_array_equal(getter(), expected)
        self.assertTrue(
            wrapper.get_last_avoidable_switch_costs()[
                wrapper.external_to_internal.index(1)
            ]
        )


if __name__ == "__main__":
    unittest.main()
