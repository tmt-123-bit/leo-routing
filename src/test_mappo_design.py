from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from leo_marl_env import EnvConfig, LeoRoutingEnv, SCENARIOS
from leo_multiagent_env import (
    EVENT_ORDER,
    MultiAgentConfig,
    SynchronousLeoMultiAgentEnv,
    candidate_feature_schema,
    first_feasible_actions,
)
from mappo_design import (
    GraphAttentionCritic,
    PacketConditionedCritic,
    SharedCandidateActor,
    avoidable_switch_actual_logit_margin_penalties,
    avoidable_switch_logit_margin_penalties,
    avoidable_switch_probabilities,
    compute_gae,
    masked_standardize,
    permute_flat_graph_state,
    select_leo_validation_record,
    shuffled_transition_minibatches,
)
from mappo_evaluation import (
    GlobalDijkstraPolicy,
    QRoutingPolicy,
    load_checkpoint_policy,
    validate_policy_environment_schema,
)
from ns3_policy_bridge import Ns3PolicyBridge, SCHEMA_VERSION
from tle_topology_builder import (
    build_snapshots,
    read_three_line_tle,
    select_four_by_six,
    validate_snapshot_connectivity,
)


class ObservationTests(unittest.TestCase):
    def test_actor_observation_distinguishes_switch_context(self):
        env = LeoRoutingEnv.from_scenario("medium_load")
        obs = env.reset(src=1, dst=12)
        neighbor = obs["neighbor_ids"][0]
        env.packet.last_next_hop = neighbor
        same_hop = env.as_mappo_inputs(env._make_obs())["actor_obs"]
        env.packet.last_next_hop = obs["neighbor_ids"][-1]
        switched_hop = env.as_mappo_inputs(env._make_obs())["actor_obs"]
        self.assertNotEqual(same_hop, switched_hop)

    def test_actor_and_critic_distinguish_ttl_context(self):
        env = LeoRoutingEnv.from_scenario("medium_load")
        env.reset(src=1, dst=12)
        env.packet.hop_count = 0
        low = env.as_mappo_inputs(env._make_obs())
        env.packet.hop_count = env.cfg.max_local_hops - 1
        high = env.as_mappo_inputs(env._make_obs())
        self.assertNotEqual(low["actor_obs"], high["actor_obs"])
        self.assertNotEqual(low["critic_state"], high["critic_state"])

    def test_single_packet_schema(self):
        env = LeoRoutingEnv.from_scenario("medium_load")
        inputs = env.as_mappo_inputs(env.reset(src=1, dst=12))
        self.assertEqual(len(inputs["candidate_obs"]), env.max_degree)
        self.assertTrue(
            all(
                len(row) == env.candidate_feature_dim
                for row in inputs["candidate_obs"]
            )
        )
        self.assertTrue(np.isfinite(inputs["critic_state"]).all())


class NetworkMathTests(unittest.TestCase):
    def test_candidate_logits_are_permutation_equivariant(self):
        torch.manual_seed(7)
        actor = SharedCandidateActor(20, 32, 1)
        candidates = torch.randn(4, 6, 20)
        mask = torch.tensor(
            [[True, True, False, True, True, False]] * 4,
            dtype=torch.bool,
        )
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        inverse = torch.argsort(permutation)
        original = actor(candidates, mask)
        permuted = actor(candidates[:, permutation], mask[:, permutation])
        restored = permuted[:, inverse]
        self.assertTrue(torch.allclose(original, restored, atol=1e-6))

    def test_zero_variance_standardization_is_finite(self):
        values = torch.ones(2, 3, 4)
        mask = torch.ones_like(values, dtype=torch.bool)
        normalized = masked_standardize(values, mask)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.allclose(normalized, torch.zeros_like(normalized)))

    def test_avoidable_switch_probability_excludes_first_use_and_forced_reroutes(self):
        logits = torch.zeros(4, 5)
        candidates = torch.zeros(4, 5, 26)
        candidates[..., 17] = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, False, True, False, False],
            ]
        )

        probabilities, eligible = avoidable_switch_probabilities(
            logits, candidates, mask, 17
        )

        self.assertTrue(
            torch.allclose(probabilities[0], torch.tensor(0.5))
        )
        self.assertEqual(eligible.tolist(), [True, False, False, False])

    def test_avoidable_switch_probability_gradient_favors_cached_route(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        candidates = torch.zeros(1, 3, 26)
        candidates[..., 17] = torch.tensor([[0.0, 0.0, 1.0]])
        probability, eligible = avoidable_switch_probabilities(
            logits,
            candidates,
            torch.tensor([[False, True, True]]),
            17,
        )
        self.assertTrue(bool(eligible.item()))
        probability[eligible].mean().backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    def test_avoidable_switch_margin_has_zero_gradient_after_stability(self):
        logits = torch.tensor([[0.0, 0.3, 0.2]], requires_grad=True)
        candidates = torch.zeros(1, 3, 26)
        candidates[..., 17] = torch.tensor([[0.0, 0.0, 1.0]])
        penalties, eligible = avoidable_switch_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[False, True, True]]),
            17,
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_beta=0.2,
        )
        self.assertTrue(bool(eligible.item()))
        self.assertEqual(float(penalties.item()), 0.0)
        penalties.sum().backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_avoidable_switch_margin_gradient_corrects_greedy_switch(self):
        logits = torch.tensor([[0.0, 0.0, 0.2]], requires_grad=True)
        candidates = torch.zeros(1, 3, 26)
        candidates[..., 17] = torch.tensor([[0.0, 0.0, 1.0]])
        penalties, eligible = avoidable_switch_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[False, True, True]]),
            17,
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_beta=0.2,
        )
        self.assertTrue(bool(eligible.item()))
        self.assertTrue(torch.allclose(penalties, torch.tensor([0.2])))
        penalties.sum().backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    def test_avoidable_switch_margin_uses_positive_margin_to_resolve_ties(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        candidates = torch.zeros(1, 3, 26)
        candidates[..., 17] = torch.tensor([[0.0, 0.0, 1.0]])
        penalties, eligible = avoidable_switch_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[True, True, True]]),
            17,
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_beta=0.2,
            margin=0.01,
        )
        self.assertTrue(bool(eligible.item()))
        self.assertTrue(torch.allclose(penalties, torch.tensor([0.01])))
        penalties.sum().backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)

    def test_avoidable_switch_margin_excludes_no_op_and_masked_cached_route(self):
        logits = torch.tensor([[100.0, 0.0, 0.2], [100.0, 0.0, 0.2]])
        candidates = torch.zeros(2, 3, 26)
        candidates[..., 17] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        penalties, eligible = avoidable_switch_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[True, True, True], [True, False, True]]),
            17,
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_beta=0.2,
            margin=0.01,
        )
        self.assertEqual(eligible.tolist(), [True, False])
        self.assertTrue(torch.allclose(penalties, torch.tensor([0.21, 0.0])))

    def test_avoidable_switch_margin_applies_urgency_relief(self):
        logits = torch.tensor([[0.0, 0.2, 0.25], [0.0, 0.1, 0.25]])
        candidates = torch.zeros(2, 3, 26)
        candidates[..., 17] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        candidates[1, :, 20] = 1.0
        penalties, eligible = avoidable_switch_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[False, True, True], [False, True, True]]),
            17,
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_beta=0.2,
            route_hysteresis_urgency_relief=0.5,
            margin=0.01,
        )
        self.assertEqual(eligible.tolist(), [True, True])
        self.assertTrue(torch.allclose(penalties, torch.tensor([0.06, 0.03])))

    def test_avoidable_switch_margin_relief_does_not_create_new_pressure(self):
        logits = torch.tensor([[0.0, 0.2, 0.15], [0.0, 0.1, 0.15]])
        candidates = torch.zeros(2, 3, 26)
        candidates[..., 17] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        candidates[1, :, 20] = 1.0
        penalties, eligible = avoidable_switch_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[False, True, True], [False, True, True]]),
            17,
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_beta=0.2,
            route_hysteresis_urgency_relief=0.5,
            margin=0.01,
        )
        self.assertEqual(eligible.tolist(), [True, True])
        self.assertTrue(torch.equal(penalties, torch.zeros_like(penalties)))

    def test_route_hysteresis_residual_is_direct_and_projected(self):
        actor = SharedCandidateActor(
            26,
            8,
            route_switch_feature_index=17,
            route_hysteresis_mode="decoupled_adaptive",
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_residual_init=0.0,
            route_hysteresis_residual_cap=0.2,
        )
        self.assertIn("route_hysteresis_residual_bias", dict(actor.named_parameters()))
        self.assertEqual(
            float(actor.route_hysteresis_residual_bias.detach()), 0.0
        )

        with torch.no_grad():
            actor.route_hysteresis_residual_bias.fill_(0.3)
        actor.project_route_hysteresis_residual_()
        projected = float(actor.route_hysteresis_residual_bias.detach())
        self.assertGreater(projected, 0.0)
        self.assertLessEqual(projected, actor.route_hysteresis_residual_cap)
        with torch.no_grad():
            actor.route_hysteresis_residual_bias.fill_(-0.1)
        actor.project_route_hysteresis_residual_()
        self.assertEqual(
            float(actor.route_hysteresis_residual_bias.detach()), 0.0
        )

        without_residual = SharedCandidateActor(
            26,
            8,
            route_switch_feature_index=17,
            route_hysteresis_mode="decoupled_adaptive",
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
        )
        self.assertNotIn(
            "route_hysteresis_residual_bias",
            dict(without_residual.named_parameters()),
        )
        with self.assertRaisesRegex(ValueError, "init must not exceed"):
            SharedCandidateActor(
                26,
                8,
                route_switch_feature_index=17,
                route_hysteresis_mode="decoupled_adaptive",
                route_urgency_feature_index=20,
                route_class_2_feature_index=23,
                route_hysteresis_residual_init=0.21,
                route_hysteresis_residual_cap=0.2,
            )
        with self.assertRaisesRegex(ValueError, "only for decoupled_adaptive"):
            SharedCandidateActor(
                26,
                8,
                route_switch_feature_index=17,
                route_hysteresis_residual_cap=0.2,
            )

    def test_route_hysteresis_residual_uses_adaptive_relief(self):
        actor = SharedCandidateActor(
            26,
            8,
            route_switch_feature_index=17,
            route_hysteresis_beta=0.2,
            route_hysteresis_mode="decoupled_adaptive",
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_urgency_relief=0.5,
            route_hysteresis_residual_init=0.1,
            route_hysteresis_residual_cap=0.2,
        )
        candidates = torch.zeros(2, 3, 26)
        candidates[..., 17] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        candidates[1, :, 20] = 1.0
        mask = torch.tensor([[False, True, True], [False, True, True]])

        logits = actor(candidates, mask)

        self.assertTrue(
            torch.allclose(logits[:, 1] - logits[:, 2], torch.tensor([0.3, 0.15]))
        )

    def test_actual_margin_excludes_no_op_and_masked_cached_route(self):
        logits = torch.tensor([[100.0, 0.1, 0.2], [100.0, 0.1, 0.2]])
        candidates = torch.zeros(2, 3, 26)
        candidates[..., 17] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        penalties, eligible = avoidable_switch_actual_logit_margin_penalties(
            logits,
            candidates,
            torch.tensor([[True, True, True], [True, False, True]]),
            17,
            margin=0.01,
        )

        self.assertEqual(eligible.tolist(), [True, False])
        self.assertTrue(torch.allclose(penalties, torch.tensor([0.11, 0.0])))

    def test_isolated_regularizer_updates_only_route_residual(self):
        actor = SharedCandidateActor(
            26,
            8,
            route_switch_feature_index=17,
            route_hysteresis_mode="decoupled_adaptive",
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_residual_init=0.01,
            route_hysteresis_residual_cap=0.2,
        )
        candidates = torch.zeros(1, 3, 26)
        candidates[..., 17] = torch.tensor([[0.0, 0.0, 1.0]])
        mask = torch.tensor([[False, True, True]])

        actual_logits = actor(candidates, mask)
        isolated_logits = actor.isolated_regularizer_logits(candidates, mask)
        self.assertTrue(torch.equal(actual_logits, isolated_logits))
        penalties, eligible = avoidable_switch_actual_logit_margin_penalties(
            isolated_logits,
            candidates,
            mask,
            17,
            margin=0.02,
        )
        self.assertTrue(bool(eligible.item()))
        penalties.mean().backward()

        quality_parameters = tuple(actor.encoder.parameters()) + tuple(
            actor.scorer.parameters()
        )
        self.assertTrue(all(parameter.grad is None for parameter in quality_parameters))
        self.assertLess(
            float(actor.route_hysteresis_residual_bias.grad.detach()), 0.0
        )

    def test_policy_gradient_reaches_quality_scorer_and_route_residual(self):
        torch.manual_seed(7)
        actor = SharedCandidateActor(
            26,
            8,
            route_switch_feature_index=17,
            route_hysteresis_mode="decoupled_adaptive",
            route_urgency_feature_index=20,
            route_class_2_feature_index=23,
            route_hysteresis_residual_init=0.0,
            route_hysteresis_residual_cap=0.2,
        )
        candidates = torch.randn(4, 3, 26)
        candidates[..., 17] = torch.tensor(
            [[0.0, 0.0, 1.0]] * candidates.shape[0]
        )
        mask = torch.tensor([[False, True, True]] * candidates.shape[0])
        distribution = torch.distributions.Categorical(
            logits=actor(candidates, mask)
        )

        policy_loss = -distribution.log_prob(torch.ones(4, dtype=torch.long)).mean()
        policy_loss.backward()

        quality_gradients = [
            parameter.grad
            for parameter in tuple(actor.encoder.parameters())
            + tuple(actor.scorer.parameters())
        ]
        self.assertTrue(
            any(
                gradient is not None and bool(torch.any(gradient != 0.0))
                for gradient in quality_gradients
            )
        )
        self.assertIsNotNone(actor.route_hysteresis_residual_bias.grad)
        self.assertNotEqual(
            float(actor.route_hysteresis_residual_bias.grad.detach()), 0.0
        )

    def test_terminal_and_truncation_bootstrap(self):
        rewards = torch.tensor([[[1.0]]])
        values = torch.tensor([[[0.5]]])
        next_values = torch.tensor([[[2.0]]])
        valid = torch.tensor([[True]])

        terminal_adv, terminal_ret = compute_gae(
            rewards,
            values,
            next_values,
            terminated=torch.tensor([[[1.0]]]),
            truncated=torch.tensor([[[0.0]]]),
            valid=valid,
        )
        trunc_adv, trunc_ret = compute_gae(
            rewards,
            values,
            next_values,
            terminated=torch.tensor([[[0.0]]]),
            truncated=torch.tensor([[[1.0]]]),
            valid=valid,
        )
        self.assertAlmostEqual(terminal_ret.item(), 1.0, places=6)
        self.assertAlmostEqual(terminal_adv.item(), 0.5, places=6)
        self.assertAlmostEqual(trunc_ret.item(), 2.98, places=6)
        self.assertAlmostEqual(trunc_adv.item(), 2.48, places=6)

    def test_supervised_candidate_capacity(self):
        torch.manual_seed(9)
        actor = SharedCandidateActor(20, 32, 1)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
        candidates = torch.randn(128, 6, 20)
        mask = torch.ones(128, 6, dtype=torch.bool)
        target = candidates[:, :, 8].argmax(dim=-1)
        for _ in range(120):
            logits = actor(candidates, mask)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = (actor(candidates, mask).argmax(dim=-1) == target).float().mean()
        self.assertGreater(float(accuracy), 0.95)

    def test_graph_critic_is_invariant_to_satellite_relabeling(self):
        torch.manual_seed(12)
        wrapper = CleanMARLLeoMultiAgentWrapper("medium_load", seed=71)
        wrapper.reset()
        state = torch.from_numpy(wrapper.get_state()).float().unsqueeze(0)
        spec = wrapper.get_critic_spec()
        critic = GraphAttentionCritic(
            n_nodes=spec["n_nodes"],
            node_feature_dim=spec["node_feature_dim"],
            edge_feature_dim=spec["edge_feature_dim"],
            global_feature_dim=spec["global_feature_dim"],
            hidden_dim=32,
            num_layers=2,
        )
        permutation = torch.randperm(spec["n_nodes"])
        relabeled = permute_flat_graph_state(
            state,
            permutation,
            n_nodes=spec["n_nodes"],
            node_feature_dim=spec["node_feature_dim"],
            edge_feature_dim=spec["edge_feature_dim"],
            global_feature_dim=spec["global_feature_dim"],
        )
        self.assertTrue(
            torch.allclose(critic(state), critic(relabeled), atol=1e-6)
        )

    def test_minibatches_cover_each_valid_transition_once(self):
        valid = torch.tensor(
            [[True, True, False, True], [True, False, False, True]]
        )
        generator = torch.Generator().manual_seed(13)
        minibatches = shuffled_transition_minibatches(valid, 3, generator)
        actual = sorted(tuple(x.tolist()) for batch in minibatches for x in batch)
        expected = sorted(tuple(x.tolist()) for x in valid.nonzero())
        self.assertEqual(actual, expected)
        self.assertEqual(len(minibatches), 3)


class ValidationSelectionTests(unittest.TestCase):
    @staticmethod
    def record(
        step: int,
        delivery: float,
        class_2: float,
        switches: float,
        reward: float = 1.0,
        avoidable_switch_rate: float = 0.0,
    ) -> dict:
        return {
            "environment_steps": step,
            "delivery_ratio": delivery,
            "class_2_delivery_ratio": class_2,
            "routing_switches": switches,
            "avoidable_switch_rate": avoidable_switch_rate,
            "mean_reward": reward,
            "drop_rate": 1.0 - delivery,
            "average_delay_slots": 4.0,
        }

    def test_stability_selection_respects_delivery_and_class_2_safety(self):
        records = [
            self.record(100, 0.900, 0.80, 100.0),
            self.record(200, 0.895, 0.79, 60.0),
            self.record(300, 0.889, 0.95, 1.0),
            self.record(400, 0.897, 0.77, 2.0),
        ]
        selected = select_leo_validation_record(
            records,
            mode="stability_constrained",
            delivery_tolerance=0.01,
            class_2_tolerance=0.02,
        )
        self.assertIs(selected, records[1])

    def test_stability_selection_has_deterministic_earliest_step_tie_break(self):
        later = self.record(200, 0.90, 0.80, 50.0)
        earlier = self.record(100, 0.90, 0.80, 50.0)
        selected = select_leo_validation_record(
            [later, earlier],
            mode="stability_constrained",
        )
        self.assertIs(selected, earlier)

    def test_avoidable_stability_selection_prioritizes_eligible_switch_rate(self):
        lower_total = self.record(
            100,
            0.90,
            0.80,
            20.0,
            avoidable_switch_rate=0.40,
        )
        lower_avoidable_rate = self.record(
            200,
            0.90,
            0.80,
            30.0,
            avoidable_switch_rate=0.10,
        )
        selected = select_leo_validation_record(
            [lower_total, lower_avoidable_rate],
            mode="avoidable_stability_constrained",
        )
        self.assertIs(selected, lower_avoidable_rate)

    def test_stability_selection_is_input_order_independent(self):
        records = [
            self.record(100, 0.90, 0.80, 80.0),
            self.record(200, 0.895, 0.79, 40.0),
            self.record(300, 0.89, 0.81, 60.0),
        ]
        selected_forward = select_leo_validation_record(
            records,
            mode="stability_constrained",
            delivery_tolerance=0.01,
            class_2_tolerance=0.02,
        )
        selected_reverse = select_leo_validation_record(
            list(reversed(records)),
            mode="stability_constrained",
            delivery_tolerance=0.01,
            class_2_tolerance=0.02,
        )
        self.assertEqual(
            selected_forward["environment_steps"],
            selected_reverse["environment_steps"],
        )

    def test_stability_selection_includes_exact_tolerance_boundaries(self):
        best = self.record(100, 0.500, 0.750, 80.0)
        boundary = self.record(200, 0.375, 0.500, 20.0)
        selected = select_leo_validation_record(
            [best, boundary],
            mode="stability_constrained",
            delivery_tolerance=0.125,
            class_2_tolerance=0.250,
        )
        self.assertIs(selected, boundary)

    def test_legacy_selection_preserves_four_metric_lexicographic_order(self):
        records = [
            self.record(100, 0.90, 0.80, 1.0, reward=2.0),
            self.record(200, 0.90, 0.80, 100.0, reward=3.0),
        ]
        selected = select_leo_validation_record(
            records,
            mode="legacy_lexicographic",
        )
        self.assertIs(selected, records[1])

    def test_stability_selection_rejects_missing_or_nonfinite_metrics(self):
        missing = self.record(100, 0.90, 0.80, 20.0)
        missing.pop("class_2_delivery_ratio")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            select_leo_validation_record(
                [missing],
                mode="stability_constrained",
            )
        nonfinite = self.record(100, 0.90, 0.80, float("nan"))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            select_leo_validation_record(
                [nonfinite],
                mode="stability_constrained",
            )


class MultiAgentEnvironmentTests(unittest.TestCase):
    pairs = [(1, 12), (2, 13), (3, 14), (4, 15), (5, 16), (6, 17)]

    def test_candidate_feature_schema_is_ordered_and_hash_frozen(self):
        proposed = CleanMARLLeoMultiAgentWrapper("medium_load", variant="proposed")
        congestion = CleanMARLLeoMultiAgentWrapper(
            "medium_load", variant="with_congestion_context"
        )
        proposed_schema = proposed.get_candidate_feature_schema()
        congestion_schema = congestion.get_candidate_feature_schema()

        self.assertEqual(proposed_schema, candidate_feature_schema())
        self.assertEqual(
            proposed_schema["feature_names"][17], "route_switch_indicator"
        )
        self.assertEqual(
            proposed_schema["feature_names"][20],
            "deadline_normalized_waiting_age",
        )
        self.assertEqual(proposed_schema["feature_names"][23], "traffic_class_2")
        self.assertEqual(len(proposed_schema["feature_names"]), 26)
        self.assertEqual(len(congestion_schema["feature_names"]), 28)
        self.assertNotEqual(proposed_schema["sha256"], congestion_schema["sha256"])

    def test_fixed_candidate_schema_rejects_non_three_class_configuration(self):
        cfg = MultiAgentConfig(
            packet_class_probabilities=(0.5, 0.5),
            packet_class_deadlines=(30, 12),
            packet_class_delay_weights=(1.0, 1.8),
            packet_class_reliability_floor=(0.86, 0.88),
        )
        with self.assertRaisesRegex(ValueError, "exactly three traffic classes"):
            SynchronousLeoMultiAgentEnv(cfg)

    def test_concurrent_actions_and_packet_conservation(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        obs, _ = env.reset(seed=33, initial_pairs=self.pairs)
        actions = first_feasible_actions(obs)
        _, _, _, _, info = env.step(actions)
        self.assertGreaterEqual(info["concurrent_non_noop"], 2)
        env.validate_invariants()
        self.assertEqual(
            len(env.generated),
            len(env.delivered) + len(env.dropped) + info["backlog"],
        )

    def test_route_switch_accounting_distinguishes_avoidable_and_forced(self):
        def run_with_cached_hop(cached_hop, selected_action):
            cfg = MultiAgentConfig(
                env=EnvConfig(seed=3301, scenario=SCENARIOS["medium_load"]),
                initial_packets=1,
                exogenous_packets_per_slot=0,
                seed=3301,
            )
            env = SynchronousLeoMultiAgentEnv(cfg)
            observations, _ = env.reset(
                seed=3301,
                initial_pairs=[(1, 12, 0)],
            )
            packet = env.packets[1]
            env.route_cache[(1, packet.dst, packet.traffic_class)] = cached_hop
            actions = [0] * env.n_agents
            actions[0] = selected_action
            _, _, _, _, info = env.step(actions)
            self.assertIn(1, info["accepted"])
            return info

        probe = SynchronousLeoMultiAgentEnv.from_scenario("medium_load", seed=3301)
        observations, _ = probe.reset(
            seed=3301,
            initial_pairs=[(1, 12, 0)],
        )
        feasible = [
            action
            for action in range(1, len(observations[0]["action_mask"]))
            if observations[0]["action_mask"][action]
        ]
        self.assertGreaterEqual(len(feasible), 2)
        cached_action, switch_action = feasible[:2]
        cached_hop = observations[0]["neighbor_ids"][cached_action - 1]

        avoidable = run_with_cached_hop(cached_hop, switch_action)
        self.assertEqual(avoidable["routing_switches"], 1)
        self.assertEqual(avoidable["avoidable_routing_switches"], 1)
        self.assertEqual(avoidable["forced_routing_switches"], 0)
        self.assertEqual(avoidable["switch_opportunities"], 1)

        forced = run_with_cached_hop(999, cached_action)
        self.assertEqual(forced["routing_switches"], 1)
        self.assertEqual(forced["avoidable_routing_switches"], 0)
        self.assertEqual(forced["forced_routing_switches"], 1)
        self.assertEqual(forced["switch_opportunities"], 0)

    def test_six_objective_team_reward_and_centered_credit(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load", seed=34)
        observations, _ = env.reset(seed=34, initial_pairs=self.pairs)
        _, agent_rewards, _, _, info = env.step(
            first_feasible_actions(observations)
        )
        components = info["reward_components"]
        expected_keys = {
            "team_reward",
            "delay_cost",
            "queue_cost",
            "load_imbalance",
            "switch_cost",
            "throughput_reward",
            "control_overhead_ratio",
            "drop_cost",
        }
        self.assertEqual(set(components), expected_keys)
        cfg = env.cfg
        expected_reward = (
            -cfg.global_delay_weight * components["delay_cost"]
            -cfg.global_queue_weight * components["queue_cost"]
            -cfg.global_imbalance_weight * components["load_imbalance"]
            -cfg.global_switch_weight * components["switch_cost"]
            +cfg.global_throughput_weight * components["throughput_reward"]
            -cfg.global_control_weight * components["control_overhead_ratio"]
            -cfg.global_drop_weight * components["drop_cost"]
        )
        self.assertAlmostEqual(info["global_reward"], expected_reward, places=7)
        self.assertAlmostEqual(
            float(np.mean(agent_rewards)), info["global_reward"], places=7
        )
        self.assertGreaterEqual(components["control_overhead_ratio"], 0.0)
        self.assertLessEqual(components["control_overhead_ratio"], 1.0)

    def test_no_credit_returns_identical_team_rewards(self):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=35, scenario=SCENARIOS["medium_load"]),
            initial_packets=6,
            exogenous_packets_per_slot=0,
            variant="no_credit",
            seed=35,
        )
        env = SynchronousLeoMultiAgentEnv(cfg)
        observations, _ = env.reset(seed=35, initial_pairs=self.pairs)
        _, rewards, _, _, info = env.step(first_feasible_actions(observations))
        self.assertTrue(
            all(abs(reward - info["global_reward"]) < 1e-9 for reward in rewards)
        )

    def test_controlled_variants_disable_only_declared_inputs(self):
        queue_cfg = MultiAgentConfig(
            env=EnvConfig(seed=36, scenario=SCENARIOS["medium_load"]),
            variant="no_queue",
            exogenous_packets_per_slot=0,
            seed=36,
        )
        queue_env = SynchronousLeoMultiAgentEnv(queue_cfg)
        queue_obs, _ = queue_env.reset(seed=36, initial_pairs=[(1, 12, 0)])
        self.assertTrue(
            all(row[0] == 0.0 and row[1] == 0.0 for row in queue_obs[0]["candidate_features"] if any(row))
        )
        self.assertTrue(
            all(node[0] == 0.0 for node in queue_env.global_state()["node_features"])
        )

        context_cfg = MultiAgentConfig(
            env=EnvConfig(seed=36, scenario=SCENARIOS["medium_load"]),
            variant="no_packet_context",
            exogenous_packets_per_slot=0,
            seed=36,
        )
        context_env = SynchronousLeoMultiAgentEnv(context_cfg)
        context_obs, _ = context_env.reset(seed=36, initial_pairs=[(1, 12, 1)])
        active_rows = [row for row in context_obs[0]["candidate_features"] if any(row)]
        self.assertTrue(all(row[15:26] == [0.0] * 11 for row in active_rows))

        flat_wrapper = CleanMARLLeoMultiAgentWrapper(
            "medium_load", seed=36, variant="flat_critic"
        )
        flat_wrapper.reset(seed=36)
        self.assertIsNone(flat_wrapper.get_critic_spec())

        lifetime_cfg = MultiAgentConfig(
            env=EnvConfig(seed=36, scenario=SCENARIOS["medium_load"]),
            variant="no_lifetime",
            exogenous_packets_per_slot=0,
            seed=36,
        )
        lifetime_env = SynchronousLeoMultiAgentEnv(lifetime_cfg)
        lifetime_env.reset(seed=36, initial_pairs=[(1, 12, 0)])
        packet = lifetime_env.packets[1]
        neighbor = lifetime_env.base._neighbors(1)[0]
        lifetime_env.graph[(1, neighbor)].t_rem = 0.5
        self.assertNotEqual(
            lifetime_env._mask_reason(packet, 1, neighbor), "lifetime"
        )
        self.assertEqual(
            lifetime_env._candidate_features(packet, 1, neighbor)[6], 0.0
        )

    def test_agent_iteration_order_does_not_change_transition(self):
        env_a = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        env_b = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        obs_a, _ = env_a.reset(seed=44, initial_pairs=self.pairs)
        obs_b, _ = env_b.reset(seed=44, initial_pairs=self.pairs)
        actions = first_feasible_actions(obs_a)
        self.assertEqual(actions, first_feasible_actions(obs_b))
        env_a.step(actions, agent_order=list(range(1, env_a.n_agents + 1)))
        env_b.step(actions, agent_order=list(range(env_b.n_agents, 0, -1)))
        self.assertEqual(env_a.state_digest(), env_b.state_digest())
        self.assertEqual(env_a.trace_hash(), env_b.trace_hash())

    def test_deterministic_replay(self):
        digests = []
        for _ in range(2):
            env = SynchronousLeoMultiAgentEnv.from_scenario("fault_links", seed=55)
            obs, _ = env.reset(seed=55, initial_pairs=self.pairs)
            for _ in range(4):
                obs, _, terminated, truncated, _ = env.step(
                    first_feasible_actions(obs)
                )
                if terminated or truncated:
                    break
            digests.append((env.state_digest(), env.trace_hash()))
        self.assertEqual(digests[0], digests[1])

    def test_inactive_agents_are_forced_noop_and_excluded_from_policy(self):
        wrapper = CleanMARLLeoMultiAgentWrapper("medium_load")
        wrapper.reset()
        avail = wrapper.get_avail_actions()
        active = wrapper.get_policy_active_mask().astype(bool)
        self.assertTrue((avail[~active, 0] == 1).all())
        self.assertTrue((avail[~active, 1:] == 0).all())
        self.assertTrue((avail[active, 0] == 0).all())

    def test_actor_observation_has_no_global_state_field(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        observations, _ = env.reset(seed=66, initial_pairs=self.pairs)
        forbidden = {"global_state", "all_queues", "future_topology"}
        for obs in observations:
            self.assertTrue(forbidden.isdisjoint(obs.keys()))

    def test_packet_class_waiting_time_and_path_context_are_observed(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        observations, _ = env.reset(
            seed=68, initial_pairs=[(1, 12, 1)]
        )
        active = observations[0]
        self.assertEqual(active["packet_class"], 1)
        self.assertEqual(active["hol_waiting_ratio"], 0.0)
        self.assertEqual(len(active["candidate_features"][0]), 26)
        env.slot = 7
        active = env.observe()[0]
        self.assertAlmostEqual(active["hol_waiting_ratio"], 0.5)
        self.assertEqual(active["candidate_features"][0][21:24], [0.0, 1.0, 0.0])

    def test_central_state_contains_masks_previous_actions_and_contention(self):
        env = SynchronousLeoMultiAgentEnv.from_scenario("medium_load")
        observations, _ = env.reset(seed=69, initial_pairs=self.pairs)
        env.step(first_feasible_actions(observations))
        state = env.global_state()
        self.assertEqual(state["schema"]["node_feature_dim"], 25)
        self.assertEqual(state["schema"]["edge_feature_dim"], 11)
        self.assertEqual(len(state["node_features"]), env.n_agents)
        self.assertTrue(
            any(edge[8] == 1.0 for row in state["edge_features"] for edge in row)
        )
        current_obs = env.observe()
        for sat, node in enumerate(state["node_features"]):
            self.assertEqual(
                node[16:23],
                [float(x) for x in current_obs[sat]["action_mask"]],
            )

    def test_slot_event_order_and_exogenous_packet_conservation(self):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=70, scenario=SCENARIOS["medium_load"]),
            initial_packets=8,
            exogenous_packets_per_slot=4,
            seed=70,
        )
        env = SynchronousLeoMultiAgentEnv(cfg)
        observations, _ = env.reset(seed=70)
        for _ in range(4):
            observations, _, _, _, info = env.step(
                first_feasible_actions(observations)
            )
            self.assertEqual(tuple(info["event_order"]), EVENT_ORDER)
            env.validate_invariants()

    def test_packet_deadline_is_an_absorbing_drop(self):
        cfg = MultiAgentConfig(
            env=EnvConfig(seed=73, scenario=SCENARIOS["medium_load"]),
            initial_packets=1,
            exogenous_packets_per_slot=0,
            packet_class_deadlines=(30, 2, 20),
            seed=73,
        )
        env = SynchronousLeoMultiAgentEnv(cfg)
        env.reset(seed=73, initial_pairs=[(1, 12, 1)])
        env.slot = 2
        expired = env._expire_deadline_packets()
        self.assertEqual(expired, [1])
        self.assertEqual(env.drop_reasons[1], "deadline_exceeded")
        env.validate_invariants()

    def test_no_route_is_forced_noop_and_not_a_policy_sample(self):
        env_cfg = EnvConfig(
            seed=74,
            scenario=SCENARIOS["medium_load"],
            topology_provider=lambda _slot, _env: {},
        )
        cfg = MultiAgentConfig(
            env=env_cfg,
            initial_packets=1,
            exogenous_packets_per_slot=0,
            seed=74,
        )
        wrapper = CleanMARLLeoMultiAgentWrapper(cfg=cfg, seed=74)
        wrapper.reset(seed=74, initial_pairs=[(1, 12, 0)])
        self.assertEqual(wrapper.get_avail_actions()[0].tolist(), [1.0] + [0.0] * 6)
        self.assertEqual(wrapper.get_policy_active_mask()[0], 0.0)
        wrapper.step(np.zeros(wrapper.n_agents, dtype=int))
        self.assertEqual(wrapper.env.drop_reasons[1], "no_route")
        self.assertIn(1, wrapper.env.dropped)

    def test_satellite_agent_relabeling_preserves_transition(self):
        permutation = list(range(24, 0, -1))
        reference = CleanMARLLeoMultiAgentWrapper("medium_load", seed=72)
        relabeled = CleanMARLLeoMultiAgentWrapper(
            "medium_load", seed=72, agent_permutation=permutation
        )
        reference_obs, _ = reference.reset(seed=72, initial_pairs=self.pairs)
        relabeled.reset(seed=72, initial_pairs=self.pairs)
        reference_actions = np.asarray(
            [np.flatnonzero(mask)[0] for mask in reference.get_avail_actions()]
        )
        relabeled_actions = np.asarray(
            [reference_actions[internal_sat - 1] for internal_sat in permutation]
        )
        reference.step(reference_actions)
        relabeled.step(relabeled_actions)
        self.assertEqual(
            reference.env.state_digest(), relabeled.env.state_digest()
        )


class RealTopologyTests(unittest.TestCase):
    def test_selected_tle_builds_connected_snapshots(self):
        path = Path(__file__).resolve().parent.parent / "data" / "starlink_24_selected.tle"
        if not path.exists():
            self.skipTest("selected TLE fixture is not present")
        satellites = read_three_line_tle(path)
        for index, satellite in enumerate(satellites):
            satellite.plane = index // 6
            satellite.local_index = index % 6
        from sgp4.conveniences import sat_epoch_datetime

        start = max(sat_epoch_datetime(satellite.satrec) for satellite in satellites)
        rows = build_snapshots(satellites, start, slots=3, slot_seconds=10)
        validate_snapshot_connectivity(rows, n_nodes=24)
        self.assertTrue(all(row["delay_ms"] > 0 for row in rows))

    def test_oneweb_selection_uses_configurable_altitude(self):
        path = Path(__file__).resolve().parent.parent / "data" / "oneweb_2026-07-15.tle"
        if not path.exists():
            self.skipTest("OneWeb TLE fixture is not present")
        selected = select_four_by_six(
            read_three_line_tle(path),
            inclination_range=(87.0, 89.0),
            plane_spacing_deg=15.2,
            altitude_range=(1000.0, 1400.0),
        )
        self.assertEqual(len(selected), 24)
        self.assertEqual({satellite.plane for satellite in selected}, set(range(4)))


class TrainingSmokeTests(unittest.TestCase):
    def test_dijkstra_and_q_routing_respect_action_masks(self):
        for policy in [GlobalDijkstraPolicy(), QRoutingPolicy(seed=14)]:
            wrapper = CleanMARLLeoMultiAgentWrapper("medium_load", seed=75)
            observation, _ = wrapper.reset(seed=75)
            policy.bind(wrapper)
            mask = wrapper.get_avail_actions()
            actions = policy(observation, mask)
            self.assertTrue(
                all(mask[agent, action] for agent, action in enumerate(actions))
            )

    def test_ns3_bridge_preserves_agent_order_and_mask(self):
        bridge = Ns3PolicyBridge.__new__(Ns3PolicyBridge)
        bridge.n_agents = 24
        bridge.action_size = 7
        bridge.feature_dim = 26
        bridge.policy = lambda observation, mask: np.asarray(
            [np.flatnonzero(row)[-1] for row in mask], dtype=np.int64
        )
        agents = []
        for agent_id in range(24, 0, -1):
            mask = [True, True, agent_id % 2 == 0, False, False, False, False]
            agents.append(
                {
                    "agent_id": agent_id,
                    "packet_id": 1000 + agent_id,
                    "candidate_features": [[0.0] * 26 for _ in range(7)],
                    "action_mask": mask,
                    "candidate_next_hops": [0, (agent_id % 24) + 1, 1, 0, 0, 0, 0],
                }
            )
        response = bridge.decide(
            {
                "schema_version": SCHEMA_VERSION,
                "type": "slot_state",
                "episode_id": "unit-test",
                "time_slot": 4,
                "agents": agents,
            }
        )
        self.assertEqual(
            [row["agent_id"] for row in response["decisions"]],
            list(range(1, 25)),
        )
        self.assertTrue(
            all(
                row["action_slot"] == (2 if row["agent_id"] % 2 == 0 else 1)
                for row in response["decisions"]
            )
        )

    def test_checkpoint_schema_mismatch_is_rejected_before_evaluation(self):
        actor = SharedCandidateActor(26, 32, 1)
        checkpoint = {
            "candidate_feature_dim": 26,
            "action_size": 7,
            "obs_size": 182,
            "n_agents": 24,
            "args": {
                "actor_hidden_dim": 32,
                "actor_num_layers": 1,
                "leo_variant": "no_lifetime",
            },
            "actor": {
                f"shared_candidate_actor.{key}": value
                for key, value in actor.state_dict().items()
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "old_proposed.pt"
            torch.save(checkpoint, checkpoint_path)
            checkpoint_policy, _ = load_checkpoint_policy(checkpoint_path)

        proposed = CleanMARLLeoMultiAgentWrapper(
            "medium_load",
            variant="proposed",
        )
        validate_policy_environment_schema(checkpoint_policy, proposed)

        same_dimension_wrong_variant = CleanMARLLeoMultiAgentWrapper(
            "medium_load",
            variant="no_queue",
        )
        with self.assertRaisesRegex(
            ValueError,
            "variant=proposed.*no_queue",
        ):
            validate_policy_environment_schema(
                checkpoint_policy,
                same_dimension_wrong_variant,
            )

        congestion = CleanMARLLeoMultiAgentWrapper(
            "medium_load",
            variant="with_congestion_context",
        )
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint/environment schema mismatch",
        ):
            validate_policy_environment_schema(checkpoint_policy, congestion)

    def _assert_multiagent_single_ppo_update_is_finite(self, variant: str):
        torch.manual_seed(10)
        wrapper = CleanMARLLeoMultiAgentWrapper(
            "medium_load",
            variant=variant,
        )
        obs, _ = wrapper.reset()
        mask = torch.from_numpy(wrapper.get_avail_actions()).bool()
        active = torch.from_numpy(wrapper.get_policy_active_mask()).bool()
        candidates = torch.from_numpy(obs).float().reshape(
            wrapper.n_agents,
            wrapper.get_action_size(),
            wrapper.get_candidate_feature_dim(),
        )
        actor = SharedCandidateActor(wrapper.get_candidate_feature_dim(), 32, 1)
        spec = wrapper.get_critic_spec()
        critic = GraphAttentionCritic(
            n_nodes=spec["n_nodes"],
            node_feature_dim=spec["node_feature_dim"],
            edge_feature_dim=spec["edge_feature_dim"],
            global_feature_dim=spec["global_feature_dim"],
            hidden_dim=64,
            num_layers=1,
        )
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=8e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=8e-4)

        old_logits = actor(candidates, mask)
        old_dist = torch.distributions.Categorical(logits=old_logits)
        actions = old_dist.sample()
        old_log_prob = old_dist.log_prob(actions).detach()
        state = torch.from_numpy(wrapper.get_state()).float()
        _, reward, terminated, truncated, _ = wrapper.step(actions.numpy())
        next_state = torch.from_numpy(wrapper.get_state()).float()

        values = critic(state).reshape(1, 1, 1).expand(1, 1, wrapper.n_agents)
        next_values = critic(next_state).reshape(1, 1, 1).expand_as(values)
        rewards = torch.full_like(values, reward)
        advantages, returns = compute_gae(
            rewards,
            values.detach(),
            next_values.detach(),
            terminated=torch.full_like(values, float(terminated)),
            truncated=torch.full_like(values, float(truncated)),
            valid=torch.tensor([[True]]),
        )
        active_mask = active.reshape(1, 1, -1)
        advantages = masked_standardize(advantages, active_mask)

        new_logits = actor(candidates, mask)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        ratio = torch.exp(new_dist.log_prob(actions) - old_log_prob)
        ratio = ratio.reshape(1, 1, -1)
        actor_loss = -torch.min(
            ratio * advantages,
            ratio.clamp(0.8, 1.2) * advantages,
        )[active_mask].mean()
        critic_loss = F.mse_loss(
            critic(state).reshape(1), returns[active_mask].mean().reshape(1)
        )
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        actor_optimizer.step()
        critic_optimizer.step()
        self.assertTrue(torch.isfinite(actor_loss))
        self.assertTrue(torch.isfinite(critic_loss))

    def test_multiagent_single_ppo_update_is_finite(self):
        self._assert_multiagent_single_ppo_update_is_finite("proposed")

    def test_congestion_context_single_ppo_update_is_finite(self):
        self._assert_multiagent_single_ppo_update_is_finite(
            "with_congestion_context"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
