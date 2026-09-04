from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper
from cleanmarl_mappo_leo import Actor, promote_validation_checkpoint
from hysteresis_policy import load_hysteresis_policy
from leo_multiagent_env import (
    ROUTE_CLASS_2_FEATURE_INDEX,
    ROUTE_SWITCH_FEATURE_INDEX,
    ROUTE_URGENCY_FEATURE_INDEX,
    candidate_feature_schema,
)
from mappo_design import (
    SharedCandidateActor,
    route_hysteresis_residual_dtype_cap,
)
from mappo_evaluation import (
    load_checkpoint_policy,
    validate_policy_environment_schema,
)


FEATURE_DIM = 26
ACTION_SIZE = 4
HIDDEN_DIM = 16
BETA = 0.20


def actor_spec(beta: float = BETA, switch_index: int = ROUTE_SWITCH_FEATURE_INDEX):
    return {
        "schema_version": 1,
        "type": "shared_candidate_actor",
        "route_switch_feature_index": switch_index,
        "route_hysteresis_beta": beta,
    }


def adaptive_actor_spec(
    beta: float = BETA,
    urgency_relief: float = 0.5,
    class_2_relief: float = 0.0,
):
    feature_schema = candidate_feature_schema()
    return {
        **actor_spec(beta),
        "schema_version": 2,
        "route_hysteresis_mode": "decoupled_adaptive",
        "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
        "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
        "route_hysteresis_urgency_relief": urgency_relief,
        "route_hysteresis_class_2_relief": class_2_relief,
        "candidate_feature_schema_id": feature_schema["schema_id"],
        "candidate_feature_schema_sha256": feature_schema["sha256"],
    }


def regularized_actor_spec(coefficient: float = 0.1):
    return {
        **adaptive_actor_spec(),
        "schema_version": 3,
        "avoidable_switch_probability_coef": coefficient,
    }


def residual_actor_spec(
    coefficient: float = 0.1,
    residual_init: float = 0.0,
    residual_cap: float = 0.2,
):
    return {
        **regularized_actor_spec(coefficient),
        "schema_version": 4,
        "route_hysteresis_residual_parameterization": (
            "projected_nonnegative_scalar"
        ),
        "route_hysteresis_residual_init": residual_init,
        "route_hysteresis_residual_cap": residual_cap,
    }


def urgency_residual_actor_spec(
    coefficient: float = 0.1,
    residual_init: float = 0.0,
    residual_cap: float = 0.2,
):
    return {
        **regularized_actor_spec(coefficient),
        "schema_version": 5,
        "route_hysteresis_residual_parameterization": (
            "projected_nonnegative_urgency_linear_endpoints"
        ),
        "route_hysteresis_residual_init": residual_init,
        "route_hysteresis_residual_cap": residual_cap,
    }


def candidate_actor(beta: float = 0.0, switch_index=None):
    return SharedCandidateActor(
        candidate_feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        route_switch_feature_index=switch_index,
        route_hysteresis_beta=beta,
    )


def adaptive_candidate_actor(
    beta: float = BETA,
    urgency_relief: float = 0.5,
    class_2_relief: float = 0.0,
):
    return SharedCandidateActor(
        candidate_feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        route_switch_feature_index=ROUTE_SWITCH_FEATURE_INDEX,
        route_hysteresis_beta=beta,
        route_hysteresis_mode="decoupled_adaptive",
        route_urgency_feature_index=ROUTE_URGENCY_FEATURE_INDEX,
        route_class_2_feature_index=ROUTE_CLASS_2_FEATURE_INDEX,
        route_hysteresis_urgency_relief=urgency_relief,
        route_hysteresis_class_2_relief=class_2_relief,
    )


def trainer_actor(beta: float = BETA):
    return Actor(
        input_dim=ACTION_SIZE * FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layer=1,
        output_dim=ACTION_SIZE,
        candidate_feature_dim=FEATURE_DIM,
        candidate_actor_spec=actor_spec(beta),
    )


def checkpoint_for(actor: Actor, beta: float = BETA, spec_marker=...):
    checkpoint = {
        "candidate_feature_dim": FEATURE_DIM,
        "action_size": ACTION_SIZE,
        "obs_size": ACTION_SIZE * FEATURE_DIM,
        "n_agents": 3,
        "args": {
            "actor_hidden_dim": HIDDEN_DIM,
            "actor_num_layers": 1,
            "leo_variant": "with_congestion_context",
            "route_hysteresis_beta": beta,
        },
        "actor": actor.state_dict(),
    }
    if spec_marker is ...:
        checkpoint["candidate_actor_spec"] = actor_spec(beta)
    elif spec_marker is not None:
        checkpoint["candidate_actor_spec"] = spec_marker
    return checkpoint


def adaptive_checkpoint_for(
    actor: Actor,
    beta: float = BETA,
    urgency_relief: float = 0.5,
    class_2_relief: float = 0.0,
):
    spec = adaptive_actor_spec(beta, urgency_relief, class_2_relief)
    checkpoint = checkpoint_for(actor, beta=beta, spec_marker=spec)
    checkpoint["args"].update(
        {
            "leo_variant": "proposed",
            "route_hysteresis_mode": "decoupled_adaptive",
            "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
            "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
            "route_hysteresis_urgency_relief": urgency_relief,
            "route_hysteresis_class_2_relief": class_2_relief,
        }
    )
    return checkpoint


def regularized_checkpoint_for(actor: Actor, coefficient: float = 0.1):
    checkpoint = adaptive_checkpoint_for(actor)
    checkpoint["candidate_actor_spec"] = regularized_actor_spec(coefficient)
    checkpoint["args"]["avoidable_switch_probability_coef"] = coefficient
    return checkpoint


def explicit_switch_regularizer_spec(
    coefficient: float = 0.1,
    *,
    mode: str = "greedy_logit_margin",
    margin: float = 0.01,
):
    return {
        "schema_version": 1,
        "mode": mode,
        "coefficient": coefficient,
        "logit_margin": margin,
        "reduction": "conditional_mean_over_eligible_active_decisions",
        "no_op_action_index": 0,
        "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX,
        "route_urgency_feature_index": ROUTE_URGENCY_FEATURE_INDEX,
        "route_class_2_feature_index": ROUTE_CLASS_2_FEATURE_INDEX,
        "route_hysteresis_beta": BETA,
        "route_hysteresis_urgency_relief": 0.5,
        "route_hysteresis_class_2_relief": 0.0,
        "relief_weighting": (
            "none"
            if mode == "conditional_probability"
            else "restore_full_hysteresis_before_hinge_then_multiply_adaptive_relief_scale"
        ),
    }


def margin_regularized_checkpoint_for(actor: Actor, coefficient: float = 0.1):
    checkpoint = regularized_checkpoint_for(actor, coefficient)
    checkpoint["args"].update(
        {
            "avoidable_switch_regularization_mode": "greedy_logit_margin",
            "avoidable_switch_logit_margin": 0.01,
        }
    )
    checkpoint["switch_regularizer_spec"] = explicit_switch_regularizer_spec(
        coefficient
    )
    return checkpoint


def isolated_switch_regularizer_spec(coefficient: float = 0.1):
    return {
        **explicit_switch_regularizer_spec(
            coefficient,
            mode="greedy_logit_margin",
            margin=0.01,
        ),
        "schema_version": 2,
        "mode": "isolated_greedy_logit_margin",
        "relief_weighting": "actual_policy_logits_no_extra_weighting",
        "logit_source": "actual_policy_logits",
        "gradient_scope": "cached_route_residual_only",
    }


def urgency_linear_switch_regularizer_spec(coefficient: float = 0.1):
    return {
        **isolated_switch_regularizer_spec(coefficient),
        "schema_version": 3,
        "reduction": (
            "rollout_micro_mean_over_eligible_pre_contention_decisions"
        ),
        "gradient_scope": "cached_route_residual_endpoints_only",
        "eligibility_stage": "pre_contention",
        "minibatch_weighting": (
            "eligible_sum_scaled_to_rollout_micro_mean"
        ),
    }


def residual_checkpoint_for(actor: Actor, coefficient: float = 0.1):
    checkpoint = adaptive_checkpoint_for(actor)
    checkpoint["candidate_actor_spec"] = residual_actor_spec(coefficient)
    checkpoint["args"].update(
        {
            "avoidable_switch_probability_coef": coefficient,
            "avoidable_switch_regularization_mode": (
                "isolated_greedy_logit_margin"
            ),
            "avoidable_switch_logit_margin": 0.01,
            "route_hysteresis_residual_init": 0.0,
            "route_hysteresis_residual_cap": 0.2,
        }
    )
    checkpoint["switch_regularizer_spec"] = isolated_switch_regularizer_spec(
        coefficient
    )
    residual = checkpoint["actor"][
        "shared_candidate_actor.route_hysteresis_residual_bias"
    ]
    checkpoint["candidate_actor_learned_state"] = {
        "route_hysteresis_residual_bias": float(residual.item())
    }
    return checkpoint


def urgency_residual_checkpoint_for(actor: Actor, coefficient: float = 0.1):
    checkpoint = adaptive_checkpoint_for(actor)
    checkpoint["candidate_actor_spec"] = urgency_residual_actor_spec(coefficient)
    checkpoint["args"].update(
        {
            "avoidable_switch_probability_coef": coefficient,
            "avoidable_switch_regularization_mode": (
                "isolated_greedy_logit_margin"
            ),
            "avoidable_switch_logit_margin": 0.01,
            "avoidable_switch_reduction": "rollout_micro_mean",
            "route_hysteresis_residual_init": 0.0,
            "route_hysteresis_residual_cap": 0.2,
            "route_hysteresis_residual_parameterization": "urgency_linear",
        }
    )
    checkpoint["switch_regularizer_spec"] = (
        urgency_linear_switch_regularizer_spec(coefficient)
    )
    checkpoint["candidate_actor_learned_state"] = {
        parameter_name: float(
            checkpoint["actor"][
                f"shared_candidate_actor.{parameter_name}"
            ].item()
        )
        for parameter_name in (
            "route_hysteresis_residual_calm_bias",
            "route_hysteresis_residual_urgent_bias",
        )
    }
    return checkpoint


def synthetic_schema_wrapper():
    schema = candidate_feature_schema()
    return SimpleNamespace(
        n_agents=3,
        variant="proposed",
        get_candidate_feature_dim=lambda: FEATURE_DIM,
        get_action_size=lambda: ACTION_SIZE,
        get_obs_size=lambda: ACTION_SIZE * FEATURE_DIM,
        get_route_switch_feature_index=lambda: ROUTE_SWITCH_FEATURE_INDEX,
        get_route_urgency_feature_index=lambda: ROUTE_URGENCY_FEATURE_INDEX,
        get_route_class_2_feature_index=lambda: ROUTE_CLASS_2_FEATURE_INDEX,
        get_candidate_feature_schema=lambda: schema,
    )


class IntegratedRouteHysteresisMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8103)
        self.candidates = torch.randn(3, ACTION_SIZE, FEATURE_DIM)

    def test_beta_zero_is_state_dict_logits_softmax_and_greedy_compatible(self):
        legacy = candidate_actor()
        configured_zero = candidate_actor(0.0, ROUTE_SWITCH_FEATURE_INDEX)

        self.assertEqual(set(legacy.state_dict()), set(configured_zero.state_dict()))
        configured_zero.load_state_dict(legacy.state_dict(), strict=True)
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, configured_zero.state_dict()[key]))

        mask = torch.tensor(
            [[True, True, False, True], [True, False, True, True], [True] * 4]
        )
        for action_mask in (None, mask):
            legacy_logits = legacy(self.candidates, action_mask)
            zero_logits = configured_zero(self.candidates, action_mask)
            self.assertTrue(torch.equal(legacy_logits, zero_logits))
            self.assertTrue(
                torch.equal(
                    legacy_logits.softmax(dim=-1), zero_logits.softmax(dim=-1)
                )
            )
            self.assertTrue(
                torch.equal(legacy_logits.argmax(dim=-1), zero_logits.argmax(dim=-1))
            )

    def test_bias_is_exactly_beta_only_for_cached_candidates(self):
        baseline = candidate_actor()
        integrated = candidate_actor(BETA, ROUTE_SWITCH_FEATURE_INDEX)
        integrated.load_state_dict(baseline.state_dict())
        switch_values = torch.tensor(
            [[0.0, 1.0, 0.25, 0.75], [1.0, 0.0, 0.5, 0.49], [0.0] * 4]
        )
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = switch_values

        shift = integrated(self.candidates) - baseline(self.candidates)
        expected = BETA * (switch_values < 0.5).to(shift.dtype)
        self.assertTrue(torch.allclose(shift, expected, atol=1e-6, rtol=0.0))

    def test_no_cached_route_leaves_probability_distribution_unchanged(self):
        baseline = candidate_actor()
        integrated = candidate_actor(BETA, ROUTE_SWITCH_FEATURE_INDEX)
        integrated.load_state_dict(baseline.state_dict())
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = 0.0

        baseline_probs = baseline(self.candidates).softmax(dim=-1)
        integrated_probs = integrated(self.candidates).softmax(dim=-1)
        self.assertTrue(
            torch.allclose(baseline_probs, integrated_probs, atol=1e-7, rtol=0.0)
        )

    def test_route_switch_index_requires_an_exact_integer(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            candidate_actor(BETA, float(ROUTE_SWITCH_FEATURE_INDEX))

    def test_masked_cached_candidate_is_not_revived(self):
        baseline = candidate_actor()
        integrated = candidate_actor(BETA, ROUTE_SWITCH_FEATURE_INDEX)
        integrated.load_state_dict(baseline.state_dict())
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = 1.0
        self.candidates[:, 1, ROUTE_SWITCH_FEATURE_INDEX] = 0.0
        mask = torch.tensor([[True, False, True, True]] * 3)

        logits = integrated(self.candidates, mask)
        self.assertTrue(torch.equal(logits[:, 1], torch.full((3,), -1e9)))
        self.assertTrue((logits.argmax(dim=-1) != 1).all())

    def test_integrated_logits_remain_permutation_equivariant(self):
        actor = candidate_actor(BETA, ROUTE_SWITCH_FEATURE_INDEX)
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * 3
        )
        mask = torch.tensor(
            [[True, True, False, True], [True, False, True, True], [True] * 4]
        )
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)

        original = actor(self.candidates, mask)
        permuted = actor(
            self.candidates[:, permutation], mask[:, permutation]
        )
        self.assertTrue(torch.allclose(original, permuted[:, inverse], atol=1e-6))

    def test_rollout_logprob_equals_ppo_recomputation(self):
        actor = trainer_actor()
        observations = self.candidates.reshape(3, -1)
        mask = torch.tensor(
            [[True, True, False, True], [True, False, True, True], [True] * 4]
        )

        actions, rollout_logprob = actor.act(observations, mask)
        ppo_logprob = Categorical(logits=actor.logits(observations, mask)).log_prob(
            actions
        )
        self.assertTrue(torch.equal(rollout_logprob, ppo_logprob))

    def test_adaptive_scorer_cannot_learn_from_the_switch_feature(self):
        actor = adaptive_candidate_actor(beta=0.0)
        changed = self.candidates.clone()
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = 0.0
        changed[..., ROUTE_SWITCH_FEATURE_INDEX] = 1.0

        self.assertTrue(torch.equal(actor(self.candidates), actor(changed)))

    def test_adaptive_bonus_is_reduced_by_packet_urgency(self):
        baseline = adaptive_candidate_actor(beta=0.0, urgency_relief=0.5)
        integrated = adaptive_candidate_actor(beta=BETA, urgency_relief=0.5)
        integrated.load_state_dict(baseline.state_dict())
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0]] * 3
        )
        self.candidates[..., ROUTE_URGENCY_FEATURE_INDEX] = 0.8
        self.candidates[..., ROUTE_CLASS_2_FEATURE_INDEX] = 0.0

        shift = integrated(self.candidates) - baseline(self.candidates)
        cached = (
            self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] < 0.5
        ).to(shift.dtype)
        expected = BETA * (1.0 - 0.5 * 0.8) * cached
        self.assertTrue(torch.allclose(shift, expected, atol=1e-6, rtol=0.0))

    def test_adaptive_bonus_applies_class_2_relief_multiplicatively(self):
        baseline = adaptive_candidate_actor(
            beta=0.0,
            urgency_relief=0.0,
            class_2_relief=0.25,
        )
        integrated = adaptive_candidate_actor(
            beta=BETA,
            urgency_relief=0.0,
            class_2_relief=0.25,
        )
        integrated.load_state_dict(baseline.state_dict())
        self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0]] * 3
        )
        self.candidates[..., ROUTE_URGENCY_FEATURE_INDEX] = 0.0
        self.candidates[..., ROUTE_CLASS_2_FEATURE_INDEX] = 1.0

        shift = integrated(self.candidates) - baseline(self.candidates)
        cached = (
            self.candidates[..., ROUTE_SWITCH_FEATURE_INDEX] < 0.5
        ).to(shift.dtype)
        expected = BETA * (1.0 - 0.25) * cached
        self.assertTrue(torch.allclose(shift, expected, atol=1e-6, rtol=0.0))

    @staticmethod
    def _urgency_linear_actor() -> SharedCandidateActor:
        actor = SharedCandidateActor(
            candidate_feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=1,
            route_switch_feature_index=ROUTE_SWITCH_FEATURE_INDEX,
            route_hysteresis_beta=BETA,
            route_hysteresis_mode="decoupled_adaptive",
            route_urgency_feature_index=ROUTE_URGENCY_FEATURE_INDEX,
            route_class_2_feature_index=ROUTE_CLASS_2_FEATURE_INDEX,
            route_hysteresis_residual_init=0.0,
            route_hysteresis_residual_cap=0.2,
            route_hysteresis_residual_parameterization="urgency_linear",
        )
        with torch.no_grad():
            actor.route_hysteresis_residual_calm_bias.fill_(0.04)
            actor.route_hysteresis_residual_urgent_bias.fill_(0.16)
        return actor

    @staticmethod
    def _urgency_probe() -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.zeros(3, ACTION_SIZE, FEATURE_DIM)
        candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[1.0, 0.0, 1.0, 1.0]] * 3
        )
        candidates[..., ROUTE_URGENCY_FEATURE_INDEX] = torch.tensor(
            [0.0, 0.5, 1.0]
        ).reshape(3, 1)
        mask = torch.tensor([[False, True, True, False]] * 3)
        return candidates, mask

    def test_urgency_linear_residual_interpolates_and_matches_isolated_logits(self):
        actor = self._urgency_linear_actor()
        with torch.no_grad():
            for parameter in actor.encoder.parameters():
                parameter.zero_()
            for parameter in actor.scorer.parameters():
                parameter.zero_()
        candidates, mask = self._urgency_probe()

        deployed = actor(candidates, mask)
        isolated = actor.isolated_regularizer_logits(candidates, mask)

        self.assertTrue(torch.equal(deployed, isolated))
        self.assertTrue(
            torch.allclose(
                deployed[:, 1] - deployed[:, 2],
                torch.tensor([0.24, 0.30, 0.36]),
            )
        )

    def test_urgency_linear_ppo_gradient_excludes_residual_endpoints(self):
        actor = self._urgency_linear_actor()
        candidates, mask = self._urgency_probe()

        actor(candidates, mask)[:, 2].sum().backward()

        self.assertIsNone(actor.route_hysteresis_residual_calm_bias.grad)
        self.assertIsNone(actor.route_hysteresis_residual_urgent_bias.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0.0))
                for parameter in actor.scorer.parameters()
            )
        )

    def test_urgency_linear_regularizer_gradient_updates_only_endpoints(self):
        actor = self._urgency_linear_actor()
        candidates, mask = self._urgency_probe()

        actor.isolated_regularizer_logits(candidates, mask)[:, 1].sum().backward()

        quality_parameters = tuple(actor.encoder.parameters()) + tuple(
            actor.scorer.parameters()
        )
        self.assertTrue(all(parameter.grad is None for parameter in quality_parameters))
        self.assertGreater(
            float(actor.route_hysteresis_residual_calm_bias.grad), 0.0
        )
        self.assertGreater(
            float(actor.route_hysteresis_residual_urgent_bias.grad), 0.0
        )

    def test_residual_dtype_cap_is_strict_at_float32_upper_boundary(self):
        dtype_cap = route_hysteresis_residual_dtype_cap(0.2, dtype=torch.float32)
        next_upper = float(
            torch.nextafter(
                torch.tensor(dtype_cap, dtype=torch.float32),
                torch.tensor(torch.inf, dtype=torch.float32),
            ).item()
        )

        self.assertLessEqual(dtype_cap, 0.2)
        self.assertGreater(next_upper, 0.2)
        actor = self._urgency_linear_actor()
        with torch.no_grad():
            actor.route_hysteresis_residual_calm_bias.fill_(next_upper)
            actor.route_hysteresis_residual_urgent_bias.fill_(next_upper)
        actor.project_route_hysteresis_residual_()
        self.assertEqual(
            float(actor.route_hysteresis_residual_calm_bias.detach()), dtype_cap
        )
        self.assertEqual(
            float(actor.route_hysteresis_residual_urgent_bias.detach()), dtype_cap
        )


class IntegratedRouteHysteresisCheckpointTests(unittest.TestCase):
    def _save(self, directory: str, checkpoint: dict, name: str = "policy.pt") -> Path:
        path = Path(directory) / name
        torch.save(checkpoint, path)
        return path

    def test_validation_candidate_promotion_is_atomic_and_step_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.pt"
            destination = Path(directory) / "validation_best.pt"
            torch.save({"step": 200, "payload": torch.arange(3)}, source)

            digest = promote_validation_checkpoint(source, destination, 200)

            self.assertEqual(
                digest,
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(torch.load(destination, weights_only=False)["step"], 200)
            with self.assertRaisesRegex(ValueError, "step mismatch"):
                promote_validation_checkpoint(source, destination, 201)

    def test_integrated_checkpoint_round_trip_preserves_policy_and_spec(self):
        torch.manual_seed(9004)
        actor = trainer_actor()
        observations = torch.randn(3, ACTION_SIZE * FEATURE_DIM)
        candidates = observations.reshape(3, ACTION_SIZE, FEATURE_DIM)
        candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * 3
        )
        observations = candidates.reshape(3, -1)
        mask = torch.tensor([[True, True, False, True]] * 3)
        expected = actor.greedy(observations, mask).numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint_for(actor))
            policy, loaded = load_checkpoint_policy(path)
            actual = policy(observations.numpy(), mask.numpy())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(loaded["candidate_actor_spec"], actor_spec())
        self.assertEqual(policy.checkpoint_schema["candidate_actor_spec"], actor_spec())

    def test_schema_v2_checkpoint_round_trip_preserves_adaptive_policy(self):
        torch.manual_seed(9006)
        spec = adaptive_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        candidates = torch.randn(3, ACTION_SIZE, FEATURE_DIM)
        candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * 3
        )
        observations = candidates.reshape(3, -1)
        mask = torch.tensor([[True, True, False, True]] * 3)
        expected = actor.greedy(observations, mask).numpy()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = adaptive_checkpoint_for(actor)
            policy, loaded = load_checkpoint_policy(
                self._save(directory, checkpoint, "adaptive.pt")
            )
            actual = policy(observations.numpy(), mask.numpy())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(loaded["candidate_actor_spec"], spec)
        self.assertEqual(policy.checkpoint_schema["candidate_actor_spec"], spec)
        self.assertNotIn("switch_regularizer_spec", policy.checkpoint_schema)
        validate_policy_environment_schema(policy, synthetic_schema_wrapper())

    def test_explicit_margin_regularizer_contract_round_trip(self):
        spec = regularized_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = margin_regularized_checkpoint_for(actor)
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(
                self._save(directory, checkpoint, "margin_regularized.pt")
            )
        self.assertEqual(
            policy.checkpoint_schema["switch_regularizer_spec"],
            explicit_switch_regularizer_spec(),
        )

    def test_schema_v4_residual_checkpoint_round_trip_preserves_policy(self):
        spec = residual_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        with torch.no_grad():
            actor.shared_candidate_actor.route_hysteresis_residual_bias.fill_(
                0.125
            )
        self.assertIsNone(
            actor.shared_candidate_actor.route_hysteresis_residual_calm_bias
        )
        self.assertIsNone(
            actor.shared_candidate_actor.route_hysteresis_residual_urgent_bias
        )
        candidates = torch.randn(3, ACTION_SIZE, FEATURE_DIM)
        candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * 3
        )
        observations = candidates.reshape(3, -1)
        mask = torch.tensor([[True, True, False, True]] * 3)
        expected = actor.greedy(observations, mask).numpy()
        checkpoint = residual_checkpoint_for(actor)

        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(
                self._save(directory, checkpoint, "residual.pt")
            )
            actual = policy(observations.numpy(), mask.numpy())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(policy.checkpoint_schema["candidate_actor_spec"], spec)
        self.assertNotIn(
            "candidate_actor_learned_state", policy.checkpoint_schema
        )
        self.assertEqual(
            policy.checkpoint_schema["switch_regularizer_spec"],
            isolated_switch_regularizer_spec(),
        )
        self.assertEqual(
            policy.candidate_actor_learned_state,
            {"route_hysteresis_residual_bias": 0.125},
        )

    def test_schema_v5_endpoint_checkpoint_round_trip_preserves_policy(self):
        spec = urgency_residual_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        with torch.no_grad():
            actor.shared_candidate_actor.route_hysteresis_residual_calm_bias.fill_(
                0.05
            )
            actor.shared_candidate_actor.route_hysteresis_residual_urgent_bias.fill_(
                0.15
            )
        candidates = torch.randn(3, ACTION_SIZE, FEATURE_DIM)
        candidates[..., ROUTE_SWITCH_FEATURE_INDEX] = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0]] * 3
        )
        candidates[..., ROUTE_URGENCY_FEATURE_INDEX] = torch.tensor(
            [0.0, 0.5, 1.0]
        ).reshape(3, 1)
        observations = candidates.reshape(3, -1)
        mask = torch.tensor([[True, True, False, True]] * 3)
        expected = actor.greedy(observations, mask).numpy()
        checkpoint = urgency_residual_checkpoint_for(actor)

        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(
                self._save(directory, checkpoint, "endpoint_residual.pt")
            )
            actual = policy(observations.numpy(), mask.numpy())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(policy.checkpoint_schema["candidate_actor_spec"], spec)
        self.assertEqual(
            policy.checkpoint_schema["switch_regularizer_spec"],
            urgency_linear_switch_regularizer_spec(),
        )
        self.assertEqual(
            policy.candidate_actor_learned_state,
            checkpoint["candidate_actor_learned_state"],
        )
        validate_policy_environment_schema(policy, synthetic_schema_wrapper())

    def test_schema_v5_loader_accepts_dtype_cap_and_rejects_next_upper(self):
        spec = urgency_residual_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        dtype_cap = route_hysteresis_residual_dtype_cap(0.2, dtype=torch.float32)
        next_upper = float(
            torch.nextafter(
                torch.tensor(dtype_cap, dtype=torch.float32),
                torch.tensor(torch.inf, dtype=torch.float32),
            ).item()
        )
        with torch.no_grad():
            actor.shared_candidate_actor.route_hysteresis_residual_calm_bias.fill_(
                dtype_cap
            )
            actor.shared_candidate_actor.route_hysteresis_residual_urgent_bias.fill_(
                dtype_cap
            )
        checkpoint = urgency_residual_checkpoint_for(actor)
        with tempfile.TemporaryDirectory() as directory:
            load_checkpoint_policy(
                self._save(directory, checkpoint, "dtype_cap.pt")
            )

        upper_checkpoint = copy.deepcopy(checkpoint)
        upper_name = "route_hysteresis_residual_urgent_bias"
        upper_checkpoint["candidate_actor_learned_state"][upper_name] = next_upper
        upper_checkpoint["actor"][
            f"shared_candidate_actor.{upper_name}"
        ].fill_(next_upper)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exceeds its cap"):
                load_checkpoint_policy(
                    self._save(directory, upper_checkpoint, "next_upper.pt")
                )

    def test_schema_v5_requires_both_learned_endpoint_values(self):
        spec = urgency_residual_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = urgency_residual_checkpoint_for(actor)
        checkpoint["candidate_actor_learned_state"].pop(
            "route_hysteresis_residual_urgent_bias"
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "learned actor state mismatch"):
                load_checkpoint_policy(
                    self._save(directory, checkpoint, "missing_endpoint.pt")
                )

    def test_schema_v5_reduction_contract_must_match_actor_args(self):
        spec = urgency_residual_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = urgency_residual_checkpoint_for(actor)
        checkpoint["args"]["avoidable_switch_reduction"] = (
            "minibatch_conditional_mean"
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "disagrees with actor args"):
                load_checkpoint_policy(
                    self._save(directory, checkpoint, "reduction_drift.pt")
                )

    def test_schema_v4_residual_state_drift_is_rejected(self):
        spec = residual_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        corruptions = (
            lambda checkpoint: checkpoint.pop(
                "candidate_actor_learned_state"
            ),
            lambda checkpoint: checkpoint[
                "candidate_actor_learned_state"
            ].update(route_hysteresis_residual_bias=0.1),
            lambda checkpoint: checkpoint["actor"][
                "shared_candidate_actor.route_hysteresis_residual_bias"
            ].fill_(0.3),
        )
        for index, corrupt in enumerate(corruptions):
            with self.subTest(index=index):
                checkpoint = residual_checkpoint_for(actor)
                corrupt(checkpoint)
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        load_checkpoint_policy(
                            self._save(
                                directory,
                                checkpoint,
                                f"residual_drift_{index}.pt",
                            )
                        )

    def test_new_regularized_checkpoint_cannot_omit_explicit_spec(self):
        spec = regularized_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = margin_regularized_checkpoint_for(actor)
        checkpoint.pop("switch_regularizer_spec")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing switch_regularizer_spec"):
                load_checkpoint_policy(
                    self._save(directory, checkpoint, "missing_switch_spec.pt")
                )

    def test_margin_regularizer_contract_drift_is_rejected(self):
        spec = regularized_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        corruptions = {
            "coefficient": 0.2,
            "logit_margin": 0.0,
            "reduction": "mean_over_all_decisions",
            "no_op_action_index": 1,
            "route_switch_feature_index": ROUTE_SWITCH_FEATURE_INDEX - 1,
            "route_hysteresis_beta": 0.3,
            "route_hysteresis_urgency_relief": 0.4,
            "relief_weighting": "none",
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                checkpoint = margin_regularized_checkpoint_for(actor)
                checkpoint["switch_regularizer_spec"][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        load_checkpoint_policy(
                            self._save(
                                directory,
                                checkpoint,
                                f"drift_{field}.pt",
                            )
                        )

    def test_schema_v2_environment_rejects_feature_schema_hash_drift(self):
        spec = adaptive_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = adaptive_checkpoint_for(actor)
        checkpoint["candidate_actor_spec"][
            "candidate_feature_schema_sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(
                self._save(directory, checkpoint, "schema_drift.pt")
            )
        with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
            validate_policy_environment_schema(policy, synthetic_schema_wrapper())

    def test_schema_v3_checkpoint_round_trip_preserves_regularization_contract(self):
        spec = regularized_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        candidates = torch.randn(3, ACTION_SIZE, FEATURE_DIM)
        observations = candidates.reshape(3, -1)
        mask = torch.tensor([[True, True, False, True]] * 3)
        expected = actor.greedy(observations, mask).numpy()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = regularized_checkpoint_for(actor)
            policy, loaded = load_checkpoint_policy(
                self._save(directory, checkpoint, "regularized.pt")
            )
            actual = policy(observations.numpy(), mask.numpy())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(loaded["candidate_actor_spec"], spec)
        self.assertEqual(policy.checkpoint_schema["candidate_actor_spec"], spec)
        self.assertNotIn("switch_regularizer_spec", policy.checkpoint_schema)
        migrated = policy.switch_regularizer_spec
        self.assertEqual(migrated["schema_version"], 0)
        self.assertEqual(migrated["mode"], "conditional_probability")
        self.assertEqual(migrated["relief_weighting"], "none")
        self.assertEqual(
            migrated["migration"],
            "legacy_schema_v3_probability_without_explicit_spec",
        )
        validate_policy_environment_schema(policy, synthetic_schema_wrapper())

    def test_schema_v3_regularization_disagreement_is_rejected(self):
        spec = regularized_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = regularized_checkpoint_for(actor)
        checkpoint["args"]["avoidable_switch_probability_coef"] = 0.2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "regularization disagrees"):
                load_checkpoint_policy(
                    self._save(directory, checkpoint, "regularizer_drift.pt")
                )

    def test_schema_v2_rejects_hidden_switch_regularization(self):
        spec = adaptive_actor_spec()
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = adaptive_checkpoint_for(actor)
        checkpoint["args"]["avoidable_switch_probability_coef"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "schema-v2.*regularization"):
                load_checkpoint_policy(
                    self._save(directory, checkpoint, "hidden_regularizer.pt")
                )

    def test_legacy_checkpoint_without_spec_falls_back_only_to_beta_zero(self):
        torch.manual_seed(9005)
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
        )
        checkpoint = checkpoint_for(actor, beta=0.0, spec_marker=None)
        checkpoint["args"].pop("route_hysteresis_beta")
        observations = torch.randn(3, ACTION_SIZE * FEATURE_DIM)
        mask = torch.tensor([[True, True, False, True]] * 3)
        expected = actor.greedy(observations, mask).numpy()

        with tempfile.TemporaryDirectory() as directory:
            policy, _ = load_checkpoint_policy(self._save(directory, checkpoint))
            actual = policy(observations.numpy(), mask.numpy())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            policy.checkpoint_schema["candidate_actor_spec"],
            {
                "schema_version": 0,
                "type": "shared_candidate_actor",
                "route_switch_feature_index": None,
                "route_hysteresis_beta": 0.0,
            },
        )

    def test_checkpoint_without_variant_is_rejected(self):
        checkpoint = checkpoint_for(trainer_actor(), beta=0.0, spec_marker=None)
        checkpoint["args"].pop("leo_variant")
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "omit leo_variant"):
                load_checkpoint_policy(path)

    def test_positive_beta_checkpoint_without_spec_is_rejected(self):
        checkpoint = checkpoint_for(trainer_actor(), spec_marker=None)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "positive-beta.*no candidate actor spec"):
                load_checkpoint_policy(path)

    def test_checkpoint_beta_disagreement_is_rejected(self):
        checkpoint = checkpoint_for(trainer_actor(), beta=BETA)
        checkpoint["args"]["route_hysteresis_beta"] = 0.10
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "beta disagrees"):
                load_checkpoint_policy(path)

    def test_schema_v1_rejects_actor_args_that_claim_adaptive_mode(self):
        checkpoint = checkpoint_for(trainer_actor())
        checkpoint["args"]["route_hysteresis_mode"] = "decoupled_adaptive"
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "legacy.*disagrees"):
                load_checkpoint_policy(path)

    def test_checkpoint_beta_type_is_not_coerced_from_text(self):
        checkpoint = checkpoint_for(trainer_actor())
        checkpoint["args"]["route_hysteresis_beta"] = "0.20"
        checkpoint["candidate_actor_spec"]["route_hysteresis_beta"] = "0.20"
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "finite non-negative number"):
                load_checkpoint_policy(path)

    def test_post_hoc_loader_rejects_an_integrated_checkpoint(self):
        checkpoint = checkpoint_for(trainer_actor())
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "integrated beta"):
                load_hysteresis_policy(path, stay_bonus=BETA)

    def test_post_hoc_loader_rejects_schema_v2_even_when_beta_is_zero(self):
        spec = adaptive_actor_spec(beta=0.0)
        actor = Actor(
            input_dim=ACTION_SIZE * FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layer=1,
            output_dim=ACTION_SIZE,
            candidate_feature_dim=FEATURE_DIM,
            candidate_actor_spec=spec,
        )
        checkpoint = adaptive_checkpoint_for(actor, beta=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "adaptive checkpoint"):
                load_hysteresis_policy(path, stay_bonus=BETA)

    def test_bad_or_incomplete_candidate_actor_spec_is_rejected(self):
        malformed_specs = {
            "missing field": {
                key: value
                for key, value in actor_spec().items()
                if key != "route_hysteresis_beta"
            },
            "extra field": {**actor_spec(), "post_hoc": True},
            "wrong version": {**actor_spec(), "schema_version": 2},
            "wrong type": {**actor_spec(), "type": "flat_actor"},
        }
        with tempfile.TemporaryDirectory() as directory:
            for index, (label, spec) in enumerate(malformed_specs.items()):
                with self.subTest(label=label):
                    checkpoint = checkpoint_for(trainer_actor(), spec_marker=spec)
                    path = self._save(directory, checkpoint, f"bad_{index}.pt")
                    with self.assertRaises(ValueError):
                        load_checkpoint_policy(path)

    def test_wrong_route_switch_feature_contract_is_rejected(self):
        bad_spec = actor_spec(switch_index=ROUTE_SWITCH_FEATURE_INDEX + 1)
        checkpoint = checkpoint_for(trainer_actor(), spec_marker=bad_spec)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, checkpoint)
            with self.assertRaisesRegex(ValueError, "route-switch feature contract"):
                load_checkpoint_policy(path)

    def test_environment_route_switch_contract_is_validated(self):
        policy = lambda observation, mask: np.zeros(observation.shape[0], dtype=int)
        policy.checkpoint_schema = {
            "candidate_feature_dim": FEATURE_DIM,
            "action_size": ACTION_SIZE,
            "obs_size": FEATURE_DIM * ACTION_SIZE,
            "n_agents": 3,
            "variant": "with_congestion_context",
            "candidate_actor_spec": actor_spec(),
        }
        wrapper = SimpleNamespace(
            n_agents=3,
            variant="with_congestion_context",
            get_candidate_feature_dim=lambda: FEATURE_DIM,
            get_action_size=lambda: ACTION_SIZE,
            get_obs_size=lambda: FEATURE_DIM * ACTION_SIZE,
            get_route_switch_feature_index=lambda: ROUTE_SWITCH_FEATURE_INDEX + 1,
        )
        with self.assertRaisesRegex(ValueError, "route-switch schema mismatch"):
            validate_policy_environment_schema(policy, wrapper)

    def test_wrapper_exposes_frozen_route_switch_feature_index(self):
        wrapper = CleanMARLLeoMultiAgentWrapper(
            "medium_load", seed=8104, variant="with_congestion_context"
        )
        self.assertEqual(
            wrapper.get_route_switch_feature_index(), ROUTE_SWITCH_FEATURE_INDEX
        )
        self.assertEqual(ROUTE_SWITCH_FEATURE_INDEX, 17)
        self.assertLess(ROUTE_SWITCH_FEATURE_INDEX, wrapper.get_candidate_feature_dim())


if __name__ == "__main__":
    unittest.main()
