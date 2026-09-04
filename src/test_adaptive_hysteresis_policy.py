from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_hysteresis_policy import (
    AdaptiveActorScoreHysteresisPolicy,
    load_adaptive_hysteresis_policy,
)
from leo_multiagent_env import BASE_CANDIDATE_FEATURE_NAMES, candidate_feature_schema
from mappo_design import SharedCandidateActor


class FeatureScoreActor(torch.nn.Module):
    def forward(self, candidates: torch.Tensor, action_mask=None):
        logits = candidates[..., 0]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


FEATURE_DIM = len(BASE_CANDIDATE_FEATURE_NAMES)


def canonical_checkpoint_schema(*, action_size: int = 3) -> dict:
    feature_schema = candidate_feature_schema()
    return {
        "candidate_feature_dim": FEATURE_DIM,
        "action_size": action_size,
        "obs_size": action_size * FEATURE_DIM,
        "n_agents": 1,
        "variant": "proposed",
        "candidate_feature_schema_id": feature_schema["schema_id"],
        "candidate_feature_schema_sha256": feature_schema["sha256"],
        "candidate_feature_names": list(feature_schema["feature_names"]),
    }


def policy(
    *,
    bonus: float = 0.40,
    urgency_relief: float = 0.0,
    class_2_relief: float = 0.0,
) -> AdaptiveActorScoreHysteresisPolicy:
    return AdaptiveActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=FEATURE_DIM,
        action_size=3,
        stay_bonus=bonus,
        urgency_relief=urgency_relief,
        class_2_relief=class_2_relief,
        device="cpu",
        checkpoint_schema=canonical_checkpoint_schema(),
    )


def observation(
    *,
    incumbent_score: float = 0.50,
    switch_score: float = 0.80,
    urgency: float = 0.0,
    class_2: float = 0.0,
    first_use: bool = False,
) -> np.ndarray:
    candidates = np.zeros((1, 3, FEATURE_DIM), dtype=np.float32)
    candidates[0, 1, 0] = incumbent_score
    candidates[0, 1, 17] = 0.0
    candidates[0, 2, 0] = switch_score
    candidates[0, 2, 17] = 0.0 if first_use else 1.0
    candidates[0, 1:, 20] = urgency
    candidates[0, 1:, 23] = class_2
    return candidates.reshape(1, -1)


MASK = np.asarray([[False, True, True]])


class AdaptiveHysteresisPolicyTests(unittest.TestCase):
    def test_calm_class_zero_uses_the_full_base_bonus(self):
        action = policy()(observation(), MASK)
        np.testing.assert_array_equal(action, [1])

    def test_class_two_full_relief_exactly_releases_the_raw_switch(self):
        action = policy(class_2_relief=1.0)(
            observation(class_2=1.0), MASK
        )
        np.testing.assert_array_equal(action, [2])

    def test_urgency_relief_reduces_the_effective_threshold(self):
        adaptive = policy(urgency_relief=0.50)
        np.testing.assert_array_equal(
            adaptive(observation(urgency=0.0), MASK), [1]
        )
        np.testing.assert_array_equal(
            adaptive(observation(urgency=1.0), MASK), [2]
        )

    def test_urgency_is_clipped_before_relief(self):
        adaptive = policy(bonus=0.80, urgency_relief=0.50)
        obs = observation(switch_score=0.95, urgency=2.0)
        np.testing.assert_array_equal(adaptive(obs, MASK), [2])
        diagnostics = adaptive.diagnostics()
        self.assertAlmostEqual(diagnostics["effective_bonus_min"], 0.40)

    def test_first_use_and_forced_switch_are_never_suppressed(self):
        adaptive = policy(bonus=100.0)
        np.testing.assert_array_equal(
            adaptive(observation(first_use=True), MASK), [2]
        )
        unavailable_mask = np.asarray([[False, False, True]])
        np.testing.assert_array_equal(
            adaptive(observation(incumbent_score=10.0), unavailable_mask), [2]
        )

    def test_diagnostics_record_effective_bonus_and_state(self):
        adaptive = policy(urgency_relief=0.50, class_2_relief=1.0)
        adaptive(observation(urgency=0.5, class_2=1.0), MASK)
        diagnostics = adaptive.diagnostics()
        self.assertEqual(diagnostics["raw_switch_choices"], 1)
        self.assertEqual(diagnostics["class_2_raw_switch_choices"], 1)
        self.assertEqual(diagnostics["urgent_raw_switch_choices"], 1)
        self.assertEqual(diagnostics["zero_bonus_raw_switch_choices"], 1)
        self.assertEqual(diagnostics["effective_bonus_mean"], 0.0)

    def test_invalid_relief_and_feature_contracts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "urgency relief"):
            policy(urgency_relief=1.01)
        with self.assertRaisesRegex(ValueError, "class-2 relief"):
            policy(class_2_relief=float("nan"))
        with self.assertRaisesRegex(ValueError, "canonical"):
            AdaptiveActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM,
                action_size=3,
                stay_bonus=0.4,
                urgency_relief=0.5,
                class_2_relief=1.0,
                urgency_feature_index=17,
                device="cpu",
                checkpoint_schema=canonical_checkpoint_schema(),
            )

    def test_checkpoint_schema_contract_rejects_dimension_and_semantic_drift(self):
        bad_hash = canonical_checkpoint_schema()
        bad_hash["candidate_feature_schema_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "schema hash"):
            AdaptiveActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM,
                action_size=3,
                stay_bonus=0.4,
                urgency_relief=0.5,
                class_2_relief=1.0,
                device="cpu",
                checkpoint_schema=bad_hash,
            )

        wrong_variant = canonical_checkpoint_schema()
        wrong_variant["variant"] = "with_congestion_context"
        with self.assertRaisesRegex(ValueError, "proposed checkpoint"):
            AdaptiveActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM,
                action_size=3,
                stay_bonus=0.4,
                urgency_relief=0.5,
                class_2_relief=1.0,
                device="cpu",
                checkpoint_schema=wrong_variant,
            )

        wrong_dimension = canonical_checkpoint_schema()
        wrong_dimension["candidate_feature_dim"] = FEATURE_DIM - 1
        wrong_dimension["obs_size"] = 3 * (FEATURE_DIM - 1)
        with self.assertRaisesRegex(ValueError, "26-dimensional"):
            AdaptiveActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM - 1,
                action_size=3,
                stay_bonus=0.4,
                urgency_relief=0.5,
                class_2_relief=1.0,
                device="cpu",
                checkpoint_schema=wrong_dimension,
            )

    def test_packet_state_features_must_be_consistent_and_binary(self):
        adaptive = policy(urgency_relief=0.5, class_2_relief=1.0)

        inconsistent_urgency = observation(urgency=0.25).reshape(
            1, 3, FEATURE_DIM
        )
        inconsistent_urgency[0, 2, 20] = 0.50
        with self.assertRaisesRegex(ValueError, "urgency.*inconsistent"):
            adaptive(inconsistent_urgency.reshape(1, -1), MASK)

        inconsistent_class = observation(class_2=0.0).reshape(
            1, 3, FEATURE_DIM
        )
        inconsistent_class[0, 2, 23] = 1.0
        with self.assertRaisesRegex(ValueError, "class-2.*inconsistent"):
            adaptive(inconsistent_class.reshape(1, -1), MASK)

        nonbinary_class = observation(class_2=0.25)
        with self.assertRaisesRegex(ValueError, "class-2.*binary"):
            adaptive(nonbinary_class, MASK)

        nonbinary_switch = observation().reshape(1, 3, FEATURE_DIM)
        nonbinary_switch[0, 2, 17] = 0.25
        with self.assertRaisesRegex(ValueError, "route-switch.*binary"):
            adaptive(nonbinary_switch.reshape(1, -1), MASK)

        out_of_range_urgency = observation(urgency=2.01)
        with self.assertRaisesRegex(ValueError, "schema range"):
            adaptive(out_of_range_urgency, MASK)

    def test_suppressed_and_allowed_diagnostics_reconcile_by_state(self):
        adaptive = policy(urgency_relief=0.5, class_2_relief=0.5)
        cases = (
            (0.5, 1.0, 0.30),
            (0.0, 1.0, 0.10),
            (0.5, 0.0, 0.10),
            (0.0, 0.0, 0.50),
        )
        for urgency, class_2, gap in cases:
            adaptive(
                observation(
                    switch_score=0.50 + gap,
                    urgency=urgency,
                    class_2=class_2,
                ),
                MASK,
            )

        diagnostics = adaptive.diagnostics()
        self.assertEqual(diagnostics["raw_switch_choices"], 4)
        self.assertEqual(diagnostics["suppressed_switch_choices"], 2)
        self.assertEqual(diagnostics["allowed_switch_choices"], 2)
        for stratum in ("class_2", "non_class_2", "urgent", "non_urgent"):
            self.assertEqual(
                diagnostics[f"{stratum}_suppressed_switch_choices"], 1
            )
            self.assertEqual(
                diagnostics[f"{stratum}_allowed_switch_choices"], 1
            )
        self.assertEqual(
            diagnostics["class_2_urgent_allowed_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["class_2_non_urgent_suppressed_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["non_class_2_urgent_suppressed_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["non_class_2_non_urgent_allowed_switch_choices"], 1
        )

    def test_checkpoint_loader_preserves_frozen_actor_schema(self):
        torch.manual_seed(7)
        actor = SharedCandidateActor(26, 16, 1)
        checkpoint = {
            "candidate_feature_dim": 26,
            "action_size": 7,
            "obs_size": 182,
            "n_agents": 24,
            "args": {
                "actor_hidden_dim": 16,
                "actor_num_layers": 1,
                "leo_variant": "proposed",
            },
            "actor": {
                f"shared_candidate_actor.{key}": value
                for key, value in actor.state_dict().items()
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded, payload = load_adaptive_hysteresis_policy(
                path,
                stay_bonus=0.8,
                urgency_relief=0.5,
                class_2_relief=1.0,
            )
        self.assertEqual(payload["candidate_feature_dim"], 26)
        self.assertEqual(payload["action_size"], 7)
        self.assertEqual(payload["args"], checkpoint["args"])
        self.assertEqual(set(payload["actor"]), set(checkpoint["actor"]))
        self.assertEqual(loaded.feature_dim, 26)
        self.assertEqual(loaded.stay_bonus, 0.8)
        self.assertEqual(loaded.urgency_feature_index, 20)
        self.assertEqual(loaded.class_2_feature_index, 23)

    def test_checkpoint_loader_requires_the_exact_legacy_actor_contract(self):
        torch.manual_seed(8)
        actor = SharedCandidateActor(26, 16, 1)
        checkpoint = {
            "candidate_feature_dim": 26,
            "action_size": 7,
            "obs_size": 182,
            "n_agents": 24,
            "args": {
                "actor_hidden_dim": 16,
                "actor_num_layers": 1,
                "leo_variant": "proposed",
            },
            "actor": {
                f"shared_candidate_actor.{key}": value
                for key, value in actor.state_dict().items()
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            actor_specs = [
                {
                    "schema_version": schema_version,
                    "route_hysteresis_beta": 0.0,
                }
                for schema_version in (2, 3, 4, 5, 6, "2", True)
            ]
            actor_specs.extend(
                (
                    {},
                    {
                        "schema_version": 1,
                        "route_hysteresis_beta": 0.0,
                        "unexpected_field": True,
                    },
                )
            )
            for actor_spec in actor_specs:
                checkpoint["candidate_actor_spec"] = actor_spec
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(
                    ValueError, "post-hoc (adaptive )?hysteresis"
                ):
                    load_adaptive_hysteresis_policy(
                        path,
                        stay_bonus=0.8,
                        urgency_relief=0.5,
                        class_2_relief=1.0,
                    )


if __name__ == "__main__":
    unittest.main()
