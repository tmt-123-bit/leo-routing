from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from adaptive_hysteresis_policy import AdaptiveActorScoreHysteresisPolicy
from calm_selective_hysteresis_policy import (
    AGE_BIN_EDGES,
    AGE_BIN_LABELS,
    ANCHOR_MARGIN_BIN_LABELS,
    ANCHOR_MARGIN_UPPER_EDGES,
    CalmSelectiveActorScoreHysteresisPolicy,
    load_calm_selective_hysteresis_policy,
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
MASK = np.asarray([[False, True, True]])


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
    urgency_relief: float = 0.50,
    class_2_relief: float = 0.75,
    calm_bonus: float = 0.20,
) -> CalmSelectiveActorScoreHysteresisPolicy:
    return CalmSelectiveActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=FEATURE_DIM,
        action_size=3,
        stay_bonus=bonus,
        urgency_relief=urgency_relief,
        class_2_relief=class_2_relief,
        calm_bonus=calm_bonus,
        device="cpu",
        checkpoint_schema=canonical_checkpoint_schema(),
    )


def observation(
    *,
    incumbent_score: float = 0.50,
    switch_score: float = 1.00,
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


class CalmSelectiveHysteresisPolicyTests(unittest.TestCase):
    def test_anchor_is_action_equivalent_to_adaptive_policy(self):
        for urgency_relief in (0.50, 0.625, 0.75):
            calm = policy(
                urgency_relief=urgency_relief, calm_bonus=0.0
            )
            adaptive = AdaptiveActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM,
                action_size=3,
                stay_bonus=0.40,
                urgency_relief=urgency_relief,
                class_2_relief=0.75,
                device="cpu",
                checkpoint_schema=canonical_checkpoint_schema(),
            )
            for urgency in (0.0, 0.25, 0.5, 1.0, 2.0):
                for class_2 in (0.0, 1.0):
                    for gap in (0.05, 0.20, 0.50, 1.00):
                        obs = observation(
                            switch_score=0.50 + gap,
                            urgency=urgency,
                            class_2=class_2,
                        )
                        np.testing.assert_array_equal(
                            calm(obs, MASK), adaptive(obs, MASK)
                        )

    def test_addon_suppresses_only_calm_non_class_two_switches(self):
        cases = (
            (0.0, 0.0, 1),
            (0.5, 0.0, 2),
            (0.0, 1.0, 2),
        )
        for urgency, class_2, expected in cases:
            with self.subTest(urgency=urgency, class_2=class_2):
                action = policy()(observation(
                    switch_score=1.00,
                    urgency=urgency,
                    class_2=class_2,
                ), MASK)
                np.testing.assert_array_equal(action, [expected])

    def test_calm_weight_fades_linearly_to_the_half_age_cutoff(self):
        calm = policy(bonus=0.0, urgency_relief=0.0, calm_bonus=0.40)
        calm(observation(switch_score=0.65, urgency=0.25), MASK)
        diagnostics = calm.diagnostics()
        self.assertAlmostEqual(diagnostics["effective_bonus_mean"], 0.20)
        self.assertAlmostEqual(diagnostics["calm_addon_mean"], 0.20)
        self.assertEqual(
            diagnostics["calm_non_class_2_suppressed_switch_choices"], 1
        )

    def test_raw_urgency_is_clipped_before_calm_weight(self):
        calm = policy(bonus=0.8, urgency_relief=0.5, calm_bonus=0.75)
        calm(observation(switch_score=0.95, urgency=2.0), MASK)
        diagnostics = calm.diagnostics()
        self.assertAlmostEqual(diagnostics["effective_bonus_mean"], 0.40)
        self.assertAlmostEqual(diagnostics["calm_addon_mean"], 0.0)

    def test_first_use_and_forced_switch_are_unchanged(self):
        calm = policy(bonus=1.0, calm_bonus=1.0)
        np.testing.assert_array_equal(
            calm(observation(first_use=True), MASK), [2]
        )
        unavailable = np.asarray([[False, False, True]])
        np.testing.assert_array_equal(calm(observation(), unavailable), [2])

    def test_diagnostics_reconcile_calm_outcomes(self):
        calm = policy()
        calm(observation(switch_score=1.00), MASK)
        calm(observation(switch_score=1.20), MASK)
        calm(observation(switch_score=1.00, urgency=0.5), MASK)
        diagnostics = calm.diagnostics()
        self.assertEqual(diagnostics["raw_switch_choices"], 3)
        self.assertEqual(
            diagnostics["calm_non_class_2_raw_switch_choices"], 2
        )
        self.assertEqual(
            diagnostics["calm_non_class_2_suppressed_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["calm_non_class_2_allowed_switch_choices"], 1
        )

    def test_zero_k_control_still_counts_calm_state_membership(self):
        calm = policy(calm_bonus=0.0)
        calm(observation(switch_score=1.00, urgency=0.25), MASK)
        diagnostics = calm.diagnostics()
        self.assertEqual(
            diagnostics["calm_non_class_2_raw_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["calm_non_class_2_allowed_switch_choices"], 1
        )
        self.assertAlmostEqual(diagnostics["calm_addon_mean"], 0.0)

    def test_age_and_relative_anchor_margin_bins_are_frozen(self):
        calm = policy(calm_bonus=0.0)
        cases = (
            (0.0, 0.55),
            (0.125, 0.525),
            (0.25, 0.65),
            (0.375, 0.775),
            (0.5, 0.90),
            (1.0, 1.25),
        )
        for urgency, switch_score in cases:
            calm(observation(
                switch_score=switch_score,
                urgency=urgency,
            ), MASK)
        diagnostics = calm.diagnostics()
        self.assertEqual(diagnostics["age_bin_edges"], list(AGE_BIN_EDGES))
        self.assertEqual(
            set(diagnostics["age_choice_bins"]), set(AGE_BIN_LABELS)
        )
        self.assertEqual(
            diagnostics["relative_anchor_margin_upper_edges"],
            list(ANCHOR_MARGIN_UPPER_EDGES),
        )
        self.assertEqual(
            set(diagnostics["relative_anchor_margin_choice_bins"]),
            set(ANCHOR_MARGIN_BIN_LABELS),
        )
        self.assertEqual(
            sum(
                sum(counts.values())
                for counts in diagnostics["age_choice_bins"].values()
            ),
            len(cases),
        )
        self.assertEqual(
            sum(
                sum(counts.values())
                for counts in diagnostics[
                    "relative_anchor_margin_choice_bins"
                ].values()
            ),
            len(cases),
        )

    def test_invalid_calm_bonus_fails_closed(self):
        for invalid in (-0.01, 1.01, float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "calm bonus"):
                    policy(calm_bonus=invalid)

    def test_checkpoint_loader_preserves_frozen_schema(self):
        torch.manual_seed(9)
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
            loaded, payload = load_calm_selective_hysteresis_policy(
                path,
                stay_bonus=1.0,
                urgency_relief=0.625,
                class_2_relief=0.75,
                calm_bonus=0.50,
            )
        self.assertEqual(payload["args"], checkpoint["args"])
        self.assertEqual(loaded.feature_dim, 26)
        self.assertEqual(loaded.stay_bonus, 1.0)
        self.assertEqual(loaded.urgency_relief, 0.625)
        self.assertEqual(loaded.class_2_relief, 0.75)
        self.assertEqual(loaded.calm_bonus, 0.50)


if __name__ == "__main__":
    unittest.main()
