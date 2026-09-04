from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from age_band_hysteresis_policy import (
    AGE_BIN_EDGES,
    AGE_BIN_LABELS,
    AgeBandActorScoreHysteresisPolicy,
    load_age_band_hysteresis_policy,
)
from calm_selective_hysteresis_policy import (
    CalmSelectiveActorScoreHysteresisPolicy,
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
V9_CANDIDATE_5_PARAMETERS = {
    "stay_bonus": 1.0,
    "urgency_relief": 0.625,
    "class_2_relief": 0.75,
    "calm_bonus": 0.75,
}


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


def age_band_policy(
    age_band_mix: float = 1.0,
) -> AgeBandActorScoreHysteresisPolicy:
    return AgeBandActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=FEATURE_DIM,
        action_size=3,
        age_band_mix=age_band_mix,
        device="cpu",
        checkpoint_schema=canonical_checkpoint_schema(),
    )


def v9_candidate_5_policy() -> CalmSelectiveActorScoreHysteresisPolicy:
    return CalmSelectiveActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=FEATURE_DIM,
        action_size=3,
        device="cpu",
        checkpoint_schema=canonical_checkpoint_schema(),
        **V9_CANDIDATE_5_PARAMETERS,
    )


def observation(
    *,
    incumbent_score: float = 0.5,
    switch_gap: float = 2.0,
    urgency: float = 0.0,
    class_2: float = 0.0,
    first_use: bool = False,
) -> np.ndarray:
    candidates = np.zeros((1, 3, FEATURE_DIM), dtype=np.float32)
    candidates[0, 1, 0] = incumbent_score
    candidates[0, 1, 17] = 0.0
    candidates[0, 2, 0] = incumbent_score + switch_gap
    candidates[0, 2, 17] = 0.0 if first_use else 1.0
    candidates[0, 1:, 20] = urgency
    candidates[0, 1:, 23] = class_2
    return candidates.reshape(1, -1)


def legacy_checkpoint() -> dict:
    torch.manual_seed(11)
    actor = SharedCandidateActor(26, 16, 1)
    return {
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


class AgeBandHysteresisPolicyTests(unittest.TestCase):
    def test_mix_zero_is_action_equivalent_to_v9_candidate_five(self):
        age_band = age_band_policy(age_band_mix=0.0)
        v9 = v9_candidate_5_policy()
        for urgency in (0.0, 0.125, 0.2, 0.25, 0.375, 0.5, 1.0, 2.0):
            for class_2 in (0.0, 1.0):
                for gap in (0.05, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0):
                    with self.subTest(
                        urgency=urgency, class_2=class_2, gap=gap
                    ):
                        obs = observation(
                            switch_gap=gap,
                            urgency=urgency,
                            class_2=class_2,
                        )
                        np.testing.assert_array_equal(
                            age_band(obs, MASK), v9(obs, MASK)
                        )
        diagnostics = age_band.diagnostics()
        self.assertEqual(
            diagnostics["threshold_unchanged_raw_switch_choices"],
            diagnostics["raw_switch_choices"],
        )
        self.assertAlmostEqual(diagnostics["threshold_shift_sum"], 0.0)

    def test_mix_one_has_frozen_band_shape_and_boundary_semantics(self):
        expected = (
            (0.0, 0.0, 1.0),
            (0.125, 0.0, 0.921875),
            (0.1875, 0.5, 1.2578125),
            (0.25, 1.0, 1.59375),
            (0.375, 0.5, 1.140625),
            (0.5, 0.0, 0.6875),
            (1.0, 0.0, 0.375),
        )
        for urgency, band_weight, beta in expected:
            with self.subTest(urgency=urgency):
                policy = age_band_policy(age_band_mix=1.0)
                policy(observation(urgency=urgency), MASK)
                diagnostics = policy.diagnostics()
                self.assertAlmostEqual(
                    diagnostics["band_weight_mean"], band_weight
                )
                self.assertAlmostEqual(
                    diagnostics["effective_bonus_mean"], beta
                )

        peak = age_band_policy(age_band_mix=1.0)
        np.testing.assert_array_equal(
            peak(observation(urgency=0.25, switch_gap=1.593749), MASK),
            [1],
        )
        exact = age_band_policy(age_band_mix=1.0)
        np.testing.assert_array_equal(
            exact(observation(urgency=0.25, switch_gap=1.59375), MASK),
            [2],
        )

    def test_class_two_threshold_is_independent_of_mix(self):
        for urgency in (0.0, 0.25, 0.5, 1.0):
            zero = age_band_policy(age_band_mix=0.0)
            one = age_band_policy(age_band_mix=1.0)
            obs = observation(
                urgency=urgency,
                class_2=1.0,
                switch_gap=0.1,
            )
            np.testing.assert_array_equal(zero(obs, MASK), one(obs, MASK))
            self.assertAlmostEqual(
                zero.diagnostics()["effective_bonus_mean"],
                one.diagnostics()["effective_bonus_mean"],
            )
            self.assertEqual(
                one.diagnostics()["threshold_unchanged_raw_switch_choices"],
                1,
            )

    def test_raw_urgency_is_clipped_before_age_weights(self):
        policy = age_band_policy(age_band_mix=1.0)
        policy(observation(urgency=2.0), MASK)
        diagnostics = policy.diagnostics()
        self.assertAlmostEqual(diagnostics["old_weight_mean"], 0.0)
        self.assertAlmostEqual(diagnostics["band_weight_mean"], 0.0)
        self.assertAlmostEqual(diagnostics["effective_bonus_mean"], 0.375)
        self.assertEqual(
            diagnostics["age_choice_bins"]["0p500_to_1p000"]["allowed"],
            1,
        )

    def test_first_use_and_forced_switch_remain_raw_actor_actions(self):
        policy = age_band_policy(age_band_mix=1.0)
        np.testing.assert_array_equal(
            policy(observation(first_use=True), MASK), [2]
        )
        unavailable_mask = np.asarray([[False, False, True]])
        np.testing.assert_array_equal(
            policy(
                observation(incumbent_score=10.0, switch_gap=-9.0),
                unavailable_mask,
            ),
            [2],
        )
        self.assertEqual(policy.diagnostics()["raw_switch_choices"], 0)

    def test_diagnostics_reconcile_weights_shifts_outcomes_and_age_bins(self):
        policy = age_band_policy(age_band_mix=1.0)
        cases = (
            (0.0, 0.0, 1.2),
            (0.25, 0.0, 1.4),
            (0.5, 0.0, 0.8),
            (0.25, 1.0, 0.1),
        )
        for urgency, class_2, gap in cases:
            policy(
                observation(
                    urgency=urgency,
                    class_2=class_2,
                    switch_gap=gap,
                ),
                MASK,
            )
        diagnostics = policy.diagnostics()
        self.assertEqual(diagnostics["raw_switch_choices"], 4)
        self.assertEqual(diagnostics["suppressed_switch_choices"], 2)
        self.assertEqual(
            diagnostics["threshold_increased_raw_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["threshold_decreased_raw_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["threshold_unchanged_raw_switch_choices"], 2
        )
        self.assertEqual(diagnostics["v9_would_suppress_switch_choices"], 2)
        self.assertEqual(
            diagnostics["created_by_band_shift_switch_choices"], 1
        )
        self.assertEqual(
            diagnostics["released_by_band_shift_switch_choices"], 1
        )
        self.assertAlmostEqual(diagnostics["old_weight_sum"], 2.0)
        self.assertAlmostEqual(diagnostics["band_weight_sum"], 2.0)
        self.assertAlmostEqual(diagnostics["mixed_weight_sum"], 2.0)
        self.assertEqual(diagnostics["age_bin_edges"], list(AGE_BIN_EDGES))
        self.assertEqual(set(diagnostics["age_choice_bins"]), set(AGE_BIN_LABELS))
        self.assertEqual(
            sum(
                sum(counts.values())
                for counts in diagnostics["age_choice_bins"].values()
            ),
            diagnostics["raw_switch_choices"],
        )
        self.assertEqual(
            sum(
                sum(counts.values())
                for counts in diagnostics["age_shift_bins"].values()
            ),
            diagnostics["raw_switch_choices"],
        )

    def test_invalid_mix_and_checkpoint_schema_fail_closed(self):
        for invalid in (-0.01, 1.01, float("nan"), True, "0.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "age-band mix"):
                    age_band_policy(age_band_mix=invalid)

        bad_schema = canonical_checkpoint_schema()
        bad_schema["candidate_feature_schema_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "schema hash"):
            AgeBandActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM,
                action_size=3,
                age_band_mix=0.5,
                device="cpu",
                checkpoint_schema=bad_schema,
            )

    def test_loader_preserves_frozen_legacy_actor_and_rejects_integrated(self):
        checkpoint = legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded, payload = load_age_band_hysteresis_policy(
                path, age_band_mix=1.0
            )
        self.assertEqual(payload["args"], checkpoint["args"])
        self.assertEqual(set(payload["actor"]), set(checkpoint["actor"]))
        self.assertEqual(loaded.feature_dim, 26)
        self.assertEqual(loaded.action_size, 7)
        self.assertEqual(loaded.age_band_mix, 1.0)
        self.assertEqual(loaded.stay_bonus, 1.0)
        self.assertEqual(loaded.urgency_relief, 0.625)
        self.assertEqual(loaded.class_2_relief, 0.75)
        self.assertEqual(loaded.checkpoint_schema["variant"], "proposed")

        integrated = legacy_checkpoint()
        integrated["candidate_actor_spec"] = {
            "schema_version": 1,
            "route_hysteresis_beta": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrated.pt"
            torch.save(integrated, path)
            with self.assertRaisesRegex(
                ValueError, "post-hoc adaptive hysteresis"
            ):
                load_age_band_hysteresis_policy(path, age_band_mix=0.5)


if __name__ == "__main__":
    unittest.main()
