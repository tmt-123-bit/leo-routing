from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from calm_selective_hysteresis_policy import (
    CalmSelectiveActorScoreHysteresisPolicy,
)
from leo_multiagent_env import BASE_CANDIDATE_FEATURE_NAMES, candidate_feature_schema
from mappo_design import SharedCandidateActor
from margin_capped_hysteresis_policy import (
    MarginCappedActorScoreHysteresisPolicy,
    load_margin_capped_hysteresis_policy,
)


class FeatureScoreActor(torch.nn.Module):
    def forward(self, candidates: torch.Tensor, action_mask=None):
        logits = candidates[..., 0]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


FEATURE_DIM = len(BASE_CANDIDATE_FEATURE_NAMES)
MASK = np.asarray([[False, True, True]])
V9_PARAMETERS = {
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


def capped_policy(
    *, margin_cap: float = 0.25
) -> MarginCappedActorScoreHysteresisPolicy:
    return MarginCappedActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=FEATURE_DIM,
        action_size=3,
        margin_cap=margin_cap,
        device="cpu",
        checkpoint_schema=canonical_checkpoint_schema(),
        **V9_PARAMETERS,
    )


def uncapped_policy() -> CalmSelectiveActorScoreHysteresisPolicy:
    return CalmSelectiveActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=FEATURE_DIM,
        action_size=3,
        device="cpu",
        checkpoint_schema=canonical_checkpoint_schema(),
        **V9_PARAMETERS,
    )


def observation(
    *,
    incumbent_score: float = 0.50,
    switch_gap: float = 1.50,
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
    torch.manual_seed(10)
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


class MarginCappedHysteresisPolicyTests(unittest.TestCase):
    def test_cap_point_seven_five_is_action_equivalent_to_v9_candidate_five(self):
        capped = capped_policy(margin_cap=0.75)
        uncapped = uncapped_policy()
        for urgency in (0.0, 0.125, 0.25, 0.49, 0.5, 1.0, 2.0):
            for class_2 in (0.0, 1.0):
                for gap in (0.05, 0.20, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0):
                    with self.subTest(
                        urgency=urgency, class_2=class_2, gap=gap
                    ):
                        obs = observation(
                            switch_gap=gap,
                            urgency=urgency,
                            class_2=class_2,
                        )
                        np.testing.assert_array_equal(
                            capped(obs, MASK), uncapped(obs, MASK)
                        )
        diagnostics = capped.diagnostics()
        self.assertEqual(diagnostics["cap_active_raw_switch_choices"], 0)
        self.assertAlmostEqual(diagnostics["cap_reduction_sum"], 0.0)

    def test_cap_point_two_five_has_exact_boundary_semantics(self):
        cases = (
            (1.249, 1, 1),
            (1.250, 2, 1),
            (1.749, 2, 1),
            (1.750, 2, 2),
        )
        for gap, expected_capped, expected_uncapped in cases:
            with self.subTest(gap=gap):
                capped = capped_policy(margin_cap=0.25)
                uncapped = uncapped_policy()
                obs = observation(switch_gap=gap)
                np.testing.assert_array_equal(
                    capped(obs, MASK), [expected_capped]
                )
                np.testing.assert_array_equal(
                    uncapped(obs, MASK), [expected_uncapped]
                )
                self.assertAlmostEqual(
                    capped.diagnostics()["effective_bonus_mean"], 1.25
                )

    def test_cap_can_only_release_and_never_create_a_suppression(self):
        for urgency in (0.0, 0.125, 0.25, 0.49, 0.5, 0.75, 1.0):
            for class_2 in (0.0, 1.0):
                for gap in np.linspace(0.05, 2.0, 40):
                    with self.subTest(
                        urgency=urgency,
                        class_2=class_2,
                        gap=float(gap),
                    ):
                        obs = observation(
                            switch_gap=float(gap),
                            urgency=urgency,
                            class_2=class_2,
                        )
                        uncapped_action = int(uncapped_policy()(obs, MASK)[0])
                        capped_action = int(
                            capped_policy(margin_cap=0.25)(obs, MASK)[0]
                        )
                        if uncapped_action == 2:
                            self.assertEqual(capped_action, 2)

    def test_class_two_and_urgent_states_are_not_changed_by_the_cap(self):
        capped = capped_policy(margin_cap=0.0)
        uncapped = uncapped_policy()
        cases = (
            (0.0, 1.0, 0.10),
            (0.25, 1.0, 0.20),
            (0.50, 0.0, 0.60),
            (1.00, 0.0, 0.30),
            (0.75, 1.0, 0.10),
        )
        for urgency, class_2, gap in cases:
            obs = observation(
                switch_gap=gap,
                urgency=urgency,
                class_2=class_2,
            )
            np.testing.assert_array_equal(
                capped(obs, MASK), uncapped(obs, MASK)
            )
        diagnostics = capped.diagnostics()
        self.assertEqual(diagnostics["cap_active_raw_switch_choices"], 0)
        self.assertEqual(diagnostics["released_by_cap_switch_choices"], 0)

    def test_first_use_and_forced_switch_remain_raw_actor_actions(self):
        capped = capped_policy(margin_cap=0.0)
        np.testing.assert_array_equal(
            capped(observation(first_use=True), MASK), [2]
        )
        unavailable_mask = np.asarray([[False, False, True]])
        np.testing.assert_array_equal(
            capped(
                observation(incumbent_score=10.0, switch_gap=-9.0),
                unavailable_mask,
            ),
            [2],
        )

    def test_margin_cap_diagnostics_strictly_reconcile(self):
        capped = capped_policy(margin_cap=0.25)
        cases = (
            (0.0, 0.0, 1.50),
            (0.0, 0.0, 1.00),
            (0.0, 1.0, 0.10),
            (0.0, 0.0, 2.00),
        )
        for urgency, class_2, gap in cases:
            capped(
                observation(
                    switch_gap=gap,
                    urgency=urgency,
                    class_2=class_2,
                ),
                MASK,
            )
        diagnostics = capped.diagnostics()
        self.assertEqual(diagnostics["raw_switch_choices"], 4)
        self.assertEqual(diagnostics["suppressed_switch_choices"], 2)
        self.assertEqual(
            diagnostics["uncapped_would_suppress_switch_choices"], 3
        )
        self.assertEqual(diagnostics["released_by_cap_switch_choices"], 1)
        self.assertEqual(diagnostics["cap_active_raw_switch_choices"], 3)
        self.assertEqual(diagnostics["cap_inactive_raw_switch_choices"], 1)
        self.assertEqual(
            diagnostics["released_by_cap_switch_choices"]
            + diagnostics["suppressed_switch_choices"],
            diagnostics["uncapped_would_suppress_switch_choices"],
        )
        self.assertAlmostEqual(diagnostics["cap_reduction_sum"], 1.50)
        self.assertAlmostEqual(diagnostics["cap_reduction_min"], 0.0)
        self.assertAlmostEqual(diagnostics["cap_reduction_max"], 0.50)
        self.assertAlmostEqual(diagnostics["cap_reduction_mean"], 0.375)
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
                for counts in diagnostics[
                    "relative_anchor_margin_choice_bins"
                ].values()
            ),
            diagnostics["raw_switch_choices"],
        )

    def test_invalid_margin_cap_fails_closed(self):
        for invalid in (-0.01, 1.01, float("nan"), True, "0.25"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "margin cap"):
                    capped_policy(margin_cap=invalid)

    def test_checkpoint_loader_preserves_frozen_actor_and_schema(self):
        checkpoint = legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            loaded, payload = load_margin_capped_hysteresis_policy(
                path,
                margin_cap=0.25,
                **V9_PARAMETERS,
            )
        self.assertEqual(payload["candidate_feature_dim"], 26)
        self.assertEqual(payload["action_size"], 7)
        self.assertEqual(payload["args"], checkpoint["args"])
        self.assertEqual(set(payload["actor"]), set(checkpoint["actor"]))
        self.assertEqual(loaded.feature_dim, 26)
        self.assertEqual(loaded.action_size, 7)
        self.assertEqual(loaded.checkpoint_schema["variant"], "proposed")
        self.assertEqual(
            loaded.checkpoint_schema["candidate_feature_schema_sha256"],
            candidate_feature_schema()["sha256"],
        )
        self.assertEqual(loaded.margin_cap, 0.25)
        self.assertEqual(loaded.urgency_relief, 0.625)
        self.assertEqual(loaded.class_2_relief, 0.75)
        self.assertEqual(loaded.calm_bonus, 0.75)

    def test_schema_and_nonlegacy_checkpoint_contracts_fail_closed(self):
        bad_schema = canonical_checkpoint_schema()
        bad_schema["candidate_feature_schema_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "schema hash"):
            MarginCappedActorScoreHysteresisPolicy(
                FeatureScoreActor(),
                feature_dim=FEATURE_DIM,
                action_size=3,
                margin_cap=0.25,
                device="cpu",
                checkpoint_schema=bad_schema,
                **V9_PARAMETERS,
            )

        checkpoint = legacy_checkpoint()
        checkpoint["candidate_actor_spec"] = {
            "schema_version": 1,
            "route_hysteresis_beta": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integrated.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(
                ValueError, "post-hoc (adaptive )?hysteresis"
            ):
                load_margin_capped_hysteresis_policy(
                    path,
                    margin_cap=0.25,
                    **V9_PARAMETERS,
                )


if __name__ == "__main__":
    unittest.main()
