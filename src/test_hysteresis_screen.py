"""Contract tests for actor-score hysteresis and its two-stage screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from typing import get_type_hints

import numpy as np
import torch

from hysteresis_policy import ActorScoreHysteresisPolicy, load_hysteresis_policy
from mappo_design import SharedCandidateActor
from mappo_evaluation import EpisodeMetrics, load_checkpoint_policy
import run_hysteresis_screen as screen


class FeatureScoreActor(torch.nn.Module):
    def forward(self, candidates: torch.Tensor, action_mask=None):
        logits = candidates[..., 0]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


def policy_for(stay_bonus: float) -> ActorScoreHysteresisPolicy:
    return ActorScoreHysteresisPolicy(
        FeatureScoreActor(),
        feature_dim=18,
        action_size=3,
        stay_bonus=stay_bonus,
        device="cpu",
        checkpoint_schema={
            "candidate_feature_dim": 18,
            "action_size": 3,
            "obs_size": 54,
            "n_agents": 1,
            "variant": "proposed",
        },
    )


def observation(
    first_score: float,
    first_switch: float,
    second_score: float,
    second_switch: float,
) -> np.ndarray:
    candidates = np.zeros((1, 3, 18), dtype=np.float32)
    candidates[0, 1, 0] = first_score
    candidates[0, 1, 17] = first_switch
    candidates[0, 2, 0] = second_score
    candidates[0, 2, 17] = second_switch
    return candidates.reshape(1, -1)


def episode(
    *,
    scenario: str,
    policy: str,
    policy_seed: int,
    workload_seed: int,
    delivery_ratio: float,
    routing_switches: int,
) -> EpisodeMetrics:
    values = {}
    field_types = get_type_hints(EpisodeMetrics)
    for field in fields(EpisodeMetrics):
        if field.name == "scenario":
            values[field.name] = scenario
        elif field.name == "policy":
            values[field.name] = policy
        elif field.name == "policy_seed":
            values[field.name] = policy_seed
        elif field.name == "workload_seed":
            values[field.name] = workload_seed
        elif field.name == "delivery_ratio":
            values[field.name] = delivery_ratio
        elif field.name == "routing_switches":
            values[field.name] = routing_switches
        elif field_types[field.name] is int:
            values[field.name] = 0
        else:
            values[field.name] = 0.0
    return EpisodeMetrics(**values)


def synthetic_tuning_rows() -> list[EpisodeMetrics]:
    rows = []
    policies = [
        (screen.proposed_policy_name(), 0.800, 0.300, 100),
        (screen.raw_context_policy_name(), 0.800, 0.305, 110),
    ]
    for beta in screen.TUNING_BETAS:
        switches = 106 if beta == 0.02 else 104
        policies.append(
            (
                screen.hysteresis_policy_name(beta, final=False),
                0.799,
                0.304,
                switches,
            )
        )
    for scenario in screen.SCENARIOS:
        for policy, medium_delivery, hotspot_delivery, switches in policies:
            delivery = (
                medium_delivery
                if scenario == "medium_load"
                else hotspot_delivery
            )
            for policy_seed in screen.POLICY_SEEDS:
                for workload_seed in screen.TUNING_WORKLOAD_SEEDS:
                    rows.append(
                        episode(
                            scenario=scenario,
                            policy=policy,
                            policy_seed=policy_seed,
                            workload_seed=workload_seed,
                            delivery_ratio=delivery,
                            routing_switches=switches,
                        )
                    )
    return rows


class HysteresisPolicyTests(unittest.TestCase):
    def test_beta_zero_exactly_matches_raw_greedy(self):
        policy = policy_for(0.0)
        obs = observation(0.50, 0.0, 0.56, 1.0)
        mask = np.asarray([[False, True, True]])

        actions = policy(obs, mask)

        np.testing.assert_array_equal(actions, [2])

    def test_switch_is_suppressed_below_margin_and_allowed_above_it(self):
        obs = observation(0.50, 0.0, 0.56, 1.0)
        mask = np.asarray([[False, True, True]])

        np.testing.assert_array_equal(policy_for(0.07)(obs, mask), [1])
        np.testing.assert_array_equal(policy_for(0.05)(obs, mask), [2])

    def test_exact_margin_boundary_allows_the_raw_switch(self):
        obs = observation(0.50, 0.0, 0.75, 1.0)
        mask = np.asarray([[False, True, True]])

        np.testing.assert_array_equal(policy_for(0.25)(obs, mask), [2])

    def test_no_cache_does_not_bias_candidate_ordering(self):
        policy = policy_for(100.0)
        obs = observation(0.50, 0.0, 0.56, 0.0)
        mask = np.asarray([[False, True, True]])

        np.testing.assert_array_equal(policy(obs, mask), [2])

    def test_unavailable_cached_route_is_never_revived(self):
        policy = policy_for(100.0)
        obs = observation(10.0, 0.0, 0.56, 1.0)
        mask = np.asarray([[False, False, True]])

        np.testing.assert_array_equal(policy(obs, mask), [2])

    def test_beta_zero_checkpoint_loader_matches_existing_loader(self):
        torch.manual_seed(7)
        actor = SharedCandidateActor(26, 16, 1)
        checkpoint = {
            "candidate_feature_dim": 26,
            "action_size": 7,
            "obs_size": 182,
            "n_agents": 3,
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
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(checkpoint, path)
            existing, _ = load_checkpoint_policy(path)
            hysteresis, _ = load_hysteresis_policy(path, stay_bonus=0.0)
            rng = np.random.default_rng(11)
            obs = rng.normal(size=(3, 182)).astype(np.float32)
            mask = np.asarray(
                [
                    [False, True, True, False, False, False, False],
                    [False, True, False, True, False, False, False],
                    [True, False, False, False, False, False, False],
                ],
                dtype=bool,
            )

            np.testing.assert_array_equal(existing(obs, mask), hysteresis(obs, mask))


class HysteresisScreenTests(unittest.TestCase):
    def test_job_grids_are_complete_disjoint_and_stable(self):
        tuning = screen.build_tuning_jobs()
        final = screen.build_final_jobs(0.10)

        self.assertEqual(len(tuning), 56)
        self.assertEqual(len(final), 24)
        self.assertEqual(len({job.job_id for job in tuning}), 56)
        self.assertEqual(len({job.job_id for job in final}), 24)
        self.assertTrue(set(screen.TUNING_WORKLOAD_SEEDS).isdisjoint(
            screen.FINAL_WORKLOAD_SEEDS
        ))
        self.assertTrue(set(range(33001, 33051)).isdisjoint(
            screen.TUNING_WORKLOAD_SEEDS + screen.FINAL_WORKLOAD_SEEDS
        ))

    def test_selection_uses_smallest_beta_passing_every_gate(self):
        rows = synthetic_tuning_rows()

        selection_rows, selected = screen.select_tuning_beta(rows)

        self.assertFalse(selection_rows[0]["eligible"])
        self.assertEqual(selected, 0.05)
        self.assertTrue(selection_rows[1]["eligible"])

    def test_output_cannot_be_nested_inside_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "experiments" / "archive" / "source"
            source.mkdir(parents=True)
            args = argparse.Namespace(
                project=root / "src",
                source=source,
                output=source / "checkpoints" / "new-output",
                device="cpu",
            )
            with (
                mock.patch.dict("os.environ", {"LEO_REWARD_OVERRIDES": ""}),
                self.assertRaisesRegex(RuntimeError, "disjoint directory trees"),
            ):
                screen.validate_environment(args)

    def test_final_only_replays_selection_instead_of_trusting_self_hash(self):
        rows = synthetic_tuning_rows()
        selection_rows, selected = screen.select_tuning_beta(rows)
        self.assertEqual(selected, 0.05)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            episode_path = output / "tuning_episode_metrics.csv"
            aggregate_path = output / "tuning_aggregate_metrics.csv"
            selection_path = output / "tuning_selection.csv"
            screen.atomic_write_csv(episode_path, (asdict(row) for row in rows))
            screen.atomic_write_csv(aggregate_path, [{"placeholder": 1}])
            screen.atomic_write_csv(selection_path, selection_rows)
            freeze = {
                "schema_version": screen.SCHEMA_VERSION,
                "screen_name": screen.SCREEN_NAME,
                "spec_sha256": "spec",
                "selection_status": "selected",
                "selected_beta": selected,
                "selection_rule": "smallest_beta_passing_every_tuning_gate",
                "candidate_betas": list(screen.TUNING_BETAS),
                "artifacts": {
                    "episode_metrics": {
                        "path": str(episode_path),
                        "sha256": screen.sha256_file(episode_path),
                        "row_count": len(rows),
                    },
                    "aggregate_metrics": {
                        "path": str(aggregate_path),
                        "sha256": screen.sha256_file(aggregate_path),
                        "row_count": 1,
                    },
                    "selection": {
                        "path": str(selection_path),
                        "sha256": screen.sha256_file(selection_path),
                        "row_count": len(selection_rows),
                    },
                },
            }
            freeze["selection_freeze_sha256"] = screen.sha256_json(freeze)
            screen.atomic_write_json(output / "selection_freeze.json", freeze)
            loaded = screen.load_selection_freeze(output, "spec")
            self.assertEqual(loaded["selected_beta"], 0.05)

            freeze["selected_beta"] = 0.10
            freeze.pop("selection_freeze_sha256")
            freeze["selection_freeze_sha256"] = screen.sha256_json(freeze)
            screen.atomic_write_json(output / "selection_freeze.json", freeze)
            with self.assertRaisesRegex(ValueError, "replayed smallest"):
                screen.load_selection_freeze(output, "spec")


if __name__ == "__main__":
    unittest.main()
