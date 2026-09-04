"""Contract tests for the exploratory congestion-context screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_congestion_screen as screen


def decision_rows(
    *,
    hotspot_delta: float = 0.010,
    hotspot_positive_fraction: float = 0.75,
    medium_delta: float = -0.003,
    medium_seed_differences: tuple[float, ...] = (-0.009, -0.009, -0.009, -0.020),
) -> list[dict[str, object]]:
    """Build the complete metric surface consumed by ``decide_screen``."""

    rows: list[dict[str, object]] = []
    for scenario in screen.SCENARIOS:
        for metric in screen.SCREEN_METRICS:
            row: dict[str, object] = {
                "scenario": scenario,
                "metric": metric,
                "reference_mean": 1.0,
                "mean_difference": 0.0,
            }
            if metric == "delivery_ratio":
                if scenario == "hotspot_high_load":
                    row.update(
                        {
                            "mean_difference": hotspot_delta,
                            "positive_policy_seed_fraction": (
                                hotspot_positive_fraction
                            ),
                            "policy_seed_mean_differences": [
                                hotspot_delta,
                                hotspot_delta,
                                hotspot_delta,
                                -0.001,
                            ],
                        }
                    )
                else:
                    row.update(
                        {
                            "mean_difference": medium_delta,
                            "positive_policy_seed_fraction": 0.25,
                            "policy_seed_mean_differences": list(
                                medium_seed_differences
                            ),
                        }
                    )
            rows.append(row)
    return rows


def row_for(
    rows: list[dict[str, object]], scenario: str, metric: str
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["scenario"] == scenario and row["metric"] == metric
    )


class CongestionScreenContractTests(unittest.TestCase):
    def test_policy_seeds_follow_frozen_sha256_derivation(self):
        independently_derived = tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"{screen.POLICY_SEED_NAMESPACE}{index}".encode("ascii")
                ).digest()[:4],
                "big",
            )
            & 0x7FFFFFFF
            for index in range(4)
        )

        self.assertEqual(screen.POLICY_SEEDS, screen.EXPECTED_POLICY_SEEDS)
        self.assertEqual(screen.POLICY_SEEDS, independently_derived)
        self.assertEqual(len(set(screen.POLICY_SEEDS)), 4)
        self.assertTrue(all(seed > 0 for seed in screen.POLICY_SEEDS))

    def test_build_jobs_has_all_16_unique_cells_in_stable_order(self):
        jobs = screen.build_jobs()
        expected_cells = {
            (scenario, variant, policy_seed)
            for scenario in screen.SCENARIOS
            for variant in screen.VARIANTS
            for policy_seed in screen.POLICY_SEEDS
        }
        observed_cells = {
            (job.scenario, job.variant, job.policy_seed) for job in jobs
        }

        self.assertEqual(len(jobs), 2 * 2 * 4)
        self.assertEqual(observed_cells, expected_cells)
        self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
        self.assertEqual([job.index for job in jobs], list(range(16)))
        self.assertEqual(
            jobs[0].job_id,
            f"medium_load/proposed/seed_{screen.POLICY_SEEDS[0]}",
        )
        self.assertEqual(
            jobs[-1].job_id,
            "hotspot_high_load/with_congestion_context/"
            f"seed_{screen.POLICY_SEEDS[-1]}",
        )

    def test_screen_spec_is_self_hashed_resumable_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                project=root / "src",
                cleanmarl=root / "cleanmarl",
                device="cuda",
            )
            with (
                mock.patch.object(
                    screen,
                    "code_fingerprint",
                    return_value={
                        "trainer": "a" * 64,
                        "design": "b" * 64,
                    },
                ),
                mock.patch.object(screen, "sha256_file", return_value="b" * 64),
            ):
                spec = screen.build_screen_spec(args, screen.build_jobs())

            unhashed = dict(spec)
            recorded_hash = unhashed.pop("spec_sha256")
            self.assertEqual(recorded_hash, screen.sha256_json(unhashed))
            self.assertEqual(spec["expected_training_jobs"], 16)
            self.assertEqual(spec["expected_evaluation_rows"], 800)
            self.assertEqual(spec["training_device"], "cuda")
            self.assertEqual(spec["train_workload_seeds"], list(range(9001, 9201)))
            self.assertFalse(spec["confirmatory"])
            self.assertEqual(json.loads(json.dumps(spec)), spec)

            spec_path = root / "screen_spec.json"
            screen.ensure_immutable_json(spec_path, spec)
            screen.ensure_immutable_json(spec_path, spec)
            self.assertEqual(
                json.loads(spec_path.read_text(encoding="utf-8")), spec
            )

            changed = dict(spec)
            changed["expected_training_jobs"] = 15
            with self.assertRaisesRegex(RuntimeError, "immutable artifact mismatch"):
                screen.ensure_immutable_json(spec_path, changed)

    def test_decision_promotes_at_all_inclusive_boundaries(self):
        rows = decision_rows()
        for scenario in screen.SCENARIOS:
            row_for(rows, scenario, "average_delay_slots")[
                "mean_difference"
            ] = 0.10
            row_for(rows, scenario, "class_0_delivery_ratio")[
                "mean_difference"
            ] = -0.020

        decision = screen.decide_screen(rows)

        self.assertEqual(
            decision["decision"],
            "promote_to_separately_frozen_50k_head_to_head",
        )
        self.assertEqual(decision["reject_reasons"], [])
        self.assertEqual(
            decision["evidence"]["medium_noninferior_policy_seed_count"], 3
        )
        self.assertFalse(decision["paper_claim_allowed"])

    def test_decision_is_inconclusive_between_reject_and_promote(self):
        decision = screen.decide_screen(decision_rows(hotspot_delta=0.009))

        self.assertEqual(decision["decision"], "inconclusive")
        self.assertEqual(decision["reject_reasons"], [])

    def test_decision_rejects_hard_direction_and_cost_failures(self):
        cases = []

        no_hotspot_gain = decision_rows(hotspot_delta=0.0)
        cases.append(("hotspot", no_hotspot_gain, "hotspot delivery did not improve"))

        medium_failure = decision_rows(medium_delta=-0.011)
        cases.append(
            (
                "medium",
                medium_failure,
                "medium-load delivery crossed the hard reject margin",
            )
        )

        cost_failure = decision_rows()
        row_for(cost_failure, "medium_load", "average_delay_slots")[
            "mean_difference"
        ] = 0.1001
        cases.append(("cost", cost_failure, "one or more predeclared cost gates failed"))

        for label, rows, reason in cases:
            with self.subTest(case=label):
                decision = screen.decide_screen(rows)
                self.assertEqual(decision["decision"], "reject")
                self.assertIn(reason, decision["reject_reasons"])

    def test_validate_environment_rejects_frozen_ablation_output_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project = root / "src"
            formal_output = root / "experiments" / "ablation-50k-v2"

            def args_for(output: Path) -> argparse.Namespace:
                return argparse.Namespace(
                    project=project,
                    output=output.resolve(),
                    device="cpu",
                )

            with mock.patch.dict("os.environ", {"LEO_REWARD_OVERRIDES": ""}):
                for forbidden in (formal_output, formal_output / "screen"):
                    with self.subTest(output=forbidden), self.assertRaisesRegex(
                        RuntimeError, "must not be inside"
                    ):
                        screen.validate_environment(args_for(forbidden))

                allowed = root / "experiments" / "archive" / "screen"
                screen.validate_environment(args_for(allowed))

                for broad in (root, project, root / "experiments", root / "experiments" / "archive"):
                    with self.subTest(output=broad), self.assertRaisesRegex(
                        RuntimeError, "dedicated experiment directory"
                    ):
                        screen.validate_environment(args_for(broad))

    def test_orphan_lock_recovery_never_overlaps_a_live_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            job = screen.build_jobs()[0]
            path = screen.lock_path(output, job)
            path.parent.mkdir(parents=True)
            payload = {
                "pid": 987654321,
                "hostname": socket.gethostname(),
                "job_id": job.job_id,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            with (
                mock.patch.object(screen, "_process_is_alive", return_value=False),
                mock.patch.object(screen, "_matching_trainer_pids", return_value=(4321,)),
                self.assertRaisesRegex(screen.JobLockBusy, "trainer may still be active"),
            ):
                screen._recover_orphaned_job_lock(output, job)
            self.assertTrue(path.exists())

            with (
                mock.patch.object(screen, "_process_is_alive", return_value=False),
                mock.patch.object(screen, "_matching_trainer_pids", return_value=()),
            ):
                screen._recover_orphaned_job_lock(output, job)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
