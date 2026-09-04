from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ablation_matrix_runner import atomic_write_json
from mappo_evaluation import EpisodeMetrics
import run_adaptive_shield_design as v8
import run_calm_selective_shield_design as design


def episode(
    scenario: str,
    policy: str,
    policy_seed: int,
    workload_seed: int,
    *,
    delivery: float = 0.80,
    class_2: float = 0.80,
    avoidable: int = 4,
    opportunities: int = 100,
    switches: int = 15,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        scenario=scenario,
        policy=policy,
        policy_seed=policy_seed,
        workload_seed=workload_seed,
        generated=100,
        delivered=80,
        dropped=5,
        backlog=15,
        delivery_ratio=delivery,
        drop_rate=0.05,
        throughput_packets_per_slot=1.0,
        average_delay_slots=5.0,
        p95_delay_slots=8.0,
        mean_queue_packets=1.0,
        max_queue_packets=4,
        routing_switches=switches,
        episode_reward=0.0,
        global_delay_cost=0.0,
        global_queue_cost=0.0,
        global_load_imbalance=0.0,
        global_switch_cost=0.0,
        global_throughput_reward=0.0,
        global_control_overhead_ratio=0.0,
        global_drop_cost=0.0,
        class_0_delivery_ratio=0.80,
        class_1_delivery_ratio=0.80,
        class_2_delivery_ratio=class_2,
        avoidable_routing_switches=avoidable,
        forced_routing_switches=switches - avoidable,
        switch_opportunities=opportunities,
        avoidable_switch_rate=avoidable / opportunities,
    )


def passing_rows(*, shield_delivery: float = 0.80) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for scenario in design.SCENARIOS:
        for policy_seed in design.POLICY_SEEDS:
            for workload_seed in design.DESIGN_WORKLOAD_SEEDS:
                rows.append(
                    episode(
                        scenario,
                        design.REFERENCE_POLICIES["proposed"],
                        policy_seed,
                        workload_seed,
                        avoidable=10,
                        switches=20,
                    )
                )
                rows.append(
                    episode(
                        scenario,
                        design.REFERENCE_POLICIES["raw_context"],
                        policy_seed,
                        workload_seed,
                        avoidable=8,
                        switches=18,
                    )
                )
                for candidate in design.DESIGN_GRID:
                    rows.append(
                        episode(
                            scenario,
                            design.candidate_policy_name(candidate),
                            policy_seed,
                            workload_seed,
                            delivery=shield_delivery,
                            avoidable=4,
                            switches=15,
                        )
                    )
    return rows


class CalmSelectiveShieldDesignTests(unittest.TestCase):
    def test_frozen_grid_controls_and_new_job_count(self):
        self.assertEqual(
            design.CANDIDATE_PARAMETER_PAIRS,
            (
                (0.500, 0.000),
                (0.625, 0.000),
                (0.750, 0.000),
                (0.625, 0.250),
                (0.625, 0.500),
                (0.625, 0.750),
                (0.750, 0.250),
                (0.750, 0.500),
                (0.750, 0.750),
            ),
        )
        self.assertEqual(design.V8_CONTROL_CANDIDATE_MAP, {0: 12, 2: 14})
        self.assertEqual(len(design.DESIGN_GRID), 9)
        self.assertEqual(len(design.build_design_jobs()), 72)
        self.assertEqual(len(design.build_new_evaluation_jobs()), 56)
        self.assertEqual(
            {job.candidate_index for job in design.build_new_evaluation_jobs()},
            {1, 3, 4, 5, 6, 7, 8},
        )
        self.assertEqual(design.DESIGN_WORKLOAD_SEEDS, tuple(range(60001, 60011)))

    def test_zero_k_controls_match_v8_candidate_parameters(self):
        for v9_index, v8_index in design.V8_CONTROL_CANDIDATE_MAP.items():
            v9_candidate = design.DESIGN_GRID[v9_index]
            v8_candidate = v8.DESIGN_GRID[v8_index]
            self.assertEqual(v9_candidate["calm_bonus"], 0.0)
            for field in ("stay_bonus", "urgency_relief", "class_2_relief"):
                self.assertEqual(v9_candidate[field], v8_candidate[field])

    def test_logical_plan_marks_reuse_without_scheduling_it(self):
        jobs = design.build_design_jobs()
        reused = [job for job in jobs if job.candidate_index in (0, 2)]
        self.assertEqual(len(reused), 16)
        self.assertTrue(
            all(job.as_dict()["execution_source"] == "v8_row_reuse" for job in reused)
        )
        with self.assertRaisesRegex(RuntimeError, "must be reused"):
            design.evaluate_job(None, reused[0], {}, {})

    def test_buffers_and_selection_key_are_frozen(self):
        self.assertEqual(design.RAW_SWITCH_HEADROOM_BUFFER, 2.0)
        self.assertEqual(design.AVOIDABLE_MICRO_IMPROVEMENT_BUFFER, 0.005)
        self.assertEqual(design.DELIVERY_SLACK_BUFFER, 1.0 / 1920.0)
        gates = design.compute_candidate_cell_gates(passing_rows())
        self.assertEqual(len(gates), 72)
        self.assertTrue(all(row["cell_eligible"] for row in gates))
        results, selected = design.select_candidate(gates)
        self.assertEqual(len(results), 9)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["candidate_index"], 2)
        expected_key = [0.003, 0.010, 3.0, 0.06, 0.0, 0.75, -2]
        for observed, expected in zip(
            selected["selection_key"], expected_key, strict=True
        ):
            self.assertAlmostEqual(observed, expected)

    def test_one_packet_delivery_design_buffer_is_required(self):
        gates = design.compute_candidate_cell_gates(
            passing_rows(shield_delivery=0.7974)
        )
        self.assertTrue(all(row["delivery_gate_pass"] for row in gates))
        self.assertTrue(all(row["five_gates_pass"] for row in gates))
        self.assertTrue(all(not row["delivery_buffer_pass"] for row in gates))
        self.assertTrue(all(not row["cell_eligible"] for row in gates))
        _, selected = design.select_candidate(gates)
        self.assertIsNone(selected)

    def test_shard_metadata_binds_v7_v8_and_policy_schema(self):
        job = design.build_new_evaluation_jobs()[0]
        spec = {
            "design_spec_sha256": "a" * 64,
            "runtime_code_fingerprint_sha256": "b" * 64,
            "source_training_freeze_sha256": "c" * 64,
            "v7_baseline": {"validation_freeze_sha256": "d" * 64},
            "v8_failure_selection": {"design_selection_sha256": "e" * 64},
        }
        checkpoint = {
            "checkpoint_path": "F:/frozen/validation_best.pt",
            "checkpoint_sha256": "f" * 64,
        }
        metadata = design._expected_shard_metadata(job, spec, checkpoint)
        self.assertEqual(metadata["v8_design_selection_sha256"], "e" * 64)
        self.assertEqual(metadata["parameters"]["stay_bonus"], 1.0)
        self.assertEqual(metadata["parameters"]["class_2_relief"], 0.75)
        self.assertEqual(
            metadata["policy_schema"]["candidate_feature_schema_id"],
            "leo_multi_candidate_features_v1_dim_26",
        )

    def test_written_selection_never_opens_a_fresh_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            atomic_write_json(
                output / "calm_selective_shield_design_spec.json",
                {"placeholder": True},
            )
            spec = {
                "design_spec_sha256": "a" * 64,
                "runtime_code_fingerprint_sha256": "b" * 64,
                "v7_baseline": {"validation_freeze_sha256": "c" * 64},
                "v8_failure_selection": {
                    "design_selection_sha256": "d" * 64
                },
            }
            source = {"training_freeze_sha256": "e" * 64}
            with mock.patch.object(
                design.source_runner,
                "aggregate_rows",
                return_value=[{"metric": "placeholder"}],
            ):
                selection = design.write_design_artifacts(
                    output, spec, source, passing_rows()
                )
        self.assertTrue(selection["all_cells_pass"])
        self.assertFalse(selection["fresh_validation_allowed"])
        self.assertFalse(selection["fresh_test_allowed"])
        self.assertFalse(selection["test_allowed"])

    def test_selection_self_hash_fails_before_artifact_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design_selection.json"
            atomic_write_json(
                path,
                {
                    "screen_name": design.SCREEN_NAME,
                    "design_selection_sha256": "bad",
                },
            )
            with self.assertRaisesRegex(ValueError, "design_selection_sha256"):
                design.load_design_selection(path)


if __name__ == "__main__":
    unittest.main()
