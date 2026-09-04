from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ablation_matrix_runner import atomic_write_json
from mappo_evaluation import EpisodeMetrics
import run_adaptive_shield_design as design


ROOT = Path(__file__).resolve().parent.parent


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


def passing_rows(
    *,
    shield_delivery: float = 0.80,
    shield_class_2: float = 0.80,
) -> list[EpisodeMetrics]:
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
                            class_2=shield_class_2,
                            avoidable=4,
                            switches=15,
                        )
                    )
    return rows


class AdaptiveShieldDesignTests(unittest.TestCase):
    def test_frozen_grid_and_job_count(self):
        self.assertEqual(design.STAY_BONUSES, (0.60, 0.80, 1.00))
        self.assertEqual(design.URGENCY_RELIEFS, (0.50, 0.75, 1.00))
        self.assertEqual(design.CLASS_2_RELIEFS, (0.75, 1.00))
        self.assertEqual(len(design.DESIGN_GRID), 18)
        self.assertEqual(len(design.build_design_jobs()), 144)
        self.assertEqual(design.DESIGN_WORKLOAD_SEEDS, tuple(range(60001, 60011)))

    def test_buffers_and_selection_key_are_frozen(self):
        self.assertEqual(design.RAW_SWITCH_HEADROOM_BUFFER, 2.0)
        self.assertEqual(design.AVOIDABLE_MICRO_IMPROVEMENT_BUFFER, 0.005)
        gates = design.compute_candidate_cell_gates(passing_rows())
        self.assertEqual(len(gates), 144)
        self.assertTrue(all(row["cell_eligible"] for row in gates))
        results, selected = design.select_candidate(gates)
        self.assertEqual(len(results), 18)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["candidate_index"], 5)
        expected_key = [0.003, 0.010, 3.0, 0.06, -0.60, 1.0, 1.0, -5]
        for observed, expected in zip(
            selected["selection_key"], expected_key, strict=True
        ):
            self.assertAlmostEqual(observed, expected)

    def test_shard_metadata_binds_the_canonical_policy_schema(self):
        job = design.build_design_jobs()[0]
        spec = {
            "design_spec_sha256": "a" * 64,
            "runtime_code_fingerprint_sha256": "b" * 64,
            "source_training_freeze_sha256": "c" * 64,
            "v7_baseline": {"validation_freeze_sha256": "d" * 64},
        }
        checkpoint = {
            "checkpoint_path": "F:/frozen/validation_best.pt",
            "checkpoint_sha256": "e" * 64,
        }
        metadata = design._expected_shard_metadata(job, spec, checkpoint)
        schema = metadata["policy_schema"]
        self.assertEqual(schema["candidate_feature_dim"], 26)
        self.assertEqual(schema["candidate_feature_schema_id"],
                         "leo_multi_candidate_features_v1_dim_26")
        self.assertEqual(schema["candidate_feature_schema_sha256"],
                         design.EXPECTED_FEATURE_SCHEMA["sha256"])
        self.assertEqual(schema["candidate_feature_names"],
                         list(design.EXPECTED_FEATURE_NAMES))

    def test_no_unregistered_delivery_or_class_two_buffer(self):
        gates = design.compute_candidate_cell_gates(
            passing_rows(shield_delivery=0.7971, shield_class_2=0.7901)
        )
        self.assertTrue(all(row["delivery_gate_pass"] for row in gates))
        self.assertTrue(all(row["class_2_gate_pass"] for row in gates))
        self.assertTrue(all(row["cell_eligible"] for row in gates))

    def test_raw_switch_buffer_is_required_in_every_cell(self):
        rows = passing_rows()
        target = design.DESIGN_GRID[5]
        target_policy = design.candidate_policy_name(target)
        for row in rows:
            if row.policy == target_policy:
                row.routing_switches = 17
                row.forced_routing_switches = (
                    row.routing_switches - row.avoidable_routing_switches
                )
        gates = design.compute_candidate_cell_gates(rows)
        target_gates = [row for row in gates if row["candidate_index"] == 5]
        self.assertTrue(all(row["five_gates_pass"] for row in target_gates))
        self.assertTrue(all(not row["buffer_gate_pass"] for row in target_gates))
        _, selected = design.select_candidate(gates)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["candidate_index"], 4)

    def test_avoidable_buffer_uses_micro_rate_not_episode_macro_mean(self):
        rows = passing_rows()
        target_policy = design.candidate_policy_name(design.DESIGN_GRID[0])
        target_rows = [row for row in rows if row.policy == target_policy]
        for index, row in enumerate(target_rows):
            row.avoidable_routing_switches = 0 if index % 10 == 0 else 10
            row.switch_opportunities = 1000 if index % 10 == 0 else 10
            row.avoidable_switch_rate = (
                row.avoidable_routing_switches / row.switch_opportunities
            )
            row.forced_routing_switches = (
                row.routing_switches - row.avoidable_routing_switches
            )
        gates = design.compute_candidate_cell_gates(rows)
        target_gates = [row for row in gates if row["candidate_index"] == 0]
        self.assertTrue(
            all(
                row["shield_avoidable_switch_micro_rate"]
                < row["proposed_avoidable_switch_micro_rate"]
                for row in target_gates
            )
        )

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

    def test_actual_historical_v7_source_fingerprint_evidence(self):
        source = design.source_runner.audit_source_screen(
            ROOT
            / "experiments"
            / "archive"
            / "congestion-context-screen-20k-v1"
        )
        spec = design.source_runner._load_json(
            ROOT
            / "experiments"
            / "proposed-shield-preflight-v7"
            / "proposed_shield_preflight_spec.json"
        )
        for name in design.HISTORICAL_SOURCE_FINGERPRINT_FIELDS:
            filename = design.BASELINE_FINGERPRINT_FILES[name]
            record = design._audit_historical_v7_source_fingerprint(
                ROOT / "src",
                source,
                name,
                filename,
                spec["code_fingerprint"][name],
            )
            self.assertEqual(
                record["historical_sha256"],
                spec["code_fingerprint"][name],
            )
            self.assertGreater(record["historical_length"], 0)
            self.assertEqual(
                record["equivalence_sha256"],
                design.EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_SHA256,
            )

    def test_historical_v7_source_evidence_fails_closed(self):
        source = {"spec_sha256": "source-spec"}
        evidence = {
            "source_runtime_equivalence_sha256": "not-a-valid-self-hash",
        }
        with (
            mock.patch.object(
                design,
                "sha256_file",
                return_value=(
                    design.EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_FILE_SHA256
                ),
            ),
            mock.patch.object(
                design.source_runner,
                "_load_json",
                return_value=evidence,
            ),
            self.assertRaisesRegex(
                ValueError,
                "source_runtime_equivalence_sha256",
            ),
        ):
            design._audit_historical_v7_source_fingerprint(
                ROOT / "src",
                source,
                "evaluation",
                "mappo_evaluation.py",
                "a" * 64,
            )

    def test_historical_source_rejects_unapproved_field(self):
        with self.assertRaisesRegex(
            ValueError,
            "unsupported v7 historical source field: statistics",
        ):
            design._audit_historical_v7_source_fingerprint(
                ROOT / "src",
                {"spec_sha256": "unused"},
                "statistics",
                "hierarchical_statistics.py",
                "a" * 64,
            )

    def test_rehashed_equivalence_tamper_still_fails_fixed_hash(self):
        evidence_path = (
            ROOT / design.SOURCE_RUNTIME_EQUIVALENCE_RELATIVE_PATH
        )
        evidence = design.source_runner._load_json(evidence_path)
        evidence["decision"] = dict(evidence["decision"])
        evidence["decision"]["all_pass"] = False
        body = dict(evidence)
        body.pop("source_runtime_equivalence_sha256")
        evidence["source_runtime_equivalence_sha256"] = (
            design.source_runner.sha256_json(body)
        )
        with (
            mock.patch.object(
                design,
                "sha256_file",
                return_value=(
                    design.EXPECTED_SOURCE_RUNTIME_EQUIVALENCE_FILE_SHA256
                ),
            ),
            mock.patch.object(
                design.source_runner,
                "_load_json",
                return_value=evidence,
            ),
            self.assertRaisesRegex(
                ValueError,
                "unexpected v7 source equivalence evidence",
            ),
        ):
            design._audit_historical_v7_source_fingerprint(
                ROOT / "src",
                {"spec_sha256": "unused"},
                "evaluation",
                "mappo_evaluation.py",
                "a" * 64,
            )

    def test_v8_loader_requires_explicit_historical_mode(self):
        selection_path = (
            ROOT
            / "experiments"
            / "adaptive-shield-design-v8"
            / "design_selection.json"
        )
        with self.assertRaisesRegex(
            ValueError,
            "design runtime code fingerprint changed",
        ):
            design.load_design_selection(selection_path)
        selection = design.load_design_selection(
            selection_path,
            allow_historical_runtime=True,
        )
        self.assertEqual(
            selection["design_selection_sha256"],
            design.EXPECTED_V8_SELECTION_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
