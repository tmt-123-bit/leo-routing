from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from mappo_evaluation import EpisodeMetrics
import run_age_band_shield_design as design


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


def complete_rows(
    *,
    shield_delivery: float = 0.80,
    shield_avoidable: int = 4,
) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for scenario in design.SCENARIOS:
        for policy_seed in design.POLICY_SEEDS:
            for workload_seed in design.DESIGN_WORKLOAD_SEEDS:
                rows.extend(
                    (
                        episode(
                            scenario,
                            design.REFERENCE_POLICIES["proposed"],
                            policy_seed,
                            workload_seed,
                            avoidable=10,
                            switches=20,
                        ),
                        episode(
                            scenario,
                            design.REFERENCE_POLICIES["raw_context"],
                            policy_seed,
                            workload_seed,
                            avoidable=8,
                            switches=18,
                        ),
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
                            avoidable=shield_avoidable,
                            switches=15,
                        )
                    )
    return rows


def minimal_spec() -> dict:
    return {
        "design_spec_sha256": "a" * 64,
        "runtime_code_fingerprint_sha256": "b" * 64,
        "source_runtime_equivalence": {
            "source_runtime_equivalence_sha256": "c" * 64,
        },
        "source_training_freeze_sha256": "d" * 64,
        "v7_baseline": {"validation_freeze_sha256": "e" * 64},
        "v10_failure_audit": {
            "design_selection_sha256": "f" * 64,
            "design_spec_sha256": "1" * 64,
            "v10_failure_audit_sha256": "2" * 64,
        },
    }


def self_hashed(record: dict, field: str) -> dict:
    result = dict(record)
    result.pop(field, None)
    result[field] = design.source_runner.sha256_json(result)
    return result


class AgeBandShieldDesignTests(unittest.TestCase):
    def test_frozen_grid_ids_reuse_and_job_counts(self):
        self.assertEqual(design.AGE_BAND_MIXES, (0.0, 1.0))
        self.assertEqual(len(design.DESIGN_GRID), 2)
        self.assertEqual(
            [design.candidate_id(row) for row in design.DESIGN_GRID],
            [
                "b1p000_u0p625_c0p750_k0p750_m0p000",
                "b1p000_u0p625_c0p750_k0p750_m1p000",
            ],
        )
        self.assertEqual(len(design.build_design_jobs()), 16)
        self.assertEqual(len(design.build_new_evaluation_jobs()), 8)
        self.assertEqual(
            {job.candidate_index for job in design.build_new_evaluation_jobs()},
            {1},
        )
        self.assertEqual(design.DESIGN_WORKLOAD_SEEDS, tuple(range(60001, 60011)))
        reused = [
            job for job in design.build_design_jobs() if job.candidate_index == 0
        ]
        self.assertEqual(len(reused), 8)
        self.assertTrue(
            all(
                job.as_dict()["execution_source"]
                == "v9_candidate_5_row_reuse"
                for job in reused
            )
        )

    def test_v10_candidate_zero_mapping_is_exact(self):
        source = design.v10.DESIGN_GRID[design.V10_REUSE_CANDIDATE_INDEX]
        target = design.DESIGN_GRID[0]
        for field in (
            "stay_bonus",
            "urgency_relief",
            "class_2_relief",
            "calm_bonus",
        ):
            self.assertEqual(source[field], target[field])
        self.assertEqual(source["margin_cap"], 0.75)
        self.assertEqual(target["age_band_mix"], 0.0)
        source_policy = design.v10.candidate_policy_name(source)
        rows = [episode("medium_load", source_policy, 1, 60001)]
        mapped = design.reuse_v10_control_rows(rows)
        self.assertEqual(
            mapped[0].policy,
            design.candidate_policy_name(target),
        )
        self.assertEqual(asdict(mapped[0]) | {"policy": source_policy}, asdict(rows[0]))

    def test_actual_v10_full_loader_and_combined_control_extraction(self):
        source = design.source_runner.audit_source_screen(
            ROOT
            / "experiments"
            / "archive"
            / "congestion-context-screen-20k-v1"
        )
        baseline = design.audit_v7_baseline(
            ROOT / "experiments" / "proposed-shield-preflight-v7",
            source,
            ROOT / "src",
        )
        record, rows = design.audit_v10_failure_and_control(
            ROOT / "experiments" / "margin-capped-shield-design-v10",
            source,
            baseline,
        )
        self.assertEqual(record["selection_status"], "no_eligible_candidate")
        self.assertEqual(
            record["design_selection_sha256"],
            design.EXPECTED_V10_SELECTION_SHA256,
        )
        self.assertEqual(record["reuse_v10_candidate_index"], 0)
        self.assertEqual(record["reuse_v9_candidate_index"], 5)
        self.assertEqual(record["reuse_row_count"], 80)
        self.assertEqual(len(rows), 80)
        self.assertTrue(
            record["combined_rows_reconstructed_from_sources_by_full_loader"]
        )
        self.assertTrue(record["control_rows_extracted_from_combined_artifact"])
        self.assertFalse(record["all_cells_pass"])
        for field in design.CLOSED_PERMISSION_FIELDS:
            self.assertFalse(record[field])

    def test_actual_v11_historical_loader_replays_frozen_selection(self):
        selection = design.load_design_selection(
            ROOT
            / "experiments"
            / "age-band-shield-design-v11"
            / "design_selection.json",
            allow_historical_runtime=True,
        )
        self.assertEqual(selection["selection_status"], "no_eligible_candidate")
        self.assertEqual(
            selection["design_selection_sha256"],
            design.EXPECTED_V11_SELECTION_SHA256,
        )

    def test_five_gates_three_buffers_and_selection_key_are_frozen(self):
        gates = design.compute_candidate_cell_gates(complete_rows())
        self.assertEqual(len(gates), 16)
        self.assertTrue(all(row["cell_eligible"] for row in gates))
        results, selected = design.select_candidate(gates)
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["candidate_index"], 0)
        expected_key = [
            0.003 - 1.0 / 1920.0,
            0.010,
            1.0,
            0.055,
            0.0,
            0,
        ]
        for observed, expected in zip(
            selected["selection_key"], expected_key, strict=True
        ):
            self.assertAlmostEqual(observed, expected)

    def test_strict_avoidable_gate_and_additional_delivery_buffer(self):
        equal_micro = design.compute_candidate_cell_gates(
            complete_rows(shield_avoidable=10)
        )
        self.assertTrue(all(not row["avoidable_gate_pass"] for row in equal_micro))
        self.assertTrue(all(not row["cell_eligible"] for row in equal_micro))
        narrow_delivery = design.compute_candidate_cell_gates(
            complete_rows(shield_delivery=0.7974)
        )
        self.assertTrue(all(row["delivery_gate_pass"] for row in narrow_delivery))
        self.assertTrue(
            all(not row["delivery_buffer_pass"] for row in narrow_delivery)
        )

    def test_shard_metadata_binds_v10_and_all_closed_flags(self):
        job = design.build_new_evaluation_jobs()[0]
        checkpoint = {
            "checkpoint_path": "F:/frozen/validation_best.pt",
            "checkpoint_sha256": "3" * 64,
        }
        metadata = design._expected_shard_metadata(
            job, minimal_spec(), checkpoint
        )
        self.assertEqual(metadata["v10_design_selection_sha256"], "f" * 64)
        self.assertEqual(metadata["v10_design_spec_sha256"], "1" * 64)
        self.assertEqual(metadata["v10_failure_audit_sha256"], "2" * 64)
        self.assertEqual(metadata["job"], job.as_dict())
        self.assertEqual(metadata["parameters"], job.as_parameters())
        self.assertEqual(metadata["workload_seeds"], list(range(60001, 60011)))
        self.assertEqual(metadata["policy_schema"]["candidate_feature_dim"], 26)
        self.assertEqual(metadata["policy_schema"]["action_size"], 7)
        for field in design.CLOSED_PERMISSION_FIELDS:
            self.assertFalse(metadata[field])

    def test_evaluator_receives_exact_exposed_seeds_and_only_candidate_one(self):
        job = design.build_new_evaluation_jobs()[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint_path = output / "validation_best.pt"
            checkpoint_path.write_bytes(b"frozen checkpoint fixture")
            checkpoint = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": design.sha256_file(checkpoint_path),
            }
            schema = design._expected_shard_metadata(
                job, minimal_spec(), checkpoint
            )["policy_schema"]
            policy = SimpleNamespace(
                checkpoint_schema=schema,
                diagnostics=lambda: {"all_reconciled": True},
            )
            rows = [
                episode(
                    job.scenario,
                    job.policy_name,
                    job.policy_seed,
                    workload,
                )
                for workload in range(60001, 60011)
            ]
            with (
                mock.patch.object(
                    design,
                    "load_age_band_hysteresis_policy",
                    return_value=(policy, {}),
                ) as loader,
                mock.patch.object(
                    design, "evaluate_policy", return_value=rows
                ) as evaluator,
            ):
                observed = design.evaluate_job(
                    SimpleNamespace(output=output, device="cpu"),
                    job,
                    minimal_spec(),
                    checkpoint,
                )
        self.assertEqual(observed, rows)
        loader.assert_called_once_with(
            checkpoint_path,
            age_band_mix=1.0,
            device="cpu",
        )
        evaluator.assert_called_once_with(
            job.scenario,
            job.policy_name,
            policy,
            job.policy_seed,
            tuple(range(60001, 60011)),
            variant="proposed",
        )
        workloads = tuple(evaluator.call_args.args[4])
        self.assertTrue(all(60001 <= seed <= 60010 for seed in workloads))
        self.assertTrue(all(seed < 71001 for seed in workloads))
        control_job = design.build_design_jobs()[0]
        with self.assertRaisesRegex(RuntimeError, "must be reused"):
            design.evaluate_job(
                SimpleNamespace(output=Path("unused"), device="cpu"),
                control_job,
                minimal_spec(),
                {},
            )

    def test_valid_existing_shard_resumes_and_tampered_metadata_fails(self):
        job = design.build_new_evaluation_jobs()[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = {
                "checkpoint_path": str(output / "missing-checkpoint.pt"),
                "checkpoint_sha256": "3" * 64,
            }
            expected = design._expected_shard_metadata(
                job, minimal_spec(), checkpoint
            )
            rows = [
                episode(
                    job.scenario,
                    job.policy_name,
                    job.policy_seed,
                    workload,
                )
                for workload in design.DESIGN_WORKLOAD_SEEDS
            ]
            csv_path, metadata_path = design._evaluation_paths(output, job)
            design.atomic_write_csv(csv_path, (asdict(row) for row in rows))
            metadata = {
                **expected,
                "csv_sha256": design.sha256_file(csv_path),
                "row_count": 10,
                "policy_diagnostics": {"all_reconciled": True},
            }
            design.atomic_write_json(metadata_path, metadata)
            with mock.patch.object(design, "evaluate_policy") as evaluator:
                loaded = design.evaluate_job(
                    SimpleNamespace(output=output, device="cpu"),
                    job,
                    minimal_spec(),
                    checkpoint,
                )
            evaluator.assert_not_called()
            self.assertEqual(loaded, rows)

            metadata["v10_design_selection_sha256"] = "9" * 64
            design.atomic_write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "v10_design_selection_sha256"):
                design.evaluate_job(
                    SimpleNamespace(output=output, device="cpu"),
                    job,
                    minimal_spec(),
                    checkpoint,
                )

    def test_partial_existing_shard_fails_closed(self):
        job = design.build_new_evaluation_jobs()[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            csv_path, _ = design._evaluation_paths(output, job)
            design.atomic_write_csv(csv_path, [{"partial": True}])
            with self.assertRaisesRegex(ValueError, "partial v11 shard"):
                design.evaluate_job(
                    SimpleNamespace(output=output, device="cpu"),
                    job,
                    minimal_spec(),
                    {
                        "checkpoint_path": "missing",
                        "checkpoint_sha256": "3" * 64,
                    },
                )

    def test_selection_self_hash_and_rehashed_permission_tamper_fail_first(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design_selection.json"
            design.atomic_write_json(
                path,
                {
                    "screen_name": design.SCREEN_NAME,
                    "design_selection_sha256": "bad",
                },
            )
            with self.assertRaisesRegex(ValueError, "design_selection_sha256"):
                design.load_design_selection(path)

            tampered = self_hashed(
                {
                    "screen_name": design.SCREEN_NAME,
                    "selection_rule_id": design.SELECTION_RULE_ID,
                    "confirmatory": False,
                    "paper_claim_allowed": False,
                    "promotion_decision_allowed": False,
                    "fresh_validation_allowed": True,
                    "fresh_test_allowed": False,
                    "test_allowed": False,
                },
                "design_selection_sha256",
            )
            design.atomic_write_json(path, tampered)
            with self.assertRaisesRegex(
                ValueError, "fresh_validation_allowed=false"
            ):
                design.load_design_selection(path)

    def test_source_runtime_failure_precedes_v7_v10_and_all_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = SimpleNamespace(
                source=Path(directory) / "source",
                source_runtime_equivalence=Path(directory) / "equivalence.json",
                v7_output=Path(directory) / "v7",
                v10_output=Path(directory) / "v10",
                output=output,
                project=Path(directory) / "src",
                device="cpu",
                dry_run=False,
            )
            with (
                mock.patch.object(design, "parse_args", return_value=args),
                mock.patch.object(design, "validate_environment"),
                mock.patch.object(
                    design,
                    "load_source_runtime_equivalence",
                    side_effect=RuntimeError("equivalence drift"),
                ),
                mock.patch.object(
                    design.source_runner, "audit_source_screen"
                ) as source_audit,
                mock.patch.object(design, "audit_v7_baseline") as v7_audit,
                mock.patch.object(
                    design, "audit_v10_failure_and_control"
                ) as v10_audit,
                mock.patch.object(design, "atomic_write_csv") as write_csv,
                mock.patch.object(design, "atomic_write_json") as write_json,
                mock.patch.object(Path, "mkdir") as mkdir,
                self.assertRaisesRegex(RuntimeError, "equivalence drift"),
            ):
                design.main()
            source_audit.assert_not_called()
            v7_audit.assert_not_called()
            v10_audit.assert_not_called()
            mkdir.assert_not_called()
            write_csv.assert_not_called()
            write_json.assert_not_called()
            self.assertFalse(output.exists())

    def test_dry_run_writes_nothing_and_never_evaluates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = SimpleNamespace(
                source=Path(directory) / "source",
                source_runtime_equivalence=Path(directory) / "equivalence.json",
                v7_output=Path(directory) / "v7",
                v10_output=Path(directory) / "v10",
                output=output,
                project=Path(directory) / "src",
                device="cpu",
                dry_run=True,
            )
            v10_audit = {
                "design_selection_sha256": design.EXPECTED_V10_SELECTION_SHA256,
                "combined_rows_reconstructed_from_sources_by_full_loader": True,
            }
            spec = {
                "design_spec_sha256": "a" * 64,
                "source_runtime_equivalence": {
                    "source_runtime_equivalence_sha256": "b" * 64,
                },
            }
            with (
                mock.patch.object(design, "parse_args", return_value=args),
                mock.patch.object(design, "validate_environment"),
                mock.patch.object(
                    design, "load_source_runtime_equivalence", return_value={}
                ),
                mock.patch.object(
                    design.source_runner,
                    "audit_source_screen",
                    return_value={"checkpoints": {}},
                ),
                mock.patch.object(
                    design,
                    "audit_v7_baseline",
                    return_value={"reference_row_count": 160},
                ),
                mock.patch.object(
                    design,
                    "audit_v10_failure_and_control",
                    return_value=(v10_audit, []),
                ),
                mock.patch.object(
                    design, "reuse_v10_control_rows", return_value=[]
                ),
                mock.patch.object(design, "build_spec", return_value=spec),
                mock.patch.object(design, "evaluate_jobs") as evaluate,
                mock.patch.object(design, "atomic_write_csv") as write_csv,
                mock.patch.object(design, "atomic_write_json") as write_json,
                mock.patch.object(
                    design.source_runner, "ensure_immutable_json"
                ) as write_json_immutable,
                mock.patch.object(
                    design.source_runner, "invocation_lock"
                ) as invocation_lock,
                mock.patch.object(Path, "mkdir") as mkdir,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(design.main(), 0)
            evaluate.assert_not_called()
            mkdir.assert_not_called()
            invocation_lock.assert_not_called()
            write_csv.assert_not_called()
            write_json.assert_not_called()
            write_json_immutable.assert_not_called()
            self.assertFalse(output.exists())

    def test_completed_reentry_performs_zero_output_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "completed"
            output.mkdir()
            selection_path = output / "design_selection.json"
            design.atomic_write_json(selection_path, {"placeholder": True})
            args = SimpleNamespace(
                source=Path(directory) / "source",
                source_runtime_equivalence=Path(directory) / "runtime.json",
                v7_output=Path(directory) / "v7",
                v10_output=Path(directory) / "v10",
                output=output,
                project=Path(directory) / "src",
                device="cpu",
                dry_run=False,
            )
            with (
                mock.patch.object(design, "parse_args", return_value=args),
                mock.patch.object(design, "validate_environment"),
                mock.patch.object(
                    design, "load_source_runtime_equivalence", return_value={}
                ),
                mock.patch.object(
                    design.source_runner,
                    "audit_source_screen",
                    return_value={"checkpoints": {}},
                ),
                mock.patch.object(
                    design,
                    "audit_v7_baseline",
                    return_value={"references": [], "reference_row_count": 160},
                ),
                mock.patch.object(
                    design,
                    "audit_v10_failure_and_control",
                    return_value=({}, []),
                ),
                mock.patch.object(
                    design, "reuse_v10_control_rows", return_value=[]
                ),
                mock.patch.object(design, "build_spec", return_value=minimal_spec()),
                mock.patch.object(
                    design,
                    "load_design_selection",
                    return_value={"all_cells_pass": False},
                ) as load_selection,
                mock.patch.object(design, "evaluate_jobs") as evaluate,
                mock.patch.object(design, "write_design_artifacts") as write_results,
                mock.patch.object(design, "atomic_write_csv") as write_csv,
                mock.patch.object(
                    design.source_runner, "ensure_immutable_json"
                ) as write_spec,
                mock.patch.object(
                    design.source_runner, "invocation_lock"
                ) as invocation_lock,
                mock.patch.object(Path, "mkdir") as mkdir,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(design.main(), 2)
            load_selection.assert_called_once_with(selection_path.resolve())
            mkdir.assert_not_called()
            invocation_lock.assert_not_called()
            evaluate.assert_not_called()
            write_results.assert_not_called()
            write_csv.assert_not_called()
            write_spec.assert_not_called()

    def test_policy_and_protocol_are_explicitly_fingerprinted(self):
        fingerprint = design._runtime_code_fingerprint(
            ROOT / "src",
            ROOT / "docs" / "AGE_BAND_SHIELD_DESIGN_V11.md",
        )
        self.assertEqual(
            fingerprint["age_band_policy"],
            design.EXPECTED_AGE_BAND_POLICY_SHA256,
        )
        self.assertEqual(fingerprint["protocol"], design.EXPECTED_PROTOCOL_SHA256)
        self.assertIn("runner", fingerprint)
        self.assertIn("v10_runner", fingerprint)


if __name__ == "__main__":
    unittest.main()
