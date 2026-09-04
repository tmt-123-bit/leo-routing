from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from mappo_evaluation import EpisodeMetrics
import run_margin_capped_shield_design as design


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
        "v9_failure_audit": {
            "design_selection_sha256": "f" * 64,
            "v9_failure_audit_sha256": "1" * 64,
        },
    }


def self_hashed(record: dict, field: str) -> dict:
    result = dict(record)
    result.pop(field, None)
    result[field] = design.source_runner.sha256_json(result)
    return result


def loader_replay_fixture(output: Path) -> tuple[dict, dict, dict, dict, dict]:
    source_runtime_record = {
        "source_runtime_equivalence_sha256": "c" * 64,
    }
    baseline = {
        "spec_sha256": "4" * 64,
        "validation_freeze_sha256": "e" * 64,
        "episode_metrics_path": "F:/frozen/v7.csv",
        "episode_metrics_sha256": "5" * 64,
        "reference_row_count": 160,
        "references": [],
    }
    v9_audit = {
        "design_selection_sha256": "f" * 64,
        "v9_failure_audit_sha256": "1" * 64,
    }
    source = {
        "training_freeze_sha256": "d" * 64,
        "checkpoints": {},
    }
    project = output / "src"
    spec = self_hashed(
        {
            "runtime_code_fingerprint": {"runner": "frozen"},
            "runtime_code_fingerprint_sha256": "b" * 64,
            "source_runtime_equivalence": source_runtime_record,
            "source_training_freeze_sha256": source["training_freeze_sha256"],
            "v7_baseline": {
                key: baseline[key]
                for key in (
                    "spec_sha256",
                    "validation_freeze_sha256",
                    "episode_metrics_path",
                    "episode_metrics_sha256",
                    "reference_row_count",
                )
            },
            "v9_failure_audit": v9_audit,
            "candidate_grid": [dict(candidate) for candidate in design.DESIGN_GRID],
            "paths": {
                "project": str(project),
                "protocol": str(
                    project.parent / "docs" / "MARGIN_CAPPED_SHIELD_DESIGN_V10.md"
                ),
                "source": str(output / "source"),
                "source_runtime_equivalence": str(output / "runtime.json"),
                "v7_output": str(output / "v7"),
                "v9_output": str(output / "v9"),
                "output": str(output),
            },
            "device": "cpu",
        },
        "design_spec_sha256",
    )
    return spec, source, source_runtime_record, baseline, v9_audit


def minimal_selection(spec: dict, source: dict, baseline: dict, v9_audit: dict) -> dict:
    return self_hashed(
        {
            "screen_name": design.SCREEN_NAME,
            "selection_rule_id": design.SELECTION_RULE_ID,
            "confirmatory": False,
            "paper_claim_allowed": False,
            "promotion_decision_allowed": False,
            "fresh_validation_allowed": False,
            "fresh_test_allowed": False,
            "test_allowed": False,
            "candidate_grid": [dict(candidate) for candidate in design.DESIGN_GRID],
            "design_spec_sha256": spec["design_spec_sha256"],
            "runtime_code_fingerprint_sha256": spec[
                "runtime_code_fingerprint_sha256"
            ],
            "source_runtime_equivalence_sha256": spec[
                "source_runtime_equivalence"
            ]["source_runtime_equivalence_sha256"],
            "source_training_freeze_sha256": source[
                "training_freeze_sha256"
            ],
            "v7_validation_freeze_sha256": baseline[
                "validation_freeze_sha256"
            ],
            "v9_design_selection_sha256": v9_audit[
                "design_selection_sha256"
            ],
            "v9_failure_audit_sha256": v9_audit[
                "v9_failure_audit_sha256"
            ],
            "artifacts": {
                name: {}
                for name in (
                    "design_spec",
                    "design_plan",
                    "episode_metrics",
                    "aggregate_metrics",
                    "cell_gates",
                )
            },
        },
        "design_selection_sha256",
    )


class MarginCappedShieldDesignTests(unittest.TestCase):
    def test_frozen_grid_reuse_and_new_job_counts(self):
        self.assertEqual(design.MARGIN_CAPS, (0.750, 0.375, 0.250))
        self.assertEqual(design.V9_REUSE_CANDIDATE_INDEX, 5)
        self.assertEqual(len(design.DESIGN_GRID), 3)
        self.assertEqual(len(design.build_design_jobs()), 24)
        self.assertEqual(len(design.build_new_evaluation_jobs()), 16)
        self.assertEqual(
            {job.candidate_index for job in design.build_new_evaluation_jobs()},
            {1, 2},
        )
        self.assertEqual(
            design.DESIGN_WORKLOAD_SEEDS, tuple(range(60001, 60011))
        )
        reused = [
            job for job in design.build_design_jobs() if job.candidate_index == 0
        ]
        self.assertEqual(len(reused), 8)
        self.assertTrue(
            all(job.as_dict()["execution_source"] == "v9_row_reuse" for job in reused)
        )

    def test_v9_candidate_five_mapping_is_exact(self):
        source = design.v9.DESIGN_GRID[design.V9_REUSE_CANDIDATE_INDEX]
        target = design.DESIGN_GRID[0]
        for field in (
            "stay_bonus",
            "urgency_relief",
            "class_2_relief",
            "calm_bonus",
        ):
            self.assertEqual(source[field], target[field])
        self.assertEqual(target["margin_cap"], 0.75)
        source_policy = design.v9.candidate_policy_name(source)
        rows = [
            episode("medium_load", source_policy, 1, 60001),
        ]
        mapped = design.reuse_v9_control_rows(rows)
        self.assertEqual(
            mapped[0].policy,
            design.candidate_policy_name(design.DESIGN_GRID[0]),
        )
        self.assertEqual(asdict(mapped[0]) | {"policy": source_policy}, asdict(rows[0]))

    def test_actual_v9_audit_checks_all_56_shards_and_combined_rows(self):
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
        record, rows = design.audit_v9_failure_and_shards(
            ROOT / "experiments" / "calm-selective-shield-design-v9",
            source,
            baseline,
        )
        self.assertEqual(record["new_shard_count"], 56)
        self.assertEqual(record["new_shard_row_count"], 560)
        self.assertTrue(record["combined_rows_reconstructed_exactly"])
        self.assertEqual(record["reuse_candidate_index"], 5)
        self.assertEqual(len(rows), 80)
        self.assertEqual(
            record["design_selection_sha256"],
            design.EXPECTED_V9_SELECTION_SHA256,
        )

    def test_five_gates_three_buffers_and_selection_key_are_frozen(self):
        gates = design.compute_candidate_cell_gates(complete_rows())
        self.assertEqual(len(gates), 24)
        self.assertTrue(all(row["cell_eligible"] for row in gates))
        results, selected = design.select_candidate(gates)
        self.assertEqual(len(results), 3)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["candidate_index"], 0)
        expected_key = [
            0.003 - 1.0 / 1920.0,
            0.010,
            1.0,
            0.055,
            0.75,
            0,
        ]
        for observed, expected in zip(
            selected["selection_key"], expected_key, strict=True
        ):
            self.assertAlmostEqual(observed, expected)

    def test_avoidable_gate_is_strict_and_delivery_buffer_is_additional(self):
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

    def test_shard_metadata_binds_every_frozen_input_and_closed_flags(self):
        job = design.build_new_evaluation_jobs()[0]
        checkpoint = {
            "checkpoint_path": "F:/frozen/validation_best.pt",
            "checkpoint_sha256": "2" * 64,
        }
        metadata = design._expected_shard_metadata(
            job, minimal_spec(), checkpoint
        )
        self.assertEqual(
            set(metadata),
            {
                "schema_version",
                "screen_name",
                "design_spec_sha256",
                "runtime_code_fingerprint_sha256",
                "source_runtime_equivalence_sha256",
                "source_training_freeze_sha256",
                "v7_validation_freeze_sha256",
                "v9_design_selection_sha256",
                "v9_failure_audit_sha256",
                "job",
                "checkpoint_path",
                "checkpoint_sha256",
                "workload_seeds",
                "parameters",
                "policy_schema",
                "paper_claim_allowed",
                "promotion_decision_allowed",
                "fresh_validation_allowed",
                "fresh_test_allowed",
                "test_allowed",
            },
        )
        self.assertEqual(metadata["design_spec_sha256"], "a" * 64)
        self.assertEqual(metadata["runtime_code_fingerprint_sha256"], "b" * 64)
        self.assertEqual(metadata["source_runtime_equivalence_sha256"], "c" * 64)
        self.assertEqual(metadata["source_training_freeze_sha256"], "d" * 64)
        self.assertEqual(metadata["v7_validation_freeze_sha256"], "e" * 64)
        self.assertEqual(metadata["v9_design_selection_sha256"], "f" * 64)
        self.assertEqual(metadata["v9_failure_audit_sha256"], "1" * 64)
        self.assertEqual(metadata["job"], job.as_dict())
        self.assertEqual(metadata["checkpoint_path"], checkpoint["checkpoint_path"])
        self.assertEqual(metadata["checkpoint_sha256"], checkpoint["checkpoint_sha256"])
        self.assertEqual(metadata["workload_seeds"], list(range(60001, 60011)))
        self.assertEqual(metadata["parameters"], job.as_parameters())
        self.assertEqual(metadata["policy_schema"]["candidate_feature_dim"], 26)
        self.assertEqual(metadata["policy_schema"]["action_size"], 7)
        self.assertEqual(metadata["policy_schema"]["obs_size"], 182)
        self.assertEqual(metadata["policy_schema"]["n_agents"], 24)
        self.assertEqual(metadata["policy_schema"]["variant"], "proposed")
        self.assertFalse(metadata["paper_claim_allowed"])
        self.assertFalse(metadata["promotion_decision_allowed"])
        self.assertFalse(metadata["fresh_validation_allowed"])
        self.assertFalse(metadata["fresh_test_allowed"])
        self.assertFalse(metadata["test_allowed"])

    def test_evaluator_receives_only_the_exposed_design_workloads(self):
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
                    "load_margin_capped_hysteresis_policy",
                    return_value=(policy, {}),
                ),
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

    def test_complete_existing_shard_resumes_without_evaluation(self):
        job = design.build_new_evaluation_jobs()[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = {
                "checkpoint_path": str(output / "missing-checkpoint.pt"),
                "checkpoint_sha256": "2" * 64,
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
            design.atomic_write_json(
                metadata_path,
                {
                    **expected,
                    "csv_sha256": design.sha256_file(csv_path),
                    "row_count": 10,
                    "policy_diagnostics": {"all_reconciled": True},
                },
            )
            args = SimpleNamespace(output=output, device="cpu")
            with mock.patch.object(design, "evaluate_policy") as evaluator:
                loaded = design.evaluate_job(
                    args, job, minimal_spec(), checkpoint
                )
            evaluator.assert_not_called()
            self.assertEqual(loaded, rows)

    def test_partial_existing_shard_fails_closed(self):
        job = design.build_new_evaluation_jobs()[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            csv_path, _ = design._evaluation_paths(output, job)
            design.atomic_write_csv(csv_path, [{"partial": True}])
            with self.assertRaisesRegex(ValueError, "partial v10 shard"):
                design.evaluate_job(
                    SimpleNamespace(output=output, device="cpu"),
                    job,
                    minimal_spec(),
                    {
                        "checkpoint_path": "missing",
                        "checkpoint_sha256": "2" * 64,
                    },
                )

    def test_written_selection_never_opens_fresh_or_test_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            design.atomic_write_json(
                output / "margin_capped_shield_design_spec.json",
                {"placeholder": True},
            )
            design.atomic_write_csv(
                output / "design_plan.csv",
                (job.as_dict() for job in design.build_design_jobs()),
            )
            spec = minimal_spec()
            source = {"training_freeze_sha256": "3" * 64}
            with (
                mock.patch.object(
                    design.source_runner,
                    "aggregate_rows",
                    return_value=[{"metric": "placeholder"}],
                ),
                mock.patch.object(design, "_shard_index", return_value=[]),
            ):
                selection = design.write_design_artifacts(
                    output, spec, source, complete_rows()
                )
        self.assertTrue(selection["all_cells_pass"])
        self.assertFalse(selection["paper_claim_allowed"])
        self.assertFalse(selection["promotion_decision_allowed"])
        self.assertFalse(selection["fresh_validation_allowed"])
        self.assertFalse(selection["fresh_test_allowed"])
        self.assertFalse(selection["test_allowed"])

    def test_source_runtime_failure_precedes_source_read_and_all_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = SimpleNamespace(
                source=Path(directory) / "source",
                source_runtime_equivalence=Path(directory) / "equivalence.json",
                v7_output=Path(directory) / "v7",
                v9_output=Path(directory) / "v9",
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
                mock.patch.object(design, "atomic_write_csv") as write_csv,
                mock.patch.object(design, "atomic_write_json") as write_json,
                mock.patch.object(Path, "mkdir") as mkdir,
                self.assertRaisesRegex(RuntimeError, "equivalence drift"),
            ):
                design.main()
            source_audit.assert_not_called()
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
                v9_output=Path(directory) / "v9",
                output=output,
                project=Path(directory) / "src",
                device="cpu",
                dry_run=True,
            )
            v9_audit = {
                "design_selection_sha256": design.EXPECTED_V9_SELECTION_SHA256,
                "new_shard_count": 56,
                "combined_rows_reconstructed_exactly": True,
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
                    "audit_v9_failure_and_shards",
                    return_value=(v9_audit, []),
                ),
                mock.patch.object(design, "reuse_v9_control_rows", return_value=[]),
                mock.patch.object(design, "build_spec", return_value=spec),
                mock.patch.object(design, "evaluate_jobs") as evaluate,
                mock.patch.object(design, "atomic_write_csv") as write_csv,
                mock.patch.object(design, "atomic_write_json") as write_json,
                mock.patch.object(Path, "mkdir") as mkdir,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(design.main(), 0)
            evaluate.assert_not_called()
            mkdir.assert_not_called()
            write_csv.assert_not_called()
            write_json.assert_not_called()
            self.assertFalse(output.exists())

    def test_selection_self_hash_fails_before_artifact_access(self):
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

    def test_rehashed_spec_semantic_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected_spec, source, runtime_record, baseline, v9_audit = (
                loader_replay_fixture(output)
            )
            tampered_spec = dict(expected_spec)
            tampered_spec["candidate_grid"] = [{"candidate_index": 99}]
            tampered_spec = self_hashed(
                tampered_spec, "design_spec_sha256"
            )
            selection = minimal_selection(
                tampered_spec, source, baseline, v9_audit
            )
            spec_path = output / "margin_capped_shield_design_spec.json"
            selection_path = output / "design_selection.json"
            design.atomic_write_json(spec_path, tampered_spec)
            design.atomic_write_json(selection_path, selection)
            with (
                mock.patch.object(
                    design,
                    "_runtime_code_fingerprint",
                    return_value=tampered_spec["runtime_code_fingerprint"],
                ),
                mock.patch.object(
                    design, "load_source_runtime_equivalence", return_value={}
                ),
                mock.patch.object(
                    design.source_runner,
                    "audit_source_screen",
                    return_value=source,
                ),
                mock.patch.object(
                    design,
                    "_source_runtime_equivalence_record",
                    return_value=runtime_record,
                ),
                mock.patch.object(
                    design, "audit_v7_baseline", return_value=baseline
                ),
                mock.patch.object(
                    design,
                    "audit_v9_failure_and_shards",
                    return_value=(v9_audit, []),
                ),
                mock.patch.object(
                    design, "build_spec", return_value=expected_spec
                ),
                self.assertRaisesRegex(
                    ValueError, "specification does not replay exactly"
                ),
            ):
                design.load_design_selection(selection_path)

    def test_rehashed_selection_binding_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            spec, source, runtime_record, baseline, v9_audit = (
                loader_replay_fixture(output)
            )
            selection = minimal_selection(spec, source, baseline, v9_audit)
            selection["source_training_freeze_sha256"] = "9" * 64
            selection = self_hashed(selection, "design_selection_sha256")
            spec_path = output / "margin_capped_shield_design_spec.json"
            selection_path = output / "design_selection.json"
            design.atomic_write_json(spec_path, spec)
            design.atomic_write_json(selection_path, selection)
            with (
                mock.patch.object(
                    design,
                    "_runtime_code_fingerprint",
                    return_value=spec["runtime_code_fingerprint"],
                ),
                mock.patch.object(
                    design, "load_source_runtime_equivalence", return_value={}
                ),
                mock.patch.object(
                    design.source_runner,
                    "audit_source_screen",
                    return_value=source,
                ),
                mock.patch.object(
                    design,
                    "_source_runtime_equivalence_record",
                    return_value=runtime_record,
                ),
                mock.patch.object(
                    design, "audit_v7_baseline", return_value=baseline
                ),
                mock.patch.object(
                    design,
                    "audit_v9_failure_and_shards",
                    return_value=(v9_audit, []),
                ),
                mock.patch.object(design, "build_spec", return_value=spec),
                self.assertRaisesRegex(
                    ValueError,
                    "selection binding changed: source_training_freeze_sha256",
                ),
            ):
                design.load_design_selection(selection_path)

    def test_completed_reentry_validates_selection_before_artifact_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "completed"
            output.mkdir()
            selection_path = output / "design_selection.json"
            design.atomic_write_json(selection_path, {"corrupt": True})
            args = SimpleNamespace(
                source=Path(directory) / "source",
                source_runtime_equivalence=Path(directory) / "runtime.json",
                v7_output=Path(directory) / "v7",
                v9_output=Path(directory) / "v9",
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
                    "audit_v9_failure_and_shards",
                    return_value=({}, []),
                ),
                mock.patch.object(design, "reuse_v9_control_rows", return_value=[]),
                mock.patch.object(design, "build_spec", return_value=minimal_spec()),
                mock.patch.object(
                    design.source_runner,
                    "invocation_lock",
                    return_value=nullcontext(),
                ),
                mock.patch.object(
                    design,
                    "load_design_selection",
                    side_effect=ValueError("existing selection drift"),
                ) as load_selection,
                mock.patch.object(design, "evaluate_jobs") as evaluate,
                mock.patch.object(design, "write_design_artifacts") as write_results,
                mock.patch.object(design, "atomic_write_csv") as write_csv,
                mock.patch.object(
                    design.source_runner, "ensure_immutable_json"
                ) as write_spec,
                self.assertRaisesRegex(ValueError, "existing selection drift"),
            ):
                design.main()
            load_selection.assert_called_once_with(selection_path.resolve())
            evaluate.assert_not_called()
            write_results.assert_not_called()
            write_csv.assert_not_called()
            write_spec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
