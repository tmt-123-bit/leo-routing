from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import get_type_hints
import unittest
from unittest import mock

from mappo_evaluation import EpisodeMetrics
import run_adaptive_shield_preflight as preflight


def selection_record(source_sha: str = "source-sha") -> dict:
    record = {
        "schema_version": 1,
        "screen_name": preflight.DESIGN_SCREEN_NAME,
        "design_spec_sha256": "design-spec",
        "source_training_freeze_sha256": source_sha,
        "v7_validation_freeze_sha256": "v7-validation",
        "selection_rule_id": "unique_lexicographic_argmax_v8",
        "all_cells_pass": True,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "selected_parameters": {
            "stay_bonus": 0.40,
            "urgency_relief": 0.50,
            "class_2_relief": 1.00,
        },
    }
    record["design_selection_sha256"] = preflight.source_runner.sha256_json(
        record
    )
    return record


def matching_runtime_fingerprints() -> dict[str, str]:
    return {
        name: f"{index:064x}"
        for index, name in enumerate(preflight.SOURCE_RUNTIME_FILES, start=1)
    }


def equivalence_artifact(source_spec_sha: str = "source-spec") -> dict:
    record = {
        "schema_version": preflight.runtime_equivalence.SCHEMA_VERSION,
        "audit_name": preflight.runtime_equivalence.AUDIT_NAME,
        "paper_claim_allowed": False,
        "promotion_decision_allowed": False,
        "new_evaluation_panel_accessed": False,
        "frozen_specs": {
            "source_screen": {"spec_sha256": source_spec_sha},
        },
        "dynamic_contract": {
            "workload_seed": 60001,
            "workload_seed_provenance": (
                "replay_of_exposed_v7_validation_seed_60001"
            ),
            "feature_contract": dict(
                preflight.runtime_equivalence.EXPECTED_CURRENT_FEATURE_CONTRACT
            ),
        },
        "decision": {
            "snapshot_current_exact_match": True,
            "all_pass": True,
        },
    }
    record["source_runtime_equivalence_sha256"] = (
        preflight.runtime_equivalence.sha256_json(record)
    )
    return record


def spec_equivalence_binding() -> dict:
    return {"source_runtime_equivalence_sha256": "equivalence-sha"}


def episode(
    *,
    scenario: str,
    policy: str,
    policy_seed: int,
    workload_seed: int,
    delivery: float,
    class_2_delivery: float,
    avoidable: int,
    forced: int,
    opportunities: int,
) -> EpisodeMetrics:
    values = {}
    type_hints = get_type_hints(EpisodeMetrics)
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
            values[field.name] = delivery
        elif field.name == "class_2_delivery_ratio":
            values[field.name] = class_2_delivery
        elif field.name == "avoidable_routing_switches":
            values[field.name] = avoidable
        elif field.name == "forced_routing_switches":
            values[field.name] = forced
        elif field.name == "routing_switches":
            values[field.name] = avoidable + forced
        elif field.name == "switch_opportunities":
            values[field.name] = opportunities
        elif field.name == "avoidable_switch_rate":
            values[field.name] = avoidable / max(1, opportunities)
        elif type_hints[field.name] is int:
            values[field.name] = 0
        elif type_hints[field.name] is float:
            values[field.name] = 0.0
        elif type_hints[field.name] is str:
            values[field.name] = ""
        else:
            default = field.default
            values[field.name] = default if default is not None else 0
    return EpisodeMetrics(**values)


def complete_rows(workloads=(1, 2)) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for scenario in preflight.SCENARIOS:
        for policy_seed in preflight.POLICY_SEEDS:
            for index, workload in enumerate(workloads):
                proposed_avoidable = 1 if index == 0 else 0
                proposed_opportunities = 1 if index == 0 else 99
                shield_avoidable = 0 if index == 0 else 2
                shield_opportunities = 1 if index == 0 else 99
                rows.extend(
                    (
                        episode(
                            scenario=scenario,
                            policy=preflight.POLICY_NAMES["proposed"],
                            policy_seed=policy_seed,
                            workload_seed=workload,
                            delivery=0.80,
                            class_2_delivery=0.80,
                            avoidable=proposed_avoidable,
                            forced=9 if index == 0 else 10,
                            opportunities=proposed_opportunities,
                        ),
                        episode(
                            scenario=scenario,
                            policy=preflight.POLICY_NAMES["raw_context"],
                            policy_seed=policy_seed,
                            workload_seed=workload,
                            delivery=0.80,
                            class_2_delivery=0.80,
                            avoidable=3,
                            forced=9,
                            opportunities=20,
                        ),
                        episode(
                            scenario=scenario,
                            policy=preflight.POLICY_NAMES["shield"],
                            policy_seed=policy_seed,
                            workload_seed=workload,
                            delivery=0.799,
                            class_2_delivery=0.795,
                            avoidable=shield_avoidable,
                            forced=8 if index == 0 else 6,
                            opportunities=shield_opportunities,
                        ),
                    )
                )
    return rows


class AdaptiveShieldPreflightTests(unittest.TestCase):
    def test_frozen_job_grids_panels_and_checkpoint_bindings(self):
        validation = preflight.build_jobs("validation")
        test = preflight.build_jobs("test")
        self.assertEqual(len(validation), 24)
        self.assertEqual(len(test), 24)
        self.assertEqual([job.index for job in validation], list(range(24)))
        self.assertEqual(
            preflight.VALIDATION_WORKLOAD_SEEDS, tuple(range(71001, 71011))
        )
        self.assertEqual(
            preflight.TEST_WORKLOAD_SEEDS, tuple(range(72001, 72026))
        )
        exposed = set().union(
            *(
                set(range(start, stop + 1))
                for start, stop in preflight.KNOWN_EXPOSED_OR_RESERVED_PANELS
            )
        )
        self.assertIn((70001, 70025), preflight.KNOWN_EXPOSED_OR_RESERVED_PANELS)
        self.assertFalse(set(preflight.VALIDATION_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(set(preflight.TEST_WORKLOAD_SEEDS) & exposed)
        for job in validation:
            expected = (
                "proposed" if job.arm in {"proposed", "shield"}
                else "with_congestion_context"
            )
            self.assertEqual(job.source_variant, expected)
            if job.arm == "shield":
                self.assertEqual(
                    job.source_job_id,
                    f"{job.scenario}/proposed/seed_{job.policy_seed}",
                )

    def test_cli_has_no_parameter_override(self):
        args = preflight.parse_args(["--dry-run", "--device", "cpu"])
        self.assertFalse(hasattr(args, "stay_bonus"))
        self.assertFalse(hasattr(args, "urgency_relief"))
        self.assertFalse(hasattr(args, "class_2_relief"))
        self.assertEqual(
            args.source_runtime_equivalence.name,
            preflight.SOURCE_RUNTIME_EQUIVALENCE_FILENAME,
        )
        with self.assertRaises(SystemExit):
            preflight.parse_args(["--stay-bonus", "0.1"])

    def test_design_selection_loader_is_replayed_and_source_bound(self):
        record = selection_record()
        source = {"training_freeze_sha256": "source-sha"}
        path = Path("F:/selection.json")
        with mock.patch.object(
            preflight.design, "load_design_selection", return_value=record
        ) as loader:
            loaded = preflight.load_design_selection(path, source)
        loader.assert_called_once_with(path.resolve())
        self.assertEqual(
            loaded["selected_parameters"], record["selected_parameters"]
        )

        corrupted = dict(record, source_training_freeze_sha256="other")
        corrupted.pop("design_selection_sha256")
        corrupted["design_selection_sha256"] = (
            preflight.source_runner.sha256_json(corrupted)
        )
        with mock.patch.object(
            preflight.design, "load_design_selection", return_value=corrupted
        ), self.assertRaisesRegex(ValueError, "source freeze"):
            preflight.load_design_selection(path, source)

    def test_design_selection_tamper_is_rejected_even_after_loader(self):
        record = selection_record()
        record["selected_parameters"] = dict(
            record["selected_parameters"], stay_bonus=0.2
        )
        with mock.patch.object(
            preflight.design, "load_design_selection", return_value=record
        ), self.assertRaisesRegex(ValueError, "design_selection_sha256"):
            preflight.load_design_selection(
                Path("F:/selection.json"),
                {"training_freeze_sha256": "source-sha"},
            )

    def test_spec_binds_selection_source_code_and_feature_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "design_selection.json"
            selection_path.write_text("{}", encoding="utf-8")
            equivalence_path = root / "source_runtime_equivalence.json"
            equivalence = equivalence_artifact()
            equivalence_path.write_text(
                json.dumps(equivalence), encoding="utf-8"
            )
            args = SimpleNamespace(
                project=Path(__file__).resolve().parent,
                source=root / "source",
                selection=selection_path,
                source_runtime_equivalence=equivalence_path,
                output=root / "output",
                device="cpu",
            )
            source = {
                "training_freeze_sha256": "source-sha",
                "spec_sha256": "source-spec",
                "source_code_fingerprint": matching_runtime_fingerprints(),
                "checkpoints": {},
            }
            selection = selection_record()
            fingerprint = {
                "runner": "runner-sha",
                **matching_runtime_fingerprints(),
            }
            with mock.patch.object(
                preflight,
                "_runtime_code_fingerprint",
                return_value=fingerprint,
            ):
                spec = preflight.build_spec(
                    args, source, selection, equivalence
                )
        self.assertFalse(spec["paper_claim_allowed"])
        self.assertFalse(spec["promotion_decision_allowed"])
        self.assertEqual(spec["training_jobs"], 0)
        self.assertEqual(spec["expected_validation_rows"], 240)
        self.assertEqual(spec["expected_test_rows"], 600)
        self.assertEqual(
            spec["design_selection"]["design_selection_sha256"],
            selection["design_selection_sha256"],
        )
        self.assertEqual(spec["source_training_freeze_sha256"], "source-sha")
        self.assertEqual(
            spec["source_runtime_equivalence"][
                "source_runtime_equivalence_sha256"
            ],
            equivalence["source_runtime_equivalence_sha256"],
        )
        contract = spec["feature_contract"]
        self.assertEqual(contract["candidate_feature_dim"], 26)
        self.assertEqual(
            contract["candidate_feature_schema_id"],
            "leo_multi_candidate_features_v1_dim_26",
        )
        self.assertEqual(
            contract["feature_indexes"],
            {"route_switch": 17, "route_urgency": 20, "route_class_2": 23},
        )

    def test_equivalence_source_screen_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "equivalence.json"
            artifact = equivalence_artifact("other-source")
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-screen"):
                preflight._source_runtime_equivalence_record(
                    path,
                    artifact,
                    {"spec_sha256": "source-spec"},
                )

    def test_equivalence_replay_fails_before_source_or_output_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = SimpleNamespace(
                source=Path(directory) / "source",
                selection=Path(directory) / "selection.json",
                source_runtime_equivalence=(
                    Path(directory) / "source_runtime_equivalence.json"
                ),
                output=output,
                project=Path(directory) / "src",
                device="cpu",
                phase="validation",
                dry_run=False,
            )
            with (
                mock.patch.object(preflight, "parse_args", return_value=args),
                mock.patch.object(preflight, "validate_environment"),
                mock.patch.object(
                    preflight.source_runner,
                    "audit_source_screen",
                ) as source_audit,
                mock.patch.object(
                    preflight,
                    "load_source_runtime_equivalence",
                    side_effect=RuntimeError("equivalence drift"),
                ),
                mock.patch.object(preflight, "atomic_write_csv") as write_csv,
                mock.patch.object(preflight, "atomic_write_json") as write_json,
                mock.patch.object(
                    preflight.source_runner, "ensure_immutable_json"
                ) as write_immutable,
                mock.patch.object(Path, "mkdir") as mkdir,
                self.assertRaisesRegex(RuntimeError, "equivalence drift"),
            ):
                preflight.main()
            source_audit.assert_not_called()
            mkdir.assert_not_called()
            write_csv.assert_not_called()
            write_json.assert_not_called()
            write_immutable.assert_not_called()
            self.assertFalse(output.exists())

    def test_shard_metadata_binds_all_frozen_inputs(self):
        job = next(job for job in preflight.build_jobs("test") if job.arm == "shield")
        selection = selection_record()
        spec = {
            "spec_sha256": "spec",
            "code_fingerprint_sha256": "code",
            "source_runtime_equivalence": spec_equivalence_binding(),
            "source_training_freeze_sha256": "source-sha",
        }
        checkpoint = {
            "checkpoint_path": "F:/checkpoint.pt",
            "checkpoint_sha256": "checkpoint-sha",
            "candidate_feature_dim": 26,
        }
        metadata = preflight._expected_shard_metadata(
            job,
            spec,
            selection,
            checkpoint,
            (72001, 72002),
            "validation-freeze",
        )
        self.assertEqual(metadata["design_selection_sha256"], selection["design_selection_sha256"])
        self.assertEqual(metadata["source_training_freeze_sha256"], "source-sha")
        self.assertEqual(metadata["code_fingerprint_sha256"], "code")
        self.assertEqual(metadata["validation_freeze_sha256"], "validation-freeze")
        self.assertEqual(metadata["checkpoint_sha256"], "checkpoint-sha")
        self.assertEqual(
            metadata["policy_schema"]["candidate_feature_schema_sha256"],
            "be660bb34d6d8579773b643f2824e6dac5069fe67cfd8a8d71a36b1b147f9f70",
        )

    def test_gate_uses_micro_rate_not_episode_rate_macro(self):
        gates = preflight.compute_cell_gates(complete_rows(), (1, 2))
        self.assertEqual(len(gates), 8)
        first = gates[0]
        self.assertAlmostEqual(first["proposed_avoidable_switch_micro_rate"], 0.01)
        self.assertAlmostEqual(first["shield_avoidable_switch_micro_rate"], 0.02)
        self.assertFalse(first["avoidable_switch_micro_gate_pass"])
        self.assertFalse(first["all_gates_pass"])

    def test_existing_mismatched_shard_fails_closed_without_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = preflight.build_jobs("validation")[0]
            csv_path, metadata_path = preflight._evaluation_paths(output, job)
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("corrupt\n", encoding="utf-8")
            metadata_path.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(output=output, device="cpu")
            spec = {
                "spec_sha256": "spec",
                "code_fingerprint_sha256": "code",
                "source_runtime_equivalence": spec_equivalence_binding(),
                "source_training_freeze_sha256": "source-sha",
            }
            checkpoint = {
                "checkpoint_path": str(output / "checkpoint.pt"),
                "checkpoint_sha256": "checkpoint",
                "candidate_feature_dim": 26,
            }
            with mock.patch.object(
                preflight, "evaluate_policy"
            ) as evaluator, self.assertRaisesRegex(ValueError, "metadata mismatch"):
                preflight.evaluate_job(
                    args,
                    job,
                    spec,
                    checkpoint,
                    (71001,),
                    selection_record(),
                )
            evaluator.assert_not_called()

    def test_test_guard_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = SimpleNamespace(
                source=Path(directory) / "source",
                selection=Path(directory) / "selection.json",
                source_runtime_equivalence=(
                    Path(directory) / "source_runtime_equivalence.json"
                ),
                output=output,
                project=Path(__file__).resolve().parent,
                device="cpu",
                phase="test",
                dry_run=False,
            )
            with (
                mock.patch.object(preflight, "parse_args", return_value=args),
                mock.patch.object(preflight, "validate_environment"),
                mock.patch.object(
                    preflight,
                    "load_source_runtime_equivalence",
                    return_value=equivalence_artifact(),
                ),
                mock.patch.object(
                    preflight.source_runner,
                    "audit_source_screen",
                    return_value={"training_freeze_sha256": "source", "checkpoints": {}},
                ),
                mock.patch.object(
                    preflight,
                    "load_design_selection",
                    return_value=selection_record("source"),
                ),
                mock.patch.object(
                    preflight,
                    "build_spec",
                    return_value={
                        "spec_sha256": "spec",
                        "code_fingerprint_sha256": "code",
                    },
                ),
                mock.patch.object(
                    preflight,
                    "load_validation_freeze",
                    side_effect=RuntimeError("validation failed"),
                ),
                mock.patch.object(preflight, "atomic_write_csv") as write_csv,
                mock.patch.object(preflight, "atomic_write_json") as write_json,
                mock.patch.object(
                    preflight.source_runner, "ensure_immutable_json"
                ) as write_immutable,
                mock.patch.object(Path, "mkdir") as mkdir,
                self.assertRaisesRegex(RuntimeError, "validation failed"),
            ):
                preflight.main()
            mkdir.assert_not_called()
            write_csv.assert_not_called()
            write_json.assert_not_called()
            write_immutable.assert_not_called()
            self.assertFalse(output.exists())

    def test_manifest_indexes_both_phase_aggregate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names = (
                "adaptive_shield_preflight_spec.json",
                "validation_plan.csv",
                "test_plan.csv",
                "validation_episode_metrics.csv",
                "validation_aggregate_metrics.csv",
                "validation_cell_gates.csv",
                "validation_decision.json",
                "test_episode_metrics.csv",
                "test_aggregate_metrics.csv",
                "test_cell_gates.csv",
                "test_decision.json",
            )
            for name in names:
                (output / name).write_text("artifact\n", encoding="utf-8")
            spec = {
                "spec_sha256": "spec",
                "code_fingerprint_sha256": "code",
                "source_runtime_equivalence": spec_equivalence_binding(),
            }
            source = {"training_freeze_sha256": "source"}
            selection = selection_record("source")
            validation = {
                "validation_freeze_sha256": "validation",
                "all_cells_pass": True,
            }
            test = {"test_decision_sha256": "test", "all_cells_pass": True}
            with (
                mock.patch.object(preflight, "_shard_index", return_value=[]),
                mock.patch.object(
                    preflight.source_runner, "ensure_immutable_json"
                ),
            ):
                manifest = preflight.write_manifest(
                    output, spec, source, selection, validation, test
                )
        self.assertEqual(
            manifest["artifacts"]["validation_aggregate_metrics"]["row_count"],
            24,
        )
        self.assertEqual(
            manifest["artifacts"]["test_aggregate_metrics"]["row_count"], 24
        )

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            args = SimpleNamespace(
                source=Path(directory) / "source",
                selection=Path(directory) / "selection.json",
                source_runtime_equivalence=(
                    Path(directory) / "source_runtime_equivalence.json"
                ),
                output=output,
                project=Path(__file__).resolve().parent,
                device="cpu",
                phase="validation",
                dry_run=True,
            )
            source = {"training_freeze_sha256": "source", "checkpoints": {}}
            selection = selection_record("source")
            spec = {
                "spec_sha256": "spec",
                "code_fingerprint_sha256": "code",
                "source_runtime_equivalence": spec_equivalence_binding(),
            }
            with (
                mock.patch.object(preflight, "parse_args", return_value=args),
                mock.patch.object(preflight, "validate_environment"),
                mock.patch.object(
                    preflight,
                    "load_source_runtime_equivalence",
                    return_value=equivalence_artifact(),
                ),
                mock.patch.object(
                    preflight.source_runner,
                    "audit_source_screen",
                    return_value=source,
                ),
                mock.patch.object(
                    preflight, "load_design_selection", return_value=selection
                ),
                mock.patch.object(preflight, "build_spec", return_value=spec),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(preflight.main(), 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
