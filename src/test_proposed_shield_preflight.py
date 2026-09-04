from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mappo_evaluation import EpisodeMetrics
import run_proposed_shield_preflight as preflight


def episode(
    scenario: str,
    arm: str,
    policy_seed: int,
    workload_seed: int,
    *,
    delivery: float = 0.80,
    class_2: float = 0.80,
    avoidable: int = 2,
    opportunities: int = 10,
    switches: int = 20,
    forced: int | None = None,
) -> EpisodeMetrics:
    if forced is None:
        forced = switches - avoidable
    return EpisodeMetrics(
        scenario=scenario,
        policy=preflight.POLICY_NAMES[arm],
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
        class_0_delivery_ratio=0.8,
        class_1_delivery_ratio=0.8,
        class_2_delivery_ratio=class_2,
        avoidable_routing_switches=avoidable,
        forced_routing_switches=forced,
        switch_opportunities=opportunities,
        avoidable_switch_rate=avoidable / max(1, opportunities),
    )


def passing_rows(workloads: tuple[int, ...] = (1, 2)) -> list[EpisodeMetrics]:
    rows: list[EpisodeMetrics] = []
    for scenario in preflight.SCENARIOS:
        for policy_seed in preflight.POLICY_SEEDS:
            for workload_seed in workloads:
                rows.append(
                    episode(
                        scenario,
                        "proposed",
                        policy_seed,
                        workload_seed,
                        avoidable=3,
                        switches=21,
                    )
                )
                rows.append(
                    episode(
                        scenario,
                        "raw_context",
                        policy_seed,
                        workload_seed,
                        avoidable=4,
                        switches=25,
                    )
                )
                rows.append(
                    episode(
                        scenario,
                        "shield",
                        policy_seed,
                        workload_seed,
                        delivery=0.799,
                        class_2=0.795,
                        avoidable=1,
                        switches=19,
                    )
                )
    return rows


def build_validation_fixture(
    output: Path,
) -> tuple[SimpleNamespace, dict, dict, list[EpisodeMetrics]]:
    args = SimpleNamespace(output=output, device="cpu")
    spec = {"spec_sha256": "b" * 64}
    source: dict = {
        "training_freeze_sha256": "a" * 64,
        "checkpoints": {},
    }
    all_rows: list[EpisodeMetrics] = []
    for job in preflight.build_jobs("validation"):
        source["checkpoints"].setdefault(
            job.source_job_id,
            {
                "checkpoint_path": f"F:/frozen/{job.source_job_id}/validation_best.pt",
                "checkpoint_sha256": "c" * 64,
                "candidate_feature_dim": (
                    26 if job.source_variant == "proposed" else 28
                ),
            },
        )
        job_rows: list[EpisodeMetrics] = []
        for workload_seed in preflight.VALIDATION_WORKLOAD_SEEDS:
            kwargs = {
                "proposed": {"avoidable": 3, "switches": 21},
                "raw_context": {"avoidable": 4, "switches": 25},
                "shield": {
                    "delivery": 0.799,
                    "class_2": 0.795,
                    "avoidable": 1,
                    "switches": 19,
                },
            }[job.arm]
            job_rows.append(
                episode(
                    job.scenario,
                    job.arm,
                    job.policy_seed,
                    workload_seed,
                    **kwargs,
                )
            )
        all_rows.extend(job_rows)
        csv_path, metadata_path = preflight._evaluation_paths(output, job)
        preflight.atomic_write_csv(
            csv_path, (row.__dict__ for row in job_rows)
        )
        metadata = {
            **preflight._expected_shard_metadata(
                job,
                spec["spec_sha256"],
                source["checkpoints"][job.source_job_id],
                preflight.VALIDATION_WORKLOAD_SEEDS,
                None,
            ),
            "csv_sha256": preflight.sha256_file(csv_path),
            "row_count": len(job_rows),
            "policy_diagnostics": {},
        }
        preflight.atomic_write_json(metadata_path, metadata)

    gates = preflight.compute_cell_gates(
        all_rows, preflight.VALIDATION_WORKLOAD_SEEDS
    )
    paths = preflight._phase_artifact_paths(output, "validation")
    preflight.atomic_write_csv(
        paths["episode_metrics"], (row.__dict__ for row in all_rows)
    )
    preflight.atomic_write_csv(
        paths["aggregate_metrics"], ({"fixture": "aggregate"},)
    )
    preflight.atomic_write_csv(paths["cell_gates"], gates)
    artifacts = {
        name: {
            "path": str(paths[name].resolve()),
            "sha256": preflight.sha256_file(paths[name]),
        }
        for name in ("episode_metrics", "aggregate_metrics", "cell_gates")
    }
    artifacts["episode_metrics"]["row_count"] = len(all_rows)
    artifacts["aggregate_metrics"]["row_count"] = 1
    artifacts["cell_gates"]["row_count"] = len(gates)
    freeze = {
        "schema_version": preflight.SCHEMA_VERSION,
        "screen_name": preflight.SCREEN_NAME,
        "phase": "validation",
        "spec_sha256": spec["spec_sha256"],
        "source_training_freeze_sha256": source[
            "training_freeze_sha256"
        ],
        "all_cells_pass": True,
        "test_allowed": True,
        "cell_gates": gates,
        "evaluation_shards": preflight._shard_index(
            output, preflight.build_jobs("validation")
        ),
        "artifacts": artifacts,
    }
    freeze["validation_freeze_sha256"] = preflight.v1.sha256_json(freeze)
    preflight.atomic_write_json(paths["decision"], freeze)
    return args, spec, source, all_rows


class ProposedShieldPreflightTests(unittest.TestCase):
    def test_frozen_grid_and_fresh_panels(self):
        self.assertEqual(preflight.STAY_BONUS, 0.40)
        self.assertEqual(
            preflight.POLICY_SEEDS,
            (1710210210, 2078783072, 1047581915, 1245825580),
        )
        self.assertEqual(
            preflight.VALIDATION_WORKLOAD_SEEDS, tuple(range(60001, 60011))
        )
        self.assertEqual(
            preflight.TEST_WORKLOAD_SEEDS, tuple(range(70001, 70026))
        )
        self.assertEqual(len(preflight.build_jobs("validation")), 24)
        self.assertEqual(len(preflight.build_jobs("test")), 24)
        exposed = set().union(
            *(
                set(range(start, stop + 1))
                for start, stop in preflight.KNOWN_EXPOSED_PANELS
            )
        )
        self.assertFalse(set(preflight.VALIDATION_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(set(preflight.TEST_WORKLOAD_SEEDS) & exposed)
        self.assertIn((61001, 61004), preflight.KNOWN_EXPOSED_PANELS)

    def test_shield_is_bound_to_the_proposed_checkpoint(self):
        for phase in ("validation", "test"):
            for job in preflight.build_jobs(phase):
                if job.arm in {"proposed", "shield"}:
                    self.assertEqual(job.source_variant, "proposed")
                    self.assertIn("/proposed/", f"/{job.source_job_id}/")
                else:
                    self.assertEqual(
                        job.source_variant, "with_congestion_context"
                    )
                expected_bonus = 0.40 if job.arm == "shield" else 0.0
                self.assertEqual(job.stay_bonus, expected_bonus)

    def test_shard_metadata_uses_v7_contract_and_test_freeze_binding(self):
        job = next(
            job for job in preflight.build_jobs("test") if job.arm == "shield"
        )
        checkpoint = {
            "checkpoint_path": "F:/frozen/validation_best.pt",
            "checkpoint_sha256": "a" * 64,
            "candidate_feature_dim": 26,
        }
        metadata = preflight._expected_shard_metadata(
            job, "b" * 64, checkpoint, (70001, 70002), "c" * 64
        )
        self.assertEqual(metadata["screen_name"], preflight.SCREEN_NAME)
        self.assertEqual(metadata["schema_version"], preflight.SCHEMA_VERSION)
        self.assertEqual(metadata["validation_freeze_sha256"], "c" * 64)
        self.assertEqual(metadata["policy_schema"]["variant"], "proposed")
        self.assertEqual(metadata["policy_schema"]["stay_bonus"], 0.40)

    def test_all_eight_cells_pass_the_frozen_gates(self):
        rows = passing_rows()
        gates = preflight.compute_cell_gates(rows, (1, 2))
        self.assertEqual(len(gates), 8)
        self.assertTrue(all(row["all_gates_pass"] for row in gates))
        self.assertTrue(
            all(
                row["shield_avoidable_switch_micro_rate"]
                < row["proposed_avoidable_switch_micro_rate"]
                for row in gates
            )
        )

    def test_total_switches_must_not_exceed_either_reference(self):
        rows = passing_rows()
        scenario = preflight.SCENARIOS[0]
        seed = preflight.POLICY_SEEDS[0]
        for row in rows:
            if row.scenario != scenario or row.policy_seed != seed:
                continue
            if row.policy == preflight.POLICY_NAMES["proposed"]:
                row.routing_switches = 10
                row.forced_routing_switches = (
                    10 - row.avoidable_routing_switches
                )
            elif row.policy == preflight.POLICY_NAMES["raw_context"]:
                row.routing_switches = 20
                row.forced_routing_switches = (
                    20 - row.avoidable_routing_switches
                )
            elif row.policy == preflight.POLICY_NAMES["shield"]:
                row.routing_switches = 15
                row.forced_routing_switches = (
                    15 - row.avoidable_routing_switches
                )
        gate = next(
            row
            for row in preflight.compute_cell_gates(rows, (1, 2))
            if row["scenario"] == scenario and row["policy_seed"] == seed
        )
        self.assertFalse(gate["routing_switch_vs_proposed_gate_pass"])
        self.assertTrue(gate["routing_switch_vs_raw_context_gate_pass"])
        self.assertFalse(gate["all_gates_pass"])

    def test_micro_gate_does_not_use_macro_episode_rate(self):
        rows = passing_rows()
        scenario = preflight.SCENARIOS[0]
        seed = preflight.POLICY_SEEDS[0]
        rows = [
            row
            for row in rows
            if not (
                row.scenario == scenario
                and row.policy_seed == seed
                and row.policy
                in {
                    preflight.POLICY_NAMES["proposed"],
                    preflight.POLICY_NAMES["shield"],
                }
            )
        ]
        # Proposed macro rate is 0.45 but micro rate is 0.009. Shield macro
        # rate is 0.10 but micro rate is 0.10, so only the exact micro gate fails.
        rows.extend(
            (
                episode(
                    scenario,
                    "proposed",
                    seed,
                    1,
                    avoidable=9,
                    opportunities=10,
                    switches=21,
                ),
                episode(
                    scenario,
                    "proposed",
                    seed,
                    2,
                    avoidable=0,
                    opportunities=990,
                    switches=21,
                ),
                episode(
                    scenario,
                    "shield",
                    seed,
                    1,
                    delivery=0.799,
                    class_2=0.795,
                    avoidable=1,
                    opportunities=10,
                    switches=19,
                ),
                episode(
                    scenario,
                    "shield",
                    seed,
                    2,
                    delivery=0.799,
                    class_2=0.795,
                    avoidable=1,
                    opportunities=10,
                    switches=19,
                ),
            )
        )
        gate = next(
            row
            for row in preflight.compute_cell_gates(rows, (1, 2))
            if row["scenario"] == scenario and row["policy_seed"] == seed
        )
        self.assertAlmostEqual(
            gate["proposed_avoidable_switch_micro_rate"], 9 / 1000
        )
        self.assertAlmostEqual(
            gate["shield_avoidable_switch_micro_rate"], 2 / 20
        )
        self.assertFalse(gate["avoidable_switch_micro_gate_pass"])
        self.assertFalse(gate["all_gates_pass"])

    def test_episode_rate_inconsistency_is_rejected(self):
        row = episode(
            "medium_load", "shield", preflight.POLICY_SEEDS[0], 1
        )
        row.avoidable_switch_rate = 0.0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            preflight.validate_switch_rows([row])

    def test_route_switch_partition_inconsistency_is_rejected(self):
        row = episode(
            "medium_load", "shield", preflight.POLICY_SEEDS[0], 1
        )
        row.forced_routing_switches += 1
        with self.assertRaisesRegex(ValueError, "accounting"):
            preflight.validate_switch_rows([row])

    def test_valid_evaluation_shard_is_resumed_without_policy_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(output=output, device="cpu")
            job = preflight.build_jobs("test")[0]
            workloads = (70001, 70002)
            checkpoint = {
                "checkpoint_path": "F:/frozen/validation_best.pt",
                "checkpoint_sha256": "a" * 64,
                "candidate_feature_dim": 26,
            }
            rows = [
                episode(
                    job.scenario,
                    job.arm,
                    job.policy_seed,
                    workload_seed,
                )
                for workload_seed in workloads
            ]
            csv_path, metadata_path = preflight._evaluation_paths(output, job)
            preflight.atomic_write_csv(
                csv_path, (row.__dict__ for row in rows)
            )
            freeze_sha = "c" * 64
            metadata = {
                **preflight._expected_shard_metadata(
                    job,
                    "b" * 64,
                    checkpoint,
                    workloads,
                    freeze_sha,
                ),
                "csv_sha256": preflight.sha256_file(csv_path),
                "row_count": len(rows),
                "policy_diagnostics": {},
            }
            preflight.atomic_write_json(metadata_path, metadata)
            with (
                mock.patch.object(preflight, "load_hysteresis_policy") as load,
                mock.patch.object(preflight, "evaluate_policy") as evaluate,
            ):
                observed = preflight.evaluate_job(
                    args,
                    job,
                    "b" * 64,
                    checkpoint,
                    workloads,
                    validation_freeze_sha256=freeze_sha,
                )
            self.assertEqual(observed, rows)
            load.assert_not_called()
            evaluate.assert_not_called()

    def test_validation_freeze_replay_rejects_all_material_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args, spec, source, _ = build_validation_fixture(output)
            loaded = preflight.load_validation_freeze(args, spec, source)
            self.assertTrue(loaded["all_cells_pass"])

            job = preflight.build_jobs("validation")[0]
            _, metadata_path = preflight._evaluation_paths(output, job)
            original_metadata = preflight.v1._load_json(metadata_path)
            tampered_metadata = dict(original_metadata)
            tampered_metadata["policy_diagnostics"] = {"tampered": True}
            preflight.atomic_write_json(metadata_path, tampered_metadata)
            with self.assertRaisesRegex(ValueError, "shard index does not replay"):
                preflight.load_validation_freeze(args, spec, source)
            preflight.atomic_write_json(metadata_path, original_metadata)

            paths = preflight._phase_artifact_paths(output, "validation")
            original_episode_bytes = paths["episode_metrics"].read_bytes()
            paths["episode_metrics"].write_bytes(
                original_episode_bytes + b"tampered\n"
            )
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                preflight.load_validation_freeze(args, spec, source)
            paths["episode_metrics"].write_bytes(original_episode_bytes)

            original_freeze = preflight.v1._load_json(paths["decision"])
            bad_self_hash = dict(original_freeze)
            bad_self_hash["test_allowed"] = False
            preflight.atomic_write_json(paths["decision"], bad_self_hash)
            with self.assertRaisesRegex(ValueError, "validation_freeze_sha256"):
                preflight.load_validation_freeze(args, spec, source)
            preflight.atomic_write_json(paths["decision"], original_freeze)

            forged_gate = json.loads(json.dumps(original_freeze))
            forged_gate["cell_gates"][0][
                "shield_delivery_difference_vs_proposed"
            ] = 1.0
            forged_gate.pop("validation_freeze_sha256")
            forged_gate["validation_freeze_sha256"] = preflight.v1.sha256_json(
                forged_gate
            )
            preflight.atomic_write_json(paths["decision"], forged_gate)
            with self.assertRaisesRegex(ValueError, "does not replay"):
                preflight.load_validation_freeze(args, spec, source)

    def test_spec_is_explicitly_diagnostic(self):
        args = SimpleNamespace(
            project=Path(__file__).resolve().parent,
            source=Path("F:/source"),
            output=Path("F:/output"),
            device="cuda",
        )
        source = {"training_freeze_sha256": "a" * 64, "checkpoints": {}}
        with mock.patch.object(
            preflight, "_runtime_code_fingerprint", return_value={"runner": "hash"}
        ):
            spec = preflight.build_spec(args, source)
        self.assertEqual(spec["training_jobs"], 0)
        self.assertEqual(spec["device"], "cuda")
        self.assertFalse(spec["confirmatory"])
        self.assertFalse(spec["paper_claim_allowed"])
        self.assertFalse(spec["promotion_decision_allowed"])
        self.assertEqual(spec["expected_validation_rows"], 240)
        self.assertEqual(spec["expected_test_rows"], 600)
        self.assertFalse(spec["beta_provenance"]["validation_used_for_tuning"])
        self.assertFalse(
            spec["beta_provenance"][
                "immutable_beta_0p40_selection_artifact_available"
            ]
        )
        self.assertTrue(
            spec["test_isolation"][
                "test_requires_replayed_passing_validation_freeze"
            ]
        )

    def test_device_is_bound_into_the_spec_hash(self):
        base = {
            "project": Path(__file__).resolve().parent,
            "source": Path("F:/source"),
            "output": Path("F:/output"),
        }
        source = {"training_freeze_sha256": "a" * 64, "checkpoints": {}}
        with mock.patch.object(
            preflight, "_runtime_code_fingerprint", return_value={"runner": "hash"}
        ):
            cpu = preflight.build_spec(
                SimpleNamespace(**base, device="cpu"), source
            )
            cuda = preflight.build_spec(
                SimpleNamespace(**base, device="cuda"), source
            )
        self.assertNotEqual(cpu["spec_sha256"], cuda["spec_sha256"])

    def test_dry_run_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must-not-exist"
            source = {
                "training_freeze_sha256": "a" * 64,
                "checkpoints": {},
            }
            spec = {
                "spec_sha256": "b" * 64,
                "inferential_status": "diagnostic_exploratory_preflight",
            }
            stdout = io.StringIO()
            with (
                mock.patch.object(preflight, "validate_environment"),
                mock.patch.object(
                    preflight.v1, "audit_source_screen", return_value=source
                ),
                mock.patch.object(preflight, "build_spec", return_value=spec),
                redirect_stdout(stdout),
            ):
                status = preflight.main(
                    [
                        "--dry-run",
                        "--device",
                        "cpu",
                        "--source",
                        str(root / "source"),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())
            self.assertIn('"dry_run_writes_output": false', stdout.getvalue())

    def test_test_phase_fails_before_output_or_evaluation_without_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must-not-exist"
            source = {
                "training_freeze_sha256": "a" * 64,
                "checkpoints": {},
            }
            spec = {"spec_sha256": "b" * 64}
            with (
                mock.patch.object(preflight, "validate_environment"),
                mock.patch.object(
                    preflight.v1, "audit_source_screen", return_value=source
                ),
                mock.patch.object(preflight, "build_spec", return_value=spec),
                mock.patch.object(
                    preflight,
                    "load_validation_freeze",
                    side_effect=RuntimeError("validation freeze missing"),
                ),
                mock.patch.object(preflight, "evaluate_jobs") as evaluate,
            ):
                with self.assertRaisesRegex(RuntimeError, "freeze missing"):
                    preflight.main(
                        [
                            "--phase",
                            "test",
                            "--device",
                            "cpu",
                            "--source",
                            str(root / "source"),
                            "--output",
                            str(output),
                        ]
                    )
            evaluate.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
