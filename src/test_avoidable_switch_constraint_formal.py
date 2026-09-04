from __future__ import annotations

from copy import deepcopy
from concurrent.futures import Future
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import torch

from ablation_matrix_runner import atomic_write_csv, atomic_write_json
from mappo_evaluation import ConstraintEpisodeMetrics
import run_avoidable_switch_constraint_formal as formal


class FrozenRegistryTests(unittest.TestCase):
    def test_policy_seed_derivation_and_complete_paired_grid(self):
        self.assertEqual(formal.POLICY_SEEDS, formal.EXPECTED_POLICY_SEEDS)
        self.assertEqual(len(set(formal.POLICY_SEEDS)), 8)
        formal.validate_seed_registry()
        jobs = formal.build_jobs()
        self.assertEqual(len(jobs), 48)
        self.assertEqual(formal.EXPECTED_VALIDATION_ROWS, 960)
        for offset in range(0, len(jobs), len(formal.ARMS)):
            group = jobs[offset : offset + len(formal.ARMS)]
            self.assertEqual(tuple(job.arm for job in group), formal.ARMS)
            self.assertEqual(len({job.scenario for job in group}), 1)
            self.assertEqual(len({job.policy_seed for job in group}), 1)

    def test_workload_panels_are_disjoint_and_validation_guard_is_exact(self):
        panels = (
            set(formal.TRAIN_WORKLOAD_SEEDS),
            set(formal.SELECTION_WORKLOAD_SEEDS),
            set(formal.GATE_WORKLOAD_SEEDS),
            set(formal.SEALED_TEST_WORKLOAD_SEEDS),
        )
        for index, left in enumerate(panels):
            for right in panels[index + 1 :]:
                self.assertFalse(left & right)
        for panel in formal.VALIDATION_PANELS:
            workloads = formal.PANEL_WORKLOAD_SEEDS[panel]
            self.assertEqual(
                formal.validate_validation_workloads(workloads, panel), workloads
            )
            with self.assertRaisesRegex(formal.ConfigurationError, "is not"):
                formal.validate_validation_workloads(workloads[:-1], panel)
        with self.assertRaises(formal.ConfigurationError):
            formal.validate_validation_workloads(
                formal.GATE_WORKLOAD_SEEDS,
                formal.PANEL_SELECTION_RECHECK,
            )
        substituted = list(formal.GATE_WORKLOAD_SEEDS)
        substituted[-1] = formal.SEALED_TEST_WORKLOAD_SEEDS[0]
        with self.assertRaises(formal.ConfigurationError):
            formal.validate_validation_workloads(
                substituted, formal.PANEL_INDEPENDENT_GATE
            )

    @staticmethod
    def args(root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            output=root / "experiments" / "formal-output",
            cleanmarl=Path("F:/cleanmarl"),
            project=Path("F:/leo-routing-preliminary-matlab/src"),
            device="cuda",
            evaluation_device="cuda",
            max_parallel=2,
        )

    def test_each_arm_command_has_only_its_frozen_treatment(self):
        args = self.args(Path("F:/leo-routing-preliminary-matlab"))

        def option(command: list[str], name: str) -> str:
            return command[command.index(name) + 1]

        for arm in formal.ARMS:
            job = formal.FormalJob(0, formal.SCENARIOS[0], arm, formal.POLICY_SEEDS[0])
            command = formal.build_command(job, args, Path("F:/formal-job"))
            self.assertEqual(option(command, "--total-timesteps"), "50000")
            self.assertEqual(option(command, "--train-seed-start"), "76001")
            self.assertEqual(option(command, "--train-seed-count"), "200")
            self.assertEqual(option(command, "--validation-seed-start"), "77001")
            self.assertEqual(option(command, "--num-eval-ep"), "10")
            self.assertEqual(option(command, "--eval-steps"), "40")
            self.assertEqual(option(command, "--leo-variant"), job.environment_variant)
            self.assertEqual(
                "--avoidable-switch-constraint-enabled" in command,
                job.constrained,
            )
            self.assertEqual(option(command, "--validation-selection-mode"), job.selection_mode)
            rendered_values = {
                int(value)
                for value in command
                if value.isdigit() and len(value) == 5
            }
            self.assertFalse(rendered_values & set(formal.SEALED_TEST_WORKLOAD_SEEDS))

    def test_runtime_requires_same_cuda_device_for_training_and_validation(self):
        args = self.args(Path("F:/leo-routing-preliminary-matlab"))
        args.evaluation_device = "cpu"
        with self.assertRaisesRegex(formal.ConfigurationError, "same frozen CUDA"):
            formal.validate_runtime_environment(args)
        self.assertEqual(formal.parse_args([]).evaluation_device, "cuda")

    def test_runtime_rejects_project_source_substitution_before_python_check(self):
        args = self.args(Path("F:/leo-routing-preliminary-matlab"))
        with TemporaryDirectory() as directory:
            substituted = Path(directory) / "src"
            substituted.mkdir()
            args.project = substituted
            with patch.dict(formal.os.environ, {"LEO_REWARD_OVERRIDES": ""}):
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "source directory importing"
                ):
                    formal.validate_runtime_environment(args)

    def test_fingerprint_binds_runtime_helpers_protocol_and_dependencies(self):
        args = self.args(Path("F:/leo-routing-preliminary-matlab"))
        dependency_output = "Torch==2.0\r\npip==25.0\nTorch==2.0\n"
        with patch.object(
            formal.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=dependency_output),
        ), patch.object(formal, "sha256_file", return_value="file-hash"), patch.object(
            formal, "_git_output", side_effect=["dirty-diff", "git-head"]
        ), patch.object(formal.torch.cuda, "is_available", return_value=False):
            fingerprint = formal.code_fingerprint(args)
        self.assertIn("artifact_io_dependency", fingerprint["files"])
        self.assertIn("classical_baselines_protocol", fingerprint["files"])
        self.assertIn(
            "preflight_test_test_avoidable_switch_constraint_formal",
            fingerprint["files"],
        )
        self.assertEqual(
            fingerprint["python_dependency_inventory"],
            ["pip==25.0", "Torch==2.0"],
        )
        expected_bytes = b"pip==25.0\nTorch==2.0\n"
        self.assertEqual(
            fingerprint["python_dependency_inventory_sha256"],
            hashlib.sha256(expected_bytes).hexdigest(),
        )

    def test_exact_resume_proof_binding_rejects_current_source_drift(self):
        digest = "a" * 64
        current_files = {
            current_name: digest
            for current_name in formal.EXACT_RESUME_RUNTIME_FILE_BINDINGS
        }
        smoke_files = {
            smoke_name: digest
            for smoke_name in formal.EXACT_RESUME_RUNTIME_FILE_BINDINGS.values()
        }
        runtime = {
            "git_head": "head",
            "python": "3.13.7",
            "torch": "2.11.0+cu128",
            "platform": "windows",
        }
        current = {"files": current_files, **runtime}
        smoke_spec = {
            "code_fingerprint": {"files": smoke_files, **runtime}
        }
        binding = formal.validate_exact_resume_runtime_binding(
            smoke_spec, current
        )
        self.assertEqual(
            binding["status"], "passed_for_fields_recorded_by_smoke_v1_r1"
        )
        self.assertIn(
            "numpy", binding["current_fields_not_recorded_by_smoke_v1_r1"]
        )
        drifted = deepcopy(current)
        drifted["files"]["trainer_snapshot"] = "b" * 64
        with self.assertRaisesRegex(formal.ConfigurationError, "trainer_snapshot"):
            formal.validate_exact_resume_runtime_binding(smoke_spec, drifted)

    def test_output_isolation_rejects_smoke_source_and_non_dedicated_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "docs").mkdir()
            (root / "experiments").mkdir()
            args = SimpleNamespace(
                output=root / "experiments" / "formal-v1",
                project=root / "src",
                cleanmarl=root / "cleanmarl",
            )
            formal.validate_output_isolation(args)
            for output in (
                root / "experiments",
                root / "src",
                root / "experiments" / formal.SMOKE_EVIDENCE_DIRECTORY_NAME,
                root / "outside" / "formal-v1",
            ):
                args.output = output
                with self.assertRaises(formal.ConfigurationError):
                    formal.validate_output_isolation(args)

    def test_nonempty_output_without_preregistration_is_rejected(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "foreign.txt").write_text("foreign", encoding="utf-8")
            with self.assertRaisesRegex(formal.ConfigurationError, "no immutable"):
                formal.validate_new_output_state(output)

    def test_spec_is_validation_only_and_never_authorizes_test(self):
        args = self.args(Path("F:/leo-routing-preliminary-matlab"))
        smoke_evidence = {
            "directory": "smoke-v1-r1",
            "files": {},
            "exact_resume_proof_sha256": "proof",
        }
        fingerprint = {
            "files": {
                "runner": "hash",
                **{
                    f"preflight_test_{module}": "test-hash"
                    for module in formal.PREFLIGHT_TEST_MODULES
                },
            }
        }
        with patch.object(formal, "validate_smoke_prerequisite", return_value=smoke_evidence), patch.object(
            formal, "code_fingerprint", return_value=fingerprint
        ):
            spec = formal.build_spec(args)
        formal.validate_self_hash(spec, "spec_sha256")
        self.assertEqual(spec["expected_training_jobs"], 48)
        self.assertEqual(spec["expected_validation_rows"], 960)
        self.assertEqual(spec["expected_test_evaluations"], 0)
        self.assertFalse(spec["sealed_test_access_authorized"])
        self.assertFalse(spec["paper_claim_allowed"])
        self.assertEqual(
            spec["preflight_tests"]["modules"],
            list(formal.PREFLIGHT_TEST_MODULES),
        )
        self.assertFalse(
            spec["preflight_tests"]["discovery_or_globbing_allowed"]
        )

    def test_preflight_uses_only_hash_bound_allowlist_and_fails_on_drift(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "src"
            project.mkdir()
            (project / "test_not_allowlisted.py").write_text(
                "raise AssertionError('must not be imported')\n", encoding="utf-8"
            )
            args = SimpleNamespace(output=root / "output", project=project)
            test_hashes = {
                module: "a" * 64 for module in formal.PREFLIGHT_TEST_MODULES
            }
            spec = {
                "spec_sha256": "spec",
                "code_fingerprint": {
                    "files": {
                        f"preflight_test_{module}": digest
                        for module, digest in test_hashes.items()
                    }
                },
                "preflight_tests": {
                    "modules": list(formal.PREFLIGHT_TEST_MODULES),
                    "expected_test_count": formal.EXPECTED_PREFLIGHT_TEST_COUNT,
                    "file_sha256": test_hashes,
                    "discovery_or_globbing_allowed": False,
                    "formal_or_sealed_workload_instantiation_allowed": False,
                },
                "smoke_prerequisite": {
                    "exact_resume_proof_sha256": "proof"
                },
            }
            output = (
                f"Ran {formal.EXPECTED_PREFLIGHT_TEST_COUNT} tests in 1.000s\n\nOK\n"
            )
            with patch.object(
                formal, "assert_runtime_fingerprint"
            ) as fingerprint, patch.object(
                formal.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=output),
            ) as run:
                manifest = formal.run_preflight_tests(args, spec)
            command = run.call_args.args[0]
            self.assertNotIn("discover", command)
            self.assertFalse(any("*" in value for value in command))
            self.assertNotIn("test_not_allowlisted", command)
            self.assertEqual(
                command[-len(formal.PREFLIGHT_TEST_MODULES) :],
                list(formal.PREFLIGHT_TEST_MODULES),
            )
            self.assertEqual(fingerprint.call_count, 2)
            self.assertEqual(
                manifest["test_file_sha256"], test_hashes
            )

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory) / "output", project=Path(directory)
            )
            with patch.object(
                formal,
                "assert_runtime_fingerprint",
                side_effect=formal.ConfigurationError("test hash drift"),
            ), patch.object(formal.subprocess, "run") as run:
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "test hash drift"
                ):
                    formal.run_preflight_tests(args, spec)
            run.assert_not_called()


class ResumeProvenanceTests(unittest.TestCase):
    @staticmethod
    def training_attempt(status: str = "running") -> dict:
        timestamp = "2026-09-03T00:00:00Z"
        attempt = {
            "attempt": 1,
            "started_at_utc": timestamp,
            "status": status,
            "resume_checkpoint": None,
            "resume_checkpoint_sha256": None,
            "command": ["python", "trainer.py"],
            "working_directory": "F:/formal-job/runtime_working_directory",
            "log_path": "F:/formal-job/runner_attempt_1.log",
        }
        if status == "completed":
            attempt["finished_at_utc"] = timestamp
        return attempt

    def test_resume_source_is_frozen_and_tamper_fails_closed(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            latest = run / "latest.pt"
            torch.save(
                {"resume_boundary": "completed_update", "step": 120}, latest
            )
            frozen, source_hash = formal.freeze_resume_source(latest, run, 2)
            self.assertEqual(formal.sha256_file(frozen), source_hash)
            frozen.write_bytes(b"tampered")
            with self.assertRaisesRegex(formal.ConfigurationError, "already differs"):
                formal.freeze_resume_source(latest, run, 2)

    def test_non_boundary_resume_is_rejected(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            latest = run / "latest.pt"
            torch.save({"resume_boundary": "inference_snapshot_only"}, latest)
            with self.assertRaisesRegex(formal.ConfigurationError, "completed-update"):
                formal.freeze_resume_source(latest, run, 1)

    def test_third_training_launch_is_rejected_before_resume_copy_or_write(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            status = {
                "schema_version": formal.SCHEMA_VERSION,
                "study_name": formal.STUDY_NAME,
                "spec_sha256": "spec",
                "phase": "training",
                "job": job.as_dict(),
                "status": "failed",
                "started_at_utc": "2026-09-03T00:00:02Z",
                "child_pid": None,
                "attempts": [
                    {
                        "attempt": number,
                        "started_at_utc": f"2026-09-03T00:00:0{number}Z",
                        "status": "failed",
                        "resume_checkpoint": None,
                        "resume_checkpoint_sha256": None,
                        "command": ["python", "trainer.py"],
                        "working_directory": str(Path(directory).resolve()),
                        "log_path": str(
                            (Path(directory) / f"attempt_{number}.log").resolve()
                        ),
                        "finished_at_utc": f"2026-09-03T00:00:0{number}Z",
                        "error": "RuntimeError('failed')",
                        "failure_category": "recognized_infrastructure_failure",
                        "retry_authorized": True,
                    }
                    for number in (1, 2)
                ],
            }
            formal.write_training_status(args, job, status)
            with patch.object(
                formal,
                "discover_run_directory",
                return_value=(Path(directory) / "run", Path(directory) / "latest.pt"),
            ), patch.object(formal, "freeze_resume_source") as freeze_source, patch.object(
                formal, "write_training_status", wraps=formal.write_training_status
            ) as write_status:
                with self.assertRaisesRegex(formal.ConfigurationError, "retry limit"):
                    formal.train_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                    )
            freeze_source.assert_not_called()
            write_status.assert_not_called()

    def test_retry_classification_and_peer_cancellation_are_fail_closed(self):
        self.assertEqual(
            formal._retry_classification(TimeoutError("temporary timeout")),
            ("recognized_infrastructure_failure", True),
        )
        self.assertEqual(
            formal._retry_classification(ValueError("bad scientific state")),
            ("scientific_or_runtime_failure", False),
        )
        self.assertEqual(
            formal._retry_classification(formal.ConfigurationError("drift")),
            ("configuration_or_audit_failure", False),
        )

        infrastructure = threading.Event()
        setattr(infrastructure, "failure_cause", TimeoutError("peer timeout"))
        cancelled = formal._cancelled(
            formal.TrainingCancelled, "peer cancelled", infrastructure
        )
        self.assertEqual(
            formal._retry_classification(cancelled),
            ("recognized_infrastructure_cancellation", True),
        )

        scientific = threading.Event()
        setattr(scientific, "failure_cause", RuntimeError("peer failed"))
        cancelled = formal._cancelled(
            formal.EvaluationCancelled, "peer cancelled", scientific
        )
        self.assertEqual(
            formal._retry_classification(cancelled),
            ("execution_interruption", False),
        )
        formal._require_retry_authorization(
            {
                "attempts": [
                    {
                        "status": "failed",
                        "failure_category": "recognized_infrastructure_failure",
                        "retry_authorized": True,
                    }
                ]
            },
            "test job",
        )

    def test_non_infrastructure_training_failure_cannot_retry_on_reentry(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        timestamp = "2026-09-03T00:00:00Z"
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            formal.write_training_status(
                args,
                job,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "phase": "training",
                    "job": job.as_dict(),
                    "status": "failed",
                    "attempts": [
                        {
                            "attempt": 1,
                            "started_at_utc": timestamp,
                            "status": "failed",
                            "resume_checkpoint": None,
                            "resume_checkpoint_sha256": None,
                            "command": ["python", "trainer.py"],
                            "working_directory": str(Path(directory).resolve()),
                            "log_path": str((Path(directory) / "attempt.log").resolve()),
                            "finished_at_utc": timestamp,
                            "error": "RuntimeError('scientific failure')",
                            "failure_category": "scientific_or_runtime_failure",
                            "retry_authorized": False,
                        }
                    ],
                    "started_at_utc": timestamp,
                    "child_pid": None,
                },
            )
            with patch.object(
                formal, "discover_run_directory", return_value=(None, None)
            ), patch.object(formal, "assert_runtime_fingerprint") as fingerprint, patch.object(
                formal, "write_training_status", wraps=formal.write_training_status
            ) as write_status:
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "not infrastructure-retry eligible"
                ):
                    formal.train_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                    )
            fingerprint.assert_not_called()
            write_status.assert_not_called()

    def test_orphaned_training_attempt_is_recorded_as_retryable(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            attempt = self.training_attempt("running")
            formal.write_training_status(
                args,
                job,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "phase": "training",
                    "job": job.as_dict(),
                    "status": "running",
                    "attempts": [attempt],
                    "started_at_utc": attempt["started_at_utc"],
                    "child_pid": 2147483646,
                },
            )
            with patch.object(formal, "_pid_is_active", return_value=False), patch.object(
                formal, "discover_run_directory", return_value=(None, None)
            ), patch.object(
                formal,
                "assert_runtime_fingerprint",
                side_effect=[None, formal.ConfigurationError("stop before relaunch")],
            ):
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "stop before relaunch"
                ):
                    formal.train_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                    )
            recovered = formal.load_training_status(args, job, "spec")
            self.assertEqual(recovered["status"], "interrupted")
            self.assertEqual(
                recovered["attempts"][0]["failure_category"],
                "orphaned_process_interruption",
            )
            self.assertIs(recovered["attempts"][0]["retry_authorized"], True)

    def test_launch_fingerprint_is_checked_before_output_state_mutation(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(
                formal, "discover_run_directory", return_value=(None, None)
            ), patch.object(
                formal,
                "assert_runtime_fingerprint",
                side_effect=formal.ConfigurationError("fingerprint drift"),
            ), patch.object(formal, "write_training_status") as write_status:
                with self.assertRaisesRegex(formal.ConfigurationError, "fingerprint drift"):
                    formal.train_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                    )
            write_status.assert_not_called()

    def test_completed_training_status_is_read_only_and_bound_to_result(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            core = {
                "job": job.as_dict(),
                "run_directory": str((Path(directory) / "run").resolve()),
            }
            attempt = self.training_attempt("completed")
            status = formal.write_training_status(
                args,
                job,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "phase": "training",
                    "job": job.as_dict(),
                    "status": "completed",
                    "attempts": [attempt],
                    "started_at_utc": attempt["started_at_utc"],
                    "child_pid": None,
                    "completed_at_utc": attempt["finished_at_utc"],
                    "artifact": core,
                },
            )
            with patch.object(
                formal, "audit_training_job", return_value=core
            ), patch.object(formal, "write_training_status") as write_status:
                result = formal.train_job(
                    args,
                    job,
                    {"spec_sha256": "spec", "code_fingerprint": {}},
                )
            write_status.assert_not_called()
            self.assertEqual(formal._training_core_artifact(result), core)
            self.assertEqual(
                result["training_status"]["status_sha256"],
                status["status_sha256"],
            )

    def test_orphan_completed_run_requires_and_closes_running_attempt(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            run_directory = Path(directory) / "run"
            core = {
                "job": job.as_dict(),
                "run_directory": str(run_directory.resolve()),
            }
            attempt = self.training_attempt("running")
            formal.write_training_status(
                args,
                job,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "phase": "training",
                    "job": job.as_dict(),
                    "status": "running",
                    "attempts": [attempt],
                    "started_at_utc": attempt["started_at_utc"],
                },
            )
            with patch.object(
                formal,
                "discover_run_directory",
                return_value=(run_directory, None),
            ), patch.object(
                formal, "audit_training_job", return_value=core
            ), patch.object(formal, "assert_runtime_fingerprint"):
                result = formal.train_job(
                    args,
                    job,
                    {"spec_sha256": "spec", "code_fingerprint": {}},
                )
            completed = formal.load_training_status(args, job, "spec")
            self.assertTrue(completed["recovered_completed_run"])
            self.assertEqual(completed["attempts"][-1]["status"], "completed")
            self.assertEqual(formal._training_core_artifact(result), core)

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(
                formal,
                "discover_run_directory",
                return_value=(Path(directory) / "untracked-run", None),
            ), patch.object(formal, "audit_training_job") as audit:
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "tracked running attempt"
                ):
                    formal.train_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                    )
            audit.assert_not_called()
            self.assertFalse(formal.job_root(args, job).exists())


def make_metric(job: formal.FormalJob, workload_seed: int) -> ConstraintEpisodeMetrics:
    return ConstraintEpisodeMetrics(
        scenario=job.scenario,
        policy=job.arm,
        policy_seed=job.policy_seed,
        workload_seed=workload_seed,
        generated=10,
        delivered=6,
        dropped=2,
        backlog=2,
        delivery_ratio=0.6,
        drop_rate=0.2,
        throughput_packets_per_slot=0.2,
        average_delay_slots=3.0,
        p95_delay_slots=5.0,
        mean_queue_packets=1.0,
        max_queue_packets=3,
        routing_switches=3,
        episode_reward=1.0,
        global_delay_cost=0.1,
        global_queue_cost=0.2,
        global_load_imbalance=0.3,
        global_switch_cost=0.0 if job.environment_variant == "qos_only" else 0.1,
        global_throughput_reward=0.4,
        global_control_overhead_ratio=0.05,
        global_drop_cost=0.2,
        class_0_delivery_ratio=0.6,
        class_1_delivery_ratio=0.5,
        class_2_delivery_ratio=0.7,
        avoidable_routing_switches=2,
        forced_routing_switches=1,
        switch_opportunities=4,
        avoidable_switch_rate=0.5,
        decision_avoidable_switches=3,
        decision_switch_opportunities=5,
        decision_forced_switches=1,
        decision_avoidable_switch_rate=0.6,
    )


def make_rows(
    job: formal.FormalJob,
    *,
    panel: str = formal.PANEL_SELECTION_RECHECK,
    spec_sha256: str = "spec",
    freeze_sha256: str = "freeze",
    checkpoint_sha256: str = "checkpoint",
) -> list[dict]:
    rows = []
    for workload_seed in formal.PANEL_WORKLOAD_SEEDS[panel]:
        rows.append(
            {
                "schema_version": formal.SCHEMA_VERSION,
                "evaluation_role": formal.PANEL_EVALUATION_ROLES[panel],
                "spec_sha256": spec_sha256,
                "training_freeze_sha256": freeze_sha256,
                "job_id": job.job_id,
                "arm": job.arm,
                "environment_variant": job.environment_variant,
                "selected_checkpoint_sha256": checkpoint_sha256,
                "decision_forced_switch_cost": 0,
                **formal.asdict(make_metric(job, workload_seed)),
            }
        )
    return rows


def write_completed_evaluation_status(
    args: SimpleNamespace,
    job: formal.FormalJob,
    panel: str,
    manifest: dict,
    *,
    spec_sha256: str = "spec",
    freeze_sha256: str = "freeze",
) -> dict:
    timestamp = "2026-09-03T00:00:00Z"
    attempt = {
        "attempt": 1,
        "started_at_utc": timestamp,
        "status": "completed",
        "evaluation_device": args.evaluation_device,
        "validation_workload_seeds": list(formal.PANEL_WORKLOAD_SEEDS[panel]),
        "finished_at_utc": timestamp,
    }
    status = {
        "schema_version": formal.SCHEMA_VERSION,
        "study_name": formal.STUDY_NAME,
        "spec_sha256": spec_sha256,
        "training_freeze_sha256": freeze_sha256,
        "phase": "post_training_validation",
        "validation_panel": panel,
        "job": job.as_dict(),
        "status": "completed",
        "attempts": [attempt],
        "started_at_utc": timestamp,
        "completed_at_utc": timestamp,
        "artifact": formal._evaluation_manifest_artifact(
            args, job, panel, manifest
        ),
    }
    return formal.write_evaluation_status(args, job, panel, status)


class CandidateValidationAuditTests(unittest.TestCase):
    @staticmethod
    def make_fixture(root: Path) -> dict:
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        steps = tuple(
            rollout * formal.ROLLOUT_STEPS
            for rollout in range(
                formal.VALIDATION_EVERY_ROLLOUTS,
                formal.EXPECTED_FINAL_ENVIRONMENT_STEPS // formal.ROLLOUT_STEPS + 1,
                formal.VALIDATION_EVERY_ROLLOUTS,
            )
        )
        validations = []
        for index, step in enumerate(steps):
            delivery = 0.9 - index * 0.01
            validations.append(
                {
                    "record_type": "validation",
                    "environment_steps": step,
                    "episodes": len(formal.SELECTION_WORKLOAD_SEEDS),
                    "seed_start": formal.SELECTION_WORKLOAD_SEEDS[0],
                    "delivery_ratio": delivery,
                    "mean_reward": delivery * 10.0,
                    "drop_rate": 1.0 - delivery,
                    "average_delay_slots": 2.0,
                    "routing_switches": 0.6,
                    "routing_switches_total": 6,
                    "avoidable_routing_switches": 0.4,
                    "avoidable_routing_switches_total": 4,
                    "forced_routing_switches": 0.2,
                    "forced_routing_switches_total": 2,
                    "switch_opportunities": 20,
                    "avoidable_switch_rate": 0.2,
                    "class_2_delivery_ratio": delivery - 0.1,
                    "decision_avoidable_switches": 1,
                    "decision_switch_opportunities": 10,
                    "decision_avoidable_switch_rate": 0.1,
                    "is_validation_best": index == 0,
                }
            )
        selected_path = root / "validation_best.pt"
        selected_path.write_bytes(b"selected")
        selected = validations[0]
        selected_metrics = {
            field: selected[field]
            for field in formal.SELECTED_VALIDATION_METRIC_FIELDS
        }
        selection_spec = {
            "schema_version": 1,
            "mode": job.selection_mode,
            "delivery_tolerance": 0.0,
            "class_2_tolerance": 0.0,
            "switch_budget": None,
            "source": "validation_only",
            "test_panel_consulted": False,
            "validation_seed_start": formal.SELECTION_WORKLOAD_SEEDS[0],
            "validation_episodes": len(formal.SELECTION_WORKLOAD_SEEDS),
            "metric_aggregation": {
                "delivery_ratio": "macro_mean_over_validation_episodes",
                "class_2_delivery_ratio": "macro_mean_over_validation_episodes",
                "routing_switches": "mean_of_episode_total_switch_counts",
            },
            "selection_timing": "online_legacy_compatible",
        }
        return {
            "job": job,
            "manifest": {
                "validation_candidate_count": len(validations),
                "validation_selection_spec": selection_spec,
                "selected_validation_metrics": selected_metrics,
                "selected_validation_checkpoint_sha256": formal.sha256_file(
                    selected_path
                ),
                "best_validation_score": list(
                    formal._legacy_validation_score(selected)
                ),
            },
            "validations": validations,
            "selections": [],
            "run_directory": root,
            "selected_path": selected_path,
            "selected_checkpoint": {
                "step": selected["environment_steps"],
                "validation_selection_spec": selection_spec,
            },
        }

    def test_candidate_count_and_rate_ledgers_fail_closed(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            formal.audit_validation_selection(**fixture)
            fixture["validations"][1]["decision_avoidable_switch_rate"] = 0.2
            with self.assertRaisesRegex(formal.ConfigurationError, "ratio-of-sums"):
                formal.audit_validation_selection(**fixture)

        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            fixture["validations"][1]["routing_switches_total"] = 7
            with self.assertRaisesRegex(formal.ConfigurationError, "accepted-switch ledger"):
                formal.audit_validation_selection(**fixture)

    def test_every_deferred_candidate_is_bound_to_path_hash_step_and_schema(self):
        job = formal.FormalJob(
            0,
            formal.SCENARIOS[0],
            formal.ARM_CONSTRAINED,
            formal.POLICY_SEEDS[0],
        )
        args = SimpleNamespace(
            output=Path("F:/formal-output"),
            project=Path("F:/leo-routing-preliminary-matlab/src"),
            device="cuda",
        )
        with TemporaryDirectory() as directory:
            run_directory = Path(directory)
            selection_spec = {"schema_version": 1, "mode": job.selection_mode}

            def payload(step):
                return {
                    "step": step,
                    "resume_boundary": "inference_snapshot_only",
                    "args": formal.expected_run_config(job, args),
                    "obs_size": 52,
                    "state_size": 10,
                    "action_size": 2,
                    "n_agents": 3,
                    "candidate_feature_dim": 26,
                    "candidate_actor_spec": {"schema_version": 1},
                    "critic_spec": {"schema_version": 1},
                    "switch_regularizer_spec": None,
                    "switch_constraint_spec": formal.constraint_contract(),
                    "switch_constraint_state": {"dual_value": 0.0},
                    "validation_selection_spec": selection_spec,
                }

            validations = []
            for step in (4800, 9600):
                path = run_directory / f"validation_candidate_step_{step}.pt"
                torch.save(payload(step), path)
                validations.append(
                    {
                        "environment_steps": step,
                        "candidate_checkpoint": str(path),
                    }
                )
            selected_checkpoint = payload(4800)
            formal.audit_constrained_validation_candidates(
                args,
                job,
                validations,
                run_directory,
                formal._checkpoint_inventory(run_directory),
                selected_checkpoint,
            )

            second = run_directory / "validation_candidate_step_9600.pt"
            decoy = run_directory / "validation_candidate_step_12345.pt"
            decoy.write_bytes(second.read_bytes())
            second.unlink()
            with self.assertRaisesRegex(formal.ConfigurationError, "inventory"):
                formal.audit_constrained_validation_candidates(
                    args,
                    job,
                    validations,
                    run_directory,
                    formal._checkpoint_inventory(run_directory),
                    selected_checkpoint,
                )

            second.write_bytes(decoy.read_bytes())
            decoy.unlink()
            tampered = torch.load(second, map_location="cpu", weights_only=False)
            tampered["step"] = 4800
            torch.save(tampered, second)
            with self.assertRaisesRegex(formal.ConfigurationError, "payload"):
                formal.audit_constrained_validation_candidates(
                    args,
                    job,
                    validations,
                    run_directory,
                    formal._checkpoint_inventory(run_directory),
                    selected_checkpoint,
                )


class ValidationRowContractTests(unittest.TestCase):
    def setUp(self):
        self.job = formal.FormalJob(
            0,
            formal.SCENARIOS[0],
            formal.ARM_BASELINE,
            formal.POLICY_SEEDS[0],
        )

    def validate(self, rows):
        return formal.validate_validation_rows(
            self.job,
            rows,
            panel=formal.PANEL_SELECTION_RECHECK,
            spec_sha256="spec",
            freeze_sha256="freeze",
            checkpoint_sha256="checkpoint",
            checkpoint_step=4800,
        )

    def test_complete_rows_pass_and_use_ratio_of_sums(self):
        summary = self.validate(make_rows(self.job))
        self.assertEqual(summary["row_count"], 10)
        self.assertEqual(summary["decision_avoidable_switches"], 30)
        self.assertEqual(summary["decision_switch_opportunities"], 50)
        self.assertAlmostEqual(summary["decision_avoidable_switch_rate"], 0.6)

    def test_missing_duplicate_or_substituted_workload_fails(self):
        rows = make_rows(self.job)
        with self.assertRaises(formal.ConfigurationError):
            self.validate(rows[:-1])
        duplicate = deepcopy(rows)
        duplicate[-1]["workload_seed"] = duplicate[-2]["workload_seed"]
        with self.assertRaises(formal.ConfigurationError):
            self.validate(duplicate)
        sealed = deepcopy(rows)
        sealed[-1]["workload_seed"] = formal.SEALED_TEST_WORKLOAD_SEEDS[0]
        with self.assertRaises(formal.ConfigurationError):
            self.validate(sealed)

    def test_ledger_conservation_forced_cost_hash_and_finiteness_fail_closed(self):
        mutations = (
            ("generated", 11),
            ("decision_avoidable_switches", 6),
            ("decision_forced_switch_cost", 1),
            ("selected_checkpoint_sha256", "different"),
            ("delivery_ratio", math.nan),
            ("routing_switches", 4),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                rows = make_rows(self.job)
                rows[0][field] = value
                with self.assertRaises(formal.ConfigurationError):
                    self.validate(rows)

    def test_schema_order_and_boolean_integer_are_rejected(self):
        rows = make_rows(self.job)
        rows[0] = dict(reversed(tuple(rows[0].items())))
        with self.assertRaisesRegex(formal.ConfigurationError, "schema/order"):
            self.validate(rows)
        rows = make_rows(self.job)
        rows[0]["decision_forced_switch_cost"] = False
        with self.assertRaises(formal.ConfigurationError):
            self.validate(rows)

    def test_jsonl_csv_roundtrip_is_exact(self):
        rows = make_rows(self.job)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "rows.jsonl"
            csv_path = root / "rows.csv"
            formal._atomic_write_jsonl(jsonl, rows)
            atomic_write_csv(csv_path, rows, fieldnames=formal.VALIDATION_ROW_FIELDS)
            self.assertEqual(formal._read_jsonl(jsonl), rows)
            self.assertEqual(formal._read_validation_csv(csv_path), rows)

    def test_selection_recheck_recomputes_all_selection_metrics(self):
        original_rows = make_rows(self.job)
        selected = self.validate(original_rows)
        artifact = {
            "selected_validation": {
                field: selected[field]
                for field in formal.SELECTED_VALIDATION_METRIC_FIELDS
            }
        }

        def change_drop(rows):
            rows[0]["dropped"] = 3
            rows[0]["backlog"] = 1
            rows[0]["drop_rate"] = 0.3

        def change_accepted_switches(rows):
            rows[0]["routing_switches"] = 4
            rows[0]["forced_routing_switches"] = 2
            rows[0]["decision_forced_switches"] = 2

        mutations = {
            "mean_reward": lambda rows: rows[0].update(episode_reward=2.0),
            "drop_rate": change_drop,
            "average_delay_slots": lambda rows: rows[0].update(
                average_delay_slots=4.0
            ),
            "routing_switches": change_accepted_switches,
        }
        for expected_field, mutate in mutations.items():
            with self.subTest(expected_field=expected_field):
                rows = deepcopy(original_rows)
                mutate(rows)
                summary = self.validate(rows)
                with self.assertRaisesRegex(
                    formal.ConfigurationError, expected_field
                ):
                    formal._audit_recheck_against_selected(
                        self.job, summary, artifact
                    )


class CompletedTrainingFreezeTests(unittest.TestCase):
    def test_swapped_seed_artifacts_fail_full_job_reaudit(self):
        jobs = [
            formal.FormalJob(
                index,
                formal.SCENARIOS[0],
                formal.ARM_BASELINE,
                formal.POLICY_SEEDS[index],
            )
            for index in range(2)
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root)
            proof_hash = "proof"
            spec = {
                "spec_sha256": "spec",
                "smoke_prerequisite": {
                    "exact_resume_proof_sha256": proof_hash
                },
            }
            log_path = root / "preflight.log"
            log_path.write_text("passed", encoding="utf-8")
            verification = formal.self_hashed(
                {
                    "spec_sha256": "spec",
                    "status": "passed",
                    "log_path": str(log_path.resolve()),
                    "log_sha256": formal.sha256_file(log_path),
                    "exact_resume_proof_sha256": proof_hash,
                },
                "verification_sha256",
            )
            atomic_write_json(root / "preflight_verification.json", verification)

            def core_artifact(job):
                placeholder = {"path": "placeholder", "sha256": "hash"}
                return {
                    "job": job.as_dict(),
                    "run_directory": str(formal.job_root(args, job) / "run"),
                    "marker_policy_seed": job.policy_seed,
                    "artifacts": {
                        "run_config": dict(placeholder),
                        "training_metrics": dict(placeholder),
                        "run_manifest": dict(placeholder),
                        "latest_checkpoint": dict(placeholder),
                        "final_checkpoint": dict(placeholder),
                        "selected_checkpoint": dict(placeholder),
                        "checkpoint_inventory": {
                            "final.pt": dict(placeholder)
                        },
                        "runner_logs": [
                            {
                                "path": str(
                                    (
                                        formal.job_root(args, job)
                                        / "runner_attempt_1.log"
                                    ).resolve()
                                ),
                                "sha256": "hash",
                            }
                        ],
                    },
                }

            def bind_status(job, core):
                timestamp = "2026-09-03T00:00:00Z"
                attempt = {
                    "attempt": 1,
                    "started_at_utc": timestamp,
                    "status": "completed",
                    "resume_checkpoint": None,
                    "resume_checkpoint_sha256": None,
                    "command": ["python", "trainer.py"],
                    "working_directory": str(
                        formal.job_root(args, job).resolve()
                    ),
                    "log_path": str(
                        (formal.job_root(args, job) / "runner_attempt_1.log").resolve()
                    ),
                    "finished_at_utc": timestamp,
                }
                status = formal.write_training_status(
                    args,
                    job,
                    {
                        "schema_version": formal.SCHEMA_VERSION,
                        "study_name": formal.STUDY_NAME,
                        "spec_sha256": "spec",
                        "phase": "training",
                        "job": job.as_dict(),
                        "status": "completed",
                        "attempts": [attempt],
                        "started_at_utc": timestamp,
                        "child_pid": None,
                        "completed_at_utc": timestamp,
                        "artifact": core,
                    },
                )
                return formal._bind_training_status(args, job, core, status)

            canonical_artifacts = {
                job.job_id: bind_status(job, core_artifact(job)) for job in jobs
            }

            def make_freeze(artifacts):
                return formal.self_hashed(
                    {
                        "schema_version": formal.SCHEMA_VERSION,
                        "study_name": formal.STUDY_NAME,
                        "spec_sha256": "spec",
                        "verification_sha256": verification[
                            "verification_sha256"
                        ],
                        "smoke_exact_resume_proof_sha256": proof_hash,
                        "training_complete": True,
                        "checkpoint_count": 2,
                        "test_panel_consulted": False,
                        "test_access_count": 0,
                        "sealed_test_instantiated": False,
                        "jobs": artifacts,
                    },
                    "freeze_sha256",
                )

            def reaudit(_args, job, _run_directory):
                return formal._training_core_artifact(
                    canonical_artifacts[job.job_id]
                )

            with patch.object(formal, "build_jobs", return_value=jobs), patch.object(
                formal, "EXPECTED_TRAINING_JOBS", 2
            ), patch.object(formal, "_validate_artifact_entry"), patch.object(
                formal, "audit_training_job", side_effect=reaudit
            ), patch.object(
                formal, "run_preflight_tests", return_value=verification
            ):
                formal.validate_training_freeze(
                    make_freeze(canonical_artifacts),
                    spec,
                    args,
                    reaudit_jobs=True,
                )
                swapped = {}
                for target, source in zip(jobs, reversed(jobs)):
                    value = formal._training_core_artifact(
                        deepcopy(canonical_artifacts[source.job_id])
                    )
                    value["job"] = target.as_dict()
                    value["run_directory"] = str(
                        formal.job_root(args, target) / "run"
                    )
                    value["artifacts"]["runner_logs"][0]["path"] = str(
                        (
                            formal.job_root(args, target)
                            / "runner_attempt_1.log"
                        ).resolve()
                    )
                    swapped[target.job_id] = bind_status(target, value)
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "job audit drifted"
                ):
                    formal.validate_training_freeze(
                        make_freeze(swapped),
                        spec,
                        args,
                        reaudit_jobs=True,
                    )


class EvaluationPipelineTests(unittest.TestCase):
    def test_structured_evaluator_receives_only_validation_and_shard_resumes(self):
        job = formal.FormalJob(
            0,
            formal.SCENARIOS[0],
            formal.ARM_BASELINE,
            formal.POLICY_SEEDS[0],
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "run" / "validation_best.pt"
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"selected")
            checkpoint_hash = formal.sha256_file(selected)
            rows = make_rows(
                job,
                spec_sha256="spec",
                freeze_sha256="freeze",
                checkpoint_sha256=checkpoint_hash,
            )
            selected_summary = formal.validate_validation_rows(
                job,
                rows,
                panel=formal.PANEL_SELECTION_RECHECK,
                spec_sha256="spec",
                freeze_sha256="freeze",
                checkpoint_sha256=checkpoint_hash,
                checkpoint_step=4800,
            )
            freeze = {
                "freeze_sha256": "freeze",
                "jobs": {
                    job.job_id: {
                        "run_directory": str(selected.parent),
                        "selected_validation": {
                            **selected_summary,
                            "environment_steps": 4800,
                            "constraint_feasible": None,
                        },
                        "checkpoint_schema": {"variant": "qos_only"},
                        "artifacts": {
                            "selected_checkpoint": {
                                "path": str(selected),
                                "sha256": checkpoint_hash,
                            }
                        },
                    }
                },
            }
            spec = {"spec_sha256": "spec", "code_fingerprint": {}}
            args = SimpleNamespace(output=root, evaluation_device="cuda")

            class Policy:
                checkpoint_schema = {"variant": "qos_only"}
                switch_constraint_spec = None

            observed_workloads = []

            def evaluate(**kwargs):
                observed_workloads.append(tuple(kwargs["workload_seeds"]))
                return [
                    make_metric(job, seed)
                    for seed in kwargs["workload_seeds"]
                ]

            with patch.object(formal, "assert_runtime_fingerprint"), patch.object(
                formal, "load_checkpoint_policy", return_value=(Policy(), {"step": 4800})
            ), patch.object(
                formal, "evaluate_policy_with_constraint_metrics", side_effect=evaluate
            ) as evaluator:
                first = formal.evaluate_validation_job(
                    args, job, spec, freeze, formal.PANEL_SELECTION_RECHECK
                )
                second = formal.evaluate_validation_job(
                    args, job, spec, freeze, formal.PANEL_SELECTION_RECHECK
                )
                gate_first = formal.evaluate_validation_job(
                    args, job, spec, freeze, formal.PANEL_INDEPENDENT_GATE
                )
                gate_second = formal.evaluate_validation_job(
                    args, job, spec, freeze, formal.PANEL_INDEPENDENT_GATE
                )
            self.assertEqual(first, second)
            self.assertEqual(gate_first, gate_second)
            self.assertEqual(evaluator.call_count, 20)
            self.assertEqual(
                observed_workloads,
                [
                    *((seed,) for seed in formal.SELECTION_WORKLOAD_SEEDS),
                    *((seed,) for seed in formal.GATE_WORKLOAD_SEEDS),
                ],
            )
            for panel in formal.VALIDATION_PANELS:
                manifest, persisted = formal.validate_evaluation_shard(
                    args, job, spec, freeze, panel
                )
                self.assertEqual(manifest["checkpoint_step"], 4800)
                self.assertEqual(manifest["checkpoint_schema"], Policy.checkpoint_schema)
                self.assertEqual(len(persisted), 10)
                status = formal.load_evaluation_status(
                    args, job, panel, "spec", "freeze"
                )
                self.assertEqual(status["status"], "completed")
                self.assertEqual(len(status["attempts"]), 1)

            manifest_path = formal.validation_shard_paths(
                args, job, formal.PANEL_INDEPENDENT_GATE
            )[2]
            original = formal.read_json(manifest_path)
            mutations = {
                "checkpoint_schema": {"variant": "proposed"},
                "checkpoint_step": 9600,
                "jsonl_path": str((root / "substituted.jsonl").resolve()),
                "paper_claim_allowed": True,
            }
            for field, value in mutations.items():
                with self.subTest(manifest_field=field):
                    body = dict(original)
                    body.pop("manifest_sha256")
                    body[field] = value
                    atomic_write_json(
                        manifest_path, formal.self_hashed(body, "manifest_sha256")
                    )
                    with self.assertRaisesRegex(
                        formal.ConfigurationError, "shard contract"
                    ):
                        formal.validate_evaluation_shard(
                            args,
                            job,
                            spec,
                            freeze,
                            formal.PANEL_INDEPENDENT_GATE,
                        )
                    atomic_write_json(manifest_path, original)

    def test_third_evaluation_attempt_is_rejected_before_launch_or_mutation(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        panel = formal.PANEL_SELECTION_RECHECK
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), evaluation_device="cuda"
            )
            attempts = []
            for number in (1, 2):
                timestamp = f"2026-09-03T00:00:0{number}Z"
                attempts.append(
                    {
                        "attempt": number,
                        "started_at_utc": timestamp,
                        "status": "failed",
                        "evaluation_device": "cuda",
                        "validation_workload_seeds": list(
                            formal.PANEL_WORKLOAD_SEEDS[panel]
                        ),
                        "finished_at_utc": timestamp,
                        "error": "RuntimeError('infrastructure failure')",
                        "failure_category": "recognized_infrastructure_failure",
                        "retry_authorized": True,
                    }
                )
            status = {
                "schema_version": formal.SCHEMA_VERSION,
                "study_name": formal.STUDY_NAME,
                "spec_sha256": "spec",
                "training_freeze_sha256": "freeze",
                "phase": "post_training_validation",
                "validation_panel": panel,
                "job": job.as_dict(),
                "status": "failed",
                "attempts": attempts,
                "started_at_utc": attempts[-1]["started_at_utc"],
            }
            formal.write_evaluation_status(args, job, panel, status)
            with patch.object(formal, "assert_runtime_fingerprint") as fingerprint, patch.object(
                formal, "load_checkpoint_policy"
            ) as load_policy, patch.object(
                formal, "evaluate_policy_with_constraint_metrics"
            ) as evaluator, patch.object(
                formal,
                "write_evaluation_status",
                wraps=formal.write_evaluation_status,
            ) as write_status:
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "retry limit"
                ):
                    formal.evaluate_validation_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                        {"freeze_sha256": "freeze"},
                        panel,
                    )
            fingerprint.assert_not_called()
            load_policy.assert_not_called()
            evaluator.assert_not_called()
            write_status.assert_not_called()

    def test_non_infrastructure_evaluation_failure_cannot_retry_on_reentry(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        panel = formal.PANEL_SELECTION_RECHECK
        timestamp = "2026-09-03T00:00:00Z"
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), evaluation_device="cuda"
            )
            attempt = {
                "attempt": 1,
                "started_at_utc": timestamp,
                "status": "failed",
                "evaluation_device": "cuda",
                "validation_workload_seeds": list(
                    formal.PANEL_WORKLOAD_SEEDS[panel]
                ),
                "finished_at_utc": timestamp,
                "error": "RuntimeError('scientific failure')",
                "failure_category": "scientific_or_runtime_failure",
                "retry_authorized": False,
            }
            formal.write_evaluation_status(
                args,
                job,
                panel,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "training_freeze_sha256": "freeze",
                    "phase": "post_training_validation",
                    "validation_panel": panel,
                    "job": job.as_dict(),
                    "status": "failed",
                    "attempts": [attempt],
                    "started_at_utc": timestamp,
                },
            )
            with patch.object(formal, "assert_runtime_fingerprint") as fingerprint, patch.object(
                formal, "load_checkpoint_policy"
            ) as load_policy, patch.object(
                formal, "evaluate_policy_with_constraint_metrics"
            ) as evaluator, patch.object(
                formal,
                "write_evaluation_status",
                wraps=formal.write_evaluation_status,
            ) as write_status:
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "not infrastructure-retry eligible"
                ):
                    formal.evaluate_validation_job(
                        args,
                        job,
                        {"spec_sha256": "spec", "code_fingerprint": {}},
                        {"freeze_sha256": "freeze"},
                        panel,
                    )
            fingerprint.assert_not_called()
            load_policy.assert_not_called()
            evaluator.assert_not_called()
            write_status.assert_not_called()

    def test_evaluation_retry_classification_tamper_is_rejected(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        panel = formal.PANEL_SELECTION_RECHECK
        timestamp = "2026-09-03T00:00:00Z"
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), evaluation_device="cuda"
            )
            formal.write_evaluation_status(
                args,
                job,
                panel,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "training_freeze_sha256": "freeze",
                    "phase": "post_training_validation",
                    "validation_panel": panel,
                    "job": job.as_dict(),
                    "status": "failed",
                    "attempts": [
                        {
                            "attempt": 1,
                            "started_at_utc": timestamp,
                            "status": "failed",
                            "evaluation_device": "cuda",
                            "validation_workload_seeds": list(
                                formal.PANEL_WORKLOAD_SEEDS[panel]
                            ),
                            "finished_at_utc": timestamp,
                            "error": "TimeoutError('temporary')",
                            "failure_category": "recognized_infrastructure_failure",
                            "retry_authorized": False,
                        }
                    ],
                    "started_at_utc": timestamp,
                },
            )
            with self.assertRaisesRegex(
                formal.ConfigurationError, "attempt history drifted"
            ):
                formal.load_evaluation_status(
                    args, job, panel, "spec", "freeze"
                )

    def test_evaluation_attempt_histories_are_panel_isolated(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), evaluation_device="cuda"
            )
            panel = formal.PANEL_SELECTION_RECHECK
            timestamp = "2026-09-03T00:00:00Z"
            attempt = {
                "attempt": 1,
                "started_at_utc": timestamp,
                "status": "failed",
                "evaluation_device": "cuda",
                "validation_workload_seeds": list(
                    formal.PANEL_WORKLOAD_SEEDS[panel]
                ),
                "finished_at_utc": timestamp,
                "error": "RuntimeError('failed')",
                "failure_category": "scientific_or_runtime_failure",
                "retry_authorized": False,
            }
            formal.write_evaluation_status(
                args,
                job,
                panel,
                {
                    "schema_version": formal.SCHEMA_VERSION,
                    "study_name": formal.STUDY_NAME,
                    "spec_sha256": "spec",
                    "training_freeze_sha256": "freeze",
                    "phase": "post_training_validation",
                    "validation_panel": panel,
                    "job": job.as_dict(),
                    "status": "failed",
                    "attempts": [attempt],
                    "started_at_utc": timestamp,
                },
            )
            gate = formal.load_evaluation_status(
                args,
                job,
                formal.PANEL_INDEPENDENT_GATE,
                "spec",
                "freeze",
            )
            self.assertEqual(gate["status"], "pending")
            self.assertEqual(gate["attempts"], [])
            self.assertNotEqual(
                formal.evaluation_status_path(args, job, panel),
                formal.evaluation_status_path(
                    args, job, formal.PANEL_INDEPENDENT_GATE
                ),
            )

    def test_validation_failure_never_submits_waiting_jobs(self):
        jobs = [
            formal.FormalJob(i, formal.SCENARIOS[0], formal.ARMS[i % 3], i + 1)
            for i in range(4)
        ]
        started = []

        def evaluate(
            _args, job, _spec, _freeze, _panel, cancellation_event
        ):
            started.append(job.index)
            if job.index == 0:
                raise RuntimeError("normal shard failure")
            cancellation_event.wait(timeout=2.0)
            raise formal.EvaluationCancelled("peer cancelled")

        with patch.object(formal, "build_jobs", return_value=jobs), patch.object(
            formal, "EXPECTED_TRAINING_JOBS", len(jobs)
        ), patch.object(formal, "validate_training_freeze"), patch.object(
            formal,
            "load_evaluation_status",
            return_value={"status": "pending", "attempts": []},
        ), patch.object(
            formal, "evaluate_validation_job", side_effect=evaluate
        ):
            with self.assertRaisesRegex(RuntimeError, "normal shard failure"):
                formal.run_all_validation_jobs(
                    SimpleNamespace(max_parallel=2),
                    {"spec_sha256": "spec"},
                    {"freeze_sha256": "freeze"},
                )
        self.assertTrue(started)
        self.assertFalse(set(started) & {2, 3})

    def test_selection_feasibility_gate_recomputes_with_named_tolerance(self):
        job = formal.FormalJob(
            0,
            formal.SCENARIOS[0],
            formal.ARM_CONSTRAINED,
            formal.POLICY_SEEDS[0],
        )
        opportunity = 10_000_000_000_000
        cost = 1_200_000_000_005
        selected = {
            "decision_avoidable_switches": cost,
            "decision_switch_opportunities": opportunity,
            "decision_avoidable_switch_rate": cost / opportunity,
            "constraint_feasible": True,
        }
        training_freeze = {
            "jobs": {job.job_id: {"selected_validation": selected}}
        }
        with patch.object(formal, "build_jobs", return_value=[job]):
            gate = formal._selection_constraint_gate(training_freeze)
            self.assertTrue(gate["passed"])
            self.assertEqual(
                gate["selection_feasibility_numerical_tolerance"], 1e-12
            )
            self.assertEqual(
                gate["independent_gate_budget_numerical_tolerance"], 0.0
            )
            selected["constraint_feasible"] = False
            with self.assertRaisesRegex(
                formal.ConfigurationError, "feasibility drifted"
            ):
                formal._selection_constraint_gate(training_freeze)

    def test_partial_training_freeze_and_partial_validation_grid_fail(self):
        spec = {"spec_sha256": "spec", "code_fingerprint": {}}
        partial_body = {
            "schema_version": 1,
            "study_name": formal.STUDY_NAME,
            "spec_sha256": "spec",
            "training_complete": True,
            "checkpoint_count": 0,
            "test_panel_consulted": False,
            "test_access_count": 0,
            "sealed_test_instantiated": False,
            "jobs": {},
        }
        partial = formal.self_hashed(partial_body, "freeze_sha256")
        with self.assertRaisesRegex(formal.ConfigurationError, "freeze contract"):
            formal.validate_training_freeze(partial, spec)
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(formal, "validate_training_freeze"), patch.object(
                formal, "assert_runtime_fingerprint"
            ):
                with self.assertRaisesRegex(formal.ConfigurationError, "partial/reordered"):
                    formal.write_validation_freeze(args, spec, partial, {})

    def test_existing_training_freeze_is_audited_before_any_new_write(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            spec = {"spec_sha256": "spec", "code_fingerprint": {}}
            atomic_write_json(output / "preregistration.json", spec)
            atomic_write_json(output / "training_freeze.json", {"damaged": True})
            args = SimpleNamespace(
                output=output,
                cleanmarl=Path("F:/cleanmarl"),
                project=Path("F:/leo-routing-preliminary-matlab/src"),
                device="cuda",
                evaluation_device="cuda",
                max_parallel=2,
                dry_run=False,
            )
            with patch.object(formal, "parse_args", return_value=args), patch.object(
                formal, "validate_runtime_environment"
            ), patch.object(formal, "validate_output_isolation"), patch.object(
                formal, "build_spec", return_value=spec
            ), patch.object(
                formal,
                "validate_training_freeze",
                side_effect=formal.ConfigurationError("damaged training freeze"),
            ), patch.object(formal, "atomic_write_json") as write_json, patch.object(
                formal, "run_preflight_tests"
            ) as preflight, patch.object(
                formal, "run_all_training_jobs"
            ) as training:
                with self.assertRaisesRegex(
                    formal.ConfigurationError, "damaged training freeze"
                ):
                    formal.main([])
            write_json.assert_not_called()
            preflight.assert_not_called()
            training.assert_not_called()

    def test_selection_recheck_failure_never_opens_independent_gate(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        observed_panels = []

        def evaluate(_args, _job, _spec, _freeze, panel, _cancellation):
            observed_panels.append(panel)
            raise formal.ConfigurationError("selection reproduction failed")

        with patch.object(formal, "build_jobs", return_value=[job]), patch.object(
            formal, "EXPECTED_TRAINING_JOBS", 1
        ), patch.object(formal, "validate_training_freeze"), patch.object(
            formal,
            "load_evaluation_status",
            return_value={"status": "pending", "attempts": []},
        ), patch.object(
            formal, "evaluate_validation_job", side_effect=evaluate
        ):
            with self.assertRaisesRegex(RuntimeError, "selection_recheck"):
                formal.run_all_validation_jobs(
                    SimpleNamespace(max_parallel=2),
                    {"spec_sha256": "spec"},
                    {"freeze_sha256": "freeze"},
                )
        self.assertEqual(observed_panels, [formal.PANEL_SELECTION_RECHECK])

    def test_freeze_uses_gate_rows_only_and_completed_tampering_fails_closed(self):
        job = formal.FormalJob(
            0,
            formal.SCENARIOS[0],
            formal.ARM_CONSTRAINED,
            formal.POLICY_SEEDS[0],
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root, evaluation_device="cuda")
            spec = {"spec_sha256": "spec", "code_fingerprint": {}}
            rows_by_panel = {
                panel: make_rows(job, panel=panel)
                for panel in formal.VALIDATION_PANELS
            }
            summaries = {
                panel: formal.validate_validation_rows(
                    job,
                    rows,
                    panel=panel,
                    spec_sha256="spec",
                    freeze_sha256="freeze",
                    checkpoint_sha256="checkpoint",
                    checkpoint_step=4800,
                )
                for panel, rows in rows_by_panel.items()
            }
            selected_validation = {
                **summaries[formal.PANEL_SELECTION_RECHECK],
                "environment_steps": 4800,
                "constraint_feasible": False,
            }
            training_freeze = {
                "freeze_sha256": "freeze",
                "jobs": {
                    job.job_id: {
                        "selected_validation": selected_validation,
                    }
                },
            }
            manifests = {
                panel: {
                    job.job_id: {
                        "manifest_sha256": f"{panel}-manifest",
                        "summary": summaries[panel],
                    }
                }
                for panel in formal.VALIDATION_PANELS
            }
            for panel in formal.VALIDATION_PANELS:
                manifest_path = formal.validation_shard_paths(args, job, panel)[2]
                atomic_write_json(manifest_path, manifests[panel][job.job_id])
                write_completed_evaluation_status(
                    args,
                    job,
                    panel,
                    manifests[panel][job.job_id],
                )

            analyzed_workloads = []
            validated_workloads = []
            report = {"report_sha256": "report", "validation_gate_passed": True}

            def validate_shard(_args, _job, _spec, _freeze, panel):
                return manifests[panel][job.job_id], rows_by_panel[panel]

            def analyze(rows):
                analyzed_workloads.append(
                    tuple(row["workload_seed"] for row in rows)
                )
                return dict(report)

            def validate_report(_report, *, rows):
                validated_workloads.append(
                    tuple(row["workload_seed"] for row in rows)
                )

            patches = (
                patch.object(formal, "build_jobs", return_value=[job]),
                patch.object(formal, "EXPECTED_TRAINING_JOBS", 1),
                patch.object(formal, "EXPECTED_SELECTION_RECHECK_ROWS", 10),
                patch.object(formal, "EXPECTED_GATE_ROWS", 10),
                patch.object(formal, "EXPECTED_VALIDATION_ROWS", 20),
                patch.object(formal, "validate_training_freeze"),
                patch.object(formal, "assert_runtime_fingerprint"),
                patch.object(
                    formal, "validate_evaluation_shard", side_effect=validate_shard
                ),
                patch.object(
                    formal.formal_statistics,
                    "analyze_validation_rows",
                    side_effect=analyze,
                ),
                patch.object(
                    formal.formal_statistics,
                    "validate_report_hash",
                    side_effect=validate_report,
                ),
            )
            for active_patch in patches:
                active_patch.start()
                self.addCleanup(active_patch.stop)
            freeze = formal.write_validation_freeze(
                args, spec, training_freeze, manifests
            )
            self.assertEqual(
                formal.load_validation_freeze(args, spec, training_freeze), freeze
            )
            self.assertFalse(
                freeze["integrity_gates"]["all_constrained_selections_feasible"]
            )
            self.assertFalse(freeze["formal_validation_gate_passed"])
            self.assertFalse(freeze["eligible_for_separate_test_authorization"])
            expected_gate_workloads = tuple(
                seed
                for seed in formal.GATE_WORKLOAD_SEEDS
            )
            self.assertTrue(analyzed_workloads)
            self.assertTrue(validated_workloads)
            self.assertTrue(
                all(values == expected_gate_workloads for values in analyzed_workloads)
            )
            self.assertTrue(
                all(values == expected_gate_workloads for values in validated_workloads)
            )

            status_path = formal.evaluation_status_path(
                args, job, formal.PANEL_INDEPENDENT_GATE
            )
            original_status = formal.read_json(status_path)
            status_body = dict(original_status)
            status_body.pop("status_sha256")
            status_body["artifact"] = dict(status_body["artifact"])
            status_body["artifact"]["manifest_sha256"] = "tampered"
            atomic_write_json(
                status_path, formal.self_hashed(status_body, "status_sha256")
            )
            with self.assertRaises(formal.ConfigurationError):
                formal.write_validation_freeze(
                    args, spec, training_freeze, manifests
                )
            atomic_write_json(status_path, original_status)

            gate_jsonl = formal.merged_validation_paths(
                args, formal.PANEL_INDEPENDENT_GATE
            )[0]
            original_gate_bytes = gate_jsonl.read_bytes()
            tampered_gate_bytes = original_gate_bytes + b" "
            gate_jsonl.write_bytes(tampered_gate_bytes)
            with self.assertRaises(formal.ConfigurationError):
                formal.write_validation_freeze(
                    args, spec, training_freeze, manifests
                )
            self.assertEqual(gate_jsonl.read_bytes(), tampered_gate_bytes)
            gate_jsonl.write_bytes(original_gate_bytes)

            statistics_path = root / "formal_validation_statistics.json"
            original_statistics = formal.read_json(statistics_path)
            tampered_statistics = dict(original_statistics)
            tampered_statistics["validation_gate_passed"] = False
            tampered_statistics["report_sha256"] = "tampered-self-hash"
            atomic_write_json(statistics_path, tampered_statistics)
            with self.assertRaisesRegex(
                formal.ConfigurationError, "do not recompute"
            ):
                formal.write_validation_freeze(
                    args, spec, training_freeze, manifests
                )
            self.assertEqual(formal.read_json(statistics_path), tampered_statistics)

    def test_interrupt_never_submits_waiting_training_jobs(self):
        jobs = [
            formal.FormalJob(i, formal.SCENARIOS[0], formal.ARMS[i % 3], i + 1)
            for i in range(4)
        ]
        args = SimpleNamespace(max_parallel=2)
        started = []

        def train(_args, job, _spec, cancellation_event: threading.Event):
            started.append(job.index)
            if job.index == 0:
                raise KeyboardInterrupt()
            cancellation_event.wait(timeout=2.0)
            raise formal.TrainingCancelled("cancelled")

        with patch.object(formal, "build_jobs", return_value=jobs), patch.object(
            formal, "audit_no_active_training_children"
        ), patch.object(
            formal,
            "load_training_status",
            return_value={"status": "pending", "attempts": []},
        ), patch.object(formal, "train_job", side_effect=train):
            with self.assertRaises(KeyboardInterrupt):
                formal.run_all_training_jobs(args, {"spec_sha256": "spec"})
        self.assertTrue(started)
        self.assertFalse(set(started) & {2, 3})

    def test_training_stage_preflight_rejects_exhausted_job_before_submission(self):
        jobs = [
            formal.FormalJob(i, formal.SCENARIOS[0], formal.ARMS[i], i + 1)
            for i in range(2)
        ]
        pending = {"status": "pending", "attempts": []}
        for state in ("failed", "running"):
            with self.subTest(exhausted_status=state):
                exhausted = {"status": state, "attempts": [{}, {}]}
                with patch.object(
                    formal, "build_jobs", return_value=jobs
                ), patch.object(
                    formal, "audit_no_active_training_children"
                ), patch.object(
                    formal,
                    "load_training_status",
                    side_effect=[pending, exhausted],
                ), patch.object(
                    formal, "ThreadPoolExecutor"
                ) as executor, patch.object(
                    formal, "train_job"
                ) as train:
                    with self.assertRaisesRegex(
                        formal.ConfigurationError, "retry limit exhausted"
                    ):
                        formal.run_all_training_jobs(
                            SimpleNamespace(max_parallel=2),
                            {"spec_sha256": "spec"},
                        )
                executor.assert_not_called()
                train.assert_not_called()

    def test_validation_stage_preflight_rejects_exhausted_shard_before_submission(self):
        job = formal.FormalJob(
            0, formal.SCENARIOS[0], formal.ARM_BASELINE, formal.POLICY_SEEDS[0]
        )
        for state in ("failed", "running"):
            with self.subTest(exhausted_status=state):
                exhausted = {"status": state, "attempts": [{}, {}]}
                with patch.object(
                    formal, "build_jobs", return_value=[job]
                ), patch.object(
                    formal, "validate_training_freeze"
                ), patch.object(
                    formal, "load_evaluation_status", return_value=exhausted
                ), patch.object(
                    formal, "ThreadPoolExecutor"
                ) as executor, patch.object(
                    formal, "evaluate_validation_job"
                ) as evaluate:
                    with self.assertRaisesRegex(
                        formal.ConfigurationError, "retry limit exhausted"
                    ):
                        formal.run_all_validation_jobs(
                            SimpleNamespace(max_parallel=2),
                            {"spec_sha256": "spec"},
                            {"freeze_sha256": "freeze"},
                        )
                executor.assert_not_called()
                evaluate.assert_not_called()

    def test_completed_success_is_not_replaced_before_done_failure_is_drained(self):
        jobs = [
            formal.FormalJob(i, formal.SCENARIOS[0], formal.ARMS[i % 3], i + 1)
            for i in range(4)
        ]
        submitted = []

        class CompletedBatchExecutor:
            def __init__(self, *, max_workers):
                self.max_workers = max_workers

            def submit(self, _function, _args, job, _spec, _cancellation):
                submitted.append(job.index)
                future = Future()
                if job.index == 0:
                    future.set_result({"job": job.job_id})
                elif job.index == 1:
                    future.set_exception(RuntimeError("simultaneous failure"))
                else:
                    future.set_result({"job": job.job_id})
                return future

            def shutdown(self, *, wait, cancel_futures):
                self.wait = wait
                self.cancel_futures = cancel_futures

        with patch.object(formal, "build_jobs", return_value=jobs), patch.object(
            formal, "audit_no_active_training_children"
        ), patch.object(
            formal,
            "load_training_status",
            return_value={"status": "pending", "attempts": []},
        ), patch.object(formal, "ThreadPoolExecutor", CompletedBatchExecutor):
            with self.assertRaisesRegex(RuntimeError, "simultaneous failure"):
                formal.run_all_training_jobs(
                    SimpleNamespace(max_parallel=2), {"spec_sha256": "spec"}
                )
        self.assertEqual(submitted, [0, 1])


if __name__ == "__main__":
    unittest.main()
