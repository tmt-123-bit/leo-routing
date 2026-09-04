from __future__ import annotations

import ast
from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from mappo_evaluation import ConstraintEpisodeMetrics, GlobalDijkstraPolicy, OspfEcmpPolicy
import run_avoidable_switch_classical_baselines_formal as classical


def fake_spec() -> dict:
    return {
        "spec_sha256": "a" * 64,
        "code_fingerprint": {"frozen": True},
    }


def fake_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=root,
        project=Path(classical.__file__).resolve().parent,
        max_parallel=1,
    )


def fake_training_freeze() -> dict:
    return {
        "freeze_sha256": "b" * 64,
        "jobs": {},
    }


def metric(job: classical.EvaluationJob, workload_seed: int) -> ConstraintEpisodeMetrics:
    return ConstraintEpisodeMetrics(
        scenario=job.scenario,
        policy=job.method,
        policy_seed=job.policy_seed,
        workload_seed=workload_seed,
        generated=10,
        delivered=8,
        dropped=1,
        backlog=1,
        delivery_ratio=0.8,
        drop_rate=0.1,
        throughput_packets_per_slot=0.3,
        average_delay_slots=2.0,
        p95_delay_slots=3.0,
        mean_queue_packets=1.5,
        max_queue_packets=4,
        routing_switches=3,
        episode_reward=1.0,
        global_delay_cost=0.2,
        global_queue_cost=0.1,
        global_load_imbalance=0.05,
        global_switch_cost=0.0,
        global_throughput_reward=0.4,
        global_control_overhead_ratio=0.01,
        global_drop_cost=0.1,
        class_0_delivery_ratio=0.8,
        class_1_delivery_ratio=0.75,
        class_2_delivery_ratio=1.0,
        avoidable_routing_switches=2,
        forced_routing_switches=1,
        switch_opportunities=4,
        avoidable_switch_rate=0.5,
        decision_avoidable_switches=3,
        decision_switch_opportunities=5,
        decision_forced_switches=2,
        decision_avoidable_switch_rate=0.6,
    )


def gate_rows(
    job: classical.EvaluationJob,
    spec: dict | None = None,
    freeze: dict | None = None,
) -> list[dict]:
    spec = fake_spec() if spec is None else spec
    freeze = fake_training_freeze() if freeze is None else freeze
    rows = []
    for workload_seed in classical.GATE_WORKLOAD_SEEDS:
        rows.append(
            {
                "schema_version": classical.SCHEMA_VERSION,
                "study_name": classical.STUDY_NAME,
                "evaluation_role": "classical_independent_gate_descriptive",
                "spec_sha256": spec["spec_sha256"],
                "training_freeze_sha256": freeze["freeze_sha256"],
                "method": job.method,
                "replicate_kind": job.replicate_kind,
                "environment_variant": classical.ENVIRONMENT_VARIANT,
                "source_model_sha256": None,
                "decision_forced_switch_cost": 0,
                **asdict(metric(job, workload_seed)),
            }
        )
    return rows


class FrozenRegistryTests(unittest.TestCase):
    def test_training_and_evaluation_grids_are_exact(self) -> None:
        classical.validate_seed_registry()
        training = classical.build_training_jobs()
        evaluations = classical.build_evaluation_jobs()
        self.assertEqual(len(training), 16)
        self.assertEqual(len(evaluations), 34)
        self.assertEqual(classical.EXPECTED_Q_TRAIN_EPISODES, 8000)
        self.assertEqual(classical.EXPECTED_GATE_ROWS, 340)
        self.assertEqual(classical.EXPECTED_Q_GATE_ROWS, 160)
        self.assertEqual(classical.EXPECTED_OSPF_GATE_ROWS, 160)
        self.assertEqual(classical.EXPECTED_DIJKSTRA_GATE_ROWS, 20)

    def test_q_workloads_traverse_two_cycles_plus_first_hundred(self) -> None:
        order = classical.q_training_workload_order()
        self.assertEqual(len(order), 500)
        self.assertEqual(
            order,
            classical.TRAIN_WORKLOAD_SEEDS * 2
            + classical.TRAIN_WORKLOAD_SEEDS[:100],
        )

    def test_dijkstra_is_not_duplicated_as_eight_seeds(self) -> None:
        dijkstra = [
            job
            for job in classical.build_evaluation_jobs()
            if job.method == classical.GLOBAL_DIJKSTRA
        ]
        self.assertEqual(len(dijkstra), len(classical.SCENARIOS))
        self.assertTrue(
            all(job.policy_seed == classical.GLOBAL_DIJKSTRA_SENTINEL_SEED for job in dijkstra)
        )
        self.assertTrue(
            all(job.replicate_kind == "deterministic_workload_reference" for job in dijkstra)
        )

    def test_ospf_seeds_are_labeled_routing_replicates(self) -> None:
        ospf = [
            job
            for job in classical.build_evaluation_jobs()
            if job.method == classical.OSPF_ECMP
        ]
        self.assertEqual(len(ospf), 16)
        self.assertTrue(
            all(job.replicate_kind == "stochastic_routing_replicate" for job in ospf)
        )

    def test_only_exact_gate_workloads_are_executable(self) -> None:
        self.assertEqual(
            classical.validate_gate_workloads(classical.GATE_WORKLOAD_SEEDS),
            classical.GATE_WORKLOAD_SEEDS,
        )
        with self.assertRaises(classical.ConfigurationError):
            classical.validate_gate_workloads(classical.SELECTION_WORKLOAD_SEEDS_METADATA_ONLY)
        with self.assertRaises(classical.ConfigurationError):
            classical.validate_gate_workloads(range(78001, 78051))

    def test_dry_run_counts_zero_selection_and_sealed_evaluations(self) -> None:
        summary = classical.dry_run_summary(fake_spec(), Path("formal-output"))
        self.assertEqual(summary["q_training_jobs"], 16)
        self.assertEqual(summary["q_training_episodes"], 8000)
        self.assertEqual(summary["checkpoint_selection_evaluations"], 0)
        self.assertEqual(summary["gate_evaluation_jobs"], 34)
        self.assertEqual(summary["gate_evaluations"], 340)
        self.assertEqual(summary["sealed_test_evaluations"], 0)
        self.assertIs(summary["test_panel_consulted"], False)

    def test_cli_exposes_no_panel_or_seed_override(self) -> None:
        parsed = classical.parse_args(["--dry-run"])
        self.assertFalse(hasattr(parsed, "panel"))
        self.assertFalse(hasattr(parsed, "workload_seed"))
        self.assertFalse(hasattr(parsed, "method"))
        self.assertFalse(hasattr(parsed, "project"))

    def test_main_dry_run_never_enters_training_or_gate_execution(self) -> None:
        output = io.StringIO()
        with (
            patch.object(classical, "validate_runtime_environment"),
            patch.object(classical, "validate_output_isolation"),
            patch.object(classical, "validate_new_output_state"),
            patch.object(classical, "build_spec", return_value=fake_spec()),
            patch.object(
                classical,
                "run_all_training_jobs",
                side_effect=AssertionError("training reached from dry-run"),
            ),
            patch.object(
                classical,
                "run_all_gate_jobs",
                side_effect=AssertionError("gate reached from dry-run"),
            ),
            redirect_stdout(output),
        ):
            result = classical.main(["--dry-run"])
        self.assertEqual(result, 0)
        observed = json.loads(output.getvalue())
        self.assertEqual(observed["gate_evaluations"], 340)
        self.assertEqual(observed["sealed_test_evaluations"], 0)


class SourceContractTests(unittest.TestCase):
    def test_source_contract_binds_distinct_q_formulas(self) -> None:
        contract = classical.source_contract()
        q = contract["q_routing"]
        self.assertIn("candidate_feature[4]", q["action_score_formula"])
        self.assertIn("link_load_rho", q["action_score_components"])
        self.assertNotIn("link_load_rho", q["transition_target_formula"])
        self.assertEqual(
            contract["object_source_sha256"]["QRoutingPolicy"],
            classical.Q_ROUTING_CLASS_SOURCE_SHA256,
        )

    def test_source_hash_drift_fails_closed(self) -> None:
        with patch.object(classical, "Q_ROUTING_CLASS_SOURCE_SHA256", "0" * 64):
            with self.assertRaises(classical.ConfigurationError):
                classical.source_contract()

    def test_spec_records_no_classical_selection(self) -> None:
        args = SimpleNamespace(
            project=Path(classical.__file__).resolve().parent,
            output=Path("unused"),
            max_parallel=2,
        )
        with patch.object(classical, "code_fingerprint", return_value={"frozen": True}):
            spec = classical.build_spec(args)
        classical.validate_self_hash(spec, "spec_sha256")
        self.assertEqual(spec["execution"]["checkpoint_selection_evaluations"], 0)
        self.assertEqual(spec["execution"]["sealed_test_evaluations"], 0)
        self.assertIs(spec["performance_gate_applied"], False)
        self.assertEqual(
            spec["q_routing"]["action_score_formula"],
            classical.Q_ACTION_SCORE_FORMULA,
        )
        self.assertEqual(
            spec["q_routing"]["transition_target_formula"],
            classical.Q_TRANSITION_TARGET_FORMULA,
        )

    def test_sibling_project_source_is_rejected(self) -> None:
        sibling = classical.ACTUAL_PROJECT_SOURCE.parent / "sibling" / "src"
        with self.assertRaises(classical.ConfigurationError):
            classical.validate_project_source(sibling)
        self.assertEqual(
            classical.validate_project_source(classical.ACTUAL_PROJECT_SOURCE),
            classical.ACTUAL_PROJECT_SOURCE,
        )

    def test_dependency_inventory_is_normalized_and_hashed(self) -> None:
        args = fake_args(Path("unused"))
        process = SimpleNamespace(stdout="zeta==2\nAlpha==1\nzeta==2\n")
        with (
            patch.object(classical, "_git_output", return_value="head"),
            patch.object(classical, "sha256_file", return_value="f" * 64),
            patch.object(classical.subprocess, "run", return_value=process),
        ):
            fingerprint = classical.code_fingerprint(args)
        inventory = fingerprint["dependencies"]["python_dependency_inventory"]
        self.assertEqual(inventory, ["Alpha==1", "zeta==2"])
        expected = classical.hashlib.sha256(b"Alpha==1\nzeta==2\n").hexdigest()
        self.assertEqual(
            fingerprint["dependencies"]["python_dependency_inventory_sha256"],
            expected,
        )

    def test_runtime_source_allowlist_covers_local_import_closure(self) -> None:
        project = classical.ACTUAL_PROJECT_SOURCE
        allowlist = classical.runtime_source_paths(project)
        allowlisted_paths = {path.resolve() for path in allowlist.values()}
        self.assertEqual(allowlist["design"], project / "mappo_design.py")
        for source in sorted(path for path in allowlisted_paths if path.suffix == ".py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            imported_modules = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported_modules.add(node.module.split(".")[0])
            local_dependencies = {
                (project / f"{module}.py").resolve()
                for module in imported_modules
                if (project / f"{module}.py").is_file()
            }
            self.assertTrue(
                local_dependencies <= allowlisted_paths,
                f"{source.name} imports unhashed local files: "
                f"{sorted(str(path) for path in local_dependencies - allowlisted_paths)}",
            )

    def test_design_tamper_changes_code_fingerprint(self) -> None:
        args = fake_args(Path("unused"))
        process = SimpleNamespace(stdout="dependency==1\n")

        def fingerprint_with_design_hash(design_hash: str) -> dict:
            def source_hash(path: Path) -> str:
                return design_hash if Path(path).name == "mappo_design.py" else "f" * 64

            with (
                patch.object(classical, "_git_output", return_value="head"),
                patch.object(classical, "sha256_file", side_effect=source_hash),
                patch.object(classical.subprocess, "run", return_value=process),
            ):
                return classical.code_fingerprint(args)

        original = fingerprint_with_design_hash("d" * 64)
        tampered = fingerprint_with_design_hash("e" * 64)
        self.assertEqual(original["files"]["design"], "d" * 64)
        self.assertEqual(tampered["files"]["design"], "e" * 64)
        self.assertNotEqual(
            classical.sha256_json(original), classical.sha256_json(tampered)
        )


class ModelPersistenceTests(unittest.TestCase):
    def test_q_model_round_trip_restores_array_and_rng(self) -> None:
        job = classical.build_training_jobs()[0]
        spec = fake_spec()
        policy = classical._fresh_q_policy(job)
        policy.q[1, 2, 3] = np.float32(7.25)
        policy.rng.random()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.npz"
            classical._atomic_write_model(
                path, job, spec, policy, 17, frozen=False
            )
            loaded, metadata = classical.load_q_model(
                path, job, spec, require_frozen=False
            )
        self.assertTrue(np.array_equal(loaded.q, policy.q))
        self.assertEqual(metadata["completed_episodes"], 17)
        self.assertEqual(loaded.rng.random(), policy.rng.random())

    def test_q_array_tampering_is_rejected(self) -> None:
        job = classical.build_training_jobs()[0]
        spec = fake_spec()
        policy = classical._fresh_q_policy(job)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            classical._atomic_write_model(
                path, job, spec, policy, 1, frozen=False
            )
            with np.load(path, allow_pickle=False) as archive:
                q = np.array(archive["q"], copy=True)
                metadata = np.array(archive["metadata"], copy=True)
            q[1, 1, 1] += np.float32(1.0)
            with path.open("wb") as handle:
                np.savez_compressed(handle, q=q, metadata=metadata)
            with self.assertRaises(classical.ConfigurationError):
                classical.load_q_model(path, job, spec, require_frozen=False)

    def test_training_retry_resumes_from_immutable_episode_checkpoint(self) -> None:
        job = classical.build_training_jobs()[0]
        spec = fake_spec()
        calls = []

        def episode(policy, observed_job, workload_seed, wrapper_factory):
            calls.append(workload_seed)
            policy.q[1, 1, 1] += np.float32(1.0)
            policy.rng.random()
            if len(calls) == 2:
                raise OSError(errno.EIO, "injected infrastructure failure")

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "Q_TRAIN_EPISODES", 2),
                patch.object(classical, "_train_one_episode", side_effect=episode),
                patch.object(classical, "assert_runtime_fingerprint"),
            ):
                manifest = classical.train_q_job(
                    args,
                    job,
                    spec,
                    threading.Event(),
                    wrapper_factory=object,
                )
                status = classical.read_json(classical.training_status_path(args, job))
                final_policy, _ = classical.load_q_model(
                    Path(manifest["model"]["path"]),
                    job,
                    spec,
                    require_frozen=True,
                )
                resume_path = (
                    classical.training_job_root(args, job)
                    / "resume_source_attempt_2.npz"
                )
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(status["attempts"]), 2)
        self.assertEqual(status["attempts"][0]["status"], "failed")
        self.assertEqual(
            status["attempts"][0]["failure_category"],
            "recognized_infrastructure_failure",
        )
        self.assertIs(status["attempts"][0]["retry_authorized"], True)
        self.assertEqual(status["attempts"][1]["resumed_from_episode"], 1)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(float(final_policy.q[1, 1, 1]), 12.0)
        self.assertTrue(resume_path.is_file() or manifest["resume_sources"])

    def test_training_scientific_failure_is_fail_fast_across_reentry(self) -> None:
        job = classical.build_training_jobs()[0]
        spec = fake_spec()
        calls = []

        def episode(policy, observed_job, workload_seed, wrapper_factory):
            calls.append(workload_seed)
            raise ValueError("invalid scientific state")

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "Q_TRAIN_EPISODES", 2),
                patch.object(classical, "_train_one_episode", side_effect=episode),
                patch.object(classical, "assert_runtime_fingerprint"),
            ):
                with self.assertRaises(ValueError):
                    classical.train_q_job(
                        args, job, spec, threading.Event(), wrapper_factory=object
                    )
                with self.assertRaises(classical.ConfigurationError):
                    classical.train_q_job(
                        args, job, spec, threading.Event(), wrapper_factory=object
                    )
            status = classical.read_json(classical.training_status_path(args, job))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(status["attempts"]), 1)
        self.assertEqual(
            status["attempts"][0]["failure_category"],
            "scientific_or_runtime_failure",
        )
        self.assertIs(status["attempts"][0]["retry_authorized"], False)

    def test_replace_before_status_commit_is_reconciled_from_checkpoint(self) -> None:
        job = classical.build_training_jobs()[0]
        spec = fake_spec()
        policy = classical._fresh_q_policy(job)
        policy.q[1, 2, 3] = np.float32(6.5)
        policy.rng.random()
        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            status_path = classical.training_status_path(args, job)
            latest = classical.training_job_root(args, job) / "latest.npz"
            status = classical._new_status(job, spec["spec_sha256"])
            classical._begin_attempt(status, status_path)
            transaction = classical._prepare_latest_checkpoint(
                status,
                status_path,
                latest,
                job,
                spec,
                policy,
                1,
                frozen=False,
            )
            classical._atomic_write_model(
                latest, job, spec, policy, 1, frozen=False
            )
            # Simulate process death after os.replace and before status commit.
            persisted = classical._load_status(
                status_path, job, spec["spec_sha256"]
            )
            self.assertEqual(
                persisted["attempts"][-1]["checkpoint_transaction"]["state"],
                "prepared",
            )
            classical._recover_training_status(
                persisted, status_path, latest, job, spec
            )
            recovered_policy, metadata = classical.load_q_model(
                latest, job, spec, require_frozen=False
            )
            recovered_status = classical._load_status(
                status_path, job, spec["spec_sha256"]
            )
        recovered_attempt = recovered_status["attempts"][-1]
        self.assertEqual(transaction["target_model_state_sha256"], metadata["model_state_sha256"])
        self.assertEqual(
            recovered_attempt["checkpoint_transaction"]["state"],
            "recovered_committed",
        )
        self.assertEqual(recovered_attempt["status"], "interrupted")
        self.assertEqual(recovered_status["status"], "interrupted")
        self.assertEqual(float(recovered_policy.q[1, 2, 3]), 6.5)
        self.assertEqual(recovered_policy.rng.random(), policy.rng.random())

    def test_completed_training_status_finalizes_missing_manifest_without_retraining(self) -> None:
        job = classical.build_training_jobs()[0]
        spec = fake_spec()
        calls = []

        def episode(policy, observed_job, workload_seed, wrapper_factory):
            calls.append(workload_seed)
            policy.q[1, 1, 1] += np.float32(1.0)

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "Q_TRAIN_EPISODES", 1),
                patch.object(classical, "_train_one_episode", side_effect=episode),
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "_write_training_manifest",
                    side_effect=RuntimeError("crash before manifest write"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    classical.train_q_job(
                        args, job, spec, threading.Event(), wrapper_factory=object
                    )
            status = classical.read_json(classical.training_status_path(args, job))
            self.assertEqual(status["status"], "completed")
            self.assertFalse(classical._training_manifest_path(args, job).exists())
            with (
                patch.object(classical, "Q_TRAIN_EPISODES", 1),
                patch.object(
                    classical,
                    "_train_one_episode",
                    side_effect=AssertionError("recovery retrained an episode"),
                ),
                patch.object(classical, "assert_runtime_fingerprint"),
            ):
                manifest = classical.train_q_job(
                    args, job, spec, threading.Event(), wrapper_factory=object
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(manifest["attempt_count"], 1)

    def test_third_attempt_is_forbidden(self) -> None:
        job = classical.build_training_jobs()[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            status = classical._new_status(job, "a" * 64)
            classical._begin_attempt(status, path)
            classical._finish_attempt(
                status,
                path,
                "failed",
                error="one",
                failure_category="recognized_infrastructure_failure",
                retry_authorized=True,
            )
            classical._begin_attempt(status, path)
            classical._finish_attempt(
                status,
                path,
                "failed",
                error="two",
                failure_category="recognized_infrastructure_failure",
                retry_authorized=True,
            )
            with self.assertRaises(classical.ConfigurationError):
                classical._begin_attempt(status, path)


class RetryClassificationTests(unittest.TestCase):
    def test_fail_fast_cancellation_inherits_only_infrastructure_cause(self) -> None:
        infrastructure = threading.Event()
        setattr(infrastructure, "failure_cause", TimeoutError("peer timeout"))
        infrastructure_error = classical._job_cancelled("cancelled", infrastructure)
        self.assertTrue(infrastructure_error.infrastructure_cause)
        self.assertEqual(
            classical._failure_category(infrastructure_error),
            "recognized_infrastructure_cancellation",
        )

        scientific = threading.Event()
        setattr(scientific, "failure_cause", ValueError("peer scientific failure"))
        scientific_error = classical._job_cancelled("cancelled", scientific)
        self.assertFalse(scientific_error.infrastructure_cause)
        self.assertEqual(
            classical._failure_category(scientific_error), "execution_interruption"
        )


class InvocationLockTests(unittest.TestCase):
    def _payload(self, output: Path, pid: int) -> dict:
        return {
            "schema_version": 1,
            "study_name": classical.STUDY_NAME,
            "pid": pid,
            "host": socket.gethostname(),
            "output": str(output.resolve()),
            "token": "stale-token",
            "created_at_utc": "2026-09-03T00:00:00Z",
        }

    def test_lock_records_identity_and_rejects_live_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            lock_path = output / ".classical_runner.lock"
            with classical.invocation_lock(output):
                payload = classical.read_json(lock_path)
                self.assertEqual(payload["pid"], os.getpid())
                self.assertEqual(payload["host"], socket.gethostname())
                self.assertEqual(payload["output"], str(output))
                self.assertIsInstance(payload["created_at_utc"], str)
                with patch.object(classical, "_pid_is_active", return_value=True):
                    with self.assertRaises(classical.ConfigurationError):
                        with classical.invocation_lock(output):
                            pass
            self.assertFalse(lock_path.exists())

    def test_dead_same_output_lock_is_safely_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            lock_path = output / ".classical_runner.lock"
            classical.atomic_write_json(lock_path, self._payload(output, 987654321))
            with patch.object(classical, "_pid_is_active", return_value=False):
                with classical.invocation_lock(output):
                    replacement = classical.read_json(lock_path)
                    self.assertEqual(replacement["pid"], os.getpid())
                    self.assertNotEqual(replacement["token"], "stale-token")
            self.assertFalse(lock_path.exists())

    def test_dead_foreign_output_lock_is_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            lock_path = output / ".classical_runner.lock"
            payload = self._payload(output, 987654321)
            payload["output"] = str(output / "other")
            classical.atomic_write_json(lock_path, payload)
            with patch.object(classical, "_pid_is_active", return_value=False):
                with self.assertRaises(classical.ConfigurationError):
                    with classical.invocation_lock(output):
                        pass
            self.assertEqual(classical.read_json(lock_path), payload)

    def test_lock_setup_guard_rejects_concurrent_stale_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            lock_path = output / ".classical_runner.lock"
            payload = self._payload(output, 987654321)
            classical.atomic_write_json(lock_path, payload)
            with classical._invocation_lock_guard(output):
                with self.assertRaises(classical.ConfigurationError):
                    with classical.invocation_lock(output):
                        pass
            self.assertEqual(classical.read_json(lock_path), payload)


class FreezeAuditTests(unittest.TestCase):
    def test_descriptive_summary_is_identity_equal_and_preserves_zero_denominator(self) -> None:
        rows = [
            {
                "scenario": classical.SCENARIOS[0],
                "method": classical.Q_ROUTING,
                "policy_seed": 1,
                "generated": 10,
                "delivered": 5,
                "delivery_ratio": 0.5,
                "decision_avoidable_switches": 0,
                "decision_switch_opportunities": 0,
            },
            {
                "scenario": classical.SCENARIOS[0],
                "method": classical.Q_ROUTING,
                "policy_seed": 1,
                "generated": 100,
                "delivered": 70,
                "delivery_ratio": 0.7,
                "decision_avoidable_switches": 0,
                "decision_switch_opportunities": 0,
            },
            {
                "scenario": classical.SCENARIOS[0],
                "method": classical.Q_ROUTING,
                "policy_seed": 2,
                "generated": 10,
                "delivered": 9,
                "delivery_ratio": 0.9,
                "decision_avoidable_switches": 1,
                "decision_switch_opportunities": 2,
            },
        ]
        summary = classical._descriptive_summaries(rows)[
            f"{classical.SCENARIOS[0]}/{classical.Q_ROUTING}"
        ]
        identity = summary["policy_identity_equal_descriptive"]
        self.assertEqual(identity["delivery_ratio"], 0.75)
        self.assertIsNone(identity["decision_avoidable_switch_rate"])
        self.assertEqual(identity["undefined_rate_policy_seeds"], [1])
        self.assertEqual(
            summary["pooled_count_audit"]["inferential_role"],
            "pooled_integrity_audit_only",
        )

    def test_training_freeze_rejects_total_episode_drift(self) -> None:
        job = classical.build_training_jobs()[0]
        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            manifest_path = classical._training_manifest_path(args, job)
            classical.atomic_write_json(manifest_path, {"artifact": True})
            manifest = {
                "manifest_sha256": "c" * 64,
                "model": {"path": "model", "sha256": "d" * 64},
                "status": {"path": "status", "sha256": "e" * 64},
                "attempt_count": 1,
            }
            entry = {
                "manifest_path": str(manifest_path.resolve()),
                "manifest_file_sha256": classical.sha256_file(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "model": manifest["model"],
                "status": manifest["status"],
                "attempt_count": 1,
            }
            body = {
                "schema_version": classical.SCHEMA_VERSION,
                "study_name": classical.STUDY_NAME,
                "spec_sha256": fake_spec()["spec_sha256"],
                "training_complete": True,
                "training_job_count": 1,
                "total_training_episodes": 2,
                "jobs": {job.job_id: entry},
                "test_panel_consulted": False,
                "test_access_count": 0,
                "sealed_test_instantiated": False,
                "completed_at_utc": "2026-09-03T00:00:00Z",
            }
            freeze = classical.self_hashed(body, "freeze_sha256")
            with (
                patch.object(classical, "EXPECTED_TRAINING_JOBS", 1),
                patch.object(classical, "EXPECTED_Q_TRAIN_EPISODES", 2),
                patch.object(classical, "build_training_jobs", return_value=[job]),
                patch.object(
                    classical, "validate_training_manifest", return_value=manifest
                ),
            ):
                classical.validate_training_freeze(args, fake_spec(), freeze)
                drifted = dict(freeze)
                drifted.pop("freeze_sha256")
                drifted["total_training_episodes"] = 3
                drifted = classical.self_hashed(drifted, "freeze_sha256")
                with self.assertRaises(classical.ConfigurationError):
                    classical.validate_training_freeze(args, fake_spec(), drifted)

    def test_gate_freeze_rejects_method_counts_and_external_merged_path(self) -> None:
        job = next(
            item
            for item in classical.build_evaluation_jobs()
            if item.method == classical.OSPF_ECMP
        )
        spec = fake_spec()
        training_freeze = fake_training_freeze()
        rows = gate_rows(job, spec, training_freeze)
        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            merged_jsonl = args.output / "classical_gate_rows.jsonl"
            merged_csv = args.output / "classical_gate_rows.csv"
            classical._atomic_write_jsonl(merged_jsonl, rows)
            classical.atomic_write_csv(
                merged_csv, rows, fieldnames=classical.GATE_ROW_FIELDS
            )
            shard_path = classical.evaluation_shard_paths(args, job)[2]
            classical.atomic_write_json(shard_path, {"artifact": True})
            shard_manifest = {
                "manifest_sha256": "f" * 64,
                "status": {"path": "status", "sha256": "1" * 64},
                "attempt_count": 1,
            }
            expected_counts = {
                classical.Q_ROUTING: 0,
                classical.OSPF_ECMP: 10,
                classical.GLOBAL_DIJKSTRA: 0,
            }
            body = {
                "schema_version": classical.SCHEMA_VERSION,
                "study_name": classical.STUDY_NAME,
                "spec_sha256": spec["spec_sha256"],
                "training_freeze_sha256": training_freeze["freeze_sha256"],
                "gate_complete": True,
                "gate_row_count": 10,
                "gate_rows_by_method": expected_counts,
                "checkpoint_selection_evaluations": 0,
                "performance_gate_applied": False,
                "descriptive_summaries": classical._descriptive_summaries(rows),
                "merged_jsonl": {
                    "path": str(merged_jsonl.resolve()),
                    "sha256": classical.sha256_file(merged_jsonl),
                },
                "merged_csv": {
                    "path": str(merged_csv.resolve()),
                    "sha256": classical.sha256_file(merged_csv),
                },
                "shards": {
                    job.job_id: {
                        "manifest_path": str(shard_path.resolve()),
                        "manifest_file_sha256": classical.sha256_file(shard_path),
                        "manifest_sha256": shard_manifest["manifest_sha256"],
                        "status": shard_manifest["status"],
                        "attempt_count": 1,
                    }
                },
                "test_panel_consulted": False,
                "test_access_count": 0,
                "sealed_test_instantiated": False,
                "sealed_test_access_authorized": False,
                "paper_claim_allowed": False,
                "promotion_decision_allowed": False,
                "completed_at_utc": "2026-09-03T00:00:00Z",
            }

            def write_freeze(value: dict) -> None:
                classical.atomic_write_json(
                    args.output / "classical_gate_freeze.json",
                    classical.self_hashed(value, "gate_freeze_sha256"),
                )

            with (
                patch.object(classical, "EXPECTED_GATE_ROWS", 10),
                patch.object(classical, "EXPECTED_Q_GATE_ROWS", 0),
                patch.object(classical, "EXPECTED_OSPF_GATE_ROWS", 10),
                patch.object(classical, "EXPECTED_DIJKSTRA_GATE_ROWS", 0),
                patch.object(classical, "build_evaluation_jobs", return_value=[job]),
                patch.object(
                    classical,
                    "validate_evaluation_shard",
                    return_value=(shard_manifest, rows),
                ),
            ):
                write_freeze(dict(body))
                classical.validate_gate_freeze(args, spec, training_freeze)

                bad_counts = dict(body)
                bad_counts["gate_rows_by_method"] = {
                    **expected_counts,
                    classical.OSPF_ECMP: 9,
                }
                write_freeze(bad_counts)
                with self.assertRaises(classical.ConfigurationError):
                    classical.validate_gate_freeze(args, spec, training_freeze)

                external = args.output.parent / "external-gate.jsonl"
                classical._atomic_write_jsonl(external, rows)
                bad_path = dict(body)
                bad_path["merged_jsonl"] = {
                    "path": str(external.resolve()),
                    "sha256": classical.sha256_file(external),
                }
                write_freeze(bad_path)
                with self.assertRaises(classical.ConfigurationError):
                    classical.validate_gate_freeze(args, spec, training_freeze)


class GateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job = next(
            job
            for job in classical.build_evaluation_jobs()
            if job.method == classical.OSPF_ECMP
        )
        self.spec = fake_spec()
        self.freeze = fake_training_freeze()

    def test_valid_rows_reconstruct_integer_rates(self) -> None:
        summary = classical.validate_gate_rows(
            self.job,
            gate_rows(self.job, self.spec, self.freeze),
            self.spec,
            self.freeze,
        )
        self.assertEqual(summary["row_count"], 10)
        self.assertEqual(summary["decision_avoidable_switches"], 30)
        self.assertEqual(summary["decision_switch_opportunities"], 50)
        self.assertEqual(summary["decision_avoidable_switch_rate"], 0.6)

    def test_workload_substitution_is_rejected(self) -> None:
        rows = gate_rows(self.job, self.spec, self.freeze)
        rows[-1]["workload_seed"] = rows[-2]["workload_seed"]
        with self.assertRaises(classical.ConfigurationError):
            classical.validate_gate_rows(self.job, rows, self.spec, self.freeze)

    def test_decision_ledger_tampering_is_rejected(self) -> None:
        rows = gate_rows(self.job, self.spec, self.freeze)
        rows[0]["decision_avoidable_switches"] = 6
        with self.assertRaises(classical.ConfigurationError):
            classical.validate_gate_rows(self.job, rows, self.spec, self.freeze)

    def test_physical_metric_bounds_are_checked(self) -> None:
        mutations = (
            ("mean_queue_packets", 5.0),
            ("global_delay_cost", -0.1),
            ("global_queue_cost", -0.1),
            ("global_load_imbalance", -0.1),
            ("global_drop_cost", -0.1),
            ("global_control_overhead_ratio", 1.1),
            ("throughput_packets_per_slot", -0.1),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                rows = gate_rows(self.job, self.spec, self.freeze)
                rows[0][field] = value
                with self.assertRaises(classical.ConfigurationError):
                    classical.validate_gate_rows(
                        self.job, rows, self.spec, self.freeze
                    )

    def test_gate_job_uses_one_seeded_policy_across_exact_gate_tuple(self) -> None:
        captured = []

        def evaluate(**kwargs):
            captured.append(dict(kwargs))
            return [metric(self.job, seed) for seed in kwargs["workload_seeds"]]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                manifest = classical.evaluate_gate_job(
                    args,
                    self.job,
                    self.spec,
                    self.freeze,
                    threading.Event(),
                )
        self.assertEqual(len(captured), len(classical.GATE_WORKLOAD_SEEDS))
        self.assertTrue(all(isinstance(call["policy"], OspfEcmpPolicy) for call in captured))
        self.assertEqual(len({id(call["policy"]) for call in captured}), 1)
        self.assertTrue(all(call["policy_seed"] == self.job.policy_seed for call in captured))
        self.assertEqual(
            tuple(call["workload_seeds"][0] for call in captured),
            classical.GATE_WORKLOAD_SEEDS,
        )
        self.assertTrue(
            all(call["variant"] == classical.ENVIRONMENT_VARIANT for call in captured)
        )
        self.assertEqual(manifest["checkpoint_selection_evaluations"], 0)
        self.assertIs(manifest["performance_gate_applied"], False)
        self.assertEqual(manifest["attempt_count"], 1)
        self.assertIn("status", manifest)

    def test_gate_recognized_infrastructure_failure_retries_once(self) -> None:
        calls = []

        def evaluate(**kwargs):
            workload_seed = kwargs["workload_seeds"][0]
            calls.append(workload_seed)
            if len(calls) == 1:
                raise TimeoutError("transient evaluator timeout")
            return [metric(self.job, workload_seed)]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                manifest = classical.evaluate_gate_job(
                    args, self.job, self.spec, self.freeze, threading.Event()
                )
            status = classical.read_json(classical.evaluation_status_path(args, self.job))
        self.assertEqual(len(calls), len(classical.GATE_WORKLOAD_SEEDS) + 1)
        self.assertEqual(manifest["attempt_count"], 2)
        self.assertEqual(
            status["attempts"][0]["failure_category"],
            "recognized_infrastructure_failure",
        )
        self.assertIs(status["attempts"][0]["retry_authorized"], True)

    def test_gate_scientific_failure_is_fail_fast_across_reentry(self) -> None:
        calls = []

        def evaluate(**kwargs):
            calls.append(kwargs["workload_seeds"][0])
            raise ValueError("invalid evaluator state")

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                with self.assertRaises(ValueError):
                    classical.evaluate_gate_job(
                        args, self.job, self.spec, self.freeze, threading.Event()
                    )
                with self.assertRaises(classical.ConfigurationError):
                    classical.evaluate_gate_job(
                        args, self.job, self.spec, self.freeze, threading.Event()
                    )
            status = classical.read_json(classical.evaluation_status_path(args, self.job))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(status["attempts"]), 1)
        self.assertEqual(
            status["attempts"][0]["failure_category"],
            "scientific_or_runtime_failure",
        )
        self.assertIs(status["attempts"][0]["retry_authorized"], False)

    def test_completed_gate_reload_is_read_only(self) -> None:
        def evaluate(**kwargs):
            return [metric(self.job, kwargs["workload_seeds"][0])]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                first = classical.evaluate_gate_job(
                    args, self.job, self.spec, self.freeze, threading.Event()
                )
            status_path = classical.evaluation_status_path(args, self.job)
            before = status_path.read_bytes()
            with patch.object(
                classical,
                "atomic_write_json",
                side_effect=AssertionError("completed reload attempted a write"),
            ):
                second = classical.evaluate_gate_job(
                    args, self.job, self.spec, self.freeze, threading.Event()
                )
            after = status_path.read_bytes()
        self.assertEqual(first, second)
        self.assertEqual(before, after)

    def test_completed_gate_status_finalizes_missing_manifest_without_evaluation(self) -> None:
        def evaluate(**kwargs):
            return [metric(self.job, kwargs["workload_seeds"][0])]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
                patch.object(
                    classical,
                    "_finalize_evaluation_manifest",
                    side_effect=RuntimeError("crash before manifest write"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    classical.evaluate_gate_job(
                        args, self.job, self.spec, self.freeze, threading.Event()
                    )
            status = classical.read_json(
                classical.evaluation_status_path(args, self.job)
            )
            self.assertEqual(status["status"], "completed")
            self.assertFalse(classical.evaluation_shard_paths(args, self.job)[2].exists())
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=AssertionError("recovery evaluated a workload"),
                ),
            ):
                manifest = classical.evaluate_gate_job(
                    args, self.job, self.spec, self.freeze, threading.Event()
                )
        self.assertEqual(manifest["attempt_count"], 1)

    def test_gate_csv_is_parsed_and_must_equal_jsonl(self) -> None:
        def evaluate(**kwargs):
            return [metric(self.job, kwargs["workload_seeds"][0])]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                classical.evaluate_gate_job(
                    args, self.job, self.spec, self.freeze, threading.Event()
                )
            jsonl_path, csv_path, manifest_path = classical.evaluation_shard_paths(
                args, self.job
            )
            csv_rows = classical._read_gate_csv(csv_path)
            csv_rows[0]["episode_reward"] += 1.0
            classical.atomic_write_csv(
                csv_path, csv_rows, fieldnames=classical.GATE_ROW_FIELDS
            )
            status_path = classical.evaluation_status_path(args, self.job)
            status = classical.read_json(status_path)
            status["attempts"][-1]["csv_sha256"] = classical.sha256_file(csv_path)
            classical.atomic_write_json(status_path, status)
            manifest = classical.read_json(manifest_path)
            manifest["csv_sha256"] = classical.sha256_file(csv_path)
            manifest["status"]["sha256"] = classical.sha256_file(status_path)
            manifest.pop("manifest_sha256")
            classical.atomic_write_json(
                manifest_path, classical.self_hashed(manifest, "manifest_sha256")
            )
            with self.assertRaises(classical.ConfigurationError):
                classical.validate_evaluation_shard(
                    args, self.job, self.spec, self.freeze
                )

    def test_manifest_with_running_status_is_rejected_without_repair(self) -> None:
        def evaluate(**kwargs):
            return [metric(self.job, kwargs["workload_seeds"][0])]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                classical.evaluate_gate_job(
                    args, self.job, self.spec, self.freeze, threading.Event()
                )
            status_path = classical.evaluation_status_path(args, self.job)
            status = classical.read_json(status_path)
            status["status"] = "running"
            status["attempts"][-1]["status"] = "running"
            status["attempts"][-1].pop("finished_at_utc", None)
            classical.atomic_write_json(status_path, status)
            manifest_path = classical.evaluation_shard_paths(args, self.job)[2]
            manifest = classical.read_json(manifest_path)
            manifest["status"]["sha256"] = classical.sha256_file(status_path)
            manifest.pop("manifest_sha256")
            classical.atomic_write_json(
                manifest_path, classical.self_hashed(manifest, "manifest_sha256")
            )
            before = status_path.read_bytes()
            with patch.object(
                classical,
                "atomic_write_json",
                side_effect=AssertionError("inconsistent completed state was repaired"),
            ):
                with self.assertRaises(classical.ConfigurationError):
                    classical.evaluate_gate_job(
                        args, self.job, self.spec, self.freeze, threading.Event()
                    )
            after = status_path.read_bytes()
        self.assertEqual(before, after)

    def test_gate_cancellation_stops_before_next_workload(self) -> None:
        cancel = threading.Event()
        calls = []

        def evaluate(**kwargs):
            calls.append(kwargs["workload_seeds"][0])
            cancel.set()
            return [metric(self.job, kwargs["workload_seeds"][0])]

        with tempfile.TemporaryDirectory() as temporary:
            args = fake_args(Path(temporary))
            with (
                patch.object(classical, "assert_runtime_fingerprint"),
                patch.object(
                    classical,
                    "evaluate_policy_with_constraint_metrics",
                    side_effect=evaluate,
                ),
            ):
                with self.assertRaises(classical.JobCancelled):
                    classical.evaluate_gate_job(
                        args, self.job, self.spec, self.freeze, cancel
                    )
            status = classical.read_json(
                classical.evaluation_status_path(args, self.job)
            )
        self.assertEqual(calls, [classical.GATE_WORKLOAD_SEEDS[0]])
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(
            status["attempts"][0]["failure_category"], "execution_interruption"
        )
        self.assertIs(status["attempts"][0]["retry_authorized"], False)

    def test_dijkstra_factory_has_single_deterministic_identity(self) -> None:
        job = next(
            job
            for job in classical.build_evaluation_jobs()
            if job.method == classical.GLOBAL_DIJKSTRA
        )
        policy = classical._make_evaluation_policy(
            fake_args(Path("unused")), job, self.spec, self.freeze
        )
        self.assertIsInstance(policy, GlobalDijkstraPolicy)
        self.assertEqual(job.policy_seed, -1)

    def test_dijkstra_rows_accept_only_the_frozen_negative_sentinel(self) -> None:
        job = next(
            job
            for job in classical.build_evaluation_jobs()
            if job.method == classical.GLOBAL_DIJKSTRA
        )
        rows = gate_rows(job, self.spec, self.freeze)
        summary = classical.validate_gate_rows(job, rows, self.spec, self.freeze)
        self.assertEqual(summary["row_count"], len(classical.GATE_WORKLOAD_SEEDS))
        rows[0]["policy_seed"] = 0
        with self.assertRaises(classical.ConfigurationError):
            classical.validate_gate_rows(job, rows, self.spec, self.freeze)


if __name__ == "__main__":
    unittest.main()
