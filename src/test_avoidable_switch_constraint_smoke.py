"""Contract tests for the independent avoidable-switch constraint smoke."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_avoidable_switch_constraint_smoke as smoke


class SmokeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            project=Path("F:/leo-routing-preliminary-matlab/src"),
            cleanmarl=Path("F:/cleanmarl"),
            device="cuda",
        )

    def test_grid_is_exactly_two_arms_by_two_scenarios(self):
        jobs = smoke.build_jobs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual(
            {(job.scenario, job.arm) for job in jobs},
            {
                (scenario, arm)
                for scenario in smoke.SCENARIOS
                for arm in smoke.ARMS
            },
        )
        self.assertEqual({job.policy_seed for job in jobs}, {197359353})
        self.assertEqual(sum(job.constrained for job in jobs), 2)

    def test_seed_registry_is_disjoint_and_retires_missing_v2_smoke_panel(self):
        smoke.validate_seed_registry()
        retired = smoke._expand_ranges(smoke.RETIRED_WORKLOAD_RANGES)
        self.assertIn(62001, retired)
        self.assertIn(62002, retired)
        fresh = (
            set(smoke.SMOKE_TRAIN_WORKLOAD_SEEDS)
            | set(smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS)
            | set(smoke.FORMAL_TRAIN_WORKLOAD_SEEDS)
            | set(smoke.FORMAL_VALIDATION_WORKLOAD_SEEDS)
            | set(smoke.SEALED_TEST_WORKLOAD_SEEDS)
        )
        self.assertFalse(fresh & retired)

    def test_commands_never_pass_formal_or_sealed_workload_seeds(self):
        baseline = smoke.SmokeJob(0, smoke.SCENARIOS[0], smoke.ARM_BASELINE)
        constrained = smoke.SmokeJob(
            1, smoke.SCENARIOS[0], smoke.ARM_CONSTRAINED
        )
        baseline_command = smoke.build_command(
            baseline, self.args, Path("F:/tmp/baseline")
        )
        constrained_command = smoke.build_command(
            constrained, self.args, Path("F:/tmp/constrained")
        )
        for command in (baseline_command, constrained_command):
            rendered = " ".join(command)
            self.assertIn(str(smoke.SMOKE_TRAIN_WORKLOAD_SEEDS[0]), rendered)
            self.assertIn(
                str(smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS[0]), rendered
            )
            self.assertFalse(
                any(str(seed) in rendered for seed in smoke.SEALED_TEST_WORKLOAD_SEEDS)
            )
            self.assertFalse(
                any(str(seed) in rendered for seed in smoke.FORMAL_TRAIN_WORKLOAD_SEEDS)
            )
            self.assertFalse(
                any(
                    str(seed) in rendered
                    for seed in smoke.FORMAL_VALIDATION_WORKLOAD_SEEDS
                )
            )
        self.assertNotIn(
            "--avoidable-switch-constraint-enabled", baseline_command
        )
        self.assertIn(
            "--avoidable-switch-constraint-enabled", constrained_command
        )
        self.assertIn("qos_only", constrained_command)

    def test_self_hash_rejects_mutation(self):
        record = smoke.self_hashed({"value": 1}, "sha256")
        smoke.validate_self_hash(record, "sha256")
        record["value"] = 2
        with self.assertRaisesRegex(smoke.ConfigurationError, "mismatch"):
            smoke.validate_self_hash(record, "sha256")

    def test_run_config_contract_covers_training_semantics(self):
        job = smoke.SmokeJob(0, smoke.SCENARIOS[0], smoke.ARM_BASELINE)
        config = smoke.expected_run_config(job, self.args)
        self.assertTrue(smoke._run_config_matches(config, job, self.args))
        expected_fields = {
            "actor_hidden_dim",
            "actor_num_layers",
            "critic_hidden_dim",
            "critic_num_layers",
            "optimizer",
            "learning_rate_actor",
            "learning_rate_critic",
            "lr_decay",
            "gamma",
            "td_lambda",
            "normalize_reward",
            "normalize_advantage",
            "normalize_return",
            "ppo_clip",
            "entropy_coef",
            "clip_gradients",
            "critic_clip_gradients",
            "normalization_epsilon",
            "target_kl",
        }
        self.assertTrue(expected_fields.issubset(config))
        for field in expected_fields:
            drifted = dict(config)
            value = drifted[field]
            drifted[field] = not value if isinstance(value, bool) else "drifted"
            self.assertFalse(
                smoke._run_config_matches(drifted, job, self.args), field
            )

    def test_exact_resume_uses_only_retired_seed_60001(self):
        command = smoke._exact_resume_command(
            self.args,
            Path("F:/tmp/exact-resume"),
            total_timesteps=50,
        )
        self.assertEqual(
            command[command.index("--train-seed-start") + 1], "60001"
        )
        self.assertEqual(
            command[command.index("--validation-seed-start") + 1], "60001"
        )
        self.assertEqual(
            command[command.index("--total-timesteps") + 1], "50"
        )
        self.assertIn("--no-lr-decay", command)
        rendered = " ".join(command)
        forbidden = (
            smoke.SMOKE_TRAIN_WORKLOAD_SEEDS
            + smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS
            + smoke.FORMAL_TRAIN_WORKLOAD_SEEDS
            + smoke.FORMAL_VALIDATION_WORKLOAD_SEEDS
            + smoke.SEALED_TEST_WORKLOAD_SEEDS
        )
        self.assertFalse(any(str(seed) in rendered for seed in forbidden))

    def test_output_isolation_accepts_default_and_rejects_history_or_source(self):
        repository_root = Path(smoke.__file__).resolve().parent.parent
        args = SimpleNamespace(
            output=(
                repository_root
                / "experiments"
                / smoke.DEFAULT_OUTPUT_DIRECTORY_NAME
            ),
            project=repository_root / "src",
            cleanmarl=Path("F:/cleanmarl"),
        )
        smoke.validate_output_isolation(args)
        args.output = (
            repository_root / "experiments" / "age-band-shield-design-v11"
        )
        with self.assertRaisesRegex(smoke.ConfigurationError, "protected"):
            smoke.validate_output_isolation(args)
        args.output = repository_root / "src" / "accidental-output"
        with self.assertRaisesRegex(smoke.ConfigurationError, "protected"):
            smoke.validate_output_isolation(args)


class DualAuditTests(unittest.TestCase):
    @staticmethod
    def records() -> list[dict]:
        first_next = 0.05 * (0.2 - 0.12)
        return [
            {
                "record_type": "training_update",
                "switch_constraint_schema_version": 1,
                "decision_avoidable_switch_cost": 2,
                "decision_switch_opportunities": 10,
                "decision_avoidable_switch_rate": 0.2,
                "decision_avoidable_switch_violation": 0.08,
                "dual_multiplier_used": 0.0,
                "dual_multiplier_next": first_next,
                "dual_update_count": 1,
                "dual_update_skipped_zero_opportunity": False,
                "dual_skipped_zero_opportunity_count": 0,
                "dual_cumulative_cost": 2,
                "dual_cumulative_opportunity": 10,
                "switch_constraint_surrogate_rate": 0.25,
                "avoidable_switch_regularization_penalty": 0.25,
                "switch_constraint_actor_term": 0.0,
                "actor_primary_loss": 1.0,
                "actor_loss": 1.0,
                "rollout_avoidable_switch_eligible_decisions": 10,
            },
            {
                "record_type": "training_update",
                "switch_constraint_schema_version": 1,
                "decision_avoidable_switch_cost": 0,
                "decision_switch_opportunities": 0,
                "decision_avoidable_switch_rate": None,
                "decision_avoidable_switch_violation": None,
                "dual_multiplier_used": first_next,
                "dual_multiplier_next": first_next,
                "dual_update_count": 1,
                "dual_update_skipped_zero_opportunity": True,
                "dual_skipped_zero_opportunity_count": 1,
                "dual_cumulative_cost": 2,
                "dual_cumulative_opportunity": 10,
                "switch_constraint_surrogate_rate": 0.25,
                "avoidable_switch_regularization_penalty": 0.25,
                "switch_constraint_actor_term": first_next * 0.25,
                "actor_primary_loss": 1.0,
                "actor_loss": 1.0 + first_next * 0.25,
                "rollout_avoidable_switch_eligible_decisions": 10,
            },
        ]

    @staticmethod
    def final_state() -> dict:
        first_next = 0.05 * (0.2 - 0.12)
        return {
            "multiplier": first_next,
            "update_count": 1,
            "skipped_zero_opportunity_count": 1,
            "cumulative_cost": 2,
            "cumulative_opportunity": 10,
            "last_rollout_cost": 0,
            "last_rollout_opportunity": 0,
        }

    def test_audit_recomputes_update_and_zero_opportunity_skip(self):
        summary = smoke.audit_constraint_records(
            self.records(), self.final_state()
        )
        self.assertEqual(summary["dual_update_count"], 1)
        self.assertEqual(summary["dual_skipped_zero_opportunity_count"], 1)
        self.assertEqual(summary["cumulative_opportunity"], 10)
        self.assertAlmostEqual(summary["final_multiplier"], 0.004)

    def test_audit_rejects_actor_loss_that_omits_positive_lambda_term(self):
        records = self.records()
        records[1]["actor_loss"] = records[1]["actor_primary_loss"]
        with self.assertRaisesRegex(
            smoke.ConfigurationError, "actor loss omits"
        ):
            smoke.audit_constraint_records(records, self.final_state())

    def test_audit_rejects_cost_above_opportunity(self):
        records = self.records()
        records[0]["decision_avoidable_switch_cost"] = 11
        with self.assertRaisesRegex(
            smoke.ConfigurationError, "cost/opportunity"
        ):
            smoke.audit_constraint_records(records, self.final_state())


class ValidationSelectionAuditTests(unittest.TestCase):
    @staticmethod
    def validation_record(
        step: int,
        *,
        delivery: float,
        decision_rate: float,
        is_best: bool | None = None,
    ) -> dict:
        opportunities = 10
        cost = int(round(decision_rate * opportunities))
        record = {
            "record_type": "validation",
            "environment_steps": step,
            "episodes": len(smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS),
            "seed_start": smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS[0],
            "delivery_ratio": delivery,
            "mean_reward": delivery * 10.0,
            "drop_rate": 1.0 - delivery,
            "average_delay_slots": 2.0,
            "routing_switches": 3.0,
            "routing_switches_total": 12,
            "avoidable_routing_switches": 1.0,
            "avoidable_routing_switches_total": 4,
            "forced_routing_switches": 0.5,
            "forced_routing_switches_total": 2,
            "switch_opportunities": 20,
            "avoidable_switch_rate": 0.2,
            "class_2_delivery_ratio": delivery - 0.1,
            "decision_avoidable_switches": cost,
            "decision_switch_opportunities": opportunities,
            "decision_avoidable_switch_rate": decision_rate,
        }
        if is_best is not None:
            record["is_validation_best"] = is_best
        return record

    @staticmethod
    def selection_spec(constrained: bool) -> dict:
        aggregation = {
            "delivery_ratio": "macro_mean_over_validation_episodes",
            "class_2_delivery_ratio": (
                "macro_mean_over_validation_episodes"
            ),
            "routing_switches": "mean_of_episode_total_switch_counts",
        }
        if constrained:
            aggregation.update(
                decision_avoidable_switch_rate=(
                    "micro_ratio_over_pre_contention_validation_opportunities"
                ),
                decision_avoidable_switches=(
                    "sum_over_pre_contention_validation_decisions"
                ),
                decision_switch_opportunities=(
                    "sum_over_pre_contention_validation_decisions"
                ),
            )
        return {
            "schema_version": 1,
            "mode": (
                "avoidable_switch_budget_constrained"
                if constrained
                else "legacy_lexicographic"
            ),
            "delivery_tolerance": 0.0,
            "class_2_tolerance": 0.0,
            "switch_budget": smoke.SWITCH_BUDGET if constrained else None,
            "source": "validation_only",
            "test_panel_consulted": False,
            "validation_seed_start": smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS[0],
            "validation_episodes": len(
                smoke.SMOKE_VALIDATION_WORKLOAD_SEEDS
            ),
            "selection_timing": (
                "after_all_validation_candidates"
                if constrained
                else "online_legacy_compatible"
            ),
            "metric_aggregation": aggregation,
        }

    def make_fixture(self, root: Path, *, constrained: bool) -> dict:
        if constrained:
            first = self.validation_record(
                120, delivery=0.8, decision_rate=0.1
            )
            second = self.validation_record(
                240, delivery=0.9, decision_rate=0.2
            )
        else:
            first = self.validation_record(
                120, delivery=0.8, decision_rate=0.1, is_best=True
            )
            second = self.validation_record(
                240, delivery=0.7, decision_rate=0.2, is_best=False
            )
        selected_path = root / "validation_best.pt"
        selected_path.write_bytes(b"selected-checkpoint")
        if constrained:
            candidate_path = root / "validation_candidate_step_120.pt"
            candidate_path.write_bytes(selected_path.read_bytes())
            first["candidate_checkpoint"] = str(candidate_path)
            second["candidate_checkpoint"] = str(
                root / "validation_candidate_step_240.pt"
            )
        selected_metrics = {
            field: first[field]
            for field in smoke.SELECTED_VALIDATION_METRIC_FIELDS
        }
        spec = self.selection_spec(constrained)
        selected_sha256 = smoke.sha256_file(selected_path)
        manifest = {
            "validation_candidate_count": 2,
            "validation_selection_spec": spec,
            "selected_validation_metrics": selected_metrics,
            "selected_validation_checkpoint_sha256": selected_sha256,
            "best_validation_score": list(smoke._legacy_validation_score(first)),
        }
        selections = []
        if constrained:
            selections = [
                {
                    "record_type": "validation_selection",
                    "selection_spec": spec,
                    "candidate_count": 2,
                    "selected_candidate_checkpoint": first[
                        "candidate_checkpoint"
                    ],
                    "selected_candidate_step": 120,
                    "selected_checkpoint_sha256": selected_sha256,
                    "validation_best_checkpoint": str(selected_path),
                    "selected_metrics": dict(selected_metrics),
                    "constraint_feasible": True,
                }
            ]
        return {
            "job": smoke.SmokeJob(
                0,
                smoke.SCENARIOS[0],
                smoke.ARM_CONSTRAINED
                if constrained
                else smoke.ARM_BASELINE,
            ),
            "manifest": manifest,
            "validations": [first, second],
            "selections": selections,
            "run_directory": root,
            "selected_path": selected_path,
            "selected_checkpoint": {
                "step": 120,
                "validation_selection_spec": spec,
            },
        }

    @staticmethod
    def audit(fixture: dict) -> dict:
        return smoke.audit_validation_selection(**fixture)

    def test_legacy_baseline_accepts_zero_selection_records(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=False)
            selected = self.audit(fixture)
        self.assertEqual(selected["environment_steps"], 120)

    def test_legacy_baseline_zero_opportunity_rate_remains_undefined(self):
        baseline = smoke.SmokeJob(
            0, smoke.SCENARIOS[0], smoke.ARM_BASELINE
        )
        metrics = {
            "decision_avoidable_switches": 0,
            "decision_switch_opportunities": 0,
            "decision_avoidable_switch_rate": None,
        }
        self.assertIsNone(
            smoke.audit_selected_decision_ledger(baseline, metrics)
        )
        self.assertEqual(
            smoke.audit_validation_decision_ledgers(baseline, [metrics]), 0
        )
        metrics["decision_avoidable_switch_rate"] = 0.0
        with self.assertRaisesRegex(smoke.ConfigurationError, "undefined"):
            smoke.audit_selected_decision_ledger(baseline, metrics)
        constrained = smoke.SmokeJob(
            1, smoke.SCENARIOS[0], smoke.ARM_CONSTRAINED
        )
        metrics["decision_avoidable_switch_rate"] = None
        with self.assertRaisesRegex(
            smoke.ConfigurationError, "constrained selected"
        ):
            smoke.audit_selected_decision_ledger(constrained, metrics)

    def test_legacy_baseline_rejects_manifest_metric_tampering(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=False)
            fixture["manifest"]["selected_validation_metrics"][
                "delivery_ratio"
            ] = 0.81
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "manifest-selected metrics"
            ):
                self.audit(fixture)

    def test_legacy_baseline_rejects_unexpected_selection_record(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=False)
            fixture["selections"] = [{"record_type": "validation_selection"}]
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "unexpectedly contains"
            ):
                self.audit(fixture)

    def test_selection_rejects_metric_aggregation_drift(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=False)
            fixture["manifest"]["validation_selection_spec"][
                "metric_aggregation"
            ]["delivery_ratio"] = "unfrozen_aggregation"
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "metric aggregation"
            ):
                self.audit(fixture)

    def test_selection_rejects_checkpoint_step_or_manifest_hash_drift(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=False)
            fixture["selected_checkpoint"]["step"] = 240
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "checkpoint step"
            ):
                self.audit(fixture)
            fixture = self.make_fixture(Path(directory), constrained=False)
            fixture["manifest"]["selected_validation_checkpoint_sha256"] = (
                "0" * 64
            )
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "checkpoint hash"
            ):
                self.audit(fixture)

    def test_constrained_selection_is_fully_bound(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=True)
            selected = self.audit(fixture)
        self.assertEqual(selected["decision_avoidable_switch_rate"], 0.1)

    def test_constrained_requires_one_selection_record(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=True)
            fixture["selections"] = []
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "exactly one"
            ):
                self.audit(fixture)

    def test_constrained_rejects_selection_metrics_or_hash_drift(self):
        with TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory), constrained=True)
            fixture["selections"][0]["selected_metrics"] = deepcopy(
                fixture["selections"][0]["selected_metrics"]
            )
            fixture["selections"][0]["selected_metrics"][
                "delivery_ratio"
            ] = 0.81
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "metrics differ"
            ):
                self.audit(fixture)
            fixture = self.make_fixture(Path(directory), constrained=True)
            fixture["selections"][0]["selected_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                smoke.ConfigurationError, "checkpoint hash"
            ):
                self.audit(fixture)


class ConcurrencySafetyTests(unittest.TestCase):
    @staticmethod
    def args(output: Path) -> SimpleNamespace:
        return SimpleNamespace(output=output, max_parallel=2)

    def test_global_active_child_scan_runs_before_any_submission(self):
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            status_root = args.output / "job_status"
            status_root.mkdir(parents=True)
            smoke.atomic_write_json(
                status_root / "orphan.json",
                {
                    "status": "running",
                    "child_pid": 43210,
                    "job": {"job_id": "orphaned/job"},
                },
            )
            with patch.object(smoke, "_pid_is_active", return_value=True), patch.object(
                smoke, "train_job"
            ) as train:
                with self.assertRaisesRegex(RuntimeError, "active training"):
                    smoke.run_all_jobs(args, "spec")
                train.assert_not_called()

    def test_interrupt_does_not_start_jobs_waiting_for_a_parallel_slot(self):
        with TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            started: list[int] = []

            def interrupting_train_job(
                _args,
                job,
                _spec_sha256,
                cancellation_event,
            ):
                started.append(job.index)
                if job.index == 0:
                    raise KeyboardInterrupt()
                cancellation_event.wait(timeout=2.0)
                raise smoke.TrainingCancelled("cancelled by test")

            with patch.object(
                smoke, "train_job", side_effect=interrupting_train_job
            ):
                with self.assertRaises(KeyboardInterrupt):
                    smoke.run_all_jobs(args, "spec")
            self.assertTrue(started)
            self.assertFalse(set(started) & {2, 3})


if __name__ == "__main__":
    unittest.main()
