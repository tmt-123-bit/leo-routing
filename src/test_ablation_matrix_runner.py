"""Unit tests for the resumable full-budget ablation orchestrator."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import ablation_matrix_runner as runner
from ablation_matrix_runner import (
    EXPECTED_MAIN_FULL_CONFIG,
    POLICY_SEEDS,
    SCENARIOS,
    TEST_WORKLOAD_SEEDS,
    VARIANTS,
    ConfigurationError,
    atomic_write_csv,
    atomic_write_json,
    audit_training_artifacts,
    audit_matrix,
    build_matrix,
    collect_code_audit,
    ensure_historical_workload_seed_audit,
    ensure_matrix_spec,
    ensure_training_freeze,
    evaluation_shard_path,
    expected_phase_fingerprints,
    generate_final_statistics,
    load_job_state,
    matrix_spec,
    merge_evaluation_shards,
    phase_fingerprint,
    parse_args,
    run_evaluation_job,
    run_training_job,
    select_jobs,
    sha256_file,
    status_path,
    validate_evaluation_record,
    validate_evaluation_shard,
    validate_full_budget_contract,
    validate_variant_contract,
    validate_training_record,
    write_job_state,
)
from mappo_evaluation import EpisodeMetrics


def complete_episode_rows(job, workload_seeds=TEST_WORKLOAD_SEEDS):
    """Return finite rows with exactly the production EpisodeMetrics schema."""

    return [
        asdict(
            EpisodeMetrics(
                scenario=job.scenario,
                policy=f"mappo_{job.variant}",
                policy_seed=job.policy_seed,
                workload_seed=int(workload_seed),
                generated=100,
                delivered=70,
                dropped=20,
                backlog=10,
                delivery_ratio=0.7,
                drop_rate=0.2,
                throughput_packets_per_slot=1.4,
                average_delay_slots=4.0,
                p95_delay_slots=8.0,
                mean_queue_packets=2.0,
                max_queue_packets=5,
                routing_switches=3,
                episode_reward=12.0,
                global_delay_cost=1.0,
                global_queue_cost=2.0,
                global_load_imbalance=0.1,
                global_switch_cost=0.2,
                global_throughput_reward=3.0,
                global_control_overhead_ratio=0.05,
                global_drop_cost=0.3,
                class_0_delivery_ratio=0.75,
                class_1_delivery_ratio=0.70,
                class_2_delivery_ratio=0.65,
            )
        )
        for workload_seed in workload_seeds
    ]


class MatrixContractTests(unittest.TestCase):
    def test_policy_seeds_follow_preregistered_sha256_derivation(self):
        namespace = "ABLATION-50K-v2.2-policy-seed-"
        derived = tuple(
            int.from_bytes(
                hashlib.sha256(f"{namespace}{index}".encode("ascii")).digest()[:4],
                "big",
            )
            & 0x7FFFFFFF
            for index in range(12)
        )
        historical = {7, 42, 123, 314, 456, 789, 1024, 2024}
        self.assertEqual(POLICY_SEEDS, derived)
        self.assertEqual(len(set(derived)), 12)
        self.assertTrue(all(seed > 0 for seed in derived))
        self.assertTrue(set(derived).isdisjoint(historical))

    def test_historical_seed_audit_rejects_selected_panel_overlap(self):
        for field, selected_value in (
            ("workload_seed", TEST_WORKLOAD_SEEDS[0]),
            ("policy_seed", POLICY_SEEDS[0]),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                experiments = root / "experiments"
                historical_csv = experiments / "prior" / "episode_metrics.csv"
                atomic_write_csv(
                    historical_csv,
                    [{field: selected_value, "result_metric": 0.9}],
                )
                with self.assertRaisesRegex(
                    ConfigurationError, "prior experiment overlap"
                ):
                    ensure_historical_workload_seed_audit(
                        root / "formal-output", experiments
                    )

    def test_historical_seed_audit_is_self_hashed_and_not_rescanned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiments = root / "experiments"
            historical_csv = experiments / "prior.csv"
            atomic_write_csv(
                historical_csv,
                [{"policy_seed": 7, "workload_seed": 13001, "metric": 0.5}],
            )
            output = root / "formal-output"
            manifest = ensure_historical_workload_seed_audit(output, experiments)
            self.assertEqual(manifest["content"]["selected_prior_overlap"], [])
            self.assertEqual(manifest["content"]["historical_policy_seeds"], [7])
            self.assertEqual(
                manifest["content"]["historical_workload_seed_ranges"],
                [{"start": 13001, "end": 13001, "count": 1}],
            )

            atomic_write_csv(
                historical_csv,
                [
                    {
                        "policy_seed": POLICY_SEEDS[0],
                        "workload_seed": TEST_WORKLOAD_SEEDS[0],
                    }
                ],
            )
            resumed = ensure_historical_workload_seed_audit(output, experiments)
            self.assertEqual(resumed["content_sha256"], manifest["content_sha256"])

    def test_fixed_matrix_has_all_540_unique_cells_in_stable_order(self):
        jobs = build_matrix()
        self.assertEqual(len(jobs), 9 * 5 * 12)
        self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
        self.assertEqual(
            jobs[0].job_id, f"low_load/proposed/seed_{POLICY_SEEDS[0]}"
        )
        self.assertEqual(
            jobs[12].job_id, f"low_load/no_queue/seed_{POLICY_SEEDS[0]}"
        )
        self.assertEqual(
            jobs[-1].job_id,
            f"fault_links/with_hard_lifetime_mask/seed_{POLICY_SEEDS[-1]}",
        )
        spec = matrix_spec(EXPECTED_MAIN_FULL_CONFIG, jobs)
        self.assertEqual(
            spec["design"],
            "component_removals_plus_adjacent_lifetime_ladder",
        )
        contrasts = {row["name"]: row for row in spec["planned_contrasts"]}
        self.assertEqual(
            contrasts["remove_ppo_protection_package"]["component_kind"],
            "trainer_safeguard_package",
        )
        self.assertEqual(
            contrasts["add_lifetime_reward"]["reference"],
            "with_lifetime_feature",
        )

    def test_matrix_spec_is_json_round_trip_stable_and_resumable(self):
        spec = matrix_spec(EXPECTED_MAIN_FULL_CONFIG, build_matrix())
        self.assertEqual(json.loads(json.dumps(spec)), spec)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = ensure_matrix_spec(output, spec)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), spec)
            self.assertEqual(ensure_matrix_spec(output, spec), path)

    def test_phase_fingerprints_bind_runner_and_protocol(self):
        job = build_matrix()[0]
        code = {
            "runner": "a" * 64,
            "protocol_document": "b" * 64,
            "trainer": "c" * 64,
            "evaluation": "d" * 64,
        }
        training = phase_fingerprint(
            "training", job, config=EXPECTED_MAIN_FULL_CONFIG, code_sha256=code
        )
        evaluation = phase_fingerprint(
            "evaluation",
            job,
            config=EXPECTED_MAIN_FULL_CONFIG,
            code_sha256=code,
            checkpoint_sha256="e" * 64,
            training_freeze_sha256="f" * 64,
        )

        for component in ("runner", "protocol_document"):
            changed = {**code, component: "0" * 64}
            self.assertNotEqual(
                phase_fingerprint(
                    "training",
                    job,
                    config=EXPECTED_MAIN_FULL_CONFIG,
                    code_sha256=changed,
                ),
                training,
            )
            self.assertNotEqual(
                phase_fingerprint(
                    "evaluation",
                    job,
                    config=EXPECTED_MAIN_FULL_CONFIG,
                    code_sha256=changed,
                    checkpoint_sha256="e" * 64,
                    training_freeze_sha256="f" * 64,
                ),
                evaluation,
            )

    def test_shards_are_disjoint_and_exhaustive(self):
        jobs = build_matrix()
        shards = [
            select_jobs(jobs, shard_index=index, shard_count=7)
            for index in range(7)
        ]
        flattened = [job.index for shard in shards for job in shard]
        self.assertEqual(sorted(flattened), list(range(len(jobs))))
        self.assertEqual(len(flattened), len(set(flattened)))
        selected = select_jobs(
            jobs,
            scenarios=["fault_links"],
            variants=["proposed"],
            policy_seeds=[POLICY_SEEDS[0]],
            indices=[jobs[-108].index],
        )
        self.assertEqual(
            [job.job_id for job in selected],
            [f"fault_links/proposed/seed_{POLICY_SEEDS[0]}"],
        )

    def test_invalid_shard_and_index_are_rejected(self):
        jobs = build_matrix()
        with self.assertRaises(ConfigurationError):
            select_jobs(jobs, shard_index=0)
        with self.assertRaises(ConfigurationError):
            select_jobs(jobs, shard_index=3, shard_count=3)
        with self.assertRaises(ConfigurationError):
            select_jobs(jobs, indices=[len(jobs)])

    def test_full_budget_must_exactly_match_main_training_contract(self):
        validated = validate_full_budget_contract(
            EXPECTED_MAIN_FULL_CONFIG,
            SCENARIOS,
            POLICY_SEEDS,
        )
        self.assertEqual(validated["timesteps"], 50000)
        changed = dict(EXPECTED_MAIN_FULL_CONFIG)
        changed["timesteps"] = 49999
        with self.assertRaisesRegex(ConfigurationError, "timesteps"):
            validate_full_budget_contract(changed, SCENARIOS, POLICY_SEEDS)
        with self.assertRaisesRegex(ConfigurationError, "policy seeds"):
            validate_full_budget_contract(
                EXPECTED_MAIN_FULL_CONFIG,
                SCENARIOS,
                tuple(reversed(POLICY_SEEDS)),
            )

    def test_cli_defaults_are_cuda_two_workers_and_isolated_v2_output(self):
        args = parse_args([])
        self.assertEqual(args.output, Path("experiments/ablation-50k-v2"))
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.max_parallel, 2)

    def test_executed_trainer_must_match_repository_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = root / "executed.py"
            snapshot = root / "snapshot.py"
            trainer.write_text("executed", encoding="utf-8")
            snapshot.write_text("stale", encoding="utf-8")
            with mock.patch.object(
                runner,
                "_source_paths",
                return_value={
                    "trainer": trainer,
                    "trainer_snapshot": snapshot,
                },
            ):
                with self.assertRaisesRegex(
                    ConfigurationError, "differs from the repository snapshot"
                ):
                    collect_code_audit(root, root)

            snapshot.write_text("executed", encoding="utf-8")
            with mock.patch.object(
                runner,
                "_source_paths",
                return_value={
                    "trainer": trainer,
                    "trainer_snapshot": snapshot,
                },
            ):
                audit = collect_code_audit(root, root)
            self.assertEqual(
                audit["trainer"]["sha256"], audit["trainer_snapshot"]["sha256"]
            )

    def test_live_training_api_uses_frozen_seeds_and_proposed_ppo_baseline(self):
        import run_exp004_mappo

        validated = validate_full_budget_contract(
            run_exp004_mappo.mode_config("full"),
            run_exp004_mappo.ALL_SCENARIOS,
            run_exp004_mappo.POLICY_SEEDS,
        )
        self.assertEqual(tuple(run_exp004_mappo.POLICY_SEEDS), POLICY_SEEDS)
        validate_variant_contract()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "selected.pt"
            checkpoint.write_bytes(b"mock checkpoint")
            process = mock.Mock()
            process.pid = 4321
            process.wait.return_value = 0
            args = SimpleNamespace(
                output=root,
                cleanmarl=root / "cleanmarl",
                project=root / "src",
                skip_training=False,
                device="cuda",
            )
            with (
                mock.patch.object(
                    run_exp004_mappo,
                    "find_selected_checkpoint",
                    side_effect=[None, checkpoint],
                ),
                mock.patch.object(
                    run_exp004_mappo,
                    "code_fingerprint",
                    return_value={"mock": "hash"},
                ),
                mock.patch.object(
                    run_exp004_mappo.subprocess,
                    "Popen",
                    return_value=process,
                ) as subprocess_popen,
            ):
                selected = run_exp004_mappo.train_one(
                    args,
                    validated,
                    "low_load",
                    POLICY_SEEDS[0],
                    variant="no_ppo_protection",
                    trainer_overrides=["--clip-gradients", "0"],
                )
            self.assertEqual(selected, checkpoint)
            command = subprocess_popen.call_args.args[0]
            self.assertEqual(command[command.index("--leo-variant") + 1], "proposed")
            self.assertEqual(command[command.index("--total-timesteps") + 1], "50000")


class ArtifactValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.job = build_matrix()[0]
        self.config = dict(EXPECTED_MAIN_FULL_CONFIG)
        self.code = {"trainer": "a" * 64, "environment": "b" * 64}
        self.spec = matrix_spec(self.config, [self.job])
        ensure_historical_workload_seed_audit(
            self.root, self.root / "experiments"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _checkpoint(self) -> Path:
        checkpoint = self.root / "model.pt"
        checkpoint.write_bytes(b"checkpoint-v1")
        return checkpoint

    def _training_audit(self):
        artifacts = {}
        artifact_root = self.root / "training-audit"
        for name in (
            "run_manifest",
            "run_config",
            "training_metrics",
            "trainer_log",
            "training_code_fingerprint",
        ):
            path = artifact_root / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            artifacts[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        return {
            "budget_verified": True,
            "requested_environment_steps": 50000,
            "actual_environment_steps": 50040,
            "selected_checkpoint_step": 48000,
            "earliest_best_validation_step": 48000,
            "selected_validation_score": [0.75, 9.0, -0.01, -6.0],
            "selected_validation_metrics": {
                "delivery_ratio": 0.75,
                "mean_reward": 9.0,
                "drop_rate": 0.01,
                "average_delay_slots": 6.0,
            },
            "selected_validation_records_at_step": 1,
            "best_validation_tie_count": 1,
            "artifacts": artifacts,
        }

    def _evaluation_rows(self):
        return complete_episode_rows(self.job)

    def _completed_state(self):
        checkpoint = self._checkpoint()
        training_fingerprint = phase_fingerprint(
            "training", self.job, config=self.config, code_sha256=self.code
        )
        checkpoint_sha = sha256_file(checkpoint)
        state = load_job_state(self.root, self.job, self.spec["spec_sha256"])
        state["training"].update(
            {
                "status": "completed",
                "fingerprint": training_fingerprint,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "training_audit": self._training_audit(),
            }
        )
        write_job_state(self.root, self.job, state)
        training_freeze = ensure_training_freeze(
            self.root,
            [self.job],
            spec_sha256=self.spec["spec_sha256"],
            config=self.config,
            code_sha256=self.code,
        )
        evaluation_fingerprint = phase_fingerprint(
            "evaluation",
            self.job,
            config=self.config,
            code_sha256=self.code,
            checkpoint_sha256=checkpoint_sha,
            training_freeze_sha256=training_freeze["content_sha256"],
        )
        shard = evaluation_shard_path(self.root, self.job).resolve()
        atomic_write_csv(shard, self._evaluation_rows())
        state["evaluation"].update(
            {
                "status": "completed",
                "fingerprint": evaluation_fingerprint,
                "shard_path": str(shard),
                "shard_sha256": sha256_file(shard),
                "row_count": len(TEST_WORKLOAD_SEEDS),
            }
        )
        write_job_state(self.root, self.job, state)
        return state, training_fingerprint, evaluation_fingerprint, training_freeze

    def test_checkpoint_resume_requires_matching_fingerprint_and_sha(self):
        state, fingerprint, _, _ = self._completed_state()
        self.assertEqual(validate_training_record(state["training"], fingerprint), (True, "valid"))
        Path(state["training"]["checkpoint_path"]).write_bytes(b"changed")
        valid, reason = validate_training_record(state["training"], fingerprint)
        self.assertFalse(valid)
        self.assertIn("SHA256", reason)
        valid, reason = validate_training_record(state["training"], "new-code")
        self.assertFalse(valid)
        self.assertIn("fingerprint", reason)

    def test_training_artifact_audit_proves_actual_50k_budget_and_log_hashes(self):
        import torch

        checkpoint_root = self.root / "checkpoint-root"
        run_directory = checkpoint_root / "run-001"
        run_directory.mkdir(parents=True)
        checkpoint = run_directory / "validation_best.pt"
        final_checkpoint = run_directory / "final.pt"
        torch.save({"step": 48000}, checkpoint)
        final_checkpoint.write_bytes(b"final")
        atomic_write_json(
            run_directory / "run_manifest.json",
            {
                "run_name": "run-001",
                "validation_best_checkpoint": str(checkpoint.resolve()),
                "final_checkpoint": str(final_checkpoint.resolve()),
                "environment_steps": 50040,
                "optimizer_updates": 5004,
                "best_validation_score": [0.75, 9.0, -0.01, -6.0],
            },
        )
        atomic_write_json(
            run_directory / "run_config.json",
            {
                "env_type": "leo_multi",
                "env_name": self.job.scenario,
                "batch_size": 4,
                "total_timesteps": 50000,
                "epochs": 3,
                "num_minibatches": 4,
                "eval_steps": 40,
                "num_eval_ep": 50,
                "save_every_steps": 5000,
                "train_seed_start": 9001,
                "train_seed_count": 200,
                "validation_seed_start": 10001,
                "seed": self.job.policy_seed,
                "leo_variant": "proposed",
                "clip_gradients": 1.0,
                "target_kl": 0.02,
                "normalize_advantage": True,
            },
        )
        validation_record = {
            "record_type": "validation",
            "environment_steps": 48000,
            "episodes": 50,
            "seed_start": 10001,
            "delivery_ratio": 0.75,
            "mean_reward": 9.0,
            "drop_rate": 0.01,
            "average_delay_slots": 6.0,
            "is_validation_best": True,
        }
        metrics_path = run_directory / "training_metrics.jsonl"
        metrics_path.write_text(
            json.dumps(validation_record)
            + "\n"
            + json.dumps(validation_record)
            + "\n"
            + json.dumps(
                {
                    "record_type": "training_update",
                    "environment_steps": 50040,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (checkpoint_root / "trainer_stdout.log").write_text(
            "complete\n", encoding="utf-8"
        )
        atomic_write_json(
            checkpoint_root / "code_fingerprint.json", {"code": "test"}
        )
        audit = audit_training_artifacts(checkpoint, self.job, self.config)
        self.assertTrue(audit["budget_verified"])
        self.assertEqual(audit["actual_environment_steps"], 50040)
        self.assertEqual(audit["rollout_overshoot_steps"], 40)
        self.assertEqual(audit["selected_checkpoint_step"], 48000)
        self.assertEqual(audit["earliest_best_validation_step"], 48000)
        self.assertEqual(
            audit["selected_validation_score"], [0.75, 9.0, -0.01, -6.0]
        )
        self.assertEqual(audit["selected_validation_records_at_step"], 2)
        self.assertEqual(audit["best_validation_tie_count"], 2)
        self.assertEqual(
            set(audit["artifacts"]),
            {
                "run_manifest",
                "run_config",
                "training_metrics",
                "trainer_log",
                "training_code_fingerprint",
            },
        )

        earlier_tie = dict(validation_record)
        earlier_tie["environment_steps"] = 40000
        metrics_path.write_text(
            json.dumps(earlier_tie)
            + "\n"
            + json.dumps(validation_record)
            + "\n"
            + json.dumps({"environment_steps": 50040})
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "earliest validation-score tie"):
            audit_training_artifacts(checkpoint, self.job, self.config)

        conflicting_record = dict(validation_record)
        conflicting_record["mean_reward"] = 8.0
        metrics_path.write_text(
            json.dumps(validation_record)
            + "\n"
            + json.dumps(conflicting_record)
            + "\n"
            + json.dumps({"environment_steps": 50040})
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "conflicting validation records"):
            audit_training_artifacts(checkpoint, self.job, self.config)

    def test_evaluation_shard_requires_exact_crossed_workload_cells(self):
        shard = evaluation_shard_path(self.root, self.job)
        atomic_write_csv(shard, self._evaluation_rows())
        fields, rows = validate_evaluation_shard(shard, self.job)
        self.assertIn("delivery_ratio", fields)
        self.assertEqual(len(rows), 50)

        duplicate_rows = self._evaluation_rows()
        duplicate_rows[-1]["workload_seed"] = duplicate_rows[0]["workload_seed"]
        atomic_write_csv(shard, duplicate_rows)
        with self.assertRaisesRegex(ValueError, "duplicate workload"):
            validate_evaluation_shard(shard, self.job)

        missing_field_rows = self._evaluation_rows()
        for row in missing_field_rows:
            row.pop("global_drop_cost")
        atomic_write_csv(shard, missing_field_rows)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_evaluation_shard(shard, self.job)

        unexpected_field_rows = self._evaluation_rows()
        for row in unexpected_field_rows:
            row["unregistered_metric"] = 1.0
        atomic_write_csv(shard, unexpected_field_rows)
        with self.assertRaisesRegex(ValueError, "unexpected=.*unregistered_metric"):
            validate_evaluation_shard(shard, self.job)

        nonfinite_rows = self._evaluation_rows()
        nonfinite_rows[0]["delivery_ratio"] = float("nan")
        atomic_write_csv(shard, nonfinite_rows)
        with self.assertRaisesRegex(ValueError, "invalid <class 'float'> value"):
            validate_evaluation_shard(shard, self.job)

    def test_atomic_json_never_leaves_temporary_file(self):
        target = self.root / "nested" / "state.json"
        atomic_write_json(target, {"state": "completed", "attempt": 2})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["attempt"], 2)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_merge_and_audit_only_accept_validated_current_shards(self):
        _, _, _, training_freeze = self._completed_state()
        fingerprints = expected_phase_fingerprints(
            self.root,
            [self.job],
            self.spec["spec_sha256"],
            self.config,
            self.code,
            training_freeze=training_freeze,
        )
        manifest = merge_evaluation_shards(
            self.root,
            [self.job],
            self.spec["spec_sha256"],
            fingerprints,
            training_freeze_sha256=training_freeze["content_sha256"],
        )
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["row_count"], 50)
        self.assertTrue((self.root / "episode_metrics.csv").is_file())

        audit = audit_matrix(
            self.root,
            [self.job],
            self.spec["spec_sha256"],
            fingerprints,
            training_freeze_sha256=training_freeze["content_sha256"],
        )
        self.assertTrue(audit["matrix_complete"])
        self.assertEqual(audit["counts"]["training"]["completed"], 1)
        self.assertEqual(audit["counts"]["evaluation"]["completed"], 1)

        shard = evaluation_shard_path(self.root, self.job)
        with shard.open("a", encoding="utf-8") as handle:
            handle.write("corrupt\n")
        audit = audit_matrix(
            self.root,
            [self.job],
            self.spec["spec_sha256"],
            fingerprints,
            training_freeze_sha256=training_freeze["content_sha256"],
        )
        self.assertFalse(audit["matrix_complete"])
        self.assertEqual(audit["counts"]["evaluation"]["stale_or_invalid"], 1)

    def test_complete_merge_generates_and_resumes_statistical_outputs(self):
        _, _, _, training_freeze = self._completed_state()
        fingerprints = expected_phase_fingerprints(
            self.root,
            [self.job],
            self.spec["spec_sha256"],
            self.config,
            self.code,
            training_freeze=training_freeze,
        )
        merge_manifest = merge_evaluation_shards(
            self.root,
            [self.job],
            self.spec["spec_sha256"],
            fingerprints,
            training_freeze_sha256=training_freeze["content_sha256"],
        )

        @dataclass
        class TinyEpisode:
            scenario: str
            policy: str
            policy_seed: int
            workload_seed: int

        calls = []
        contrast = SimpleNamespace(
            name="test_contrast",
            family="component_removal",
            reference="proposed",
            treatment="no_queue",
            changed_flags=("queue_features",),
            component_kind="single_component",
        )

        def effect_row():
            return {
                "scenario": self.job.scenario,
                "contrast": contrast.name,
                "contrast_family": contrast.family,
                "reference_variant": contrast.reference,
                "treatment_variant": contrast.treatment,
                "changed_flags": ";".join(contrast.changed_flags),
                "component_kind": contrast.component_kind,
                "variant": contrast.treatment,
                "metric": "delivery_ratio",
                "paired_policy_seeds": len(POLICY_SEEDS),
                "paired_workloads": len(TEST_WORKLOAD_SEEDS),
                "paired_episode_cells": len(POLICY_SEEDS)
                * len(TEST_WORKLOAD_SEEDS),
                "bootstrap_rng_seed": 18000,
                "policy_seed_ids_json": json.dumps(sorted(POLICY_SEEDS)),
                "policy_seed_mean_differences_json": json.dumps(
                    [0.0] * len(POLICY_SEEDS)
                ),
                "policy_seed_pairing": "matched_policy_seed",
                "treatment_minus_reference": 0.0,
                "sign_flip_permutations": 1 << len(POLICY_SEEDS),
                "sign_flip_tail_tie_permutations": 1,
                "sign_flip_zero_difference_count": len(POLICY_SEEDS),
                "sign_flip_tail_ties_included": True,
                "seed_level_wilcoxon_requested_method": "auto",
                "seed_level_wilcoxon_zero_method": "wilcox",
                "seed_level_wilcoxon_zero_count": len(POLICY_SEEDS),
                "seed_level_wilcoxon_absolute_tie_group_count": 0,
                "seed_level_wilcoxon_absolute_tied_value_count": 0,
                "seed_level_wilcoxon_method": (
                    "scipy_wilcoxon_two_sided_on_workload_averaged_policy_seeds"
                ),
                "raw_p_value": 1.0,
                "confirmatory_holm_within_metric_p": 1.0,
                "within_metric_bh_sensitivity_p": 1.0,
                "within_metric_family_size": 1,
                "global_holm_sensitivity_p": 1.0,
                "global_bh_exploratory_p": 1.0,
                "global_family_size": 1,
                "multiplicity_role": "primary_confirmatory_family",
            }

        def paired(rows, scenarios, contrasts):
            calls.append((len(rows), tuple(scenarios), len(contrasts)))
            return [effect_row()]

        def malformed_seed_vector(*args, **kwargs):
            effect = effect_row()
            effect["policy_seed_ids_json"] = json.dumps(sorted(POLICY_SEEDS)[:-1])
            return [effect]

        with self.assertRaisesRegex(ValueError, "frozen 12 policy seed IDs"):
            generate_final_statistics(
                self.root,
                merge_manifest,
                code_audit={},
                paired_effects_fn=malformed_seed_vector,
                analysis_manifest_fn=lambda: {"method": "test"},
                episode_row_type=TinyEpisode,
                contrasts=[contrast],
                primary_metrics=["delivery_ratio"],
                scenarios=[self.job.scenario],
            )

        manifest = generate_final_statistics(
            self.root,
            merge_manifest,
            code_audit={},
            paired_effects_fn=paired,
            analysis_manifest_fn=lambda: {"method": "test"},
            episode_row_type=TinyEpisode,
            contrasts=[contrast],
            primary_metrics=["delivery_ratio"],
            scenarios=[self.job.scenario],
        )
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(calls, [(50, (self.job.scenario,), 1)])
        self.assertTrue((self.root / "paired_ablation_effects.csv").is_file())
        self.assertTrue(
            (self.root / "statistical_analysis_manifest.json").is_file()
        )

        def should_not_run(*args, **kwargs):
            raise AssertionError("current statistical artifact should resume")

        resumed = generate_final_statistics(
            self.root,
            merge_manifest,
            code_audit={},
            paired_effects_fn=should_not_run,
            episode_row_type=TinyEpisode,
            contrasts=[contrast],
            primary_metrics=["delivery_ratio"],
            scenarios=[self.job.scenario],
        )
        self.assertEqual(
            resumed["paired_effects_csv_sha256"],
            manifest["paired_effects_csv_sha256"],
        )

    def test_unselected_cells_are_missing_not_failed(self):
        second_job = build_matrix()[1]
        fingerprints = {
            self.job.job_id: {
                "training": phase_fingerprint(
                    "training",
                    self.job,
                    config=self.config,
                    code_sha256=self.code,
                ),
                "evaluation": None,
            },
            second_job.job_id: {
                "training": phase_fingerprint(
                    "training",
                    second_job,
                    config=self.config,
                    code_sha256=self.code,
                ),
                "evaluation": None,
            },
        }
        audit = audit_matrix(
            self.root,
            [self.job, second_job],
            self.spec["spec_sha256"],
            fingerprints,
        )
        self.assertEqual(audit["counts"]["training"]["missing"], 2)
        self.assertEqual(audit["counts"]["training"]["failed"], 0)
        self.assertEqual(audit["counts"]["evaluation"]["missing"], 2)
        self.assertEqual(audit["counts"]["evaluation"]["failed"], 0)


class RetryAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.job = build_matrix()[0]
        self.config = dict(EXPECTED_MAIN_FULL_CONFIG)
        self.code = {"trainer": "c" * 64, "evaluation": "d" * 64}
        self.spec = matrix_spec(self.config, [self.job])
        self.checkpoint = self.root / "checkpoint.pt"
        ensure_historical_workload_seed_audit(
            self.root, self.root / "experiments"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _training_kwargs(self, train_one):
        return {
            "output": self.root,
            "cleanmarl": self.root / "cleanmarl",
            "project": self.root / "src",
            "device": "cpu",
            "config": self.config,
            "spec_sha256": self.spec["spec_sha256"],
            "code_sha256": self.code,
            "invocation_id": "test-invocation",
            "max_retries": 1,
            "stale_lock_seconds": 3600,
            "train_one": train_one,
            "artifact_auditor": lambda checkpoint, job, config: self._training_audit(),
        }

    def _training_audit(self):
        artifacts = {}
        for name in (
            "run_manifest",
            "run_config",
            "training_metrics",
            "trainer_log",
            "training_code_fingerprint",
        ):
            path = self.root / "audit" / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            artifacts[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        return {
            "budget_verified": True,
            "requested_environment_steps": 50000,
            "actual_environment_steps": 50040,
            "selected_checkpoint_step": 48000,
            "earliest_best_validation_step": 48000,
            "selected_validation_score": [0.75, 9.0, -0.01, -6.0],
            "selected_validation_metrics": {
                "delivery_ratio": 0.75,
                "mean_reward": 9.0,
                "drop_rate": 0.01,
                "average_delay_slots": 6.0,
            },
            "selected_validation_records_at_step": 1,
            "best_validation_tie_count": 1,
            "artifacts": artifacts,
        }

    def test_training_failure_is_retried_once_then_resumes_from_sha(self):
        calls = []

        def flaky_train(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise RuntimeError("transient")
            self.checkpoint.write_bytes(b"valid checkpoint")
            return self.checkpoint

        outcome = run_training_job(
            self.job, **self._training_kwargs(flaky_train)
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.attempts, 2)
        state = load_job_state(self.root, self.job, self.spec["spec_sha256"])
        self.assertEqual(state["training"]["attempts_total"], 2)
        self.assertEqual(
            [attempt["status"] for attempt in state["training"]["attempt_history"]],
            ["failed", "completed"],
        )

        resumed = run_training_job(
            self.job, **self._training_kwargs(flaky_train)
        )
        self.assertEqual(resumed.status, "skipped")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][0].output, calls[1][0][0].output)

    def test_changed_fingerprint_uses_an_isolated_training_root(self):
        outputs = []

        def train(args, *unused_args, **unused_kwargs):
            outputs.append(args.output)
            self.checkpoint.write_bytes(b"checkpoint")
            return self.checkpoint

        first = run_training_job(self.job, **self._training_kwargs(train))
        self.assertEqual(first.status, "completed")

        changed_kwargs = self._training_kwargs(train)
        changed_kwargs["code_sha256"] = {
            **self.code,
            "trainer": "e" * 64,
        }
        second = run_training_job(self.job, **changed_kwargs)
        self.assertEqual(second.status, "completed")
        self.assertEqual(len(outputs), 2)
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0].parent.name, "training_runs")
        self.assertEqual(outputs[1].parent.name, "training_runs")

    def test_retry_limit_is_finite_and_failure_is_persisted(self):
        calls = []

        def always_fails(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("permanent")

        outcome = run_training_job(
            self.job, **self._training_kwargs(always_fails)
        )
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(len(calls), 2)
        state = load_job_state(self.root, self.job, self.spec["spec_sha256"])
        self.assertEqual(state["training"]["status"], "failed")
        self.assertEqual(state["training"]["attempts_total"], 2)
        self.assertEqual(state["training"]["last_error"]["message"], "permanent")
        self.assertEqual(
            [
                entry["attempt_for_fingerprint"]
                for entry in state["training"]["attempt_history"]
            ],
            [1, 2],
        )

        exhausted = run_training_job(
            self.job, **self._training_kwargs(always_fails)
        )
        self.assertEqual(exhausted.status, "failed")
        self.assertIn("retry budget exhausted", exhausted.message)
        self.assertEqual(len(calls), 2)

        changed_kwargs = self._training_kwargs(always_fails)
        changed_kwargs["code_sha256"] = {
            **self.code,
            "trainer": "e" * 64,
        }
        changed = run_training_job(self.job, **changed_kwargs)
        self.assertEqual(changed.status, "failed")
        self.assertEqual(len(calls), 4)
        state = load_job_state(self.root, self.job, self.spec["spec_sha256"])
        history = state["training"]["attempt_history"]
        self.assertEqual(
            [entry["attempt_for_fingerprint"] for entry in history],
            [1, 2, 1, 2],
        )
        self.assertEqual(len({entry["fingerprint"] for entry in history}), 2)

    def test_evaluation_writes_atomic_shard_and_resumes(self):
        def train(*args, **kwargs):
            self.checkpoint.write_bytes(b"checkpoint")
            return self.checkpoint

        trained = run_training_job(self.job, **self._training_kwargs(train))
        self.assertEqual(trained.status, "completed")
        training_freeze = ensure_training_freeze(
            self.root,
            [self.job],
            spec_sha256=self.spec["spec_sha256"],
            config=self.config,
            code_sha256=self.code,
        )
        evaluations = []

        def load_policy(path, device):
            self.assertEqual(Path(path), self.checkpoint)
            return object(), {"args": {}, "actor": {}}

        def evaluate(scenario, policy, actor, policy_seed, workload_seeds, variant):
            evaluations.append(tuple(workload_seeds))
            return complete_episode_rows(self.job, workload_seeds)

        kwargs = {
            "output": self.root,
            "cleanmarl": self.root / "cleanmarl",
            "project": self.root / "src",
            "device": "cpu",
            "config": self.config,
            "spec_sha256": self.spec["spec_sha256"],
            "code_sha256": self.code,
            "invocation_id": "test-evaluation",
            "max_retries": 1,
            "stale_lock_seconds": 3600,
            "load_checkpoint_policy": load_policy,
            "evaluate_policy": evaluate,
            "training_freeze": training_freeze,
        }
        outcome = run_evaluation_job(self.job, **kwargs)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(evaluations, [TEST_WORKLOAD_SEEDS])
        shard = evaluation_shard_path(self.root, self.job)
        self.assertTrue(shard.is_file())
        validate_evaluation_shard(shard, self.job)

        resumed = run_evaluation_job(self.job, **kwargs)
        self.assertEqual(resumed.status, "skipped")
        self.assertEqual(len(evaluations), 1)

    def test_no_ppo_evaluation_keeps_label_but_uses_proposed_environment(self):
        job = next(job for job in build_matrix() if job.variant == "no_ppo_protection")

        def train(*args, **kwargs):
            self.checkpoint.write_bytes(b"checkpoint")
            return self.checkpoint

        training_kwargs = self._training_kwargs(train)
        trained = run_training_job(job, **training_kwargs)
        self.assertEqual(trained.status, "completed")
        training_freeze = ensure_training_freeze(
            self.root,
            [job],
            spec_sha256=self.spec["spec_sha256"],
            config=self.config,
            code_sha256=self.code,
        )
        observed = {}

        def evaluate(scenario, policy, actor, policy_seed, workload_seeds, variant):
            observed.update({"policy": policy, "variant": variant})
            return complete_episode_rows(job, workload_seeds)

        outcome = run_evaluation_job(
            job,
            output=self.root,
            cleanmarl=self.root / "cleanmarl",
            project=self.root / "src",
            device="cpu",
            config=self.config,
            spec_sha256=self.spec["spec_sha256"],
            code_sha256=self.code,
            invocation_id="test-no-ppo-evaluation",
            max_retries=1,
            stale_lock_seconds=3600,
            load_checkpoint_policy=lambda path, device: (object(), {}),
            evaluate_policy=evaluate,
            training_freeze=training_freeze,
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(observed["policy"], "mappo_no_ppo_protection")
        self.assertEqual(observed["variant"], "proposed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
