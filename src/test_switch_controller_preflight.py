from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import cleanmarl_mappo_leo as trainer
import run_integrated_hysteresis_screen as v1
import run_switch_controller_preflight as controller
import run_switch_regularized_preflight as base


class SwitchControllerPreflightTests(unittest.TestCase):
    def test_frozen_jobs_and_fresh_nonoverlapping_panels(self):
        training_jobs = controller.build_training_jobs()
        evaluation_jobs = controller.build_evaluation_jobs()
        self.assertEqual(len(training_jobs), 4)
        self.assertEqual(len(evaluation_jobs), 16)
        self.assertEqual(
            {job.variant for job in training_jobs},
            {"with_avoidable_switch_reward"},
        )
        self.assertEqual(
            {job.as_dict()["environment_variant"] for job in training_jobs},
            {"with_avoidable_switch_reward"},
        )

        for job in evaluation_jobs:
            with self.subTest(arm=job.arm, seed=job.policy_seed):
                if job.arm == v1.ARM_PROPOSED:
                    self.assertEqual(job.variant, "proposed")
                    self.assertIn("/proposed/", job.source_job_id)
                elif job.arm == v1.ARM_INTEGRATED:
                    self.assertEqual(
                        job.variant, "with_avoidable_switch_reward"
                    )
                    self.assertIn("/integrated_beta_0p20/", job.source_job_id)
                else:
                    self.assertEqual(job.variant, "with_congestion_context")
                    self.assertIn(
                        "/with_congestion_context/", job.source_job_id
                    )
        self.assertEqual(controller.CONFIG["timesteps"], 6000)
        self.assertEqual(controller.SCENARIOS, ("medium_load", "hotspot_high_load"))
        self.assertEqual(controller.POLICY_SEEDS, (1710210210, 2078783072))
        self.assertEqual(
            controller.TRAIN_WORKLOAD_SEEDS, tuple(range(9001, 9201))
        )
        self.assertEqual(
            controller.VALIDATION_WORKLOAD_SEEDS, tuple(range(50001, 50011))
        )
        self.assertEqual(
            controller.TEST_WORKLOAD_SEEDS, tuple(range(51001, 51026))
        )
        self.assertIn((48001, 48010), controller.KNOWN_EXPOSED_PANELS)
        self.assertIn((49001, 49025), controller.KNOWN_EXPOSED_PANELS)
        exposed = set().union(
            *(
                set(range(start, stop + 1))
                for start, stop in controller.KNOWN_EXPOSED_PANELS
            )
        )
        train = set(controller.TRAIN_WORKLOAD_SEEDS)
        validation = set(controller.VALIDATION_WORKLOAD_SEEDS)
        test = set(controller.TEST_WORKLOAD_SEEDS)
        self.assertFalse(validation & exposed)
        self.assertFalse(test & exposed)
        self.assertFalse(validation & test)
        self.assertFalse(train & (validation | test))

    def test_schema_v5_actor_contract_is_exactly_frozen(self):
        self.assertEqual(
            controller.EXPECTED_ACTOR_SPEC,
            {
                "schema_version": 5,
                "type": "shared_candidate_actor",
                "route_switch_feature_index": 17,
                "route_hysteresis_beta": 0.2,
                "route_hysteresis_mode": "decoupled_adaptive",
                "route_urgency_feature_index": 20,
                "route_class_2_feature_index": 23,
                "route_hysteresis_urgency_relief": 0.5,
                "route_hysteresis_class_2_relief": 0.0,
                "route_hysteresis_residual_parameterization": (
                    "projected_nonnegative_urgency_linear_endpoints"
                ),
                "route_hysteresis_residual_init": 0.0,
                "route_hysteresis_residual_cap": 0.2,
                "avoidable_switch_probability_coef": 0.01,
                "candidate_feature_schema_id": (
                    "leo_multi_candidate_features_v1_dim_28"
                ),
                "candidate_feature_schema_sha256": (
                    "c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503"
                ),
            },
        )

    def test_schema_v3_regularizer_contract_is_exactly_frozen(self):
        self.assertEqual(
            controller.EXPECTED_SWITCH_REGULARIZER_SPEC,
            {
                "schema_version": 3,
                "mode": "isolated_greedy_logit_margin",
                "coefficient": 0.01,
                "logit_margin": 0.01,
                "reduction": (
                    "rollout_micro_mean_over_eligible_pre_contention_decisions"
                ),
                "no_op_action_index": 0,
                "route_switch_feature_index": 17,
                "route_urgency_feature_index": 20,
                "route_class_2_feature_index": 23,
                "route_hysteresis_beta": 0.2,
                "route_hysteresis_urgency_relief": 0.5,
                "route_hysteresis_class_2_relief": 0.0,
                "relief_weighting": "actual_policy_logits_no_extra_weighting",
                "logit_source": "actual_policy_logits",
                "gradient_scope": "cached_route_residual_endpoints_only",
                "eligibility_stage": "pre_contention",
                "minibatch_weighting": (
                    "eligible_sum_scaled_to_rollout_micro_mean"
                ),
            },
        )

    def test_cli_and_run_config_close_controller_contract(self):
        overrides = list(controller.TRAINER_OVERRIDES)
        expected_options = {
            "--route-hysteresis-residual-parameterization": "urgency_linear",
            "--avoidable-switch-reduction": "rollout_micro_mean",
            "--avoidable-switch-regularization-mode": (
                "isolated_greedy_logit_margin"
            ),
            "--avoidable-switch-logit-margin": "0.01",
            "--route-hysteresis-residual-init": "0.0",
            "--route-hysteresis-residual-cap": "0.2",
        }
        for option, expected in expected_options.items():
            self.assertEqual(overrides[overrides.index(option) + 1], expected)
        self.assertIn("--log-loss-component-gradients", overrides)

        config = controller._expected_run_config(
            controller.build_training_jobs()[0], SimpleNamespace(device="cuda")
        )
        self.assertEqual(config["leo_variant"], "with_avoidable_switch_reward")
        self.assertEqual(
            config["route_hysteresis_residual_parameterization"],
            "urgency_linear",
        )
        self.assertEqual(config["avoidable_switch_reduction"], "rollout_micro_mean")
        self.assertEqual(config["validation_seed_start"], 50001)
        self.assertEqual(controller.EXPECTED_SELECTION_SPEC["schema_version"], 5)
        self.assertTrue(config["log_loss_component_gradients"])

        args = controller.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.output.name, "switch-controller-preflight-6k-v6")

    def test_checkpoint_lookup_keeps_frozen_sources_and_treatment_separate(self):
        source_records = {}
        integrated_records = {}
        jobs = controller.build_evaluation_jobs()
        for job in jobs:
            record = {
                "source_job_id": job.source_job_id,
                "seed": job.policy_seed,
            }
            target = (
                integrated_records
                if job.arm == v1.ARM_INTEGRATED
                else source_records
            )
            target.setdefault(job.source_job_id, record)

        source = {"checkpoints": source_records}
        for job in jobs:
            with self.subTest(arm=job.arm, seed=job.policy_seed):
                observed = v1._checkpoint_record_for_evaluation(
                    job, source, integrated_records
                )
                self.assertEqual(observed["source_job_id"], job.source_job_id)
                self.assertEqual(observed["seed"], job.policy_seed)

        controller.validate_job_bindings()

    def test_rollout_micro_partition_mean_is_invariant_with_empty_partition(self):
        partitions = (
            torch.tensor([1.0, 3.0]),
            torch.empty(0),
            torch.tensor([2.0]),
            torch.tensor([4.0, 5.0, 6.0, 7.0]),
        )
        minibatch_count = len(partitions)
        rollout_denominator = sum(partition.numel() for partition in partitions)
        minibatch_terms = torch.stack(
            [
                trainer.rollout_micro_minibatch_mean(
                    partition,
                    rollout_denominator=rollout_denominator,
                    active_minibatch_count=minibatch_count,
                )
                for partition in partitions
            ]
        )
        observed = minibatch_terms.mean()
        expected = torch.cat(partitions).mean()
        torch.testing.assert_close(observed, expected)

    def test_rollout_micro_rejects_invalid_denominators(self):
        values = torch.tensor([1.0])
        for denominator in (0, -1):
            with self.subTest(rollout_denominator=denominator):
                with self.assertRaisesRegex(ValueError, "denominator"):
                    trainer.rollout_micro_minibatch_mean(
                        values,
                        rollout_denominator=denominator,
                        active_minibatch_count=1,
                    )
        for minibatch_count in (0, -1):
            with self.subTest(active_minibatch_count=minibatch_count):
                with self.assertRaisesRegex(ValueError, "minibatch count"):
                    trainer.rollout_micro_minibatch_mean(
                        values,
                        rollout_denominator=1,
                        active_minibatch_count=minibatch_count,
                    )

    def test_checkpoint_payload_requires_v6_endpoint_contracts(self):
        job = controller.build_training_jobs()[0]
        calm = torch.tensor(0.05)
        urgent = torch.tensor(0.10)
        payload = {
            "candidate_actor_spec": dict(controller.EXPECTED_ACTOR_SPEC),
            "switch_regularizer_spec": dict(
                controller.EXPECTED_SWITCH_REGULARIZER_SPEC
            ),
            "validation_selection_spec": dict(
                controller.EXPECTED_SELECTION_SPEC
            ),
            "args": controller._expected_run_config(
                job, SimpleNamespace(device="cuda")
            ),
            "candidate_feature_dim": 28,
            "action_size": 7,
            "n_agents": 24,
            "obs_size": 196,
            "critic_spec": {"node_feature_dim": 26},
            "actor": {
                "shared_candidate_actor.route_hysteresis_residual_calm_bias": calm,
                "shared_candidate_actor.route_hysteresis_residual_urgent_bias": urgent,
            },
            "candidate_actor_learned_state": {
                "route_hysteresis_residual_calm_bias": float(calm.item()),
                "route_hysteresis_residual_urgent_bias": float(urgent.item()),
            },
            "step": 6000,
        }
        with controller.configured_v6():
            base._audit_checkpoint_payload(payload, job, 6000)
            for field in ("candidate_actor_spec", "switch_regularizer_spec"):
                with self.subTest(field=field):
                    corrupted = dict(payload, **{field: {}})
                    with self.assertRaises(ValueError):
                        base._audit_checkpoint_payload(corrupted, job, 6000)
            corrupted = dict(
                payload,
                candidate_actor_learned_state={
                    "route_hysteresis_residual_calm_bias": 0.21,
                    "route_hysteresis_residual_urgent_bias": float(urgent.item()),
                },
            )
            with self.assertRaisesRegex(ValueError, "residual"):
                base._audit_checkpoint_payload(corrupted, job, 6000)

    def test_screen_spec_and_dry_run_expose_v6_contracts(self):
        args = SimpleNamespace(
            project=Path(__file__).resolve().parent,
            cleanmarl=Path("F:/cleanmarl"),
            source=Path("F:/frozen-source"),
            output=Path("F:/v6-output"),
        )
        with mock.patch.object(
            base, "screen_code_fingerprint", return_value={"runner": "hash"}
        ):
            spec = controller.build_screen_spec(args, {"source": "frozen"})
        self.assertEqual(spec["schema_version"], 6)
        self.assertEqual(spec["screen_name"], controller.SCREEN_NAME)
        self.assertEqual(
            spec["environment_variant"], "with_avoidable_switch_reward"
        )
        self.assertEqual(
            {
                job["environment_variant"]
                for job in spec["training_jobs"]
            },
            {"with_avoidable_switch_reward"},
        )
        evaluation_by_arm = {
            arm: [
                job
                for job in spec["evaluation_jobs"]
                if job["arm"] == arm
            ]
            for arm in v1.EVALUATION_ARMS
        }
        self.assertEqual(
            {
                job["environment_variant"]
                for job in evaluation_by_arm[v1.ARM_INTEGRATED]
            },
            {"with_avoidable_switch_reward"},
        )
        for arm in (v1.ARM_RAW_CONTEXT, v1.ARM_POSTHOC):
            self.assertEqual(
                {
                    job["environment_variant"]
                    for job in evaluation_by_arm[arm]
                },
                {"with_congestion_context"},
            )
            self.assertTrue(
                all(
                    "/with_congestion_context/" in job["source_job_id"]
                    for job in evaluation_by_arm[arm]
                )
            )
        self.assertEqual(spec["inferential_status"], "diagnostic_exploratory_preflight")
        self.assertFalse(spec["paper_claim_allowed"])
        self.assertFalse(spec["promotion_decision_allowed"])
        self.assertEqual(spec["integrated_policy_contract"]["schema_version"], 5)
        self.assertEqual(
            spec["switch_regularizer_contract"],
            controller.EXPECTED_SWITCH_REGULARIZER_SPEC,
        )
        self.assertEqual(spec["expected_training_jobs"], 4)
        self.assertEqual(spec["expected_evaluation_rows"], 400)
        dry = controller._dry_run_summary(args, spec)
        self.assertEqual(
            dry["switch_regularizer_spec"],
            controller.EXPECTED_SWITCH_REGULARIZER_SPEC,
        )
        self.assertFalse(dry["dry_run_writes_output"])

    def test_v6_context_restores_base_and_v1_state(self):
        base_state = {
            "screen": base.SCREEN_NAME,
            "spec": base.EXPECTED_SWITCH_REGULARIZER_SPEC,
            "overrides": base._V1_OVERRIDES,
        }
        v1_state = {
            "screen": v1.SCREEN_NAME,
            "seeds": v1.VALIDATION_WORKLOAD_SEEDS,
            "environment_variant": v1.ENVIRONMENT_VARIANT,
            "training_job": v1.TrainingJob,
            "evaluation_job": v1.EvaluationJob,
        }
        with controller.configured_v6():
            self.assertEqual(base.SCREEN_NAME, controller.SCREEN_NAME)
            self.assertEqual(base.ENVIRONMENT_VARIANT, controller.ENVIRONMENT_VARIANT)
            with base.configured_v1():
                self.assertEqual(v1.SCREEN_NAME, controller.SCREEN_NAME)
                self.assertEqual(
                    v1.ENVIRONMENT_VARIANT,
                    controller.ENVIRONMENT_VARIANT,
                )
                self.assertIs(v1.TrainingJob, controller.ControllerTrainingJob)
                self.assertIs(
                    v1.EvaluationJob, controller.ControllerEvaluationJob
                )
                self.assertEqual(
                    v1.VALIDATION_WORKLOAD_SEEDS,
                    controller.VALIDATION_WORKLOAD_SEEDS,
                )
        self.assertEqual(base.SCREEN_NAME, base_state["screen"])
        self.assertIs(base.EXPECTED_SWITCH_REGULARIZER_SPEC, base_state["spec"])
        self.assertIs(base._V1_OVERRIDES, base_state["overrides"])
        self.assertEqual(v1.SCREEN_NAME, v1_state["screen"])
        self.assertEqual(v1.VALIDATION_WORKLOAD_SEEDS, v1_state["seeds"])
        self.assertEqual(
            v1.ENVIRONMENT_VARIANT, v1_state["environment_variant"]
        )
        self.assertIs(v1.TrainingJob, v1_state["training_job"])
        self.assertIs(v1.EvaluationJob, v1_state["evaluation_job"])


if __name__ == "__main__":
    unittest.main()
