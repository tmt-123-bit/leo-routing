from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import run_integrated_hysteresis_screen as v1
import run_switch_regularized_preflight as base
import run_switch_residual_preflight as residual


class SwitchResidualPreflightTests(unittest.TestCase):
    def test_frozen_grid_and_fresh_panels(self):
        self.assertEqual(len(residual.build_training_jobs()), 4)
        self.assertEqual(len(residual.build_evaluation_jobs()), 16)
        self.assertEqual(residual.CONFIG["timesteps"], 6000)
        self.assertEqual(residual.POLICY_SEEDS, (1710210210, 2078783072))
        self.assertEqual(
            residual.TRAIN_WORKLOAD_SEEDS, tuple(range(9001, 9201))
        )
        self.assertEqual(
            residual.VALIDATION_WORKLOAD_SEEDS, tuple(range(48001, 48011))
        )
        self.assertEqual(
            residual.TEST_WORKLOAD_SEEDS, tuple(range(49001, 49026))
        )
        exposed = set().union(
            *(
                set(range(start, stop + 1))
                for start, stop in residual.KNOWN_EXPOSED_PANELS
            )
        )
        self.assertFalse(set(residual.VALIDATION_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(set(residual.TEST_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(
            set(residual.VALIDATION_WORKLOAD_SEEDS)
            & set(residual.TEST_WORKLOAD_SEEDS)
        )

    def test_schema_v4_actor_contract_is_exactly_frozen(self):
        self.assertEqual(
            residual.EXPECTED_ACTOR_SPEC,
            {
                "schema_version": 4,
                "type": "shared_candidate_actor",
                "route_switch_feature_index": 17,
                "route_hysteresis_beta": 0.2,
                "route_hysteresis_mode": "decoupled_adaptive",
                "route_urgency_feature_index": 20,
                "route_class_2_feature_index": 23,
                "route_hysteresis_urgency_relief": 0.5,
                "route_hysteresis_class_2_relief": 0.0,
                "route_hysteresis_residual_parameterization": (
                    "projected_nonnegative_scalar"
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

    def test_schema_v2_regularizer_contract_is_exactly_frozen(self):
        self.assertEqual(
            residual.EXPECTED_SWITCH_REGULARIZER_SPEC,
            {
                "schema_version": 2,
                "mode": "isolated_greedy_logit_margin",
                "coefficient": 0.01,
                "logit_margin": 0.01,
                "reduction": "conditional_mean_over_eligible_active_decisions",
                "no_op_action_index": 0,
                "route_switch_feature_index": 17,
                "route_urgency_feature_index": 20,
                "route_class_2_feature_index": 23,
                "route_hysteresis_beta": 0.2,
                "route_hysteresis_urgency_relief": 0.5,
                "route_hysteresis_class_2_relief": 0.0,
                "relief_weighting": "actual_policy_logits_no_extra_weighting",
                "logit_source": "actual_policy_logits",
                "gradient_scope": "cached_route_residual_only",
            },
        )

    def test_trainer_overrides_and_run_config_close_residual_contract(self):
        overrides = list(residual.TRAINER_OVERRIDES)
        expected_options = {
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

        config = residual._expected_run_config(
            residual.build_training_jobs()[0], SimpleNamespace(device="cuda")
        )
        self.assertEqual(
            config["avoidable_switch_regularization_mode"],
            "isolated_greedy_logit_margin",
        )
        self.assertEqual(config["avoidable_switch_logit_margin"], 0.01)
        self.assertEqual(config["route_hysteresis_residual_init"], 0.0)
        self.assertEqual(config["route_hysteresis_residual_cap"], 0.2)
        self.assertTrue(config["log_loss_component_gradients"])
        self.assertEqual(config["validation_seed_start"], 48001)
        self.assertEqual(residual.EXPECTED_SELECTION_SPEC["schema_version"], 4)

    def test_checkpoint_payload_requires_both_v5_contracts(self):
        job = residual.build_training_jobs()[0]
        residual_tensor = torch.tensor(0.1)
        payload = {
            "candidate_actor_spec": dict(residual.EXPECTED_ACTOR_SPEC),
            "switch_regularizer_spec": dict(
                residual.EXPECTED_SWITCH_REGULARIZER_SPEC
            ),
            "validation_selection_spec": dict(
                residual.EXPECTED_SELECTION_SPEC
            ),
            "args": residual._expected_run_config(
                job, SimpleNamespace(device="cuda")
            ),
            "candidate_feature_dim": 28,
            "action_size": 7,
            "n_agents": 24,
            "obs_size": 196,
            "critic_spec": {"node_feature_dim": 26},
            "actor": {
                "shared_candidate_actor.route_hysteresis_residual_bias": (
                    residual_tensor
                )
            },
            "candidate_actor_learned_state": {
                "route_hysteresis_residual_bias": float(
                    residual_tensor.item()
                )
            },
            "step": 6000,
        }
        with residual.configured_v5():
            base._audit_checkpoint_payload(payload, job, 6000)
            for field in ("candidate_actor_spec", "switch_regularizer_spec"):
                with self.subTest(field=field):
                    corrupted = dict(payload, **{field: {}})
                    with self.assertRaises(ValueError):
                        base._audit_checkpoint_payload(corrupted, job, 6000)
            corrupted = dict(
                payload,
                candidate_actor_learned_state={
                    "route_hysteresis_residual_bias": 0.21
                },
            )
            with self.assertRaisesRegex(ValueError, "residual"):
                base._audit_checkpoint_payload(corrupted, job, 6000)

    def test_screen_spec_and_dry_run_expose_v5_contracts(self):
        args = SimpleNamespace(
            project=Path(__file__).resolve().parent,
            cleanmarl=Path("F:/cleanmarl"),
            source=Path("F:/frozen-source"),
            output=Path("F:/v5-output"),
        )
        with mock.patch.object(
            base, "screen_code_fingerprint", return_value={"runner": "hash"}
        ):
            spec = residual.build_screen_spec(args, {"source": "frozen"})
        self.assertEqual(spec["schema_version"], 5)
        self.assertEqual(spec["screen_name"], residual.SCREEN_NAME)
        self.assertEqual(
            spec["switch_regularizer_contract"],
            residual.EXPECTED_SWITCH_REGULARIZER_SPEC,
        )
        self.assertEqual(spec["integrated_policy_contract"]["schema_version"], 4)
        self.assertEqual(spec["expected_training_jobs"], 4)
        self.assertEqual(spec["expected_evaluation_rows"], 400)
        self.assertEqual(
            {
                job["switch_regularizer_spec"]["mode"]
                for job in spec["training_jobs"]
            },
            {"isolated_greedy_logit_margin"},
        )
        dry = residual._dry_run_summary(args, spec)
        self.assertEqual(
            dry["switch_regularizer_spec"],
            residual.EXPECTED_SWITCH_REGULARIZER_SPEC,
        )
        self.assertFalse(dry["dry_run_writes_output"])

    def test_v5_context_restores_base_and_v1_state(self):
        base_state = {
            "screen": base.SCREEN_NAME,
            "spec": base.EXPECTED_SWITCH_REGULARIZER_SPEC,
            "overrides": base._V1_OVERRIDES,
        }
        v1_state = {
            "screen": v1.SCREEN_NAME,
            "seeds": v1.VALIDATION_WORKLOAD_SEEDS,
        }
        with residual.configured_v5():
            self.assertEqual(base.SCREEN_NAME, residual.SCREEN_NAME)
            self.assertEqual(
                base.EXPECTED_SWITCH_REGULARIZER_SPEC,
                residual.EXPECTED_SWITCH_REGULARIZER_SPEC,
            )
            with base.configured_v1():
                self.assertEqual(v1.SCREEN_NAME, residual.SCREEN_NAME)
                self.assertEqual(
                    v1.VALIDATION_WORKLOAD_SEEDS,
                    residual.VALIDATION_WORKLOAD_SEEDS,
                )
        self.assertEqual(base.SCREEN_NAME, base_state["screen"])
        self.assertIs(base.EXPECTED_SWITCH_REGULARIZER_SPEC, base_state["spec"])
        self.assertIs(base._V1_OVERRIDES, base_state["overrides"])
        self.assertEqual(v1.SCREEN_NAME, v1_state["screen"])
        self.assertEqual(v1.VALIDATION_WORKLOAD_SEEDS, v1_state["seeds"])


if __name__ == "__main__":
    unittest.main()
