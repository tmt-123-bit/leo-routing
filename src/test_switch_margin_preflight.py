from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import run_integrated_hysteresis_screen as v1
import run_switch_margin_preflight as margin
import run_switch_regularized_preflight as base


class SwitchMarginPreflightTests(unittest.TestCase):
    def test_frozen_jobs_and_fresh_panels(self):
        self.assertEqual(len(margin.build_training_jobs()), 4)
        self.assertEqual(len(margin.build_evaluation_jobs()), 16)
        self.assertEqual(margin.CONFIG["timesteps"], 6000)
        self.assertEqual(margin.POLICY_SEEDS, (1710210210, 2078783072))
        self.assertEqual(
            margin.VALIDATION_WORKLOAD_SEEDS, tuple(range(46001, 46011))
        )
        self.assertEqual(margin.TEST_WORKLOAD_SEEDS, tuple(range(47001, 47026)))
        exposed = set().union(
            *(
                set(range(start, stop + 1))
                for start, stop in margin.KNOWN_EXPOSED_PANELS
            )
        )
        self.assertFalse(set(margin.VALIDATION_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(set(margin.TEST_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(
            set(margin.VALIDATION_WORKLOAD_SEEDS)
            & set(margin.TEST_WORKLOAD_SEEDS)
        )

    def test_margin_regularizer_contract_is_exactly_frozen(self):
        self.assertEqual(
            margin.EXPECTED_SWITCH_REGULARIZER_SPEC,
            {
                "schema_version": 1,
                "mode": "greedy_logit_margin",
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
                "relief_weighting": (
                    "restore_full_hysteresis_before_hinge_then_multiply_adaptive_relief_scale"
                ),
            },
        )
        overrides = list(margin.TRAINER_OVERRIDES)
        self.assertEqual(
            overrides[
                overrides.index("--avoidable-switch-regularization-mode") + 1
            ],
            "greedy_logit_margin",
        )
        self.assertEqual(
            overrides[overrides.index("--avoidable-switch-logit-margin") + 1],
            "0.01",
        )
        self.assertIn("--log-loss-component-gradients", overrides)

    def test_expected_run_config_closes_margin_and_logging_contract(self):
        config = margin._expected_run_config(
            margin.build_training_jobs()[0], SimpleNamespace(device="cuda")
        )
        self.assertEqual(
            config["avoidable_switch_regularization_mode"],
            "greedy_logit_margin",
        )
        self.assertEqual(config["avoidable_switch_logit_margin"], 0.01)
        self.assertTrue(config["log_loss_component_gradients"])
        self.assertEqual(config["validation_seed_start"], 46001)

    def test_checkpoint_payload_requires_the_full_margin_contract(self):
        job = margin.build_training_jobs()[0]
        payload = {
            "candidate_actor_spec": dict(margin.EXPECTED_ACTOR_SPEC),
            "switch_regularizer_spec": dict(
                margin.EXPECTED_SWITCH_REGULARIZER_SPEC
            ),
            "validation_selection_spec": dict(margin.EXPECTED_SELECTION_SPEC),
            "args": margin._expected_run_config(
                job, SimpleNamespace(device="cuda")
            ),
            "candidate_feature_dim": 28,
            "action_size": 7,
            "n_agents": 24,
            "obs_size": 196,
            "critic_spec": {"node_feature_dim": 26},
            "step": 6000,
        }
        with margin.configured_v4():
            base._audit_checkpoint_payload(payload, job, 6000)
            corrupted = dict(payload, switch_regularizer_spec={})
            with self.assertRaisesRegex(
                ValueError, "switch regularizer spec mismatch"
            ):
                base._audit_checkpoint_payload(corrupted, job, 6000)

    def test_screen_spec_and_dry_run_expose_regularizer_contract(self):
        args = SimpleNamespace(
            project=Path(__file__).resolve().parent,
            cleanmarl=Path("F:/cleanmarl"),
            source=Path("F:/frozen-source"),
            output=Path("F:/v4-output"),
        )
        with mock.patch.object(
            base, "screen_code_fingerprint", return_value={"runner": "hash"}
        ):
            spec = margin.build_screen_spec(args, {"source": "frozen"})
        self.assertEqual(spec["schema_version"], 4)
        self.assertEqual(spec["screen_name"], margin.SCREEN_NAME)
        self.assertEqual(
            spec["switch_regularizer_contract"],
            margin.EXPECTED_SWITCH_REGULARIZER_SPEC,
        )
        self.assertEqual(spec["expected_training_jobs"], 4)
        self.assertEqual(spec["expected_evaluation_rows"], 400)
        self.assertEqual(
            {
                job["switch_regularizer_spec"]["mode"]
                for job in spec["training_jobs"]
            },
            {"greedy_logit_margin"},
        )
        dry = margin._dry_run_summary(args, spec)
        self.assertEqual(
            dry["switch_regularizer_spec"],
            margin.EXPECTED_SWITCH_REGULARIZER_SPEC,
        )
        self.assertFalse(dry["dry_run_writes_output"])

    def test_v4_context_restores_v3_and_v1_module_state(self):
        base_state = {
            "screen": base.SCREEN_NAME,
            "spec": base.EXPECTED_SWITCH_REGULARIZER_SPEC,
            "overrides": base._V1_OVERRIDES,
        }
        v1_state = {
            "screen": v1.SCREEN_NAME,
            "seeds": v1.VALIDATION_WORKLOAD_SEEDS,
        }
        with margin.configured_v4():
            self.assertEqual(base.SCREEN_NAME, margin.SCREEN_NAME)
            self.assertEqual(
                base.EXPECTED_SWITCH_REGULARIZER_SPEC,
                margin.EXPECTED_SWITCH_REGULARIZER_SPEC,
            )
            with base.configured_v1():
                self.assertEqual(v1.SCREEN_NAME, margin.SCREEN_NAME)
                self.assertEqual(
                    v1.VALIDATION_WORKLOAD_SEEDS,
                    margin.VALIDATION_WORKLOAD_SEEDS,
                )
        self.assertEqual(base.SCREEN_NAME, base_state["screen"])
        self.assertIs(
            base.EXPECTED_SWITCH_REGULARIZER_SPEC, base_state["spec"]
        )
        self.assertIs(base._V1_OVERRIDES, base_state["overrides"])
        self.assertEqual(v1.SCREEN_NAME, v1_state["screen"])
        self.assertEqual(v1.VALIDATION_WORKLOAD_SEEDS, v1_state["seeds"])


if __name__ == "__main__":
    unittest.main()
