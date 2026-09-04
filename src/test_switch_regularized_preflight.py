from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_switch_regularized_preflight as preflight
import run_integrated_hysteresis_screen as v1
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper


class SwitchRegularizedPreflightTests(unittest.TestCase):
    def test_short_frozen_grid_and_fresh_panels(self):
        self.assertEqual(len(preflight.build_training_jobs()), 4)
        self.assertEqual(len(preflight.build_evaluation_jobs()), 16)
        self.assertEqual(preflight.CONFIG["timesteps"], 6000)
        self.assertEqual(preflight.POLICY_SEEDS, (1710210210, 2078783072))
        self.assertEqual(preflight.VALIDATION_WORKLOAD_SEEDS, tuple(range(43001, 43011)))
        self.assertEqual(preflight.TEST_WORKLOAD_SEEDS, tuple(range(44001, 44026)))
        exposed = set().union(
            *(
                set(range(start, stop + 1))
                for start, stop in preflight.KNOWN_EXPOSED_PANELS
            )
        )
        self.assertFalse(set(preflight.VALIDATION_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(set(preflight.TEST_WORKLOAD_SEEDS) & exposed)
        self.assertFalse(
            set(preflight.VALIDATION_WORKLOAD_SEEDS)
            & set(preflight.TEST_WORKLOAD_SEEDS)
        )
        self.assertEqual(
            preflight.POLICY_NAMES[v1.ARM_INTEGRATED],
            "mappo_context_switch_regularized_v3_beta_0p20_coef_0p10",
        )
        for job in preflight.build_evaluation_jobs():
            self.assertEqual(
                preflight.evaluation_job_record(job)["policy_name"],
                preflight.POLICY_NAMES[job.arm],
            )

    def test_actor_and_selection_contracts_are_frozen(self):
        self.assertEqual(
            preflight.EXPECTED_ACTOR_SPEC,
            {
                "schema_version": 3,
                "type": "shared_candidate_actor",
                "route_switch_feature_index": 17,
                "route_hysteresis_beta": 0.20,
                "route_hysteresis_mode": "decoupled_adaptive",
                "route_urgency_feature_index": 20,
                "route_class_2_feature_index": 23,
                "route_hysteresis_urgency_relief": 0.50,
                "route_hysteresis_class_2_relief": 0.0,
                "avoidable_switch_probability_coef": 0.10,
                "candidate_feature_schema_id": "leo_multi_candidate_features_v1_dim_28",
                "candidate_feature_schema_sha256": (
                    "c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503"
                ),
            },
        )
        self.assertEqual(
            preflight.EXPECTED_SELECTION_SPEC["mode"],
            "avoidable_stability_constrained",
        )
        self.assertEqual(
            preflight.EXPECTED_SELECTION_SPEC["source"], "validation_only"
        )
        self.assertFalse(
            preflight.EXPECTED_SELECTION_SPEC["test_panel_consulted"]
        )
        self.assertEqual(
            preflight.EXPECTED_SELECTION_SPEC["metric_aggregation"][
                "avoidable_switch_rate"
            ],
            "micro_ratio_over_validation_switch_opportunities",
        )
        self.assertIn(
            "avoidable_routing_switches_total",
            preflight.SELECTED_VALIDATION_METRIC_FIELDS,
        )
        overrides = list(preflight.TRAINER_OVERRIDES)
        self.assertEqual(
            overrides[overrides.index("--route-hysteresis-mode") + 1],
            "decoupled_adaptive",
        )
        self.assertEqual(
            overrides[overrides.index("--validation-selection-mode") + 1],
            "avoidable_stability_constrained",
        )
        self.assertEqual(
            overrides[
                overrides.index("--avoidable-switch-probability-coef") + 1
            ],
            "0.1",
        )

    def test_validation_micro_rate_is_recomputed_from_exact_totals(self):
        record = {
            "episodes": 10,
            "routing_switches": 3.0,
            "routing_switches_total": 30,
            "avoidable_routing_switches": 2.0,
            "avoidable_routing_switches_total": 20,
            "forced_routing_switches": 1.0,
            "forced_routing_switches_total": 10,
            "switch_opportunities": 80,
            "avoidable_switch_rate": 0.25,
        }
        preflight._audit_validation_record_aggregation(record)

        corrupted = dict(record, avoidable_switch_rate=0.20)
        with self.assertRaisesRegex(ValueError, "switch rate aggregation mismatch"):
            preflight._audit_validation_record_aggregation(corrupted)

    def test_validation_aggregation_rejects_inconsistent_mean(self):
        record = {
            "episodes": 10,
            "routing_switches": 3.0,
            "routing_switches_total": 30,
            "avoidable_routing_switches": 2.1,
            "avoidable_routing_switches_total": 20,
            "forced_routing_switches": 1.0,
            "forced_routing_switches_total": 10,
            "switch_opportunities": 80,
            "avoidable_switch_rate": 0.25,
        }
        with self.assertRaisesRegex(
            ValueError, "avoidable_routing_switches aggregation mismatch"
        ):
            preflight._audit_validation_record_aggregation(record)

    def test_checkpoint_payload_embeds_selection_contract(self):
        job = preflight.build_training_jobs()[0]
        payload = {
            "candidate_actor_spec": dict(preflight.EXPECTED_ACTOR_SPEC),
            "validation_selection_spec": dict(preflight.EXPECTED_SELECTION_SPEC),
            "args": preflight._expected_run_config(
                job, SimpleNamespace(device="cpu")
            ),
            "candidate_feature_dim": 28,
            "action_size": 7,
            "n_agents": 24,
            "obs_size": 196,
            "critic_spec": {"node_feature_dim": 26},
            "step": 600,
        }
        preflight._audit_checkpoint_payload(payload, job, 600)

        corrupted = dict(payload, validation_selection_spec={})
        with self.assertRaisesRegex(ValueError, "validation selection spec mismatch"):
            preflight._audit_checkpoint_payload(corrupted, job, 600)

    def test_actor_spec_uses_exact_congestion_wrapper_feature_schema(self):
        wrapper = CleanMARLLeoMultiAgentWrapper(
            "medium_load", variant=preflight.ENVIRONMENT_VARIANT
        )
        try:
            schema = wrapper.get_candidate_feature_schema()
        finally:
            wrapper.close()
        self.assertEqual(
            preflight.EXPECTED_ACTOR_SPEC["candidate_feature_schema_id"],
            schema["schema_id"],
        )
        self.assertEqual(
            preflight.EXPECTED_ACTOR_SPEC["candidate_feature_schema_sha256"],
            schema["sha256"],
        )
        self.assertEqual(len(schema["feature_names"]), 28)

    def test_spec_is_explicitly_diagnostic_and_jobs_carry_v3_contract(self):
        args = SimpleNamespace(
            project=Path(__file__).resolve().parent,
            cleanmarl=Path("F:/cleanmarl"),
            source=Path("F:/frozen-source"),
        )
        with mock.patch.object(
            preflight, "screen_code_fingerprint", return_value={"runner": "hash"}
        ):
            spec = preflight.build_screen_spec(args, {"source": "frozen"})
        self.assertEqual(spec["inferential_status"], "diagnostic_exploratory_preflight")
        self.assertFalse(spec["paper_claim_allowed"])
        self.assertFalse(spec["promotion_decision_allowed"])
        self.assertEqual(spec["expected_training_jobs"], 4)
        self.assertEqual(spec["expected_evaluation_rows"], 400)
        self.assertEqual(
            {job["candidate_actor_spec"]["schema_version"] for job in spec["training_jobs"]},
            {3},
        )
        self.assertEqual(
            {job["validation_selection"]["source"] for job in spec["training_jobs"]},
            {"validation_only"},
        )

    def test_v1_module_state_is_restored(self):
        original = {
            "screen_name": v1.SCREEN_NAME,
            "schema_version": v1.SCHEMA_VERSION,
            "policy_seeds": v1.POLICY_SEEDS,
            "config": v1.CONFIG,
            "audit": v1.audit_integrated_checkpoint,
        }
        with preflight.configured_v1():
            self.assertEqual(v1.SCREEN_NAME, preflight.SCREEN_NAME)
            self.assertEqual(v1.EXPECTED_ACTOR_SPEC["schema_version"], 3)
            self.assertIs(
                v1.audit_integrated_checkpoint,
                preflight.audit_switch_regularized_checkpoint,
            )
        self.assertEqual(v1.SCREEN_NAME, original["screen_name"])
        self.assertEqual(v1.SCHEMA_VERSION, original["schema_version"])
        self.assertEqual(v1.POLICY_SEEDS, original["policy_seeds"])
        self.assertIs(v1.CONFIG, original["config"])
        self.assertIs(v1.audit_integrated_checkpoint, original["audit"])

    def test_statistics_manifest_explicitly_reports_raw_p_only(self):
        stats = preflight.raw_exploratory_statistics_manifest()
        multiple = stats["multiple_comparisons"]
        self.assertEqual(stats["inferential_status"], "raw_exploratory_only")
        self.assertEqual(multiple["adjustments_applied"], [])
        self.assertFalse(multiple["holm_applied"])
        self.assertFalse(multiple["benjamini_hochberg_applied"])
        self.assertIn("raw", multiple["reported_p_values"])
        self.assertEqual(stats["hypothesis_test"]["reported_field"], "raw_p_value")
        self.assertNotIn("sensitivity_test", stats["hypothesis_test"])

    def test_paired_rows_are_v3_diagnostic_and_raw_p_only(self):
        source = {
            "analysis": "exploratory_integrated_hysteresis_holdout",
            "contrast": "integrated_minus_proposed",
            "scenario": "medium_load",
            "metric": "delivery_ratio",
            "raw_p_value": 0.5,
            "p_value_method": "exact_policy_seed_sign_flip_on_workload_means",
            "seed_level_wilcoxon_p": 0.5,
            "seed_level_wilcoxon_method": "sensitivity",
            "holm_adjusted_p": 1.0,
            "benjamini_hochberg_p": 1.0,
            "multiplicity_family_size": 10,
        }
        prepared = preflight.prepare_paired_rows([source])
        row = prepared[0]
        self.assertEqual(row["analysis"], preflight.PAIRED_ANALYSIS)
        self.assertEqual(row["inferential_status"], "raw_exploratory_only")
        self.assertFalse(row["confirmatory"])
        self.assertEqual(row["raw_p_value"], 0.5)
        self.assertEqual(
            [field for field in row if "p_value" in field or field.endswith("_p")],
            ["raw_p_value"],
        )
        self.assertNotIn("holm_adjusted_p", row)
        self.assertNotIn("benjamini_hochberg_p", row)
        self.assertNotIn("seed_level_wilcoxon_p", row)
        self.assertNotIn("seed_level_wilcoxon_method", row)
        self.assertNotIn("multiplicity_family_size", row)
        self.assertEqual(source["analysis"], "exploratory_integrated_hysteresis_holdout")

    def test_paired_row_validator_rejects_adjusted_p_fields(self):
        row = {
            "analysis": preflight.PAIRED_ANALYSIS,
            "inferential_status": "raw_exploratory_only",
            "confirmatory": False,
            "raw_p_value": 0.5,
            "holm_adjusted_p": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "non-raw p-value fields"):
            preflight.validate_raw_only_paired_rows([row])

    def test_diagnostic_decision_never_promotes(self):
        rows = []
        for scenario in preflight.SCENARIOS:
            for metric, difference in (
                ("delivery_ratio", 0.1),
                ("routing_switches", -10.0),
                ("class_2_delivery_ratio", 0.1),
                ("avoidable_switch_rate", -0.1),
            ):
                rows.append(
                    {
                        "contrast": "integrated_minus_proposed",
                        "scenario": scenario,
                        "metric": metric,
                        "mean_difference": difference,
                        "positive_policy_seed_fraction": 1.0,
                    }
                )
            rows.append(
                {
                    "contrast": "integrated_minus_raw_context",
                    "scenario": scenario,
                    "metric": "routing_switches",
                    "mean_difference": -10.0,
                    "positive_policy_seed_fraction": 0.0,
                }
            )
        decision = preflight._diagnostic_decision(rows)
        self.assertEqual(decision["decision"], "diagnostic_only_no_promotion_decision")
        self.assertFalse(decision["paper_claim_allowed"])
        self.assertFalse(decision["promotion_decision_allowed"])


if __name__ == "__main__":
    unittest.main()
