from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_decoupled_hysteresis_preflight as preflight
import run_integrated_hysteresis_screen as v1
from cleanmarl_leo_multiagent_wrapper import CleanMARLLeoMultiAgentWrapper


class DecoupledHysteresisPreflightTests(unittest.TestCase):
    def test_short_frozen_grid_and_fresh_panels(self):
        self.assertEqual(len(preflight.build_training_jobs()), 4)
        self.assertEqual(len(preflight.build_evaluation_jobs()), 16)
        self.assertEqual(preflight.CONFIG["timesteps"], 10000)
        self.assertEqual(preflight.POLICY_SEEDS, (1710210210, 2078783072))
        self.assertEqual(preflight.VALIDATION_WORKLOAD_SEEDS, tuple(range(41001, 41011)))
        self.assertEqual(preflight.TEST_WORKLOAD_SEEDS, tuple(range(42001, 42026)))
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
            "mappo_context_decoupled_adaptive_hysteresis_v2_beta_0p20",
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
                "schema_version": 2,
                "type": "shared_candidate_actor",
                "route_switch_feature_index": 17,
                "route_hysteresis_beta": 0.20,
                "route_hysteresis_mode": "decoupled_adaptive",
                "route_urgency_feature_index": 20,
                "route_class_2_feature_index": 23,
                "route_hysteresis_urgency_relief": 0.50,
                "route_hysteresis_class_2_relief": 0.0,
                "candidate_feature_schema_id": "leo_multi_candidate_features_v1_dim_28",
                "candidate_feature_schema_sha256": (
                    "c66bbdd36bff16f16409e636f4fab2e3e0fff10e74b852e0448c89aedd5d3503"
                ),
            },
        )
        self.assertEqual(
            preflight.EXPECTED_SELECTION_SPEC["mode"], "stability_constrained"
        )
        self.assertEqual(
            preflight.EXPECTED_SELECTION_SPEC["source"], "validation_only"
        )
        self.assertFalse(
            preflight.EXPECTED_SELECTION_SPEC["test_panel_consulted"]
        )
        overrides = list(preflight.TRAINER_OVERRIDES)
        self.assertEqual(
            overrides[overrides.index("--route-hysteresis-mode") + 1],
            "decoupled_adaptive",
        )
        self.assertEqual(
            overrides[overrides.index("--validation-selection-mode") + 1],
            "stability_constrained",
        )

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

    def test_spec_is_explicitly_diagnostic_and_jobs_carry_v2_contract(self):
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
            {2},
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
            self.assertEqual(v1.EXPECTED_ACTOR_SPEC["schema_version"], 2)
            self.assertIs(v1.audit_integrated_checkpoint, preflight.audit_decoupled_checkpoint)
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

    def test_paired_rows_are_v2_diagnostic_and_raw_p_only(self):
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
        decision = preflight._diagnostic_decision(rows)
        self.assertEqual(decision["decision"], "diagnostic_only_no_promotion_decision")
        self.assertFalse(decision["paper_claim_allowed"])
        self.assertFalse(decision["promotion_decision_allowed"])


if __name__ == "__main__":
    unittest.main()
