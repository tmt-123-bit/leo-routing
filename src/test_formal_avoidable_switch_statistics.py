from __future__ import annotations

from copy import deepcopy
import math
import unittest

import formal_avoidable_switch_statistics as stats
import numpy as np


def complete_rows() -> list[dict]:
    rows = []
    for scenario_index, scenario in enumerate(stats.SCENARIOS):
        for arm in stats.ARMS:
            for policy_index, policy_seed in enumerate(stats.POLICY_SEEDS):
                for workload_index, workload_seed in enumerate(
                    stats.VALIDATION_WORKLOAD_SEEDS
                ):
                    # Unequal denominators make an erroneous mean of episode rates
                    # differ sharply from the required ratio of sums.
                    opportunity = 1 if workload_index % 2 == 0 else 9
                    if arm == stats.ARM_QOS_ONLY_CONSTRAINED:
                        cost = 1 if workload_index % 2 == 0 else 0
                        arm_delivery = 0.78
                    elif arm == stats.ARM_QOS_ONLY_BASELINE:
                        cost = 1
                        arm_delivery = 0.80
                    else:
                        cost = 1 if workload_index % 4 == 0 else 0
                        arm_delivery = 0.79
                    delivery = (
                        arm_delivery
                        - 0.20 * scenario_index
                        + 0.0002 * policy_index
                        + 0.00001 * workload_index
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "arm": arm,
                            "policy_seed": policy_seed,
                            "workload_seed": workload_seed,
                            "delivery_ratio": delivery,
                            "decision_avoidable_switches": cost,
                            "decision_switch_opportunities": opportunity,
                            "decision_forced_switches": 2,
                            "decision_forced_switch_cost": 0,
                            "decision_avoidable_switch_rate": (
                                cost / opportunity if opportunity else 0.0
                            ),
                            "checkpoint_sha256": "extra-columns-are-allowed",
                        }
                    )
    return rows


class FormalAvoidableSwitchStatisticsTests(unittest.TestCase):
    def test_complete_grid_produces_validation_only_self_hashed_report(self):
        report = stats.analyze_validation_rows(complete_rows())
        stats.validate_report_hash(report, rows=complete_rows())
        self.assertTrue(report["validation_only"])
        self.assertTrue(report["independent_validation_gate"])
        self.assertFalse(report["checkpoint_selection_rows_included"])
        self.assertFalse(report["final_test_evidence"])
        self.assertFalse(report["paper_claim_allowed"])
        self.assertEqual(report["sealed_test_access_count"], 0)
        self.assertEqual(report["sealed_test_access"], 0)
        self.assertEqual(report["test_access_count"], 0)
        self.assertFalse(report["test_panel_consulted"])
        self.assertEqual(report["grid_audit"]["expected_rows"], 480)
        self.assertEqual(
            report["grid_audit"]["selection_workload_seeds_excluded"],
            list(range(77001, 77011)),
        )
        self.assertEqual(
            report["grid_audit"]["gate_workload_seeds"],
            list(range(77011, 77021)),
        )
        self.assertEqual(
            report["statistical_contract"]["bootstrap"]["resamples"], 5000
        )
        self.assertEqual(
            report["statistical_contract"]["bootstrap"]["maximum_total_draws"],
            5_000_000,
        )
        self.assertEqual(
            report["statistical_contract"]["bootstrap"][
                "insufficient_defined_draws"
            ],
            "fail_closed",
        )
        self.assertTrue(report["grid_audit"]["complete_pairing_verified"])

        primary = report["primary"]["paired_differences"]
        self.assertEqual(len(primary), 4)
        self.assertTrue(
            all(record["multiplicity_family_size"] == 4 for record in primary)
        )
        self.assertEqual(len(report["validation_gates"]), 8)
        self.assertEqual(
            {row["gate"] for row in report["validation_gates"]},
            {
                "delivery_noninferiority_one_sided_95_lower",
                "avoidable_rate_reduction_one_sided_95_upper",
                "constrained_rate_one_sided_95_upper",
                "every_constrained_policy_seed_rate",
            },
        )
        self.assertTrue(all("holm_adjusted_p" in record for record in primary))
        self.assertTrue(
            all(
                record["treatment_arm"] == stats.ARM_QOS_ONLY_CONSTRAINED
                and record["reference_arm"] == stats.ARM_QOS_ONLY_BASELINE
                for record in primary
            )
        )
        secondary = report["secondary_reward_shaped_control"]
        self.assertFalse(secondary["confirmatory"])
        self.assertIsNone(secondary["multiplicity_adjustment"])
        self.assertTrue(
            all("holm_adjusted_p" not in row for row in secondary["paired_effects"])
        )

    def test_missing_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 480 rows"):
            stats._analyze_validation_rows(
                complete_rows()[:-1], bootstrap_resamples=8
            )

    def test_duplicate_cell_is_rejected(self):
        rows = complete_rows()
        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate formal validation cell"):
            stats._analyze_validation_rows(rows, bootstrap_resamples=8)

    def test_wrong_workload_pairing_is_rejected(self):
        rows = complete_rows()
        rows[0]["workload_seed"] = 77999
        with self.assertRaisesRegex(
            ValueError, "unexpected formal independent gate workload"
        ):
            stats._analyze_validation_rows(rows, bootstrap_resamples=8)

    def test_non_gate_workloads_are_rejected_by_gate_statistics(self):
        for workload_seed in (stats.SELECTION_WORKLOAD_SEEDS[0], 78001):
            with self.subTest(workload_seed=workload_seed):
                rows = complete_rows()
                rows[0]["workload_seed"] = workload_seed
                with self.assertRaisesRegex(
                    ValueError, "unexpected formal independent gate workload"
                ):
                    stats._analyze_validation_rows(rows, bootstrap_resamples=8)

    def test_nonfinite_value_is_rejected(self):
        rows = complete_rows()
        rows[0]["delivery_ratio"] = math.nan
        with self.assertRaisesRegex(ValueError, "delivery_ratio must be finite"):
            stats._analyze_validation_rows(rows, bootstrap_resamples=8)

    def test_cost_and_forced_cost_contracts_are_fail_closed(self):
        rows = complete_rows()
        rows[0]["decision_avoidable_switches"] = 2
        rows[0]["decision_switch_opportunities"] = 1
        rows[0]["decision_avoidable_switch_rate"] = 1.0
        with self.assertRaisesRegex(ValueError, "cost exceeds opportunity"):
            stats._analyze_validation_rows(rows, bootstrap_resamples=8)

        rows = complete_rows()
        rows[0]["decision_forced_switch_cost"] = 1
        with self.assertRaisesRegex(ValueError, "forced decision constraint cost"):
            stats._analyze_validation_rows(rows, bootstrap_resamples=8)

    def test_reported_episode_rate_must_match_counts(self):
        rows = complete_rows()
        rows[0]["decision_avoidable_switch_rate"] = 0.5
        with self.assertRaisesRegex(ValueError, "disagrees with its cost/opportunity"):
            stats._analyze_validation_rows(rows, bootstrap_resamples=8)

    def test_zero_opportunity_bootstrap_draws_are_rejected_and_replaced(self):
        rows = complete_rows()
        first_workload = stats.VALIDATION_WORKLOAD_SEEDS[0]
        for row in rows:
            if row["arm"] == stats.ARM_QOS_ONLY_CONSTRAINED:
                row["decision_avoidable_switches"] = 0
                row["decision_switch_opportunities"] = int(
                    row["workload_seed"] == first_workload
                )
                row["decision_avoidable_switch_rate"] = 0.0

        report = stats._analyze_validation_rows(rows, bootstrap_resamples=128)
        medium = next(
            row
            for row in report["primary"]["constrained_rates"]
            if row["scenario"] == "medium_load"
        )
        self.assertEqual(medium["bootstrap_resamples"], 128)
        self.assertGreater(
            medium["bootstrap_zero_denominator_draws_rejected"], 0
        )
        self.assertEqual(
            medium["bootstrap_total_draws"],
            128 + medium["bootstrap_zero_denominator_draws_rejected"],
        )

    def test_exact_sign_flip_tolerance_scales_with_tiny_effect(self):
        result = stats._exact_sign_flip(
            np.full(len(stats.POLICY_SEEDS), 1e-16), "test"
        )
        self.assertEqual(result["sign_flip_permutations"], 256)
        self.assertEqual(result["raw_p_value"], 2 / 256)

        zeros = stats._exact_sign_flip(
            np.zeros(len(stats.POLICY_SEEDS)), "all_zero_test"
        )
        self.assertEqual(zeros["raw_p_value"], 1.0)
        self.assertEqual(zeros["sign_flip_tail_tie_permutations"], 256)
        self.assertEqual(zeros["sign_flip_zero_difference_count"], 8)

    def test_holm_adjustment_matches_known_vector_and_stable_ties(self):
        np.testing.assert_allclose(
            stats._holm_adjusted([0.01, 0.04, 0.03, 0.002]),
            [0.03, 0.06, 0.06, 0.008],
        )
        np.testing.assert_allclose(
            stats._holm_adjusted([0.01, 0.01, 0.2, 0.5]),
            [0.04, 0.04, 0.4, 0.5],
        )

    def test_rate_is_ratio_of_sums_not_mean_episode_rate(self):
        report = stats._analyze_validation_rows(
            complete_rows(), bootstrap_resamples=64
        )
        medium = next(
            row
            for row in report["primary"]["constrained_rates"]
            if row["scenario"] == "medium_load"
        )
        # Per policy: 5 costs / (5*1 + 5*9) opportunities.
        self.assertAlmostEqual(medium["estimate"], 0.1)
        self.assertNotAlmostEqual(medium["estimate"], 0.5)
        self.assertIn("ratio_of_sums", medium["ci_method"])
        self.assertIn("one_sided_95_upper_bound", medium)

        rate_effect = next(
            row
            for row in report["primary"]["paired_differences"]
            if row["scenario"] == "medium_load"
            and row["endpoint"] == "decision_avoidable_switch_rate_difference"
        )
        # Baseline is 20 / 100 = 0.2, so constrained minus baseline is -0.1.
        self.assertAlmostEqual(rate_effect["estimate"], -0.1)

    def test_study_rate_equal_weights_seed_level_ratios_not_pooled_counts(self):
        rows = complete_rows()
        first_seed = stats.POLICY_SEEDS[0]
        for row in rows:
            if (
                row["arm"] == stats.ARM_QOS_ONLY_CONSTRAINED
                and row["policy_seed"] == first_seed
            ):
                row["decision_avoidable_switches"] = 0
                row["decision_switch_opportunities"] = 100
                row["decision_avoidable_switch_rate"] = 0.0

        report = stats._analyze_validation_rows(rows, bootstrap_resamples=64)
        medium = next(
            row
            for row in report["primary"]["constrained_rates"]
            if row["scenario"] == "medium_load"
        )
        expected_equal_seed_rate = (0.0 + 7 * 0.1) / 8
        pooled_rate = 35 / (1000 + 7 * 50)
        self.assertAlmostEqual(medium["estimate"], expected_equal_seed_rate)
        self.assertAlmostEqual(
            medium["pooled_decision_count_rate_for_audit_only"], pooled_rate
        )
        self.assertNotAlmostEqual(medium["estimate"], pooled_rate)

    def test_every_seed_budget_gate_cannot_be_hidden_by_pooled_or_mean_rate(self):
        rows = complete_rows()
        first_seed = stats.POLICY_SEEDS[0]
        for row in rows:
            if (
                row["arm"] == stats.ARM_QOS_ONLY_CONSTRAINED
                and row["policy_seed"] == first_seed
            ):
                row["decision_avoidable_switches"] = 1
                row["decision_switch_opportunities"] = 5
                row["decision_avoidable_switch_rate"] = 0.2

        report = stats._analyze_validation_rows(rows, bootstrap_resamples=64)
        medium = next(
            row
            for row in report["primary"]["constrained_rates"]
            if row["scenario"] == "medium_load"
        )
        self.assertLessEqual(medium["estimate"], stats.SWITCH_BUDGET)
        self.assertFalse(medium["every_policy_seed_rate_within_budget"])
        seed_gate = next(
            row
            for row in report["validation_gates"]
            if row["scenario"] == "medium_load"
            and row["gate"] == "every_constrained_policy_seed_rate"
        )
        self.assertFalse(seed_gate["passed"])
        self.assertFalse(report["validation_gate_passed"])

    def test_budget_gates_do_not_apply_an_unregistered_numeric_slack(self):
        rows = complete_rows()
        opportunity = 10_000_000_000_000
        cost = 1_200_000_000_001
        observed_rate = cost / opportunity
        self.assertGreater(observed_rate, stats.SWITCH_BUDGET)
        for row in rows:
            if row["arm"] == stats.ARM_QOS_ONLY_CONSTRAINED:
                row["decision_avoidable_switches"] = cost
                row["decision_switch_opportunities"] = opportunity
                row["decision_avoidable_switch_rate"] = observed_rate

        report = stats._analyze_validation_rows(rows, bootstrap_resamples=64)
        for constrained_rate in report["primary"]["constrained_rates"]:
            self.assertGreater(
                constrained_rate["one_sided_95_upper_bound"],
                stats.SWITCH_BUDGET,
            )
            self.assertFalse(constrained_rate["point_estimate_within_budget"])
            self.assertFalse(constrained_rate["upper_bound_within_budget"])
            self.assertFalse(
                constrained_rate["every_policy_seed_rate_within_budget"]
            )

    def test_analysis_is_deterministic_and_input_order_independent(self):
        rows = complete_rows()
        first = stats.analyze_validation_rows(rows)
        second = stats.analyze_validation_rows(list(reversed(rows)))
        self.assertEqual(first, second)
        stats.validate_report_hash(first, rows=rows)
        with self.assertRaises(TypeError):
            stats.validate_report_hash(first)  # type: ignore[call-arg]
        tampered = deepcopy(first)
        tampered["validation_only"] = False
        with self.assertRaisesRegex(ValueError, "report_sha256 mismatch"):
            stats.validate_report_hash(tampered, rows=rows)

        semantic_tamper = deepcopy(first)
        semantic_tamper["paper_claim_allowed"] = True
        semantic_tamper.pop("report_sha256")
        semantic_tamper["report_sha256"] = stats.sha256_json(semantic_tamper)
        with self.assertRaisesRegex(ValueError, "semantic field"):
            stats.validate_report_hash(semantic_tamper, rows=rows)

        gate_tamper = deepcopy(first)
        gate_tamper["validation_gate_passed"] = not gate_tamper[
            "validation_gate_passed"
        ]
        gate_tamper.pop("report_sha256")
        gate_tamper["report_sha256"] = stats.sha256_json(gate_tamper)
        with self.assertRaisesRegex(ValueError, "aggregate validation gate"):
            stats.validate_report_hash(gate_tamper, rows=rows)

        result_tamper = deepcopy(first)
        result_tamper["primary"]["paired_differences"][0]["estimate"] = 0.123
        result_tamper.pop("report_sha256")
        result_tamper["report_sha256"] = stats.sha256_json(result_tamper)
        with self.assertRaisesRegex(ValueError, "fixed-input recomputation"):
            stats.validate_report_hash(result_tamper, rows=rows)

    def test_full_input_hash_binds_extra_columns_and_rejects_extra_nan(self):
        rows = complete_rows()
        first = stats._analyze_validation_rows(rows, bootstrap_resamples=32)
        changed = deepcopy(rows)
        changed[0]["checkpoint_sha256"] = "different-checkpoint"
        second = stats._analyze_validation_rows(changed, bootstrap_resamples=32)
        self.assertEqual(
            first["analysis_input_core_sha256"],
            second["analysis_input_core_sha256"],
        )
        self.assertNotEqual(
            first["analysis_input_sha256"], second["analysis_input_sha256"]
        )

        changed[0]["extra_numeric_metric"] = math.inf
        with self.assertRaisesRegex(ValueError, "non-finite value in full input"):
            stats._analyze_validation_rows(changed, bootstrap_resamples=8)

    def test_frozen_seed_and_bootstrap_contracts(self):
        self.assertEqual(
            stats.POLICY_SEEDS,
            (
                179055553,
                626669596,
                746965870,
                183895110,
                310818925,
                2136406109,
                985998595,
                515636025,
            ),
        )
        self.assertEqual(stats.BOOTSTRAP_RESAMPLES, 5000)
        self.assertEqual(stats.TRAIN_WORKLOAD_SEEDS, tuple(range(76001, 76201)))
        self.assertEqual(
            stats.SELECTION_WORKLOAD_SEEDS, tuple(range(77001, 77011))
        )
        self.assertEqual(
            stats.GATE_WORKLOAD_SEEDS, tuple(range(77011, 77021))
        )
        self.assertEqual(stats.VALIDATION_WORKLOAD_SEEDS, stats.GATE_WORKLOAD_SEEDS)
        self.assertEqual(stats.EXPECTED_GATE_ROW_COUNT, 480)


if __name__ == "__main__":
    unittest.main(verbosity=2)
