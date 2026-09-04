from __future__ import annotations

import copy
import unittest

import run_integrated_hysteresis_screen as screen


def effect_row(
    contrast: str,
    scenario: str,
    metric: str,
    *,
    treatment_mean: float = 1.0,
    reference_mean: float = 1.0,
    seed_treatment: list[float] | None = None,
    seed_reference: list[float] | None = None,
) -> dict:
    seed_treatment = seed_treatment or [treatment_mean] * 4
    seed_reference = seed_reference or [reference_mean] * 4
    differences = [
        treatment - reference
        for treatment, reference in zip(seed_treatment, seed_reference)
    ]
    return {
        "contrast": contrast,
        "scenario": scenario,
        "metric": metric,
        "treatment_mean": treatment_mean,
        "reference_mean": reference_mean,
        "mean_difference": treatment_mean - reference_mean,
        "policy_seed_mean_differences": differences,
        "positive_policy_seed_fraction": sum(value > 0.0 for value in differences)
        / len(differences),
        "policy_seed_treatment_means": seed_treatment,
        "policy_seed_reference_means": seed_reference,
    }


def passing_effect_rows() -> list[dict]:
    rows: list[dict] = []
    for scenario in screen.SCENARIOS:
        for metric in screen.METRICS:
            treatment = 1.0
            reference = 1.0
            if metric == "routing_switches":
                treatment, reference = 80.0, 100.0
            elif metric == "delivery_ratio" and scenario == "hotspot_high_load":
                treatment, reference = 1.02, 1.0
            rows.append(
                effect_row(
                    "integrated_minus_proposed",
                    scenario,
                    metric,
                    treatment_mean=treatment,
                    reference_mean=reference,
                )
            )

        rows.append(
            effect_row(
                "integrated_minus_raw_context",
                scenario,
                "routing_switches",
                treatment_mean=80.0,
                reference_mean=100.0,
            )
        )

    rows.append(
        effect_row(
            "integrated_minus_posthoc",
            "hotspot_high_load",
            "delivery_ratio",
            treatment_mean=1.02,
            reference_mean=1.01,
        )
    )
    return rows


def row_for(rows: list[dict], contrast: str, scenario: str, metric: str) -> dict:
    return next(
        row
        for row in rows
        if row["contrast"] == contrast
        and row["scenario"] == scenario
        and row["metric"] == metric
    )


class IntegratedHysteresisScreenTests(unittest.TestCase):
    def test_frozen_grid_is_the_shortened_eight_job_screen(self):
        training = screen.build_training_jobs()
        evaluation = screen.build_evaluation_jobs()

        self.assertEqual(len(training), 8)
        self.assertEqual(len(evaluation), 32)
        self.assertEqual({job.variant for job in training}, {"with_congestion_context"})
        self.assertEqual(
            {job.as_dict()["route_hysteresis_beta"] for job in training},
            {0.20},
        )
        self.assertEqual(screen.VALIDATION_WORKLOAD_SEEDS, tuple(range(32001, 32021)))
        self.assertEqual(screen.TEST_WORKLOAD_SEEDS, tuple(range(37001, 37051)))
        self.assertTrue(
            set(screen.TEST_WORKLOAD_SEEDS).isdisjoint(
                screen.VALIDATION_WORKLOAD_SEEDS
            )
        )

    def test_every_strengthened_gate_can_pass(self):
        decision = screen.decide_screen(passing_effect_rows())

        self.assertTrue(decision["hard_gates_passed"])
        self.assertEqual(
            decision["decision"],
            "eligible_for_separately_frozen_50k_experiment",
        )
        self.assertFalse(decision["paper_claim_allowed"])

    def test_posthoc_training_advantage_is_mandatory(self):
        rows = passing_effect_rows()
        row = row_for(
            rows,
            "integrated_minus_posthoc",
            "hotspot_high_load",
            "delivery_ratio",
        )
        row.update(
            treatment_mean=1.014,
            reference_mean=1.01,
            mean_difference=0.004,
            policy_seed_mean_differences=[0.004] * 4,
            policy_seed_treatment_means=[1.014] * 4,
            policy_seed_reference_means=[1.01] * 4,
        )

        decision = screen.decide_screen(rows)

        self.assertFalse(decision["gate_results"]["hotspot_training_advantage_vs_posthoc"])
        self.assertEqual(decision["decision"], "do_not_advance")

    def test_raw_switch_reduction_requires_every_seed(self):
        rows = passing_effect_rows()
        row = row_for(
            rows,
            "integrated_minus_raw_context",
            "medium_load",
            "routing_switches",
        )
        row["policy_seed_treatment_means"] = [80.0, 80.0, 80.0, 100.0]

        decision = screen.decide_screen(rows)

        self.assertFalse(decision["gate_results"]["switch_seed_reduction_vs_raw"])

    def test_delay_and_class_two_worst_seed_limits_are_mandatory(self):
        cases = (
            ("average_delay_slots", [100.0, 100.0, 100.0, 104.0], "medium_average_delay_seed_safety"),
            ("class_2_delivery_ratio", [1.0, 1.0, 1.0, 0.984], "medium_class_2_seed_safety"),
        )
        for metric, treatment_means, gate in cases:
            with self.subTest(metric=metric):
                rows = copy.deepcopy(passing_effect_rows())
                row = row_for(
                    rows,
                    "integrated_minus_proposed",
                    "medium_load",
                    metric,
                )
                row["policy_seed_treatment_means"] = treatment_means
                row["policy_seed_reference_means"] = [1.0] * 4
                if metric == "average_delay_slots":
                    row["policy_seed_reference_means"] = [100.0] * 4
                row["policy_seed_mean_differences"] = [
                    treatment - reference
                    for treatment, reference in zip(
                        row["policy_seed_treatment_means"],
                        row["policy_seed_reference_means"],
                    )
                ]

                decision = screen.decide_screen(rows)

                self.assertFalse(decision["gate_results"][gate])


if __name__ == "__main__":
    unittest.main()
