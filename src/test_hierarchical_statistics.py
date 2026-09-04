from __future__ import annotations

import unittest

import numpy as np

from hierarchical_statistics import (
    ABLATION_BOOTSTRAP_RNG_SEED_FORMULA,
    ABLATION_CONTRAST_ORDER,
    ABLATION_ENDPOINT_ORDER,
    ABLATION_SCENARIO_ORDER,
    add_multiplicity_corrections,
    build_crossed_matrix,
    crossed_bootstrap_mean,
    exact_sign_flip_test,
    pair_crossed_matrices,
    paired_crossed_summary,
    statistical_analysis_manifest,
)


def rows_for(values, *, policy: str, policy_seeds, workload_seeds):
    return [
        {
            "scenario": "synthetic",
            "policy": policy,
            "policy_seed": policy_seed,
            "workload_seed": workload_seed,
            "score": float(values[policy_index, workload_index]),
        }
        for policy_index, policy_seed in enumerate(policy_seeds)
        for workload_index, workload_seed in enumerate(workload_seeds)
    ]


class CrossedMatrixTests(unittest.TestCase):
    def test_builds_policy_by_workload_matrix(self):
        rows = rows_for(
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            policy="learned",
            policy_seeds=[7, 42],
            workload_seeds=[101, 102],
        )
        matrix = build_crossed_matrix(rows, "score")
        self.assertEqual(matrix.policy_seeds, (7, 42))
        # Labels remain aligned with their response values after sorting.
        by_label = {
            (policy_seed, workload_seed): matrix.values[i, j]
            for i, policy_seed in enumerate(matrix.policy_seeds)
            for j, workload_seed in enumerate(matrix.workload_seeds)
        }
        self.assertEqual(by_label[(7, 101)], 1.0)
        self.assertEqual(by_label[(42, 102)], 4.0)

    def test_rejects_missing_and_duplicate_cells(self):
        incomplete = [
            {"policy_seed": 1, "workload_seed": 10, "score": 1.0},
            {"policy_seed": 1, "workload_seed": 11, "score": 1.0},
            {"policy_seed": 2, "workload_seed": 10, "score": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "incomplete crossed design"):
            build_crossed_matrix(incomplete, "score")

        duplicate = incomplete[:1] * 2
        with self.assertRaisesRegex(ValueError, "duplicate policy/workload cell"):
            build_crossed_matrix(duplicate, "score")

    def test_deterministic_reference_is_broadcast_but_not_recounted(self):
        learned = rows_for(
            np.asarray([[2.0, 4.0], [3.0, 5.0]]),
            policy="learned",
            policy_seeds=[7, 42],
            workload_seeds=[101, 102],
        )
        deterministic = rows_for(
            np.asarray([[1.0, 2.0]]),
            policy="reference",
            policy_seeds=[-1],
            workload_seeds=[101, 102],
        )
        paired = pair_crossed_matrices(learned, deterministic, "score")
        self.assertEqual(paired.policy_seed_pairing, "deterministic_reference_broadcast")
        np.testing.assert_allclose(
            paired.difference,
            np.asarray([[1.0, 2.0], [2.0, 3.0]]),
        )


class InferenceTests(unittest.TestCase):
    def test_crossed_bootstrap_is_reproducible_and_resamples_both_axes(self):
        values = np.asarray([[0.0, 2.0], [4.0, 6.0]])
        first = crossed_bootstrap_mean(
            values, rng=np.random.default_rng(123), resamples=200
        )
        second = crossed_bootstrap_mean(
            values, rng=np.random.default_rng(123), resamples=200
        )
        np.testing.assert_array_equal(first, second)
        self.assertGreater(len(np.unique(first)), 2)
        self.assertAlmostEqual(float(first.mean()), float(values.mean()), delta=0.35)

    def test_exact_sign_flip_uses_policy_seed_count(self):
        result = exact_sign_flip_test(np.arange(1.0, 9.0))
        self.assertEqual(result["permutation_count"], 256)
        self.assertAlmostEqual(result["p_value"], 2.0 / 256.0)
        self.assertEqual(result["tail_tie_permutation_count"], 2)
        self.assertEqual(result["zero_difference_count"], 0)
        self.assertTrue(result["tail_ties_included"])

        zero = exact_sign_flip_test(np.zeros(8))
        self.assertEqual(zero["p_value"], 1.0)
        self.assertEqual(zero["tail_tie_permutation_count"], 256)
        self.assertEqual(zero["zero_difference_count"], 8)

    def test_paired_summary_does_not_treat_episode_cells_as_replicates(self):
        policy_seeds = list(range(8))
        workload_seeds = list(range(50))
        reference_values = np.zeros((1, 50))
        treatment_values = np.asarray(
            [[1.0 + 0.01 * policy_seed + 0.001 * workload_seed
              for workload_seed in workload_seeds]
             for policy_seed in policy_seeds]
        )
        paired = pair_crossed_matrices(
            rows_for(
                treatment_values,
                policy="treatment",
                policy_seeds=policy_seeds,
                workload_seeds=workload_seeds,
            ),
            rows_for(
                reference_values,
                policy="reference",
                policy_seeds=[-1],
                workload_seeds=workload_seeds,
            ),
            "score",
        )
        summary = paired_crossed_summary(paired, rng_seed=9, resamples=300)
        self.assertEqual(summary["policy_seed_count"], 8)
        self.assertEqual(summary["workload_seed_count"], 50)
        self.assertEqual(summary["episode_pair_count"], 400)
        self.assertEqual(summary["bootstrap_rng_seed"], 9)
        self.assertEqual(summary["policy_seed_ids"], policy_seeds)
        self.assertEqual(len(summary["policy_seed_mean_differences"]), 8)
        self.assertEqual(summary["policy_seed_pairing"], "deterministic_reference_broadcast")
        self.assertEqual(summary["sign_flip_permutations"], 256)
        self.assertEqual(summary["sign_flip_tail_tie_permutations"], 2)
        self.assertTrue(summary["sign_flip_tail_ties_included"])
        self.assertAlmostEqual(summary["raw_p_value"], 2.0 / 256.0)
        self.assertAlmostEqual(summary["seed_level_wilcoxon_p"], 2.0 / 256.0)
        self.assertEqual(summary["seed_level_wilcoxon_requested_method"], "auto")
        self.assertEqual(summary["seed_level_wilcoxon_zero_method"], "wilcox")
        self.assertEqual(summary["seed_level_wilcoxon_zero_count"], 0)
        self.assertGreater(summary["difference_ci95_low"], 0.0)

    def test_manifest_records_frozen_ablation_rng_indexing_and_ties(self):
        manifest = statistical_analysis_manifest()
        contract = manifest["confidence_interval"][
            "confirmatory_ablation_rng_contract"
        ]
        self.assertEqual(contract["formula"], ABLATION_BOOTSTRAP_RNG_SEED_FORMULA)
        self.assertEqual(contract["scenario_order"], list(ABLATION_SCENARIO_ORDER))
        self.assertEqual(contract["contrast_order"], list(ABLATION_CONTRAST_ORDER))
        self.assertEqual(contract["endpoint_order"], list(ABLATION_ENDPOINT_ORDER))
        self.assertTrue(manifest["hypothesis_test"]["tail_ties_included"])
        self.assertEqual(
            manifest["hypothesis_test"]["wilcoxon_requested_method"], "auto"
        )
        self.assertEqual(
            manifest["hypothesis_test"]["wilcoxon_zero_method"], "wilcox"
        )

    def test_holm_and_bh_adjustments_are_monotone(self):
        rows = [
            {"metric": "m", "raw_p_value": 0.01},
            {"metric": "m", "raw_p_value": 0.04},
            {"metric": "m", "raw_p_value": 0.03},
        ]
        add_multiplicity_corrections(rows)
        np.testing.assert_allclose(
            [row["holm_adjusted_p"] for row in rows],
            [0.03, 0.06, 0.06],
        )
        np.testing.assert_allclose(
            [row["benjamini_hochberg_p"] for row in rows],
            [0.03, 0.04, 0.04],
        )
        self.assertEqual({row["multiplicity_family_size"] for row in rows}, {3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
