from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from types import SimpleNamespace
from typing import get_type_hints
import unittest
from unittest.mock import patch

from mappo_evaluation import EpisodeMetrics
from hierarchical_statistics import (
    ABLATION_CONTRAST_ORDER,
    ABLATION_ENDPOINT_ORDER,
    ABLATION_SCENARIO_ORDER,
)
import recompute_hierarchical_statistics as recompute
from run_ablation_experiments import (
    DEVELOPMENT_WORKLOAD_SEEDS,
    PLANNED_CONTRASTS,
    PRIMARY_METRICS,
    SCENARIOS,
    PlannedContrast,
    TEST_WORKLOAD_SEEDS,
    paired_effects,
    parse_args,
)


def episode_row(scenario: str, policy: str, policy_seed: int, workload_seed: int):
    values = {}
    for field, field_type in get_type_hints(EpisodeMetrics).items():
        if field_type is str:
            values[field] = "value"
        elif field_type is int:
            values[field] = 1
        elif field_type is float:
            values[field] = 0.5
        else:
            raise AssertionError(field_type)
    values.update(
        scenario=scenario,
        policy=policy,
        policy_seed=policy_seed,
        workload_seed=workload_seed,
    )
    return SimpleNamespace(**values)


class FormalGridValidationTests(unittest.TestCase):
    def test_manifest_index_orders_match_analysis_code(self):
        self.assertEqual(ABLATION_SCENARIO_ORDER, tuple(SCENARIOS))
        self.assertEqual(
            ABLATION_CONTRAST_ORDER,
            tuple(contrast.name for contrast in PLANNED_CONTRASTS),
        )
        self.assertEqual(ABLATION_ENDPOINT_ORDER, tuple(PRIMARY_METRICS))

    def setUp(self):
        self.scenarios = ("scenario_b", "scenario_a")
        self.variants = ("proposed", "no_queue")
        self.policy_seeds = (7, 42)
        self.workload_seeds = (31001, 31002)
        self.rows = [
            episode_row(scenario, f"mappo_{variant}", policy_seed, workload_seed)
            for scenario in self.scenarios
            for variant in self.variants
            for policy_seed in self.policy_seeds
            for workload_seed in self.workload_seeds
        ]

    def contract_patch(self):
        return patch.multiple(
            recompute,
            ABLATION_SCENARIOS=self.scenarios,
            ABLATION_VARIANTS=self.variants,
            POLICY_SEEDS=self.policy_seeds,
            TEST_WORKLOAD_SEEDS=self.workload_seeds,
        )

    def test_formal_path_requires_complete_finite_grid_and_fixed_order(self):
        with self.contract_patch():
            validation = recompute.validate_formal_ablation_rows(self.rows)
            self.assertEqual(validation["unique_finite_rows"], 16)
            self.assertEqual(validation["scenario_order"], list(self.scenarios))
            scenarios, contrasts, status, confirmatory, _ = (
                recompute.select_ablation_analysis_contract(self.rows)
            )
        self.assertEqual(scenarios, list(self.scenarios))
        self.assertEqual(contrasts, PLANNED_CONTRASTS)
        self.assertEqual(status, "formal_confirmatory_complete_grid")
        self.assertTrue(confirmatory)

    def test_formal_path_rejects_partial_duplicate_and_nonfinite_rows(self):
        with self.contract_patch():
            with self.assertRaisesRegex(ValueError, "exactly 16 rows"):
                recompute.validate_formal_ablation_rows(self.rows[:-1])

            duplicate = list(self.rows)
            duplicate[-1] = duplicate[0]
            with self.assertRaisesRegex(ValueError, "duplicate formal ablation cell"):
                recompute.validate_formal_ablation_rows(duplicate)

            nonfinite = list(self.rows)
            nonfinite[0] = SimpleNamespace(**vars(nonfinite[0]))
            nonfinite[0].delivery_ratio = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite 'delivery_ratio'"):
                recompute.validate_formal_ablation_rows(nonfinite)

    def test_legacy_full_path_is_explicitly_exploratory(self):
        rows = [
            SimpleNamespace(scenario="z", policy="mappo_full"),
            SimpleNamespace(scenario="z", policy="mappo_no_queue"),
        ]
        scenarios, contrasts, status, confirmatory, validation = (
            recompute.select_ablation_analysis_contract(rows)
        )
        self.assertEqual(scenarios, ["z"])
        self.assertTrue(contrasts)
        self.assertTrue(all(item.family == "legacy_post_hoc" for item in contrasts))
        self.assertEqual(status, "legacy_exploratory_reanalysis")
        self.assertFalse(confirmatory)
        self.assertEqual(validation["label"], "exploratory_only")


class DevelopmentRunnerAndSchemaTests(unittest.TestCase):
    def test_development_cli_cannot_run_full_or_default_to_formal_output(self):
        args = parse_args([])
        self.assertEqual(
            str(args.output).replace("\\", "/"),
            "experiments/archive/ablation-dev",
        )
        self.assertTrue(set(DEVELOPMENT_WORKLOAD_SEEDS).isdisjoint(TEST_WORKLOAD_SEEDS))
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(["--mode", "full"])
        self.assertEqual(raised.exception.code, 2)

    def test_paired_effect_schema_records_seed_level_audit_fields(self):
        scenario = "synthetic"
        contrast = PlannedContrast(
            name="synthetic_remove",
            family="component_removal",
            reference="proposed",
            treatment="no_queue",
            changed_flags=("queue_features",),
        )
        rows = []
        for variant, offset in (("proposed", 0.0), ("no_queue", 0.1)):
            for policy_seed in (7, 42):
                for workload_seed in (31001, 31002):
                    row = episode_row(
                        scenario,
                        f"mappo_{variant}",
                        policy_seed,
                        workload_seed,
                    )
                    for metric in (
                        "delivery_ratio",
                        "drop_rate",
                        "throughput_packets_per_slot",
                        "average_delay_slots",
                        "p95_delay_slots",
                        "mean_queue_packets",
                        "routing_switches",
                        "global_control_overhead_ratio",
                    ):
                        setattr(row, metric, float(getattr(row, metric)) + offset)
                    rows.append(row)

        effects = paired_effects(rows, [scenario], (contrast,))
        self.assertEqual(len(effects), 8)
        delivery = effects[0]
        self.assertEqual(delivery["bootstrap_rng_seed"], 18000)
        self.assertEqual(json.loads(delivery["policy_seed_ids_json"]), [7, 42])
        self.assertEqual(
            len(json.loads(delivery["policy_seed_mean_differences_json"])), 2
        )
        self.assertTrue(delivery["sign_flip_tail_ties_included"])
        self.assertEqual(delivery["seed_level_wilcoxon_requested_method"], "auto")
        self.assertEqual(delivery["seed_level_wilcoxon_zero_method"], "wilcox")
        self.assertNotIn("benjamini_hochberg_p", delivery)

        legacy_contrast = PlannedContrast(
            name="legacy_synthetic",
            family="legacy_post_hoc",
            reference="proposed",
            treatment="no_queue",
            changed_flags=(),
            component_kind="legacy_confounded",
        )
        legacy_effects = paired_effects(rows, [scenario], (legacy_contrast,))
        self.assertTrue(
            all(
                row["multiplicity_role"] == "legacy_exploratory"
                for row in legacy_effects
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
