from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from make_figures import (
    PRIMARY_ABLATION_FAMILY_SIZE,
    PRIMARY_HOLM_FIELD,
    SCENARIO_ORDER,
    fig_ablation,
    load_confirmatory_delivery_effects,
)
from variant_definitions import PLANNED_CONTRASTS


FIELDS = [
    "scenario",
    "contrast",
    "contrast_family",
    "reference_variant",
    "treatment_variant",
    "metric",
    "treatment_minus_reference",
    "difference_ci95_low",
    "difference_ci95_high",
    PRIMARY_HOLM_FIELD,
    "within_metric_family_size",
    "multiplicity_role",
    "global_bh_exploratory_p",
]


def confirmatory_rows():
    rows = []
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        for contrast_index, contrast in enumerate(PLANNED_CONTRASTS):
            effect = (scenario_index - contrast_index) / 1000.0
            rows.append(
                {
                    "scenario": scenario,
                    "contrast": contrast.name,
                    "contrast_family": contrast.family,
                    "reference_variant": contrast.reference,
                    "treatment_variant": contrast.treatment,
                    "metric": "delivery_ratio",
                    "treatment_minus_reference": effect,
                    "difference_ci95_low": effect - 0.002,
                    "difference_ci95_high": effect + 0.002,
                    PRIMARY_HOLM_FIELD: 0.01 if contrast_index == 1 else 0.20,
                    "within_metric_family_size": PRIMARY_ABLATION_FAMILY_SIZE,
                    "multiplicity_role": "primary_confirmatory_family",
                    # Deliberately conflicts with Holm to catch accidental BH use.
                    "global_bh_exploratory_p": 0.99 if contrast_index == 1 else 0.001,
                }
            )
    return rows


def write_rows(path: Path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ConfirmatoryAblationFigureTests(unittest.TestCase):
    def test_loader_uses_frozen_direction_and_primary_holm_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "paired_ablation_effects.csv"
            rows = confirmatory_rows()
            write_rows(csv_path, rows)
            records = load_confirmatory_delivery_effects(csv_path)

        first_scenario = SCENARIO_ORDER[0]
        non_significant = PLANNED_CONTRASTS[0]
        significant = PLANNED_CONTRASTS[1]
        first = records[(first_scenario, non_significant.name)]
        second = records[(first_scenario, significant.name)]
        self.assertAlmostEqual(first["effect_pp"], 0.0)
        self.assertFalse(first["significant"])
        self.assertTrue(second["significant"])

        lifetime_reward = next(
            contrast
            for contrast in PLANNED_CONTRASTS
            if contrast.name == "add_lifetime_reward"
        )
        source = next(
            row
            for row in rows
            if row["scenario"] == first_scenario
            and row["contrast"] == lifetime_reward.name
        )
        self.assertEqual(source["reference_variant"], "with_lifetime_feature")
        self.assertEqual(source["treatment_variant"], "with_lifetime_reward")

    def test_rejects_legacy_full_reference_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "paired_ablation_effects.csv"
            write_rows(
                csv_path,
                [
                    {
                        "scenario": "low_load",
                        "variant": "no_queue",
                        "metric": "delivery_ratio",
                        "ablated_minus_full": -0.01,
                        "benjamini_hochberg_p": 0.01,
                    }
                ],
                fields=[
                    "scenario",
                    "variant",
                    "metric",
                    "ablated_minus_full",
                    "benjamini_hochberg_p",
                ],
            )
            with self.assertRaisesRegex(ValueError, "confirmatory schema"):
                load_confirmatory_delivery_effects(csv_path)

    def test_forest_matrix_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "paired_ablation_effects.csv"
            output = root / "figures"
            output.mkdir()
            write_rows(csv_path, confirmatory_rows())

            fig_ablation(csv_path, output)

            for extension in ("png", "pdf"):
                figure = output / f"fig4_ablation_effect.{extension}"
                self.assertTrue(figure.is_file())
                self.assertGreater(figure.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
