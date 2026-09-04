import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import make_icc_submission_assets as icc


class IccSubmissionAssetsTests(unittest.TestCase):
    def test_experiment_contract_binds_training_and_checkpoint_selection(self):
        contract = icc._require_experiment_contract(
            icc.DEFAULT_EXPERIMENT_MANIFEST
        )
        self.assertEqual(contract["validation_workloads"], 50)
        self.assertEqual(contract["test_workloads"], 50)
        self.assertEqual(contract["checkpoint_count"], 40)
        self.assertEqual(
            contract["checkpoint_provenance_sha256"],
            icc.EXPECTED_CHECKPOINT_PROVENANCE_SHA256,
        )

    def test_experiment_contract_rejects_semantic_drift(self):
        payload = json.loads(
            icc.DEFAULT_EXPERIMENT_MANIFEST.read_text(encoding="utf-8")
        )
        payload["config"]["timesteps"] = 49_999
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment_manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                icc,
                "EXPECTED_EXPERIMENT_MANIFEST_SHA256",
                icc.sha256_file(path),
            ):
                with self.assertRaisesRegex(ValueError, "timesteps changed"):
                    icc._require_experiment_contract(path)

    def test_frozen_qrouting_grid_and_key_results(self):
        rows = icc.load_qrouting_rows()
        self.assertEqual([row["scenario"] for row in rows], list(icc.SCENARIO_ORDER))
        self.assertTrue(all(row["same_direction_seed_fraction"] == 1.0 for row in rows))
        by_scenario = {row["scenario"]: row for row in rows}
        self.assertAlmostEqual(
            by_scenario["medium_load"]["delivery_difference_pp"], 0.3424479167
        )
        self.assertAlmostEqual(
            by_scenario["hotspot_high_load"]["delivery_difference_pp"], -1.8625
        )
        reductions = [float(row["switch_reduction_percent"]) for row in rows]
        self.assertGreaterEqual(min(reductions), 25.0)
        self.assertLessEqual(max(reductions), 59.0)
        self.assertEqual(
            [float(row["delivery_holm_p"]) for row in rows],
            [0.375, 0.0390625, 0.0390625, 0.0390625, 0.4921875],
        )
        self.assertTrue(all(float(row["switch_ci_high"]) < 0.0 for row in rows))
        self.assertTrue(
            all(float(row["switch_holm_p"]) == 0.0390625 for row in rows)
        )

    def test_latex_table_is_explicitly_retrospective(self):
        rows = icc.load_qrouting_rows()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "table.tex"
            icc.write_main_table(rows, output)
            text = output.read_text(encoding="utf-8")
        self.assertIn("Hotspot", text)
        self.assertIn("-1.862500", text)
        self.assertIn("[-2.104025, -1.641988]", text)
        self.assertIn("-47.4400 [-66.9434, -28.2348]", text)
        self.assertIn("0.0078125 / 0.0390625", text)
        self.assertIn("Raw exact / retro. Holm", text)
        self.assertIn(r"$\Delta=$ MAPPO $-$ Q-routing", text)
        self.assertIn("a negative switch", text)
        self.assertIn(r"reduction is $-\Delta$ divided by the Q-routing mean", text)
        self.assertIn("Parentheses report policy seeds with fewer switches", text)
        self.assertIn("first average the 50 paired workloads", text)
        self.assertIn("two-sided policy-seed sign-flip", text)
        self.assertIn("5,000-resample crossed pigeonhole bootstrap", text)
        self.assertIn("applied separately by endpoint", text)
        self.assertIn("network-wide accepted changes", text)
        self.assertIn("Retrospective seed-level reanalysis", text)
        self.assertIn("Requires graphicx and booktabs", text)

    def test_holm_adjustment_preserves_order_and_monotonicity(self):
        adjusted = icc.holm_adjust([0.1875, 0.0078125, 0.4921875, 0.0078125])
        self.assertEqual(adjusted, [0.375, 0.03125, 0.4921875, 0.03125])

    def test_effect_figure_writes_both_publication_formats(self):
        rows = icc.load_qrouting_rows()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "effects"
            icc.write_effect_figure(rows, output)
            self.assertGreater(output.with_suffix(".png").stat().st_size, 10_000)
            self.assertGreater(output.with_suffix(".pdf").stat().st_size, 1_000)
            pdf = output.with_suffix(".pdf").read_bytes()
            self.assertNotIn(b"/Type3", pdf)
            self.assertIn(b"/CIDFontType2", pdf)


if __name__ == "__main__":
    unittest.main()
