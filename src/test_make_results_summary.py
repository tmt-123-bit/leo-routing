from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ablation_matrix_runner import POLICY_SEED_NAMESPACE
from make_results_summary import (
    ALL_REPOSITORY_SOURCE_FILES,
    ANALYSIS_SOURCE_FILES,
    CONFIGURATIONS,
    CONTRAST_VARIANTS,
    ENDPOINTS,
    EXPECTED_EFFECT_ROWS,
    LEGACY_ABLATION_CONTRASTS,
    LEGACY_BASELINES,
    LEGACY_METRICS,
    MATRIX_NAME,
    PLANNED_CONTRASTS,
    POLICY_SEEDS,
    SCENARIOS,
    TEST_WORKLOAD_SEEDS,
    TRAINING_SOURCE_FILES,
    build_summary,
    inspect_confirmatory_ablation,
    inspect_legacy_headline,
    sha256_json,
)


class ResultsSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @property
    def ablation(self) -> Path:
        return self.project / "experiments" / "ablation-50k-v2"

    def write_json(self, name: str, value: dict) -> None:
        self.ablation.mkdir(parents=True, exist_ok=True)
        (self.ablation / name).write_text(
            json.dumps(value), encoding="utf-8"
        )

    def write_reanalysis_manifest(
        self,
        *,
        source: Path,
        output_root: Path,
        output_name: str,
        output_rows: list[dict],
        kind: str,
        inferential_status: str,
    ) -> None:
        output_path = output_root / output_name
        with source.open("r", encoding="utf-8", newline="") as handle:
            source_row_count = sum(1 for _ in csv.DictReader(handle))
        manifest = {
            "analysis": "crossed_seed_statistical_reanalysis",
            "kind": kind,
            "inferential_status": inferential_status,
            "confirmatory": False,
            "source_episode_metrics": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_row_count": source_row_count,
            "output_files": {"comparisons": output_name},
            "output_artifacts": {
                "comparisons": {
                    "file": output_name,
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                    "row_count": len(output_rows),
                }
            },
            "statistical_analysis": {
                "hypothesis_test": {
                    "method": "exact_policy_seed_sign_flip_on_workload_means",
                    "episode_rows_treated_as_independent": False,
                }
            },
        }
        (output_root / "statistical_analysis_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def write_headline_reanalysis_fixture(self, name: str) -> None:
        if name == "eval-main":
            source = self.project / "experiments" / "eval-main" / (
                "episode_metrics.csv"
            )
        elif name == "repro-check":
            source = self.project / "experiments" / "archive" / (
                "repro-check"
            ) / "episode_metrics.csv"
        else:
            raise ValueError(name)
        source.parent.mkdir(parents=True, exist_ok=True)
        policy_seeds = (7, 42, 123, 314, 456, 789, 1024, 2024)
        source_policies = {
            "mappo": policy_seeds,
            "q_routing": policy_seeds,
            "delay_only": (0,),
            "full_heuristic": (0,),
            "global_dijkstra": (0,),
            "ospf_ecmp": (0,),
            "random": (0,),
        }
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "scenario",
                    "policy",
                    "policy_seed",
                    "workload_seed",
                    "delivery_ratio",
                ),
            )
            writer.writeheader()
            for scenario in SCENARIOS:
                for policy, seeds in source_policies.items():
                    for policy_seed in seeds:
                        for workload_seed in range(11001, 11051):
                            writer.writerow(
                                {
                                    "scenario": scenario,
                                    "policy": policy,
                                    "policy_seed": policy_seed,
                                    "workload_seed": workload_seed,
                                    "delivery_ratio": 0.5,
                                }
                            )

        q_delivery = {
            "low_load": (0.9128030303, 0.9121969697, 0.0006060606, -0.0014393939, 0.0028039773, 0.75, 0.1875),
            "medium_load": (0.7879817708, 0.7845572917, 0.0034244792, 0.0004293620, 0.0066666667, 1.0, 0.0078125),
            "hotspot_high_load": (0.28993, 0.308555, -0.018625000000000006, -0.02104025, -0.016419875, 0.0, 0.0078125),
            "frequent_break": (0.7585677083, 0.7520963542, 0.0064713542, 0.0028385417, 0.0104039714, 1.0, 0.0078125),
            "fault_links": (0.763359375, 0.7621354167, 0.0012239583, -0.0030598958, 0.0056643880, 0.5, 0.4921875),
        }
        q_switches = {
            "low_load": (0.9325, 2.22, -1.2875),
            "medium_load": (31.0325, 59.0875, -28.055),
            "hotspot_high_load": (138.8925, 186.3325, -47.44),
            "frequent_break": (54.7875, 76.9925, -22.205),
            "fault_links": (44.5725, 65.895, -21.3225),
        }
        dijkstra_gaps = {
            "eval-main": {
                "low_load": 0.0021969697,
                "medium_load": 0.0530859375,
                "hotspot_high_load": -0.00899,
                "frequent_break": 0.051484375,
                "fault_links": 0.0488802083,
            },
            "repro-check": {
                "low_load": -0.0045075758,
                "medium_load": 0.01,
                "hotspot_high_load": -0.02012,
                "frequent_break": 0.0033984375,
                "fault_links": 0.0084765625,
            },
        }
        archived_p = {
            "low_load": 0.0078125,
            "medium_load": 0.015625,
            "hotspot_high_load": 0.0078125,
            "frequent_break": 0.15625,
            "fault_links": 0.015625,
        }
        output_rows: list[dict] = []
        for scenario in SCENARIOS:
            scenario_mappo = q_delivery[scenario][0]
            for baseline in LEGACY_BASELINES:
                for metric in LEGACY_METRICS:
                    mappo_mean = 0.8
                    baseline_mean = 0.6
                    difference = 0.2
                    low = 0.1
                    high = 0.3
                    positive_fraction = 1.0
                    raw_p = 0.0078125
                    if metric == "delivery_ratio":
                        mappo_mean = scenario_mappo
                        if baseline == "q_routing":
                            (
                                mappo_mean,
                                baseline_mean,
                                difference,
                                low,
                                high,
                                positive_fraction,
                                raw_p,
                            ) = q_delivery[scenario]
                        elif baseline == "global_dijkstra":
                            difference = dijkstra_gaps[name][scenario]
                            baseline_mean = mappo_mean - difference
                            low = difference - 0.002
                            high = difference + 0.002
                            raw_p = (
                                archived_p[scenario]
                                if name == "repro-check"
                                else 0.0078125
                            )
                        else:
                            baseline_mean = mappo_mean - 0.1
                            difference = 0.1
                            low = 0.08
                            high = 0.12
                    elif baseline == "q_routing" and metric == "routing_switches":
                        mappo_mean, baseline_mean, difference = q_switches[scenario]
                        low = difference - 1.0
                        high = difference + 1.0
                        positive_fraction = 0.0
                    output_rows.append(
                        {
                            "scenario": scenario,
                            "baseline": baseline,
                            "metric": metric,
                            "paired_policy_seeds": 8,
                            "paired_workloads": 50,
                            "paired_episode_cells": 400,
                            "mappo_mean": mappo_mean,
                            "baseline_mean": baseline_mean,
                            "mean_difference": difference,
                            "difference_ci95_low": low,
                            "difference_ci95_high": high,
                            "difference_ci_method": (
                                "crossed_pigeonhole_bootstrap_percentile"
                            ),
                            "ci_resamples": 5000,
                            "positive_policy_seed_fraction": positive_fraction,
                            "raw_p_value": raw_p,
                            "p_value_method": (
                                "exact_policy_seed_sign_flip_on_workload_means"
                            ),
                            "sign_flip_permutations": 256,
                        }
                    )

        output_root = self.project / "experiments" / "legacy-reanalysis" / name
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / "paired_tests.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        self.write_reanalysis_manifest(
            source=source,
            output_root=output_root,
            output_name="paired_tests.csv",
            output_rows=output_rows,
            kind="headline",
            inferential_status="headline_reanalysis",
        )

    def write_ablation_reanalysis_fixture(self) -> None:
        source = self.project / "experiments" / "ablation" / (
            "episode_metrics.csv"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        policies = (
            "mappo_flat_critic",
            "mappo_full",
            "mappo_no_credit",
            "mappo_no_lifetime",
            "mappo_no_packet_context",
            "mappo_no_ppo_protection",
            "mappo_no_queue",
        )
        policy_seeds = (7, 42, 123, 314, 456, 789, 1024, 2024)
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "scenario",
                    "policy",
                    "policy_seed",
                    "workload_seed",
                    "delivery_ratio",
                ),
            )
            writer.writeheader()
            for scenario in SCENARIOS:
                for policy in policies:
                    for policy_seed in policy_seeds:
                        for workload_seed in range(13001, 13051):
                            writer.writerow(
                                {
                                    "scenario": scenario,
                                    "policy": policy,
                                    "policy_seed": policy_seed,
                                    "workload_seed": workload_seed,
                                    "delivery_ratio": 0.5,
                                }
                            )

        output_rows: list[dict] = []
        for scenario in SCENARIOS:
            for contrast in LEGACY_ABLATION_CONTRASTS:
                treatment = contrast.removeprefix("legacy_").removesuffix(
                    "_vs_full"
                )
                for metric in ENDPOINTS:
                    output_rows.append(
                        {
                            "scenario": scenario,
                            "contrast": contrast,
                            "reference_variant": "full",
                            "treatment_variant": treatment,
                            "metric": metric,
                            "paired_policy_seeds": 8,
                            "paired_workloads": 50,
                            "paired_episode_cells": 400,
                            "reference_mean": 0.5,
                            "treatment_mean": 0.58,
                            "treatment_minus_reference": 0.08,
                            "difference_ci95_low": 0.05,
                            "difference_ci95_high": 0.11,
                            "difference_ci_method": (
                                "crossed_pigeonhole_bootstrap_percentile"
                            ),
                            "ci_resamples": 5000,
                            "positive_policy_seed_fraction": 1.0,
                            "raw_p_value": 0.0078125,
                            "p_value_method": (
                                "exact_policy_seed_sign_flip_on_workload_means"
                            ),
                            "sign_flip_permutations": 256,
                            "confirmatory_holm_within_metric_p": 0.234375,
                            "within_metric_family_size": 30,
                        }
                    )
        output_root = self.project / "experiments" / "legacy-reanalysis" / (
            "ablation-5k"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / "paired_ablation_effects.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
        self.write_reanalysis_manifest(
            source=source,
            output_root=output_root,
            output_name="paired_ablation_effects.csv",
            output_rows=output_rows,
            kind="ablation",
            inferential_status="legacy_exploratory_reanalysis",
        )

    def write_legacy_provenance_fixtures(self) -> None:
        policy_seeds = (7, 42, 1024, 123, 456, 789, 2024, 314)
        common_checkpoints = {
            f"{scenario}/seed_{policy_seed}": (
                "F:/fixture/train-main/checkpoints/"
                f"{scenario}/no_lifetime/seed_{policy_seed}/validation_best.pt"
            )
            for scenario in SCENARIOS
            for policy_seed in policy_seeds
        }
        reproduction_checkpoints = {
            f"{scenario}/seed_{policy_seed}": (
                "F:/fixture/repro-check/checkpoints/"
                f"{scenario}/seed_{policy_seed}/validation_best.pt"
            )
            for scenario in SCENARIOS
            for policy_seed in policy_seeds
        }
        old_fingerprint = {"wrapper": "old-wrapper", "evaluation": "old-eval"}
        manifests = {
            "eval-main": {
                "checkpoints": common_checkpoints,
                "code_fingerprint": {
                    "wrapper": "corrected-wrapper",
                    "evaluation": "corrected-eval",
                },
            },
            "train-main": {
                "checkpoints": common_checkpoints,
                "code_fingerprint": old_fingerprint,
            },
            "repro-check": {
                "checkpoints": reproduction_checkpoints,
                "code_fingerprint": old_fingerprint,
            },
        }
        locations = {
            "eval-main": self.project / "experiments" / "eval-main",
            "train-main": self.project / "experiments" / "train-main",
            "repro-check": self.project / "experiments" / "archive" / (
                "repro-check"
            ),
        }
        for name, value in manifests.items():
            value.update(
                {
                    "policy_seeds": list(policy_seeds),
                    "config": {"timesteps": 50000, "test_episodes": 50},
                }
            )
            locations[name].mkdir(parents=True, exist_ok=True)
            (locations[name] / "experiment_manifest.json").write_text(
                json.dumps(value), encoding="utf-8"
            )

    def write_complete_fixture(self) -> None:
        source_root = self.project / "src"
        source_root.mkdir(parents=True, exist_ok=True)
        for relative_path in {
            *ALL_REPOSITORY_SOURCE_FILES.values(),
        }:
            (source_root / relative_path).write_text(
                f"frozen source fixture: {relative_path}\n", encoding="utf-8"
            )
        training_code_hashes = {
            name: hashlib.sha256(
                (source_root / relative_path).read_bytes()
            ).hexdigest()
            for name, relative_path in TRAINING_SOURCE_FILES.items()
        }
        training_code_hashes["trainer"] = training_code_hashes[
            "trainer_snapshot"
        ]
        all_code_hashes = {
            name: hashlib.sha256(
                (source_root / relative_path).read_bytes()
            ).hexdigest()
            for name, relative_path in ALL_REPOSITORY_SOURCE_FILES.items()
        }
        all_code_hashes["trainer"] = all_code_hashes["trainer_snapshot"]
        analysis_code_audit = {
            name: {
                "sha256": hashlib.sha256(
                    (source_root / relative_path).read_bytes()
                ).hexdigest()
            }
            for name, relative_path in ANALYSIS_SOURCE_FILES.items()
        }
        jobs = [
            {
                "index": index,
                "scenario": scenario,
                "variant": configuration,
                "policy_seed": policy_seed,
            }
            for index, (scenario, configuration, policy_seed) in enumerate(
                (
                    (scenario, configuration, policy_seed)
                    for scenario in SCENARIOS
                    for configuration in CONFIGURATIONS
                    for policy_seed in POLICY_SEEDS
                )
            )
        ]
        spec = {
            "matrix_name": MATRIX_NAME,
            "scenarios": list(SCENARIOS),
            "variants": list(CONFIGURATIONS),
            "policy_seeds": list(POLICY_SEEDS),
            "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
            "expected_training_jobs": 540,
            "expected_evaluation_rows": 27000,
            "full_budget_steps": 50000,
            "planned_contrasts": [
                {"name": contrast} for contrast in PLANNED_CONTRASTS
            ],
            "jobs": jobs,
            "config": {
                "timesteps": 50000,
                "validation_episodes": 50,
                "test_episodes": 50,
                "eval_every_rollouts": 40,
                "save_every_steps": 5000,
                "batch_size": 4,
            },
        }
        spec["spec_sha256"] = sha256_json(spec)
        spec_hash = spec["spec_sha256"]
        self.write_json("matrix_spec.json", spec)
        protocol_path = self.project / "docs" / "ABLATION_50K_PROTOCOL.md"
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text("frozen protocol fixture\n", encoding="utf-8")
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        all_code_hashes["protocol_document"] = protocol_hash
        historical_audit_content = {
            "selected_policy_seeds": list(POLICY_SEEDS),
            "selected_workload_seeds": list(TEST_WORKLOAD_SEEDS),
            "selected_prior_overlap": [],
            "selected_workload_seed_prior_overlap": [],
            "selected_policy_seed_prior_overlap": [],
            "scanned_csv_files": [],
            "historical_workload_seed_ranges": [],
            "historical_policy_seeds": [],
        }
        historical_audit_content_hash = sha256_json(historical_audit_content)
        self.write_json(
            "historical_workload_seed_audit.json",
            {
                "status": "passed_no_selected_workload_seed_overlap",
                "content_sha256": historical_audit_content_hash,
                "content": historical_audit_content,
            },
        )
        historical_audit_hash = hashlib.sha256(
            (self.ablation / "historical_workload_seed_audit.json").read_bytes()
        ).hexdigest()
        freeze_content = {
            "matrix_name": MATRIX_NAME,
            "protocol_spec_sha256": spec_hash,
            "expected_training_jobs": 540,
            "training_code_sha256": training_code_hashes,
            "all_code_sha256": all_code_hashes,
            "protocol_document_sha256": protocol_hash,
            "historical_workload_seed_audit_sha256": historical_audit_hash,
            "historical_workload_seed_audit_content_sha256": (
                historical_audit_content_hash
            ),
            "checkpoints": [
                {
                    "scenario": job["scenario"],
                    "variant": job["variant"],
                    "policy_seed": job["policy_seed"],
                    "training_fingerprint": "b" * 64,
                    "checkpoint_sha256": "c" * 64,
                    "selected_checkpoint_step": 40000,
                    "earliest_best_validation_step": 40000,
                    "selected_validation_score": [0.8, 1.0, -0.1, -4.0],
                    "actual_environment_steps": 50000,
                    "run_manifest_sha256": "d" * 64,
                    "run_config_sha256": "e" * 64,
                    "training_metrics_sha256": "f" * 64,
                    "trainer_log_sha256": "1" * 64,
                    "training_code_fingerprint_sha256": "2" * 64,
                }
                for job in jobs
            ],
        }
        training_freeze_hash = sha256_json(freeze_content)
        self.write_json(
            "training_freeze_manifest.json",
            {
                "status": "frozen_before_test_evaluation",
                "content_sha256": training_freeze_hash,
                "content": freeze_content,
            },
        )
        self.write_json(
            "matrix_audit.json",
            {
                "matrix_name": MATRIX_NAME,
                "spec_sha256": spec_hash,
                "training_freeze_sha256": training_freeze_hash,
                "matrix_complete": True,
                "expected_training_jobs": 540,
                "expected_evaluation_jobs": 540,
                "expected_evaluation_rows": 27000,
                "validated_evaluation_rows": 27000,
                "counts": {
                    "training": {"completed": 540},
                    "evaluation": {"completed": 540},
                },
            },
        )
        episode_path = self.ablation / "episode_metrics.csv"
        episode_path.write_text("frozen synthetic fixture\n", encoding="utf-8")
        episode_hash = hashlib.sha256(episode_path.read_bytes()).hexdigest()
        rows = []
        for scenario in SCENARIOS:
            for contrast in PLANNED_CONTRASTS:
                reference, treatment = CONTRAST_VARIANTS[contrast]
                for endpoint in ENDPOINTS:
                    rows.append(
                        {
                            "scenario": scenario,
                            "contrast": contrast,
                            "reference_variant": reference,
                            "treatment_variant": treatment,
                            "metric": endpoint,
                            "paired_policy_seeds": 12,
                            "paired_workloads": 50,
                            "paired_episode_cells": 600,
                            "treatment_minus_reference": 0.01,
                            "difference_ci95_low": 0.005,
                            "difference_ci95_high": 0.015,
                            "difference_ci_method": (
                                "crossed_pigeonhole_bootstrap_percentile"
                            ),
                            "ci_resamples": 5000,
                            "raw_p_value": 0.00048828125,
                            "p_value_method": (
                                "exact_policy_seed_sign_flip_on_workload_means"
                            ),
                            "sign_flip_permutations": 4096,
                            "confirmatory_holm_within_metric_p": 0.01953125,
                            "within_metric_family_size": 40,
                        }
                    )
        path = self.ablation / "paired_ablation_effects.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        effect_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_json(
            "episode_metrics_manifest.json",
            {
                "complete": True,
                "included_job_count": 540,
                "expected_job_count": 540,
                "row_count": 27000,
                "expected_row_count": 27000,
                "spec_sha256": spec_hash,
                "training_freeze_sha256": training_freeze_hash,
                "csv_sha256": episode_hash,
            },
        )
        self.write_json(
            "statistical_analysis_manifest.json",
            {
                "matrix_name": MATRIX_NAME,
                "status": "completed",
                "protocol_spec_sha256": spec_hash,
                "training_freeze_sha256": training_freeze_hash,
                "source_episode_rows": 27000,
                "source_episode_csv_sha256": episode_hash,
                "paired_effect_rows": EXPECTED_EFFECT_ROWS,
                "expected_paired_effect_rows": EXPECTED_EFFECT_ROWS,
                "paired_effects_csv_sha256": effect_hash,
                "policy_seed_count": 12,
                "workload_seed_count": 50,
                "planned_contrasts": [
                    {"name": contrast} for contrast in PLANNED_CONTRASTS
                ],
                "analysis_code_audit": analysis_code_audit,
            },
        )

    def test_completed_legacy_results_are_primary_and_audited(self):
        self.write_headline_reanalysis_fixture("eval-main")
        self.write_headline_reanalysis_fixture("repro-check")
        self.write_ablation_reanalysis_fixture()
        self.write_legacy_provenance_fixtures()

        status = inspect_legacy_headline(self.project, "eval-main")
        self.assertTrue(status.complete, status.issues)
        self.assertEqual(status.source_rows, 5250)
        self.assertEqual(status.output_rows, 375)

        summary = build_summary(self.project)
        self.assertLess(
            summary.index("Corrected completed 50k evaluation"),
            summary.index("Optional paused 50k ablation"),
        )
        self.assertIn("q_routing 0.7846", summary)
        self.assertIn("+0.342 [+0.043, +0.667]", summary)
        self.assertIn("-1.863 [-2.104, -1.642]", summary)
        self.assertIn("retrospective five-scenario Holm p is 0.0390625", summary)
        self.assertIn("archived run gap vs Dijkstra", summary)
        self.assertIn("+0.34", summary)
        self.assertIn("-0.45", summary)
        self.assertIn("Legacy 5k ablation (exploratory only)", summary)
        self.assertIn("cannot validate the final method configuration", summary)
        self.assertNotIn("p<", summary)
        self.assertNotIn("1e-9", summary)

    def test_tampered_legacy_output_is_hidden(self):
        self.write_headline_reanalysis_fixture("eval-main")
        output_path = self.project / "experiments" / "legacy-reanalysis" / (
            "eval-main"
        ) / "paired_tests.csv"
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        status = inspect_legacy_headline(self.project, "eval-main")
        self.assertFalse(status.complete)
        self.assertEqual(status.rows, ())
        self.assertTrue(any("SHA-256" in issue for issue in status.issues))

        summary = build_summary(self.project)
        self.assertIn("failed its hash/row-count audit", summary)
        self.assertNotIn("q_routing 0.7846", summary)

    def test_protocol_document_id_matches_machine_namespace(self):
        protocol_id = POLICY_SEED_NAMESPACE.removesuffix("-policy-seed-")
        self.assertEqual(protocol_id, "ABLATION-50K-v2.2")
        protocol_path = Path(__file__).resolve().parents[1] / "docs" / (
            "ABLATION_50K_PROTOCOL.md"
        )
        protocol = protocol_path.read_text(encoding="utf-8")
        self.assertIn(f"**Protocol ID:** `{protocol_id}`", protocol)
        self.assertIn(f'"protocol_id": "{protocol_id}"', protocol)

    def test_missing_matrix_is_explicitly_nonconfirmatory(self):
        status = inspect_confirmatory_ablation(self.project)
        self.assertFalse(status.complete)
        self.assertEqual(status.effects, ())

        summary = build_summary(self.project)
        self.assertIn("no confirmatory ablation estimate is reportable", summary)
        self.assertIn("5,000 training steps", summary)
        self.assertIn("legacy pilot", summary)
        self.assertNotIn("complete and audit-validated", summary)
        self.assertNotIn("1.2" + "e-9", summary)

    def test_complete_frozen_grid_is_reported(self):
        self.write_complete_fixture()
        status = inspect_confirmatory_ablation(self.project)
        self.assertTrue(status.complete, status.issues)
        self.assertEqual(len(status.effects), EXPECTED_EFFECT_ROWS)

        summary = build_summary(self.project)
        self.assertIn("complete and audit-validated", summary)
        self.assertIn("remove_ppo_protection_package", summary)
        self.assertIn("proposed -> no_queue", summary)
        self.assertIn("+1.00 [+0.50, +1.50]", summary)
        self.assertIn("0.00048828125", summary)
        self.assertIn("0.01953125", summary)

    def test_stale_effects_are_hidden_when_audit_is_incomplete(self):
        self.write_complete_fixture()
        audit = json.loads(
            (self.ablation / "matrix_audit.json").read_text(encoding="utf-8")
        )
        audit["matrix_complete"] = False
        audit["counts"]["evaluation"]["completed"] = 539
        audit["validated_evaluation_rows"] = 26950
        self.write_json("matrix_audit.json", audit)

        status = inspect_confirmatory_ablation(self.project)
        self.assertFalse(status.complete)
        self.assertEqual(status.effects, ())
        summary = build_summary(self.project)
        self.assertNotIn("remove_ppo_protection_package | proposed", summary)
        self.assertIn("| completed evaluations | 539 | 540 |", summary)

    def test_hash_mismatch_hides_a_stale_effect_table(self):
        self.write_complete_fixture()
        with (self.ablation / "paired_ablation_effects.csv").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("\n")

        status = inspect_confirmatory_ablation(self.project)
        self.assertFalse(status.complete)
        self.assertEqual(status.effects, ())
        self.assertTrue(any("SHA-256" in issue for issue in status.issues))

    def test_downstream_manifests_must_bind_to_training_freeze(self):
        for name in (
            "matrix_audit.json",
            "episode_metrics_manifest.json",
            "statistical_analysis_manifest.json",
        ):
            with self.subTest(manifest=name):
                self.write_complete_fixture()
                path = self.ablation / name
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["training_freeze_sha256"] = "0" * 64
                self.write_json(name, manifest)

                status = inspect_confirmatory_ablation(self.project)
                self.assertFalse(status.complete)
                self.assertEqual(status.effects, ())
                self.assertTrue(
                    any("training freeze" in issue for issue in status.issues)
                )

    def test_wrong_contrast_direction_is_rejected_even_with_matching_hash(self):
        self.write_complete_fixture()
        effect_path = self.ablation / "paired_ablation_effects.csv"
        with effect_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["reference_variant"] = "with_hard_lifetime_mask"
        with effect_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = self.ablation / "statistical_analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["paired_effects_csv_sha256"] = hashlib.sha256(
            effect_path.read_bytes()
        ).hexdigest()
        self.write_json("statistical_analysis_manifest.json", manifest)

        status = inspect_confirmatory_ablation(self.project)
        self.assertFalse(status.complete)
        self.assertTrue(
            any("reference/treatment direction" in issue for issue in status.issues)
        )

    def test_changed_training_source_invalidates_the_frozen_summary(self):
        self.write_complete_fixture()
        with (self.project / "src" / "leo_multiagent_env.py").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write("# changed after freeze\n")

        status = inspect_confirmatory_ablation(self.project)
        self.assertFalse(status.complete)
        self.assertTrue(
            any("frozen code hashes" in issue for issue in status.issues)
        )


if __name__ == "__main__":
    unittest.main()
