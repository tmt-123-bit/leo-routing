"""Build ICC paper assets from the frozen seed-level reanalysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISONS = (
    PROJECT_ROOT
    / "experiments"
    / "legacy-reanalysis"
    / "eval-main"
    / "paired_tests.csv"
)
DEFAULT_MANIFEST = DEFAULT_COMPARISONS.with_name("statistical_analysis_manifest.json")
DEFAULT_EPISODES = PROJECT_ROOT / "experiments" / "eval-main" / "episode_metrics.csv"
DEFAULT_EXPERIMENT_MANIFEST = (
    PROJECT_ROOT / "experiments" / "eval-main" / "experiment_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "figures"

EXPECTED_COMPARISONS_SHA256 = (
    "f225f9d7e5db0125ebdd230e223ad943370a9ad7f8b84840db1846ad4e143178"
)
EXPECTED_MANIFEST_SHA256 = (
    "82f51eedb81786b11e01aa355a3f704fbb986259b346cd81a886eb7fc51c8fb4"
)
EXPECTED_SOURCE_SHA256 = (
    "ef48c69602d772c17c23b1f4e3e1c7000658d6dc1b115f9491c61385cb3bea84"
)
EXPECTED_EXPERIMENT_MANIFEST_SHA256 = (
    "3d4d1b8b31555c60036a0fc3f2b1f8d6d38e368c6561c2cc2cd8e14cd38a6a8f"
)
EXPECTED_CHECKPOINT_PROVENANCE_SHA256 = (
    "0084b875ded2bffb8df2f3e5cc132fb655a204346c3e9780eec3e043b7d1ca2a"
)

SCENARIO_ORDER = (
    "low_load",
    "medium_load",
    "hotspot_high_load",
    "frequent_break",
    "fault_links",
)
SCENARIO_LABELS = {
    "low_load": "Low load",
    "medium_load": "Medium load",
    "hotspot_high_load": "Hotspot",
    "frequent_break": "Frequent break",
    "fault_links": "Fault links",
}
POLICY_SEEDS = (7, 42, 1024, 123, 456, 789, 2024, 314)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    if not p_values:
        return []
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("Holm adjustment requires p-values in [0, 1]")
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    family_size = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_experiment_contract(path: Path) -> dict[str, Any]:
    """Freeze the training budget, split, and selected-checkpoint provenance."""
    path = Path(path)
    if sha256_file(path) != EXPECTED_EXPERIMENT_MANIFEST_SHA256:
        raise ValueError("ICC experiment manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "EXP-004" or manifest.get("mode") != "full":
        raise ValueError("ICC experiment identity must remain EXP-004/full")
    config = manifest.get("config", {})
    expected_config = {
        "scenarios": list(SCENARIO_ORDER),
        "timesteps": 50_000,
        "validation_episodes": 50,
        "test_episodes": 50,
        "q_routing_train_episodes": 500,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"ICC experiment {key} changed: expected {expected!r}, "
                f"observed {config.get(key)!r}"
            )
    if manifest.get("policy_seeds") != list(POLICY_SEEDS):
        raise ValueError("ICC experiment must retain the eight frozen policy seeds")

    validation_bounds = manifest.get("validation_workload_seeds")
    test_bounds = manifest.get("test_workload_seeds")
    if validation_bounds != [10001, 10050] or test_bounds != [11001, 11050]:
        raise ValueError("ICC validation/test workload ranges changed")
    validation_workloads = set(range(validation_bounds[0], validation_bounds[1] + 1))
    test_workloads = set(range(test_bounds[0], test_bounds[1] + 1))
    if (
        len(validation_workloads) != 50
        or len(test_workloads) != 50
        or validation_workloads & test_workloads
    ):
        raise ValueError("ICC validation/test workloads must be disjoint 50-seed sets")

    checkpoints = manifest.get("checkpoints")
    expected_keys = {
        f"{scenario}/seed_{seed}"
        for scenario in SCENARIO_ORDER
        for seed in POLICY_SEEDS
    }
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_keys:
        raise ValueError("ICC experiment must bind the exact 5 x 8 checkpoint grid")

    provenance = []
    for key in sorted(expected_keys):
        checkpoint = Path(str(checkpoints[key]))
        if checkpoint.name != "validation_best.pt" or not checkpoint.is_file():
            raise ValueError(f"ICC selected checkpoint is missing or invalid: {key}")
        run_manifest_path = checkpoint.with_name("run_manifest.json")
        if not run_manifest_path.is_file():
            raise ValueError(f"ICC checkpoint selection manifest is missing: {key}")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if Path(str(run_manifest.get("validation_best_checkpoint", ""))).resolve() != (
            checkpoint.resolve()
        ):
            raise ValueError(f"ICC run manifest selects a different checkpoint: {key}")
        if run_manifest.get("run_name") != checkpoint.parent.name:
            raise ValueError(f"ICC checkpoint run identity changed: {key}")
        if int(run_manifest.get("environment_steps", -1)) != 50_040:
            raise ValueError(f"ICC checkpoint training-step record changed: {key}")
        score = run_manifest.get("best_validation_score")
        if not isinstance(score, list) or len(score) != 4:
            raise ValueError(f"ICC checkpoint validation-selection score changed: {key}")
        provenance.append(
            {
                "key": key,
                "checkpoint_sha256": sha256_file(checkpoint),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "environment_steps": run_manifest["environment_steps"],
                "best_validation_score": score,
            }
        )
    provenance_sha256 = canonical_sha256(provenance)
    if provenance_sha256 != EXPECTED_CHECKPOINT_PROVENANCE_SHA256:
        raise ValueError(
            "ICC checkpoint provenance SHA-256 mismatch: "
            f"observed {provenance_sha256}"
        )
    return {
        "manifest": manifest,
        "validation_workloads": len(validation_workloads),
        "test_workloads": len(test_workloads),
        "checkpoint_count": len(provenance),
        "checkpoint_provenance_sha256": provenance_sha256,
    }


def _require_frozen_sources(
    comparisons: Path,
    manifest_path: Path,
    experiment_manifest_path: Path,
) -> dict[str, Any]:
    if sha256_file(comparisons) != EXPECTED_COMPARISONS_SHA256:
        raise ValueError("ICC comparison source SHA-256 mismatch")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("ICC statistical manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("confirmatory") is not False:
        raise ValueError("ICC source must remain retrospective, not confirmatory")
    if manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("ICC manifest binds an unexpected episode source")
    if int(manifest.get("source_row_count", -1)) != 5250:
        raise ValueError("ICC episode source row count must be 5250")
    episode_source = Path(str(manifest.get("source_episode_metrics", "")))
    if not episode_source.is_file():
        episode_source = DEFAULT_EPISODES
    if sha256_file(episode_source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("ICC episode source SHA-256 mismatch")
    with episode_source.open("r", encoding="utf-8-sig", newline="") as handle:
        episode_rows = sum(1 for _ in handle) - 1
    if episode_rows != 5250:
        raise ValueError("ICC episode source must contain 5250 data rows")
    artifact = manifest.get("output_artifacts", {}).get("comparisons", {})
    if artifact.get("sha256") != EXPECTED_COMPARISONS_SHA256:
        raise ValueError("ICC manifest does not bind the comparison CSV")
    if int(artifact.get("row_count", -1)) != 375:
        raise ValueError("ICC comparison row count must be 375")
    return {
        "statistics": manifest,
        "experiment": _require_experiment_contract(experiment_manifest_path),
    }


def load_qrouting_rows(
    comparisons: Path = DEFAULT_COMPARISONS,
    manifest_path: Path = DEFAULT_MANIFEST,
    experiment_manifest_path: Path = DEFAULT_EXPERIMENT_MANIFEST,
) -> list[dict[str, float | str]]:
    _require_frozen_sources(comparisons, manifest_path, experiment_manifest_path)
    selected: dict[tuple[str, str], dict[str, str]] = {}
    delivery_baselines: dict[str, dict[str, float]] = {
        scenario: {} for scenario in SCENARIO_ORDER
    }
    with comparisons.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["metric"])
            if (
                row["scenario"] in delivery_baselines
                and row["metric"] == "delivery_ratio"
            ):
                delivery_baselines[row["scenario"]][row["baseline"]] = float(
                    row["baseline_mean"]
                )
            if row["baseline"] == "q_routing" and row["metric"] in {
                "delivery_ratio",
                "routing_switches",
            }:
                if key in selected:
                    raise ValueError(f"duplicate ICC comparison row: {key}")
                if row["paired_policy_seeds"] != "8" or row["paired_workloads"] != "50":
                    raise ValueError(f"unexpected ICC replication contract: {key}")
                if row["p_value_method"] != (
                    "exact_policy_seed_sign_flip_on_workload_means"
                ):
                    raise ValueError(f"unexpected ICC test method: {key}")
                selected[key] = row

    expected = {
        (scenario, metric)
        for scenario in SCENARIO_ORDER
        for metric in ("delivery_ratio", "routing_switches")
    }
    if set(selected) != expected:
        raise ValueError("ICC comparison source does not contain the exact 5 x 2 grid")

    for scenario, baselines in delivery_baselines.items():
        if not baselines:
            raise ValueError(f"no delivery baselines found for {scenario}")
        strongest = max(baselines, key=baselines.__getitem__)
        if strongest != "q_routing":
            raise ValueError(
                f"Q-routing is not the strongest delivery baseline in {scenario}"
            )

    delivery_holm = holm_adjust(
        [float(selected[(scenario, "delivery_ratio")]["raw_p_value"]) for scenario in SCENARIO_ORDER]
    )
    switch_holm = holm_adjust(
        [float(selected[(scenario, "routing_switches")]["raw_p_value"]) for scenario in SCENARIO_ORDER]
    )

    result = []
    for scenario_index, scenario in enumerate(SCENARIO_ORDER):
        delivery = selected[(scenario, "delivery_ratio")]
        switches = selected[(scenario, "routing_switches")]
        baseline_switches = float(switches["baseline_mean"])
        switch_difference = float(switches["mean_difference"])
        if baseline_switches <= 0 or switch_difference >= 0:
            raise ValueError(f"expected a routing-switch reduction in {scenario}")
        result.append(
            {
                "scenario": scenario,
                "mappo_delivery": float(delivery["mappo_mean"]),
                "q_routing_delivery": float(delivery["baseline_mean"]),
                "delivery_difference_pp": 100.0 * float(delivery["mean_difference"]),
                "delivery_ci_low_pp": 100.0 * float(delivery["difference_ci95_low"]),
                "delivery_ci_high_pp": 100.0 * float(delivery["difference_ci95_high"]),
                "delivery_raw_p": float(delivery["raw_p_value"]),
                "delivery_holm_p": delivery_holm[scenario_index],
                "delivery_higher_seed_fraction": float(
                    delivery["positive_policy_seed_fraction"]
                ),
                "mappo_switches": float(switches["mappo_mean"]),
                "q_routing_switches": baseline_switches,
                "switch_difference": switch_difference,
                "switch_ci_low": float(switches["difference_ci95_low"]),
                "switch_ci_high": float(switches["difference_ci95_high"]),
                "switch_raw_p": float(switches["raw_p_value"]),
                "switch_holm_p": switch_holm[scenario_index],
                "switch_reduction_percent": -100.0
                * switch_difference
                / baseline_switches,
                "same_direction_seed_fraction": 1.0
                - float(switches["positive_policy_seed_fraction"]),
            }
        )
    return result


def write_main_table(rows: list[dict[str, float | str]], path: Path) -> None:
    lines = [
        r"% Requires graphicx and booktabs; place inside a two-column table* environment.",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lrrrcc@{}}",
        r"\toprule",
        r"\multicolumn{6}{c}{(a) Packet delivery} \\",
        r"Scenario & MAPPO & Q-routing & $\Delta$ delivery [95\% CI] (pp) & M$>$Q seeds & Raw exact / retro. Holm $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = SCENARIO_LABELS[str(row["scenario"])]
        lines.append(
            f"{label} & {float(row['mappo_delivery']):.6f} & "
            f"{float(row['q_routing_delivery']):.6f} & "
            f"{float(row['delivery_difference_pp']):+.6f} "
            f"[{float(row['delivery_ci_low_pp']):+.6f}, "
            f"{float(row['delivery_ci_high_pp']):+.6f}] & "
            f"{round(8 * float(row['delivery_higher_seed_fraction']))}/8 & "
            f"{float(row['delivery_raw_p']):.7g} / "
            f"{float(row['delivery_holm_p']):.7g} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{6}{c}{(b) Network-wide next-hop switching} \\",
            r"Scenario & MAPPO & Q-routing & $\Delta$ switches [95\% CI] & Reduction (seeds) & Raw exact / retro. Holm $p$ \\",
            r"\midrule",
        ]
    )
    for row in rows:
        label = SCENARIO_LABELS[str(row["scenario"])]
        fewer_seeds = round(8 * float(row["same_direction_seed_fraction"]))
        lines.append(
            f"{label} & {float(row['mappo_switches']):.4f} & "
            f"{float(row['q_routing_switches']):.4f} & "
            f"{float(row['switch_difference']):+.4f} "
            f"[{float(row['switch_ci_low']):+.4f}, "
            f"{float(row['switch_ci_high']):+.4f}] & "
            f"{float(row['switch_reduction_percent']):.2f}\\% "
            f"({fewer_seeds}/8) & "
            f"{float(row['switch_raw_p']):.7g} / "
            f"{float(row['switch_holm_p']):.7g} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            r"\par\smallskip",
            r"\begin{minipage}{\linewidth}",
            r"\footnotesize Retrospective, non-confirmatory seed-level analysis. $\Delta=$ MAPPO $-$ Q-routing; a negative switch $\Delta$ means fewer accepted changes, and reduction is $-\Delta$ divided by the Q-routing mean. Parentheses report policy seeds with fewer switches. Means first average the 50 paired workloads within each policy seed and then average the eight seed means. Raw exact $p$ is from a two-sided policy-seed sign-flip test; retrospective Holm adjustment is applied separately by endpoint across its five Q-routing scenario comparisons. Intervals use a 5,000-resample crossed pigeonhole bootstrap over policy and workload seeds. Switch counts are network-wide accepted changes to per-satellite $(satellite,destination,class)$ cached next hops.",
            r"\end{minipage}",
            r"\endgroup",
            "% Retrospective seed-level reanalysis; not confirmatory evidence.",
            "% M = MAPPO; Q = Q-routing. Differences are M minus Q.",
            "% Holm p-values are a retrospective sensitivity analysis over the five Q-routing scenarios.",
            "% Every switching row has raw exact p=0.0078125 and retrospective Holm p=0.0390625.",
            "% Workload episodes are paired repeated measurements; policy seed is the inference unit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_effect_figure(rows: list[dict[str, float | str]], output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.15),
        sharey=True,
        constrained_layout=True,
    )
    y_positions = list(range(len(rows)))
    panels = (
        (
            axes[0],
            "delivery_difference_pp",
            "delivery_ci_low_pp",
            "delivery_ci_high_pp",
            "Delivery difference (pp)",
            "(a) Delivery (MAPPO - Q-routing)",
        ),
        (
            axes[1],
            "switch_difference",
            "switch_ci_low",
            "switch_ci_high",
            "Switch-count difference / episode",
            "(b) Next-hop switches (MAPPO - Q-routing)",
        ),
    )
    for ax, value_key, low_key, high_key, xlabel, title in panels:
        for y, row in zip(y_positions, rows):
            value = float(row[value_key])
            low = float(row[low_key])
            high = float(row[high_key])
            is_delivery_hotspot = (
                value_key == "delivery_difference_pp"
                and row["scenario"] == "hotspot_high_load"
            )
            color = "#b23a48" if is_delivery_hotspot else "#285f8f"
            marker = "D" if is_delivery_hotspot else "o"
            ax.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5.5,
                linewidth=1.2,
            )
        ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_yticks(
        y_positions,
        [SCENARIO_LABELS[str(row["scenario"])] for row in rows],
    )
    axes[0].invert_yaxis()
    axes[1].text(
        0.02,
        0.96,
        "negative = fewer",
        transform=axes[1].transAxes,
        fontsize=7,
        color="#444444",
        va="top",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=DEFAULT_EXPERIMENT_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = load_qrouting_rows(
        args.comparisons,
        args.manifest,
        args.experiment_manifest,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_main_table(rows, args.output / "icc_main_results.tex")
    write_effect_figure(rows, args.output / "icc_delivery_stability")
    print(
        json.dumps(
            {
                "source_sha256": EXPECTED_COMPARISONS_SHA256,
                "scenario_count": len(rows),
                "output": str(args.output),
                "inferential_status": "retrospective_seed_level_reanalysis",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
