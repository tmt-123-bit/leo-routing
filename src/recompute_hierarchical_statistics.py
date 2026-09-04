"""Recompute inferential tables from an existing episode_metrics.csv file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

from hierarchical_statistics import statistical_analysis_manifest
from mappo_evaluation import EpisodeMetrics
from run_ablation_experiments import (
    PRIMARY_METRICS,
    SCENARIOS as ABLATION_SCENARIOS,
    TEST_WORKLOAD_SEEDS,
    VARIANTS as ABLATION_VARIANTS,
    legacy_contrasts,
    paired_effects,
)
from run_exp004_mappo import (
    METRICS,
    POLICY_SEEDS,
    aggregate_rows,
    paired_tests,
    write_csv,
)
from variant_definitions import PLANNED_CONTRASTS


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute crossed-seed confidence intervals and policy-seed-level "
            "paired tests without training or evaluating policies again."
        )
    )
    parser.add_argument("--kind", choices=["headline", "ablation"], required=True)
    parser.add_argument("--episode-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_episode_rows(path: Path) -> list[SimpleNamespace]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [SimpleNamespace(**row) for row in csv.DictReader(file)]
    if not rows:
        raise ValueError(f"episode metrics file has no data rows: {path}")
    for row_number, row in enumerate(rows, start=2):
        try:
            row.policy_seed = int(row.policy_seed)
            row.workload_seed = int(row.workload_seed)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid policy/workload seed identity on CSV row {row_number}"
            ) from error
    return rows


def validate_formal_ablation_rows(rows: list[SimpleNamespace]) -> dict[str, Any]:
    """Reject anything other than the frozen 9 x 5 x 12 x 50 test grid."""

    expected_policies = {f"mappo_{variant}" for variant in ABLATION_VARIANTS}
    expected_policy_seeds = {int(seed) for seed in POLICY_SEEDS}
    expected_workload_seeds = {int(seed) for seed in TEST_WORKLOAD_SEEDS}
    expected_scenarios = set(ABLATION_SCENARIOS)
    expected_row_count = (
        len(expected_scenarios)
        * len(expected_policies)
        * len(expected_policy_seeds)
        * len(expected_workload_seeds)
    )
    if len(rows) != expected_row_count:
        raise ValueError(
            "formal ablation requires exactly "
            f"{expected_row_count} rows, observed {len(rows)}"
        )

    field_types = get_type_hints(EpisodeMetrics)
    required_fields = set(field_types)
    observed_cells: set[tuple[str, str, int, int]] = set()
    observed_scenarios: set[str] = set()
    observed_policies: set[str] = set()
    observed_policy_seeds: set[int] = set()
    observed_workload_seeds: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        values = vars(row)
        missing_fields = required_fields.difference(values)
        if missing_fields:
            raise ValueError(
                f"formal ablation CSV row {row_number} lacks fields "
                f"{sorted(missing_fields)}"
            )

        for field, field_type in field_types.items():
            raw_value = values[field]
            try:
                if field_type is int:
                    converted = int(raw_value)
                elif field_type is float:
                    converted = float(raw_value)
                    if not math.isfinite(converted):
                        raise ValueError("non-finite")
                elif field_type is str:
                    converted = str(raw_value)
                else:
                    raise TypeError(f"unsupported EpisodeMetrics type {field_type!r}")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid or non-finite {field!r} on CSV row {row_number}"
                ) from error
            setattr(row, field, converted)

        cell = (
            row.scenario,
            row.policy,
            row.policy_seed,
            row.workload_seed,
        )
        if cell in observed_cells:
            raise ValueError(f"duplicate formal ablation cell {cell!r}")
        observed_cells.add(cell)
        observed_scenarios.add(row.scenario)
        observed_policies.add(row.policy)
        observed_policy_seeds.add(row.policy_seed)
        observed_workload_seeds.add(row.workload_seed)

    level_errors = []
    for label, observed, expected in (
        ("scenarios", observed_scenarios, expected_scenarios),
        ("policies", observed_policies, expected_policies),
        ("policy seeds", observed_policy_seeds, expected_policy_seeds),
        ("test workload seeds", observed_workload_seeds, expected_workload_seeds),
    ):
        if observed != expected:
            level_errors.append(
                f"{label}: missing={sorted(expected - observed)!r}, "
                f"unexpected={sorted(observed - expected)!r}"
            )
    if level_errors:
        raise ValueError(
            "formal ablation levels differ from the frozen protocol; "
            + "; ".join(level_errors)
        )

    expected_cells = {
        (scenario, f"mappo_{variant}", policy_seed, workload_seed)
        for scenario in ABLATION_SCENARIOS
        for variant in ABLATION_VARIANTS
        for policy_seed in POLICY_SEEDS
        for workload_seed in TEST_WORKLOAD_SEEDS
    }
    if observed_cells != expected_cells:
        missing = sorted(expected_cells.difference(observed_cells))
        unexpected = sorted(observed_cells.difference(expected_cells))
        raise ValueError(
            "formal ablation is not the complete frozen crossed grid: "
            f"missing examples={missing[:5]!r}, "
            f"unexpected examples={unexpected[:5]!r}"
        )

    return {
        "status": "validated_complete_formal_grid",
        "scenario_order": list(ABLATION_SCENARIOS),
        "variants": list(ABLATION_VARIANTS),
        "policy_seeds": list(POLICY_SEEDS),
        "test_workload_seeds": list(TEST_WORKLOAD_SEEDS),
        "unique_finite_rows": len(observed_cells),
        "expected_unique_finite_rows": expected_row_count,
    }


def select_ablation_analysis_contract(
    rows: list[SimpleNamespace],
) -> tuple[list[str], tuple[Any, ...], str, bool, dict[str, Any]]:
    """Choose formal analysis only for the exact frozen proposed-method grid."""

    variants = sorted(
        policy.removeprefix("mappo_")
        for policy in {row.policy for row in rows}
        if policy.startswith("mappo_")
    )
    if "proposed" in variants:
        validation = validate_formal_ablation_rows(rows)
        return (
            list(ABLATION_SCENARIOS),
            PLANNED_CONTRASTS,
            "formal_confirmatory_complete_grid",
            True,
            validation,
        )
    if "full" in variants:
        return (
            sorted({row.scenario for row in rows}),
            legacy_contrasts(variants),
            "legacy_exploratory_reanalysis",
            False,
            {
                "status": "legacy_input_not_formal_grid",
                "label": "exploratory_only",
            },
        )
    raise ValueError(
        "ablation input contains neither proposed nor legacy full reference"
    )


def main() -> None:
    args = parse_args()
    source = args.episode_metrics.resolve()
    rows = read_episode_rows(source)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.kind == "headline":
        aggregate = aggregate_rows(rows)
        comparisons = paired_tests(rows)
        output_files = {
            "aggregate": "aggregate_metrics.csv",
            "comparisons": "paired_tests.csv",
        }
        write_csv(args.output / output_files["aggregate"], aggregate)
        write_csv(args.output / output_files["comparisons"], comparisons)
        metrics = METRICS
        inferential_status = "headline_reanalysis"
        confirmatory = False
        grid_validation = None
    else:
        (
            scenarios,
            contrasts,
            inferential_status,
            confirmatory,
            grid_validation,
        ) = select_ablation_analysis_contract(rows)
        comparisons = paired_effects(rows, scenarios, contrasts)
        output_files = {"comparisons": "paired_ablation_effects.csv"}
        write_csv(args.output / output_files["comparisons"], comparisons)
        metrics = PRIMARY_METRICS

    output_row_counts = {
        "comparisons": len(comparisons),
    }
    if args.kind == "headline":
        output_row_counts["aggregate"] = len(aggregate)
    output_artifacts = {
        name: {
            "file": filename,
            "sha256": hashlib.sha256(
                (args.output / filename).read_bytes()
            ).hexdigest(),
            "row_count": output_row_counts[name],
        }
        for name, filename in output_files.items()
    }

    manifest = {
        "analysis": "crossed_seed_statistical_reanalysis",
        "kind": args.kind,
        "inferential_status": inferential_status,
        "confirmatory": confirmatory,
        "source_episode_metrics": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_row_count": len(rows),
        "metrics": metrics,
        "output_files": output_files,
        "output_artifacts": output_artifacts,
        "formal_grid_validation": grid_validation,
        "statistical_analysis": statistical_analysis_manifest(),
    }
    (args.output / "statistical_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Recomputed {len(comparisons)} comparisons from {source} -> "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
