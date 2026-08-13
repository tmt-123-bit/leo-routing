"""Minimum-detectable-effect (MDE) reporting for paired policy comparisons.

WHY THIS EXISTS
---------------
The paper makes several "no significant difference" claims (e.g. frequent_break:
MAPPO vs Dijkstra). A reviewer will reject a bare null unless we state the
smallest effect the experiment could have detected at the chosen alpha/power.
This script computes that MDE for every (scenario, metric, policy-pair) and
flags every null whose observed effect is smaller than its MDE — i.e. claims
that are honest only up to the MDE.

It reads the per-episode metrics CSV produced by run_exp004_mappo.py and
reconstructs per-workload paired differences (averaging over policy seeds within
each workload, mirroring `paired_tests`). The MDE uses the paired-difference
standard deviation:

    MDE = (z_{1-alpha/2} + z_{1-power}) * sigma_d / sqrt(n)

where sigma_d is the std of the paired per-workload differences and n is the
number of paired workloads. This is the parametric (paired-t) detectability
bound; the main test in paired_tests is Wilcoxon, so MDE is reported as an
approximate power bound, clearly labelled — standard practice for stating the
resolution of a non-parametric null.

USAGE
-----
    py compute_mde.py \
        --episodes experiments/IEEE-EXP-004-FULL/episode_metrics.csv \
        --output   experiments/IEEE-EXP-004-FULL/mde_report.csv

Run AFTER the full re-run (run_ieee_reproduction.sh). The reference policy is
``mappo`` by default; every other policy is compared against it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


# Reportable metrics (matches run_exp004 METRICS; subset that is meaningful
# for paired power analysis).
DEFAULT_METRICS = [
    "delivery_ratio",
    "drop_rate",
    "throughput_packets_per_slot",
    "average_delay_slots",
    "p95_delay_slots",
    "mean_queue_packets",
    "routing_switches",
]


def z_two_sided(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def z_power(power: float) -> float:
    return float(stats.norm.ppf(power))


def load_episodes(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def paired_diffs(
    rows: Sequence[Dict[str, str]],
    scenario: str,
    metric: str,
    reference: str,
    baseline: str,
) -> Tuple[np.ndarray, int]:
    """Per-workload paired differences (reference - baseline), averaging over
    policy seeds within each workload. Returns (differences, n_workloads)."""
    ref_by_workload: Dict[int, List[float]] = {}
    base_by_workload: Dict[int, List[float]] = {}
    for row in rows:
        if row["scenario"] != scenario:
            continue
        try:
            value = float(row[metric])
        except (KeyError, ValueError):
            continue
        workload_seed = int(row["workload_seed"])
        if row["policy"] == reference:
            ref_by_workload.setdefault(workload_seed, []).append(value)
        elif row["policy"] == baseline:
            base_by_workload.setdefault(workload_seed, []).append(value)
    common = sorted(set(ref_by_workload) & set(base_by_workload))
    ref_mean = np.asarray([np.mean(ref_by_workload[s]) for s in common])
    base_mean = np.asarray([np.mean(base_by_workload[s]) for s in common])
    return ref_mean - base_mean, len(common)


def compute(
    rows: Sequence[Dict[str, str]],
    metrics: Sequence[str],
    reference: str,
    alpha: float,
    power: float,
) -> List[Dict]:
    scenarios = sorted({row["scenario"] for row in rows})
    policies = sorted({row["policy"] for row in rows if row["policy"] != reference})
    z_a = z_two_sided(alpha)
    z_p = z_power(power)
    z_sum = z_a + z_p
    out: List[Dict] = []
    for scenario in scenarios:
        for baseline in policies:
            for metric in metrics:
                diffs, n = paired_diffs(rows, scenario, metric, reference, baseline)
                if n < 2:
                    continue
                sigma_d = float(np.std(diffs, ddof=1))
                mde = z_sum * sigma_d / np.sqrt(n)
                observed = float(np.mean(diffs))
                try:
                    p_value = float(stats.wilcoxon(diffs).pvalue)
                except ValueError:
                    p_value = 1.0
                base_vals = []
                for row in rows:
                    if (
                        row["scenario"] == scenario
                        and row["policy"] == baseline
                    ):
                        try:
                            base_vals.append(float(row[metric]))
                        except ValueError:
                            pass
                base_scale = abs(float(np.mean(base_vals))) if base_vals else float("nan")
                out.append(
                    {
                        "scenario": scenario,
                        "reference": reference,
                        "baseline": baseline,
                        "metric": metric,
                        "n_paired_workloads": n,
                        "sigma_d_paired": sigma_d,
                        "observed_diff_ref_minus_base": observed,
                        "mde_abs": mde,
                        "mde_rel_to_baseline": (mde / base_scale) if base_scale else float("nan"),
                        "alpha": alpha,
                        "power": power,
                        "wilcoxon_p": p_value,
                        "is_null_at_alpha": p_value >= alpha,
                        "observed_below_mde": abs(observed) < mde,
                    }
                )
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict], reference: str) -> str:
    """Human-readable focus on delivery_ratio nulls — the claims a reviewer
    probes hardest."""
    lines = [
        f"MDE report (reference = {reference}). "
        "Rows where is_null_at_alpha AND observed_below_mde are honest only "
        "up to MDE — state the MDE when claiming 'no difference'.",
        "",
    ]
    nulls = [
        r
        for r in rows
        if r["metric"] == "delivery_ratio" and r["is_null_at_alpha"]
    ]
    if not nulls:
        lines.append("No delivery_ratio nulls at the chosen alpha.")
    else:
        lines.append(
            f"{'scenario':16s} {'baseline':16s} {'obs':>8s} "
            f"{'MDE':>8s} {'MDE%':>7s} {'p':>8s}"
        )
        for r in sorted(nulls, key=lambda r: r["wilcoxon_p"]):
            rel = r["mde_rel_to_baseline"]
            lines.append(
                f"{r['scenario']:16s} {r['baseline']:16s} "
                f"{r['observed_diff_ref_minus_base']:+8.4f} "
                f"{r['mde_abs']:8.4f} {rel * 100 if rel == rel else float('nan'):6.1f}% "
                f"{r['wilcoxon_p']:8.3g}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("experiments/IEEE-EXP-004-FULL/episode_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: <episodes dir>/mde_report.csv",
    )
    parser.add_argument("--reference", default="mappo")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.episodes.parent / "mde_report.csv"

    rows = load_episodes(args.episodes)
    report = compute(rows, args.metrics, args.reference, args.alpha, args.power)
    write_csv(args.output, report)
    print(f"wrote {len(report)} rows -> {args.output}")
    print()
    print(summarize(report, args.reference))


if __name__ == "__main__":
    main()
