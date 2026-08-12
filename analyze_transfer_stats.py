"""Paired statistical analysis for the transfer experiments (realism + scale).

For each topology/scale this computes, mirroring the headline paired_tests.csv
rigor:
  * MAPPO-vs-Dijkstra and MAPPO-vs-heuristic on every metric:
      mean(diff), paired Wilcoxon p, Cohen's dz, n paired workloads.
      (MAPPO is averaged across its policy seeds within each workload seed, so
       each workload seed yields one paired difference — the standard way to
       pair a multi-seed learned policy against a single-seed oracle.)
  * A bootstrap 95% CI on the MAPPO/Dijkstra THROUGHPUT RATIO (resampling
      workload seeds), which is the headline transfer-quality number — reporting
      a CI on the ratio itself, not just on the separate numerator/denominator.
  * Benjamini-Hochberg adjustment across all tests within a group.

Works on both:
  experiments/IEEE-REALISM/realism_matrix.csv   (group = scenario/topology)
  experiments/IEEE-SCALE/transfer_matrix.csv    (group = scenario/scale label)

Usage:
  python analyze_transfer_stats.py \
      --input experiments/IEEE-REALISM/realism_matrix.csv \
      --mappo-policy mappo_realism_transfer \
      --group-col scenario --output experiments/IEEE-REALISM/paired_stats.csv

  python analyze_transfer_stats.py \
      --input experiments/IEEE-SCALE/transfer_matrix.csv \
      --mappo-policy mappo_transfer_n24 \
      --group-col scenario --output experiments/IEEE-SCALE/paired_stats.csv
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = [
    "delivery_ratio", "drop_rate", "throughput_packets_per_slot",
    "average_delay_slots", "p95_delay_slots", "mean_queue_packets",
]
BASELINES = ["global_dijkstra", "full_heuristic"]


def load_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def paired_series(rows, group, policy, baseline, metric, group_col):
    """Return (mappo_vals, base_vals) arrays, one per shared workload seed.
    MAPPO is averaged across its policy seeds within each workload seed."""
    mappo_by_wseed = defaultdict(list)
    base_by_wseed = defaultdict(list)
    for r in rows:
        if r[group_col] != group:
            continue
        ws = int(r["workload_seed"])
        if r["policy"] == policy:
            mappo_by_wseed[ws].append(float(r[metric]))
        elif r["policy"] == baseline:
            base_by_wseed[ws].append(float(r[metric]))
    common = sorted(set(mappo_by_wseed) & set(base_by_wseed))
    if not common:
        return np.array([]), np.array([])
    m = np.array([np.mean(mappo_by_wseed[ws]) for ws in common])
    b = np.array([np.mean(base_by_wseed[ws]) for ws in common])
    return m, b


def ratio_bootstrap_ci(mappo_thru, base_thru, seeds=(4001,), B=5000):
    rng = np.random.default_rng(seeds[0])
    n = len(mappo_thru)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    num = mappo_thru[idx].mean(axis=1)
    den = base_thru[idx].mean(axis=1)
    ratios = num / np.maximum(den, 1e-9)
    return float(ratios.mean()), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def cohens_dz(diffs):
    if len(diffs) < 2:
        return float("nan")
    sd = np.std(diffs, ddof=1)
    return float(np.mean(diffs) / sd) if sd > 1e-12 else float("inf") * np.sign(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mappo-policy", required=True)
    ap.add_argument("--group-col", default="scenario")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_rows(args.input)
    groups = sorted({r[args.group_col] for r in rows})

    records = []
    ratio_rows = []
    for group in groups:
        for baseline in BASELINES:
            for metric in METRICS:
                m, b = paired_series(rows, group, args.mappo_policy, baseline, metric, args.group_col)
                if len(m) == 0:
                    continue
                diffs = m - b
                try:
                    w = stats.wilcoxon(diffs)
                    p = float(w.pvalue); stat = float(w.statistic)
                except ValueError:
                    p, stat = 1.0, 0.0
                records.append({
                    "group": group, "baseline": baseline, "metric": metric,
                    "n_paired": len(diffs), "mappo_mean": float(m.mean()),
                    "baseline_mean": float(b.mean()), "mean_diff": float(diffs.mean()),
                    "median_diff": float(np.median(diffs)), "cohens_dz": cohens_dz(diffs),
                    "wilcoxon_stat": stat, "raw_p": p,
                })
        # throughput ratio CI (MAPPO vs Dijkstra)
        mt, dt = paired_series(rows, group, args.mappo_policy, "global_dijkstra",
                               "throughput_packets_per_slot", args.group_col)
        rmean, rlo, rhi = ratio_bootstrap_ci(mt, dt)
        ratio_rows.append({"group": group, "ratio_mean": rmean, "ratio_ci95_low": rlo,
                           "ratio_ci95_high": rhi, "n_paired": len(mt)})

    # Benjamini-Hochberg within each (group, baseline) family of metrics.
    # Step-up BH: sort ascending, then adjusted q_(k) = min_{j>=k} (m/j * p_(j)),
    # so iterate ranks DESCENDING (largest p first) carrying a running min.
    by_fam = defaultdict(list)
    for i, rec in enumerate(records):
        by_fam[(rec["group"], rec["baseline"])].append(i)
    for fam, idxs in by_fam.items():
        idxs_sorted = sorted(idxs, key=lambda i: records[i]["raw_p"])
        m = len(idxs_sorted); running = 1.0
        for rank in range(m, 0, -1):
            i = idxs_sorted[rank - 1]
            running = min(running, records[i]["raw_p"] * m / rank)
            records[i]["bh_p"] = min(1.0, running)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    ratio_path = out.parent / (out.stem + "_ratio_ci.csv")
    with ratio_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
        w.writeheader(); w.writerows(ratio_rows)

    print(f"\n=== paired stats: {args.input} ===")
    for group in groups:
        print(f"\n[{group}]  MAPPO vs baselines (delivery_ratio):")
        for rec in records:
            if rec["group"] == group and rec["metric"] == "delivery_ratio":
                star = "***" if rec["bh_p"] < 1e-3 else ("**" if rec["bh_p"] < 1e-2 else ("*" if rec["bh_p"] < 0.05 else ""))
                print(f"  vs {rec['baseline']:16s} diff={rec['mean_diff']:+.4f}  dz={rec['cohens_dz']:+.2f}  "
                      f"p={rec['raw_p']:.2g} (BH={rec['bh_p']:.2g}) {star}  n={rec['n_paired']}")
        for rr in ratio_rows:
            if rr["group"] == group:
                print(f"  >> throughput ratio MAPPO/Dijkstra = {rr['ratio_mean']:.3f} "
                      f"[{rr['ratio_ci95_low']:.3f}, {rr['ratio_ci95_high']:.3f}]  (n={rr['n_paired']})")
    print(f"\n=> wrote {out} + {ratio_path}")


if __name__ == "__main__":
    main()
