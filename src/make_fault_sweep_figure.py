"""Fault-robustness figure: performance vs link-failure rate (the standout result).

Reads experiments/IEEE-FAULT/aggregate_fault.csv and produces a two-panel figure:
  Left  -- delivery ratio vs fault_link_ratio for MAPPO / Dijkstra / heuristic,
           with 95% CI bands. Under time-correlated link failures the reliability-
           aware learned policy crosses ABOVE the centralized shortest-path oracle
           (proactive failure avoidance > reactive re-routing), while the
           distributed heuristic collapses.
  Right -- MAPPO - Dijkstra delivery gap vs fault rate (bars) with a zero
           reference; the positive bars mark the regime where the distributed
           learned policy beats the centralized oracle.

Usage:
  python make_fault_sweep_figure.py --input experiments/IEEE-FAULT/aggregate_fault.csv \
                                    --matrix experiments/IEEE-FAULT/fault_matrix.csv \
                                    --outdir figures
"""
from __future__ import annotations
import argparse, csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POL = {
    "mappo":           ("MAPPO",            "#1f4e79", "o", "-"),
    "global_dijkstra": ("Dijkstra (oracle)", "#d62728", "s", "--"),
    "full_heuristic":  ("Queue-aware heuristic", "#2ca02c", "^", ":"),
}


def load_agg(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    cells = {}
    for r in rows:
        fr = float(re.search(r"[\d.]+", r["fault_point"]).group())
        cells[(fr, r["policy"], r["metric"])] = (
            fr, float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    frs = sorted({k[0] for k in cells})
    return cells, frs


def series(cells, frs, policy, metric):
    xs, ys, loe, hie = [], [], [], []
    for fr in frs:
        if (fr, policy, metric) not in cells:
            continue
        _, m, lo, hi = cells[(fr, policy, metric)]
        xs.append(fr); ys.append(m); loe.append(m - lo); hie.append(hi - m)
    return xs, ys, loe, hie


def gap_series(matrix_path, frs):
    """MAPPO - Dijkstra delivery gap per fault rate, paired by workload seed."""
    rows = list(csv.DictReader(open(matrix_path, encoding="utf-8-sig")))
    gaps = []
    for fr in frs:
        label = f"fr{fr:.2f}"
        m_by = defaultdict(list); d_by = defaultdict(list)
        for r in rows:
            if r["scenario"] != label:
                continue
            ws = r["workload_seed"]
            if r["policy"] == "mappo":
                m_by[ws].append(float(r["delivery_ratio"]))
            elif r["policy"] == "global_dijkstra":
                d_by[ws].append(float(r["delivery_ratio"]))
        common = sorted(set(m_by) & set(d_by))
        diffs = [np.mean(m_by[ws]) - np.mean(d_by[ws]) for ws in common]
        gaps.append((fr, float(np.mean(diffs)) if diffs else float("nan")))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/IEEE-FAULT/aggregate_fault.csv")
    ap.add_argument("--matrix", default="experiments/IEEE-FAULT/fault_matrix.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cells, frs = load_agg(args.input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    # left: delivery
    for pol, (lbl, color, marker, ls) in POL.items():
        xs, ys, loe, hie = series(cells, frs, pol, "delivery_ratio")
        ax1.errorbar(xs, ys, yerr=[loe, hie], marker=marker, color=color, linestyle=ls,
                     linewidth=1.8, capsize=3, markersize=5, label=lbl)
    ax1.set_xlabel("Faulted-link fraction"); ax1.set_ylabel("Delivery ratio")
    ax1.set_ylim(0, 0.85); ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=7, loc="lower left", frameon=False)

    # right: MAPPO - Dijkstra gap
    gaps = gap_series(args.matrix, frs)
    xs = [g[0] for g in gaps]; ys = [g[1] for g in gaps]
    colors = ["#1f4e79" if y > 0 else "#d62728" for y in ys]
    ax2.bar(xs, ys, width=0.025, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax2.axhline(0.0, color="gray", linewidth=1.0)
    for x, y in zip(xs, ys):
        ax2.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 5 if y >= 0 else -11), ha="center", fontsize=7,
                     color="#1f4e79" if y >= 0 else "#d62728")
    ax2.set_xlabel("Faulted-link fraction"); ax2.set_ylabel("Delivery gap (MAPPO $-$ Dijkstra)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_fault_sweep.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_fault_sweep] {len(frs)} fault rates -> {outdir}/fig_fault_sweep.{{png,pdf}}")


if __name__ == "__main__":
    main()
