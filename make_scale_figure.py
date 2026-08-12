"""Scale-figure generator for the constellation-scale transfer experiment.

Reads experiments/IEEE-SCALE/aggregate_scale.csv and produces a two-panel figure:
  Left  — delivery ratio vs constellation size (MAPPO-transfer, Dijkstra, heuristic)
          with 95% bootstrap bands; shows the learned policy tracks the oracle while
          the distributed heuristic collapses.
  Right — MAPPO/Dijkstra throughput ratio vs size (the scale-invariant metric);
          1.0 reference = oracle parity; ~0.95 through 110 sats, dipping at 132.

Usage:
  python make_scale_figure.py --input experiments/IEEE-SCALE/aggregate_scale.csv \
                              --outdir figures
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POL = {
    "mappo_transfer_n24": ("MAPPO (trained @24, zero-shot)", "#1f4e79", "o", "-"),
    "global_dijkstra":    ("Dijkstra (oracle)",              "#d62728", "s", "--"),
    "full_heuristic":     ("Queue-aware heuristic",          "#2ca02c", "^", ":"),
}
METRIC_Y = {"delivery_ratio": (0, 0.9, "Delivery ratio"),
            "throughput_packets_per_slot": (0, None, "Throughput (pkts/slot)")}


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    cells = {}
    for r in rows:
        cells[(r["target_scale"], r["policy"], r["metric"])] = (
            int(r["n_sats"]), float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    # scales in n_sats order
    scales = sorted({k[0] for k in cells}, key=lambda s: cells[(s, "global_dijkstra", "delivery_ratio")][0])
    return cells, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/IEEE-SCALE/aggregate_scale.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cells, scales = load(args.input)
    xs = [cells[(s, "global_dijkstra", "delivery_ratio")][0] for s in scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    # left: delivery
    for pol, (lbl, color, marker, ls) in POL.items():
        ys, lo, hi = [], [], []
        for s in scales:
            _, m, l, h = cells[(s, pol, "delivery_ratio")]
            ys.append(m); lo.append(m - l); hi.append(h - m)
        ax1.errorbar(xs, ys, yerr=[lo, hi], marker=marker, color=color, linestyle=ls,
                     linewidth=1.8, capsize=3, markersize=5, label=lbl)
    ax1.set_xlabel("Constellation size (satellites)"); ax1.set_ylabel("Delivery ratio")
    ax1.set_xticks(xs); ax1.set_ylim(0, 0.85); ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=7, loc="upper right", frameon=False)

    # right: MAPPO/Dijkstra throughput ratio
    ratios = []
    for s in scales:
        _, mt, _, _ = cells[(s, "mappo_transfer_n24", "throughput_packets_per_slot")]
        _, dt, _, _ = cells[(s, "global_dijkstra", "throughput_packets_per_slot")]
        ratios.append(mt / max(dt, 1e-9))
    ax2.plot(xs, ratios, marker="o", color="#1f4e79", linewidth=2, markersize=6, label="MAPPO / Dijkstra")
    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="oracle parity")
    ax2.axhline(0.95, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    for x, r in zip(xs, ratios):
        ax2.annotate(f"{r:.2f}", (x, r), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=7.5, color="#1f4e79")
    ax2.set_xlabel("Constellation size (satellites)"); ax2.set_ylabel("Throughput ratio (MAPPO / Dijkstra)")
    ax2.set_xticks(xs); ax2.set_ylim(0.78, 1.05); ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=7, loc="lower left", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_scale_transfer.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_scale_transfer] {len(scales)} scales -> {outdir}/fig_scale_transfer.{{png,pdf}}")


if __name__ == "__main__":
    main()
