"""Reward-weight sensitivity figure (the 'robust to reward weights' panel).

Reads experiments/IEEE-REWARD-SENSITIVITY/aggregate_sensitivity.csv and produces a 2x2
grid: delivery / throughput / P95 delay / load-imbalance, one bar per config
(baseline + 6 OAT weight perturbations), MAPPO only, with 95% CI whiskers and a
baseline reference line. A robust policy lands every bar within a tight band of the
baseline -> the result is not an artifact of a specific reward-weight setting.

Usage:
  python make_reward_sensitivity_figure.py \
      --input experiments/IEEE-REWARD-SENSITIVITY/aggregate_sensitivity.csv --outdir figures
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# display order + labels (baseline first, then each weight low/high)
CONFIG_LABEL = {
    "baseline":   "baseline",
    "deliver_lo": "w_deliver\n0.5x",
    "deliver_hi": "w_deliver\n2x",
    "load_lo":    "w_load\n0.5x",
    "load_hi":    "w_load\n2x",
    "switch_lo":  "w_switch\n0.5x",
    "switch_hi":  "w_switch\n5x",
}
ORDER = ["baseline", "deliver_lo", "deliver_hi", "load_lo", "load_hi", "switch_lo", "switch_hi"]

PANELS = [
    ("delivery_ratio",                "Delivery ratio",              (None, None)),
    ("throughput_packets_per_slot",   "Throughput (pkts/slot)",      (None, None)),
    ("p95_delay_slots",               "P95 delay (slots)",           (None, None)),
    ("global_load_imbalance",         "Load imbalance",              (None, None)),
]


def load(path):
    cells = {}  # (config, metric) -> (mean, lo, hi)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["policy"] != "mappo":
            continue
        cells[(r["config"], r["metric"])] = (
            float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/IEEE-REWARD-SENSITIVITY/aggregate_sensitivity.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cells = load(args.input)
    configs = [c for c in ORDER if any(k[0] == c for k in cells)] or ORDER
    x = np.arange(len(configs))

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.2))
    for ax, (metric, ylabel, (lo, hi)) in zip(axes.flat, PANELS):
        means, loe, hie = [], [], []
        for c in configs:
            m, cil, cih = cells.get((c, metric), (np.nan, np.nan, np.nan))
            means.append(m); loe.append(m - cil); hie.append(cih - m)
        cols = ["#1f4e79" if c == "baseline" else "#7fb1d6" for c in configs]
        ax.bar(x, means, width=0.62, color=cols, edgecolor="white", linewidth=0.6,
               yerr=[loe, hie], capsize=2.5, error_kw={"linewidth": 0.9}, zorder=3)
        if not np.isnan(means[configs.index("baseline")]) if "baseline" in configs else True:
            base = means[configs.index("baseline")]
            ax.axhline(base, color="#d62728", linestyle="--", linewidth=1.1, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([CONFIG_LABEL.get(c, c) for c in configs], fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=8)
        if lo is not None: ax.set_ylim(bottom=lo)
        if hi is not None: ax.set_ylim(top=hi)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Reward-weight sensitivity (MAPPO, medium_load) — bars within a tight band of "
                 "baseline (dashed) = robust", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_reward_sensitivity.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_reward_sensitivity] {len(configs)} configs -> "
          f"{outdir}/fig_reward_sensitivity.{{png,pdf}}")


if __name__ == "__main__":
    main()
