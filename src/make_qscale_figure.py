"""Figure: Q-routing transfer vs retraining at constellation scale (the "why MAPPO" plot).

Reads IEEE-QSCALE/qscale_matrix.csv (q_routing_transfer / q_routing_retrained /
q_routing_n24_indist) and IEEE-SCALE-EVALFIX/transfer_matrix.csv (mappo_transfer_n24
/ global_dijkstra, same workload seeds), plots throughput ratio vs Dijkstra with
bootstrap 95% CI. Output: figures/fig_qscale_transfer.{png,pdf}.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALES = ["n24", "n66", "n110", "n132"]
NSATS = {"n24": 24, "n66": 66, "n110": 110, "n132": 132}


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ratio_ci(mine, theirs, rng, boots=5000):
    """Bootstrap CI of mean(mine)/mean(theirs) over workload-mean paired rows."""
    a = np.asarray(mine, dtype=float)
    b = np.asarray(theirs, dtype=float)
    point = a.mean() / b.mean()
    bs = []
    for _ in range(boots):
        idx = rng.integers(0, len(a), len(a))
        bs.append(a[idx].mean() / max(b[idx].mean(), 1e-9))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, lo, hi


def thru_by_workload(rows, scenario, policy):
    out = {}
    for r in rows:
        if r["scenario"] == scenario and r["policy"] == policy:
            out.setdefault(int(r["workload_seed"]), []).append(
                float(r["throughput_packets_per_slot"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qscale", default="experiments/IEEE-QSCALE/qscale_matrix.csv")
    ap.add_argument("--scale", default="experiments/IEEE-SCALE-EVALFIX/transfer_matrix.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    q, s = load(args.qscale), load(args.scale)
    rng = np.random.default_rng(11)

    series = {
        "MAPPO zero-shot (n24-trained)": ("mappo_transfer_n24", s, "#1f4e79", "o", "-"),
        "Q-routing retrained at scale": ("q_routing_retrained", q, "#c0504d", "s", "--"),
        "Q-routing n24 table as-is": ("q_routing_transfer", q, "#9bbb59", "^", ":"),
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for label, (pol, src, color, marker, ls) in series.items():
        xs, ys, los, his = [], [], [], []
        for i, lab in enumerate(SCALES):
            m = thru_by_workload(src, lab, pol)
            d = thru_by_workload(s, lab, "global_dijkstra")
            if not m:
                continue
            common = sorted(set(m) & set(d))
            point, lo, hi = ratio_ci([m[k] for k in common], [d[k] for k in common], rng)
            xs.append(NSATS[lab]); ys.append(point); los.append(point - lo); his.append(hi - point)
        if xs:
            ax.errorbar(xs, ys, yerr=[los, his], marker=marker, color=color,
                        linestyle=ls, linewidth=1.8, capsize=3, label=label)
    ax.axhline(1.0, color="#444", linewidth=1.0, linestyle="-", alpha=0.6)
    ax.text(129, 1.015, "Dijkstra oracle", fontsize=8, color="#444", ha="right")
    ax.set_xticks([NSATS[l] for l in SCALES])
    ax.set_xticklabels([f"{NSATS[l]}\n({l})" for l in SCALES], fontsize=8.5)
    ax.set_xlabel("Constellation size (satellites)", fontsize=10)
    ax.set_ylabel("Throughput ratio vs Dijkstra", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.tight_layout()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_qscale_transfer.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_qscale_transfer] {len(SCALES)} scales -> {outdir}/fig_qscale_transfer.{{png,pdf}}")


if __name__ == "__main__":
    main()
