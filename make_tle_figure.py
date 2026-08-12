"""Real-topology TRAINING figure (C3): train MAPPO on real Starlink-24, eval in-distribution
(Starlink) + cross-constellation (OneWeb, unseen). Compares against the zero-shot synthetic
transfer (fig_realism_transfer) to make the "transfer suffices" point.

Two panels:
  Left  -- MAPPO/Dijkstra throughput ratio, grouped by topology, two bars each:
           "zero-shot" (synthetic-trained, from realism transfer) vs "real-trained" (C3).
           Both land ~0.93-0.96 -> training on real topology does NOT beat transfer from
           synthetic; the scale-invariant actor generalizes either way. Cross-constellation
           (train Starlink -> eval OneWeb) holds at 0.96.
  Right -- P95 delay: MAPPO (real-trained) vs Dijkstra, per topology. The learned local-info
           policy trades a little raw delivery for decisively lower tail latency (esp. OneWeb,
           -3.0 slots), and lower load imbalance (inset text).

Usage:
  python make_tle_figure.py \
      --tle experiments/TLE-NOLIFE-FULL/aggregate_metrics.csv \
      --realism experiments/IEEE-REALISM-NOLIFE/aggregate_realism.csv \
      --outdir figures
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOPOS = ["starlink", "oneweb"]
TOPO_LABEL = {"starlink": "Starlink-24\n(in-dist)", "oneweb": "OneWeb-24\n(cross-const.)"}
DIJ = "#d62728"
MAPPO_T = "#1f4e79"   # real-trained
MAPPO_Z = "#7fb1d6"   # zero-shot transfer (lighter)


def load_realism(path):
    """topology/policy/metric -> (mean, lo, hi). Keys: starlink_24/oneweb_24."""
    cells = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        cells[(r["topology"], r["policy"], r["metric"])] = (
            float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    return cells


def load_tle(path):
    """scenario/policy/metric -> (mean, lo, hi). Keys: tle_starlink_24/tle_oneweb_24."""
    cells = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        cells[(r["scenario"], r["policy"], r["metric"])] = (
            float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    return cells


def ratio(m_lo_hi, d_lo_hi):
    m, _, _ = m_lo_hi; d, _, _ = d_lo_hi
    return m / max(d, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tle", default="experiments/TLE-NOLIFE-FULL/aggregate_metrics.csv")
    ap.add_argument("--realism", default="experiments/IEEE-REALISM-NOLIFE/aggregate_realism.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cr = load_realism(args.realism)
    ct = load_tle(args.tle)

    THRU = "throughput_packets_per_slot"
    P95 = "p95_delay_slots"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})

    # ---- left: throughput ratio, transfer vs real-trained ----
    x = np.arange(len(TOPOS)); width = 0.34
    for i, cond in enumerate(["zero", "trained"]):
        ratios = []
        for t in TOPOS:
            if cond == "zero":
                m = cr[(f"{t}_24", "mappo_realism_transfer", THRU)]
                d = cr[(f"{t}_24", "global_dijkstra", THRU)]
            else:
                m = ct[(f"tle_{t}_24", "mappo_tle_trained", THRU)]
                d = ct[(f"tle_{t}_24", "global_dijkstra", THRU)]
            ratios.append(ratio(m, d))
        col = MAPPO_Z if cond == "zero" else MAPPO_T
        lbl = "Zero-shot (synthetic-trained)" if cond == "zero" else "Real-trained (Starlink)"
        ax1.bar(x + (i - 0.5) * width, ratios, width, color=col, edgecolor="white",
                linewidth=0.6, label=lbl, zorder=3)
        for xi, r in zip(x + (i - 0.5) * width, ratios):
            ax1.annotate(f"{r:.3f}", (xi, r), textcoords="offset points", xytext=(0, 4),
                         ha="center", fontsize=7, color=col)
    ax1.axhline(1.0, color=DIJ, linestyle="--", linewidth=1.2, label="oracle parity")
    ax1.set_xticks(x); ax1.set_xticklabels([TOPO_LABEL[t] for t in TOPOS], fontsize=7.5)
    ax1.set_ylabel("Throughput ratio (MAPPO / Dijkstra)")
    ax1.set_ylim(0.86, 1.02); ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=6.5, loc="lower left", frameon=False)

    # ---- right: P95 delay, MAPPO(real-trained) vs Dijkstra ----
    mp95, dp95 = [], []
    for t in TOPOS:
        m = ct[(f"tle_{t}_24", "mappo_tle_trained", P95)]
        d = ct[(f"tle_{t}_24", "global_dijkstra", P95)]
        mp95.append(m[0]); dp95.append(d[0])
    ax2.bar(x - width/2, dp95, width, color=DIJ, edgecolor="white", linewidth=0.6,
            label="Dijkstra (oracle)", zorder=3)
    ax2.bar(x + width/2, mp95, width, color=MAPPO_T, edgecolor="white", linewidth=0.6,
            label="MAPPO (real-trained)", zorder=3)
    for xi, (dm, dd) in zip(x, zip(mp95, dp95)):
        ax2.annotate(f"$-${dd-dm:.1f}", (xi, max(dm, dd) + 0.4), ha="center", fontsize=7.5,
                     color=MAPPO_T, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels([TOPO_LABEL[t] for t in TOPOS], fontsize=7.5)
    ax2.set_ylabel("P95 delay (slots)"); ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(fontsize=7, loc="upper left", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_tle_training.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_tle_training] {len(TOPOS)} topologies -> {outdir}/fig_tle_training.{{png,pdf}}")


if __name__ == "__main__":
    main()
