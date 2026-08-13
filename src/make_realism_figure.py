"""Realism-transfer figure: synthetic-grid policy -> real TLE constellations.

Three categories on the x-axis (training distribution -> real topologies):
  synthetic_24   (the 4x6 grid the policy was trained on; from IEEE-REPRO-CHECK)
  starlink_24    (real Starlink orbital geometry, zero-shot transfer)
  oneweb_24      (real OneWeb orbital geometry, zero-shot transfer)

Left  -- grouped delivery-ratio bars (MAPPO / Dijkstra / heuristic) with 95% CI.
Right -- MAPPO/Dijkstra throughput ratio across the three categories, with the
         1.0 oracle-parity line. Shows graceful transfer: 1.01 on the training
         distribution, ~0.90-0.92 zero-shot on real constellations.

Usage:
  python make_realism_figure.py \
      --realism experiments/IEEE-REALISM/aggregate_realism.csv \
      --headline experiments/IEEE-REPRO-CHECK/aggregate_metrics.csv \
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
    "mappo_realism_transfer": ("MAPPO (synthetic-trained)", "#1f4e79"),
    "mappo":                  ("MAPPO (synthetic-trained)", "#1f4e79"),
    "global_dijkstra":        ("Dijkstra (oracle)",         "#d62728"),
    "full_heuristic":         ("Queue-aware heuristic",     "#2ca02c"),
}
CATS = ["synthetic_24", "starlink_24", "oneweb_24"]
CAT_LABEL = {"synthetic_24": "Synthetic 4x6\n(training)", "starlink_24": "Starlink-24\n(real)",
             "oneweb_24": "OneWeb-24\n(real)"}


def load_realism(path):
    cells = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        cells[(r["topology"], r["policy"], r["metric"])] = (
            float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    return cells


def load_headline(path, scenario="medium_load"):
    cells = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["scenario"] != scenario:
            continue
        cells[(r["policy"], r["metric"])] = (
            float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    return cells


def get(cells_realism, cells_headline, cat, policy_key, metric):
    """Fetch (mean, lo_err, hi_err) for a category/policy/metric."""
    if cat == "synthetic_24":
        m, lo, hi = cells_headline[(policy_key, metric)]
    else:
        m, lo, hi = cells_realism[(cat, policy_key, metric)]
    return m, m - lo, hi - m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realism", default="experiments/IEEE-REALISM/aggregate_realism.csv")
    ap.add_argument("--headline", default="experiments/IEEE-REPRO-CHECK/aggregate_metrics.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cr = load_realism(args.realism)
    ch = load_headline(args.headline)

    # policy key mapping: synthetic uses "mappo"/"global_dijkstra"/"full_heuristic";
    # realism uses "mappo_realism_transfer"/"global_dijkstra"/"full_heuristic".
    pol_keys = [("mappo_realism_transfer", "mappo"), ("global_dijkstra", "global_dijkstra"),
                ("full_heuristic", "full_heuristic")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- left: grouped delivery bars ----
    x = np.arange(len(CATS))
    width = 0.26
    for i, (real_key, synth_key) in enumerate(pol_keys):
        means, loe, hie = [], [], []
        for cat in CATS:
            pk = real_key if cat != "synthetic_24" else synth_key
            m, lo, hi = get(cr, ch, cat, pk, "delivery_ratio")
            means.append(m); loe.append(lo); hie.append(hi)
        lbl, color = POL[real_key if real_key != "mappo_realism_transfer" else "mappo_realism_transfer"]
        ax1.bar(x + (i - 1) * width, means, width, yerr=[loe, hie], capsize=2.5,
                color=color, label=lbl, edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.9})
    ax1.set_xticks(x); ax1.set_xticklabels([CAT_LABEL[c] for c in CATS], fontsize=7.5)
    ax1.set_ylabel("Delivery ratio"); ax1.set_ylim(0, 0.85)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=7, loc="upper right", frameon=False)

    # ---- right: throughput ratio ----
    ratios = []
    for cat in CATS:
        pk_m = "mappo_realism_transfer" if cat != "synthetic_24" else "mappo"
        pk_d = "global_dijkstra"
        mt, _, _ = get(cr, ch, cat, pk_m, "throughput_packets_per_slot")
        dt, _, _ = get(cr, ch, cat, pk_d, "throughput_packets_per_slot")
        ratios.append(mt / max(dt, 1e-9))
    ax2.plot(x, ratios, marker="o", color="#1f4e79", linewidth=2, markersize=7, zorder=3)
    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="oracle parity")
    ax2.axhline(0.90, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    for xi, r in zip(x, ratios):
        ax2.annotate(f"{r:.2f}", (xi, r), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#1f4e79")
    ax2.set_xticks(x); ax2.set_xticklabels([CAT_LABEL[c] for c in CATS], fontsize=7.5)
    ax2.set_ylabel("Throughput ratio (MAPPO / Dijkstra)"); ax2.set_ylim(0.82, 1.08)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(fontsize=7, loc="lower left", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_realism_transfer.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_realism_transfer] {len(CATS)} categories -> {outdir}/fig_realism_transfer.{{png,pdf}}")


if __name__ == "__main__":
    main()
