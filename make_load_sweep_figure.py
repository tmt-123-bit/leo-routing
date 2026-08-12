"""Load-sweep figure: performance vs offered load (the congestion-awareness case).

Reads experiments/IEEE-LOAD/aggregate_load.csv and produces a two-panel figure:
  Left  -- delivery ratio vs offered load (exogenous packets/slot) for
           MAPPO / Dijkstra / heuristic, with 95% CI bands. Shows the operating
           regime (MAPPO tracks the oracle) and the capacity wall (all collapse
           under extreme overload), while the distributed heuristic collapses
           far earlier.
  Right -- P95 delay vs offered load. The congestion-awareness payoff: at
           moderate load MAPPO holds a markedly lower tail delay than
           shortest-path Dijkstra (fewer hot-spot queues), i.e. a Pareto
           improvement on the centralized oracle with local information only.

Usage:
  python make_load_sweep_figure.py --input experiments/IEEE-LOAD/aggregate_load.csv \
                                   --outdir figures
"""
from __future__ import annotations
import argparse, csv, re
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


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    cells = {}
    for r in rows:
        exo = int(re.search(r"\d+", r["load_point"]).group())
        cells[(exo, r["policy"], r["metric"])] = (
            exo, float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    exos = sorted({k[0] for k in cells})
    return cells, exos


def series(cells, exos, policy, metric):
    xs, ys, loe, hie = [], [], [], []
    for e in exos:
        if (e, policy, metric) not in cells:
            continue
        _, m, lo, hi = cells[(e, policy, metric)]
        xs.append(e); ys.append(m); loe.append(m - lo); hie.append(hi - m)
    return xs, ys, loe, hie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/IEEE-LOAD/aggregate_load.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cells, exos = load(args.input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    # left: delivery
    for pol, (lbl, color, marker, ls) in POL.items():
        xs, ys, loe, hie = series(cells, exos, pol, "delivery_ratio")
        ax1.errorbar(xs, ys, yerr=[loe, hie], marker=marker, color=color, linestyle=ls,
                     linewidth=1.8, capsize=3, markersize=5, label=lbl)
    ax1.set_xlabel("Offered load (exogenous pkts/slot)"); ax1.set_ylabel("Delivery ratio")
    ax1.set_ylim(0, 0.85); ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=7, loc="upper right", frameon=False)

    # right: P95 delay (tail latency — the congestion-awareness payoff)
    for pol, (lbl, color, marker, ls) in POL.items():
        xs, ys, loe, hie = series(cells, exos, pol, "p95_delay_slots")
        ax2.errorbar(xs, ys, yerr=[loe, hie], marker=marker, color=color, linestyle=ls,
                     linewidth=1.8, capsize=3, markersize=5, label=lbl)
    ax2.set_xlabel("Offered load (exogenous pkts/slot)"); ax2.set_ylabel("P95 delay (slots)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=7, loc="upper left", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_load_sweep.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_load_sweep] {len(exos)} load points -> {outdir}/fig_load_sweep.{{png,pdf}}")


if __name__ == "__main__":
    main()
