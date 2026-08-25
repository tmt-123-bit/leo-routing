"""CLOSED-LOOP ns-3 validation figure + stats.

Unlike replay (paths precomputed in the slot env), in the closed loop the
policies decide INSIDE the ns-3 event loop: each slot boundary ns-3 ships its
own queue/link state to the policy server, which rebuilds the env's candidate
features from that state and returns next-hop decisions executed on the real
FIFO data plane. The queues/bandwidth/contention the policy conditions on are
therefore ns-3's OWN, closing the loop.

Panels:
  L: per-episode delivery (dots) + bootstrap CI (error bars), MAPPO vs Dijkstra.
  R: delivered-packet delay CDF.

Stats -> experiments/<indir>/closedloop_stats.csv (bootstrap 5000, paired
Wilcoxon over episodes).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    try:
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))
    except FileNotFoundError:
        return []


def per_episode(path):
    ep = {}
    for r in load(path):
        e = int(r["packet_id"]) // 1000000
        ep.setdefault(e, [0, 0])
        ep[e][1] += 1
        ep[e][0] += int(r["delivered"])
    return [d / max(1, n) for _, (d, n) in sorted(ep.items())]


def delays(path):
    return np.asarray([float(r["delay_ms"]) for r in load(path)
                       if r["delivered"] == "1"])


def bootstrap_ci(vals, B=5000, seed=0):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(B, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="experiments/ns3-closedloop")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    m_ep = per_episode(indir / "clpkt_mappo.csv")
    d_ep = per_episode(indir / "clpkt_dijkstra.csv")
    m_del, d_del = delays(indir / "clpkt_mappo.csv"), delays(indir / "clpkt_dijkstra.csv")
    mm, mlo, mhi = bootstrap_ci(m_ep)
    dd, dlo, dhi = bootstrap_ci(d_ep)
    try:
        p = float(wilcoxon(m_ep, d_ep).pvalue)
    except ValueError:
        p = float("nan")

    with open(indir / "closedloop_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "policy", "n_episodes", "mean", "ci95_low", "ci95_high",
            "mean_delay_ms", "p95_delay_ms"])
        w.writeheader()
        for name, ep, dl in (("mappo", m_ep, m_del), ("dijkstra", d_ep, d_del)):
            mean, lo, hi = bootstrap_ci(ep)
            w.writerow({"policy": name, "n_episodes": len(ep),
                        "mean": round(mean, 4), "ci95_low": round(lo, 4),
                        "ci95_high": round(hi, 4),
                        "mean_delay_ms": round(float(dl.mean()), 1),
                        "p95_delay_ms": round(float(np.percentile(dl, 95)), 1)})
        w.writerow({"policy": "paired_gap_pp",
                    "mean": round(100 * (mm - dd), 2)})
        w.writerow({"policy": "wilcoxon_p", "mean": f"{p:.3e}"})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    cols = {"mappo": "#1f77b4", "dijkstra": "#ff7f0e"}
    names = {"mappo": "MAPPO", "dijkstra": "Dijkstra"}

    x = {"mappo": 0.0, "dijkstra": 1.0}
    for pol, ep in (("mappo", m_ep), ("dijkstra", d_ep)):
        c = cols[pol]
        axL.scatter([x[pol]] * len(ep) + np.random.default_rng(1).uniform(-0.13, 0.13, len(ep)),
                    ep, s=14, color=c, alpha=0.55, edgecolors="none", zorder=2)
        mean, lo, hi = bootstrap_ci(ep)
        axL.errorbar([x[pol]], [mean], yerr=[[mean - lo], [hi - mean]],
                     color=c, capsize=5, lw=2, zorder=3)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["MAPPO", "Dijkstra"])
    axL.set_xlim(-0.6, 1.6)
    axL.set_ylabel("delivery ratio (per episode)")
    axL.grid(True, ls=":", alpha=0.5)

    for pol, dl in (("mappo", m_del), ("dijkstra", d_del)):
        cdf = np.searchsorted(np.sort(dl), np.linspace(dl.min(), dl.max(), 300), side="right") / len(dl)
        axR.plot(np.linspace(dl.min(), dl.max(), 300), cdf, color=c, lw=1.7,
                 label=f"{names[pol]} (mean {dl.mean()/1000:.1f}s)")
    axR.set_xlabel("delivered-packet delay (ms)")
    axR.set_ylabel("CDF")
    axR.grid(True, ls=":", alpha=0.5)
    axR.legend(fontsize=8)

    fig.suptitle("CLOSED-LOOP ns-3: policies decide inside the event loop on ns-3's own "
                 f"queue state ({len(m_ep)} episodes, medium_load) — MAPPO +{100*(mm-dd):.1f} pp, "
                 f"p={p:.1e}", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_ns3_closedloop.{ext}", dpi=300)
    plt.close(fig)

    print(f"[fig_ns3_closedloop] -> {outdir}/fig_ns3_closedloop.{{png,pdf}}")
    print(f"[stats] -> {indir}/closedloop_stats.csv")
    print(f"MAPPO {mm:.4f} [{mlo:.4f}, {mhi:.4f}]  Dijkstra {dd:.4f} [{dlo:.4f}, {dhi:.4f}]"
          f"  gap {100*(mm-dd):+.2f}pp  Wilcoxon p={p:.3e}")


if __name__ == "__main__":
    main()
