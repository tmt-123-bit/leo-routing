"""ns-3 packet-level validation figure (STATIC topology, corrected).

Finding (after the partial-path harness fix, 2026-08-14): in the static-topology
continuous-time ns-3 replay, MAPPO's slot-sim advantage (medium_load env: MAPPO
0.783 > Dijkstra 0.719, +6.4pp) PERSISTS at every offered load (+7.2/+5.1/+4.3/
+2.6 pp at 1x/2x/4x/8x compression), shrinking toward saturation. The earlier
reported negative result ("Dijkstra edges ahead") was an artifact of counting
env-dropped partial-path packets as delivered at their mid-network end node —
Dijkstra, with more env drops, was inflated more. See make_ns3_dynamic_figure.py
for the dynamic-topology replay, where the env's +33pp frequent_break advantage
also transfers.

Panels:
  L: delivery vs offered load (ns-3, CI bands) + env reference (dashed).
  R: P95 delay vs offered load.

CIs are bootstrap (B=5000) over per-episode delivery (5 episodes).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    try:
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))
    except FileNotFoundError:
        return []


def bootstrap_ci(vals, B=5000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(B, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_episode_delivery(pol, ss, indir):
    """Per-episode delivery from ns-3 per-packet results. packet_id is a global id
    (gid = episode*1e6 + local_pktId), so episode = gid // 1e6 — no join needed."""
    res = load(indir / f"nspkt_{pol}_ss{ss}.csv")
    ep_del = {}
    for r in res:
        gid = int(r["packet_id"])
        ep = gid // 1000000
        delivered = int(r["delivered"])
        ep_del.setdefault(ep, [0, 0])
        ep_del[ep][1] += 1
        ep_del[ep][0] += delivered
    return [d / max(1, n) for (d, n) in ep_del.values()]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="experiments/IEEE-NS3")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    indir = Path(args.indir); outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    SLOTS = [("0.5", 1), ("0.25", 2), ("0.125", 4), ("0.0625", 8)]
    agg = load(indir / "load_sweep_bw36_q8.csv")
    p95 = {(r["policy"], r["slot_sec"]): float(r["p95_delay_ms"]) for r in agg}

    # env reference (per-episode delivery)
    env = {}
    for pol in ["mappo", "dijkstra"]:
        rows = load(indir / f"env_summary_{pol}.csv")
        env[pol] = bootstrap_ci([float(r["delivery_ratio"]) for r in rows])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    cols = {"mappo": "#1f77b4", "dijkstra": "#ff7f0e"}
    names = {"mappo": "MAPPO", "dijkstra": "Dijkstra"}

    loads = np.array([s[1] for s in SLOTS], dtype=float)
    for pol in ["mappo", "dijkstra"]:
        means, los, his = [], [], []
        for ss, _ in SLOTS:
            m, lo, hi = bootstrap_ci(per_episode_delivery(pol, ss, indir))
            means.append(m); los.append(lo); his.append(hi)
        means = np.array(means); los = np.array(los); his = np.array(his)
        axL.plot(loads, means, "-o", color=cols[pol], label=f"{names[pol]} (ns-3)", lw=1.6, ms=4)
        axL.fill_between(loads, los, his, color=cols[pol], alpha=0.18)
        # env reference (dashed horizontal)
        em, elo, ehi = env[pol]
        axL.axhline(em, color=cols[pol], ls="--", lw=1.1, alpha=0.7,
                    label=f"{names[pol]} (env, slot-sync)")
        # p95 delay
        ds = [p95.get((pol, ss), np.nan) for ss, _ in SLOTS]
        axR.plot(loads, ds, "-o", color=cols[pol], lw=1.6, ms=4)

    axL.set_xscale("log", base=2)
    axR.set_xscale("log", base=2)
    axL.set_xticks(loads); axL.set_xticklabels(["1×", "2×", "4×", "8×"])
    axR.set_xticks(loads); axR.set_xticklabels(["1×", "2×", "4×", "8×"])
    axL.set_xlabel("offered load (× medium_load)")
    axL.set_ylabel("delivery ratio")
    axR.set_xlabel("offered load (× medium_load)")
    axR.set_ylabel("P95 delay (ms)")
    axL.set_ylim(0.4, 1.02)
    axL.grid(True, ls=":", alpha=0.5); axR.grid(True, ls=":", alpha=0.5)
    axL.legend(fontsize=6.5, loc="lower left", ncol=2)
    fig.suptitle("ns-3 packet-level replay (static topology, 36 kb/s ISL, drop-tail q=8): "
                 "MAPPO's slot-model advantage persists", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_ns3_validation.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_ns3_validation] -> {outdir}/fig_ns3_validation.{{png,pdf}}")

    # honest text summary
    print("\n== ns-3 vs env delivery (bootstrap 95% CI over 5 episodes) ==")
    for pol in ["mappo", "dijkstra"]:
        em, elo, ehi = env[pol]
        print(f"  env   {pol:9s}: {em:.3f} [{elo:.3f}, {ehi:.3f}]")
    for ss, ld in SLOTS:
        line = f"  ns-3 @{ld}x:"
        for pol in ["mappo", "dijkstra"]:
            m, lo, hi = bootstrap_ci(per_episode_delivery(pol, ss, indir))
            line += f"  {names[pol]} {m:.3f}[{lo:.3f},{hi:.3f}]"
        print(line)


if __name__ == "__main__":
    main()
