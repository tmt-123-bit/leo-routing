"""ns-3 DYNAMIC-topology packet-level validation figure + stats.

Scenario: frequent_break (the env's real per-slot link break schedule is replayed
in ns-3: links go down per links_<policy>.csv; packets hitting a down link wait
and retry next slot; class deadlines are enforced).

Finding: MAPPO's env advantage TRANSFERS to packet-level replay when the
topology dynamics the policies conditioned on are replayed faithfully. Env
(frequent_break): MAPPO 0.766 vs Dijkstra 0.432 (+33.4 pp); ns-3 dynamic replay:
+33 pp at 1x/2x compression, +30 pp at 4x, +11 pp at 8x (both degrade under
congestion; the ordering never flips). This complements the corrected STATIC
replay (make_ns3_figure.py), where the (smaller) env advantage also persists.

Panels:
  L: delivery vs offered load (ns-3 dynamic, CI bands) + env reference (dashed).
  R: P95 delay vs offered load.

CIs are bootstrap (B=5000) over per-episode delivery; paired
Wilcoxon per load level; both written to ns3_dynamic_stats.csv.
"""
from __future__ import annotations
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
    res = load(indir / f"dynpkt_{pol}_ss{ss}.csv")
    ep_del = {}
    for r in res:
        gid = int(r["packet_id"])
        ep = gid // 1000000
        delivered = int(r["delivered"])
        ep_del.setdefault(ep, [0, 0])
        ep_del[ep][1] += 1
        ep_del[ep][0] += delivered
    return [d / max(1, n) for (_, (d, n)) in sorted(ep_del.items())]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="experiments/IEEE-NS3-DYN")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    indir = Path(args.indir); outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    SLOTS = [("1.0", 1), ("0.5", 2), ("0.25", 4), ("0.125", 8)]
    agg = load(indir / "dyn_sweep_bw36_q8.csv")
    p95 = {(r["policy"], r["slot_sec"]): float(r["p95_delay_ms"]) for r in agg}
    dl_drops = {(r["policy"], r["slot_sec"]): int(r["deadline_drops"]) for r in agg}
    waits = {(r["policy"], r["slot_sec"]): int(r["wait_events"]) for r in agg}

    # env reference (per-episode delivery, frequent_break)
    n_eps_env = len(load(indir / "env_summary_mappo.csv"))
    env = {}
    for pol in ["mappo", "dijkstra"]:
        rows = load(indir / f"env_summary_{pol}.csv")
        env[pol] = bootstrap_ci([float(r["delivery_ratio"]) for r in rows])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    cols = {"mappo": "#1f77b4", "dijkstra": "#ff7f0e"}
    names = {"mappo": "MAPPO", "dijkstra": "Dijkstra"}

    stats_rows = []
    loads = np.array([s[1] for s in SLOTS], dtype=float)
    for pol in ["mappo", "dijkstra"]:
        means, los, his = [], [], []
        for ss, _ in SLOTS:
            m, lo, hi = bootstrap_ci(per_episode_delivery(pol, ss, indir))
            means.append(m); los.append(lo); his.append(hi)
        means = np.array(means); los = np.array(los); his = np.array(his)
        axL.plot(loads, means, "-o", color=cols[pol], label=f"{names[pol]} (ns-3 dynamic)", lw=1.6, ms=4)
        axL.fill_between(loads, los, his, color=cols[pol], alpha=0.18)
        em, elo, ehi = env[pol]
        axL.axhline(em, color=cols[pol], ls="--", lw=1.1, alpha=0.7,
                    label=f"{names[pol]} (env, slot-sync)")
        ds = [p95.get((pol, ss), np.nan) for ss, _ in SLOTS]
        axR.plot(loads, ds, "-o", color=cols[pol], lw=1.6, ms=4)

    # stats: paired Wilcoxon per load level (same N workload episodes)
    for ss, ld in SLOTS:
        m_ep = per_episode_delivery("mappo", ss, indir)
        d_ep = per_episode_delivery("dijkstra", ss, indir)
        mm, mlo, mhi = bootstrap_ci(m_ep)
        dd, dlo, dhi = bootstrap_ci(d_ep)
        try:
            wp = float(wilcoxon(m_ep, d_ep).pvalue)
        except ValueError:
            wp = float("nan")
        stats_rows.append({
            "slot_sec": ss, "load_x": ld, "n_episodes": len(m_ep),
            "mappo_mean": round(mm, 4), "mappo_lo": round(mlo, 4), "mappo_hi": round(mhi, 4),
            "dijkstra_mean": round(dd, 4), "dijkstra_lo": round(dlo, 4), "dijkstra_hi": round(dhi, 4),
            "gap_pp": round(100 * (mm - dd), 2),
            "wilcoxon_p": f"{wp:.3e}",
            "mappo_deadline_drops": dl_drops.get(("mappo", ss), -1),
            "dijkstra_deadline_drops": dl_drops.get(("dijkstra", ss), -1),
            "mappo_wait_events": waits.get(("mappo", ss), -1),
            "dijkstra_wait_events": waits.get(("dijkstra", ss), -1),
        })
    for pol in ["mappo", "dijkstra"]:
        em, elo, ehi = env[pol]
        stats_rows.append({
            "slot_sec": "env", "load_x": 0, "n_episodes": n_eps_env,
            f"{pol}_mean": round(em, 4), f"{pol}_lo": round(elo, 4), f"{pol}_hi": round(ehi, 4),
        })
    with open(indir / "ns3_dynamic_stats.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["slot_sec", "load_x", "n_episodes",
                "mappo_mean", "mappo_lo", "mappo_hi",
                "dijkstra_mean", "dijkstra_lo", "dijkstra_hi",
                "gap_pp", "wilcoxon_p",
                "mappo_deadline_drops", "dijkstra_deadline_drops",
                "mappo_wait_events", "dijkstra_wait_events"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(stats_rows)

    axL.set_xscale("log", base=2)
    axR.set_xscale("log", base=2)
    axL.set_xticks(loads); axL.set_xticklabels(["1×", "2×", "4×", "8×"])
    axR.set_xticks(loads); axR.set_xticklabels(["1×", "2×", "4×", "8×"])
    axL.set_xlabel("offered load (× frequent_break)")
    axL.set_ylabel("delivery ratio")
    axR.set_xlabel("offered load (× frequent_break)")
    axR.set_ylabel("P95 delay (ms)")
    axL.set_ylim(0.1, 1.02)
    axL.grid(True, ls=":", alpha=0.5); axR.grid(True, ls=":", alpha=0.5)
    axL.legend(fontsize=6.5, loc="center left", ncol=2)
    fig.suptitle("ns-3 packet-level replay with the REAL break schedule (frequent_break, 36 kb/s ISL, "
                 "drop-tail q=8): MAPPO's advantage transfers", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_ns3_dynamic.{ext}", dpi=300)
    plt.close(fig)
    print(f"[fig_ns3_dynamic] -> {outdir}/fig_ns3_dynamic.{{png,pdf}}")
    print(f"[stats] -> {indir}/ns3_dynamic_stats.csv")

    print(f"\n== ns-3 dynamic replay vs env (bootstrap 95% CI over {n_eps_env} episodes) ==")
    for pol in ["mappo", "dijkstra"]:
        em, elo, ehi = env[pol]
        print(f"  env       {pol:9s}: {em:.3f} [{elo:.3f}, {ehi:.3f}]")
    for r in stats_rows:
        if r["slot_sec"] == "env":
            continue
        print(f"  ns-3 @{r['load_x']}x: MAPPO {r['mappo_mean']:.3f}[{r['mappo_lo']:.3f},{r['mappo_hi']:.3f}]"
              f"  Dij {r['dijkstra_mean']:.3f}[{r['dijkstra_lo']:.3f},{r['dijkstra_hi']:.3f}]"
              f"  gap {r['gap_pp']:+.1f}pp  Wilcoxon p={r['wilcoxon_p']}")


if __name__ == "__main__":
    main()
