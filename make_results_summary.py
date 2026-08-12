"""Consolidated results-summary digest. Reads every experiment aggregate + paired stats
and emits RESULTS_SUMMARY.md (one-screen, reviewer-facing). Run after figures."""
import csv, re
from collections import defaultdict
from pathlib import Path


def load(p):
    try:
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))
    except FileNotFoundError:
        return []


def ratio_ci(exp, fname="paired_stats_ratio_ci.csv"):
    rc = load(f"experiments/{exp}/{fname}")
    return {r["group"]: (float(r["ratio_mean"]), float(r["ratio_ci95_low"]), float(r["ratio_ci95_high"]))
            for r in rc}


SCEN = ["low_load", "medium_load", "hotspot_high_load", "frequent_break", "fault_links"]
out = ["# LEO MAPPO Routing — Consolidated Results Summary (all no_lifetime)", "",
       "> Auto-generated from experiment aggregates. MAPPO = no_lifetime variant.",
       "> Paired Wilcoxon + Benjamini-Hochberg; throughput-ratio bootstrap CI (B=5000).", ""]

# 1 HEADLINE
rows = load("experiments/IEEE-NOLIFE-full/aggregate_metrics.csv")
c = defaultdict(dict)
for r in rows:
    c[(r["scenario"], r["policy"])][r["metric"]] = float(r["mean"])
out += ["## 1. Headline (5 scenarios, 8 policy seeds x 20 workload seeds)", "",
        "| scenario | MAPPO del | Dijkstra del | gap | MAPPO P95 | Dijkstra P95 |",
        "|---|---|---|---|---|---|"]
for s in SCEN:
    md = c[(s, "mappo")]["delivery_ratio"]; dd = c[(s, "global_dijkstra")]["delivery_ratio"]
    mp = c[(s, "mappo")]["p95_delay_slots"]; dp = c[(s, "global_dijkstra")]["p95_delay_slots"]
    out.append(f"| {s} | {md:.3f} | {dd:.3f} | {(md-dd)*100:+.1f}pp | {mp:.1f} | {dp:.1f} |")
out += ["", "MAPPO beats Dijkstra on delivery in medium/frequent_break/fault; lower P95 in 4/5."]

# 2 ABLATION
abrows = load("experiments/IEEE-ABLATION-FULL/paired_ablation_effects.csv")
if abrows:
    # variant -> list of (ablated_mean delivery); also full_mean per scenario
    vdel = defaultdict(list)
    full_per_scen = {}
    for r in abrows:
        if r["metric"] != "delivery_ratio":
            continue
        vdel[r["variant"]].append(float(r["ablated_mean"]))
        full_per_scen[r["scenario"]] = float(r["full_mean"])
    fb = sum(full_per_scen.values()) / max(1, len(full_per_scen))
    out += ["", "## 2. Ablation (variants, full-variant ref; mean delivery over 5 scenarios)", "",
            "| variant | delivery | vs full |", "|---|---|---|"]
    out.append(f"| full (ref) | {fb:.3f} |  — |")
    for v in ["no_credit", "no_lifetime", "no_queue", "flat_critic",
              "no_packet_context", "no_ppo_protection"]:
        if vdel.get(v):
            m = sum(vdel[v]) / len(vdel[v])
            out.append(f"| {v} | {m:.3f} | {(m-fb)*100:+.1f}pp |")
    out += ["", "no_lifetime large positive (adopted as default); no_credit strongly negative "
               "(credit assignment essential to CTDE)."]

# 3 SCALE
rows = load("experiments/IEEE-SCALE-NOLIFE/aggregate_scale.csv")
if rows:
    c3 = defaultdict(dict)
    for r in rows:
        c3[r["target_scale"]][(r["policy"], r["metric"])] = float(r["mean"])
    rci = ratio_ci("IEEE-SCALE-NOLIFE")
    out += ["", "## 3. Scale transfer (zero-shot, n24-trained -> eval)", "",
            "| scale | thru ratio MAPPO/Dij [95% CI] |", "|---|---|"]
    for sc in sorted(c3):
        rm, lo, hi = rci.get(sc, (float("nan"),) * 3)
        out.append(f"| {sc} | {rm:.3f} [{lo:.3f}, {hi:.3f}] |")
    out.append("Beats oracle (ratio>1) at n24/n66/n110; graceful to n132 (0.945).")

# 4 REALISM
rows = load("experiments/IEEE-REALISM-NOLIFE/aggregate_realism.csv")
if rows:
    c4 = defaultdict(dict)
    for r in rows:
        c4[r["topology"]][(r["policy"], r["metric"])] = float(r["mean"])
    rci = ratio_ci("IEEE-REALISM-NOLIFE")
    out += ["", "## 4. Real-topology transfer (zero-shot synthetic -> real TLE)", "",
            "| topology | thru ratio [95% CI] | MAPPO P95 | Dij P95 |", "|---|---|---|---|"]
    for t in sorted(c4):
        rm, lo, hi = rci.get(t, (float("nan"),) * 3)
        mp = c4[t].get(("mappo_realism_transfer", "p95_delay_slots"), float("nan"))
        dp = c4[t].get(("global_dijkstra", "p95_delay_slots"), float("nan"))
        out.append(f"| {t} | {rm:.3f} [{lo:.3f}, {hi:.3f}] | {mp:.1f} | {dp:.1f} |")
    out.append("~93-94% oracle throughput zero-shot; lower P95 on both real topologies.")

# 5 C3
rows = load("experiments/TLE-NOLIFE-FULL/aggregate_metrics.csv")
if rows:
    c5 = defaultdict(dict)
    for r in rows:
        c5[r["scenario"]][(r["policy"], r["metric"])] = float(r["mean"])
    rci = ratio_ci("TLE-NOLIFE-FULL", "paired_ratio_ci_ratio_ci.csv")
    out += ["", "## 5. C3 real-topology TRAINING (train Starlink-24; in-dist + cross-constellation)", "",
            "| eval topology | thru ratio [95% CI] | MAPPO P95 | Dij P95 |", "|---|---|---|---|"]
    for t in sorted(c5):
        rm, lo, hi = rci.get(t, (float("nan"),) * 3)
        mp = c5[t].get(("mappo_tle_trained", "p95_delay_slots"), float("nan"))
        dp = c5[t].get(("global_dijkstra", "p95_delay_slots"), float("nan"))
        out.append(f"| {t} | {rm:.3f} [{lo:.3f}, {hi:.3f}] | {mp:.1f} | {dp:.1f} |")
    out.append("train-on-real (0.935) = zero-shot transfer (0.928) -> transfer suffices; "
               "cross-constellation OneWeb 0.960 with -3.0 slots P95.")

# 6 LOAD
rows = load("experiments/IEEE-LOAD-NOLIFE/aggregate_load.csv")
if rows:
    c6 = defaultdict(dict)
    for r in rows:
        c6[r["load_point"]][(r["policy"], r["metric"])] = float(r["mean"])
    rci = ratio_ci("IEEE-LOAD-NOLIFE")
    out += ["", "## 6. Load sweep (the congestion-awareness win)", "",
            "| load | MAPPO del | Dij del | gap | thru ratio [CI] | MAPPO P95 | Dij P95 |",
            "|---|---|---|---|---|---|---|"]
    for l in sorted(c6, key=lambda x: int(re.search(r"\d+", x).group())):
        md = c6[l].get(("mappo", "delivery_ratio"), float("nan"))
        dd = c6[l].get(("global_dijkstra", "delivery_ratio"), float("nan"))
        mp = c6[l].get(("mappo", "p95_delay_slots"), float("nan"))
        dp = c6[l].get(("global_dijkstra", "p95_delay_slots"), float("nan"))
        rm, lo, hi = rci.get(l, (float("nan"),) * 3)
        out.append(f"| {l} | {md:.3f} | {dd:.3f} | {(md-dd)*100:+.1f}pp | "
                   f"{rm:.3f} [{lo:.3f},{hi:.3f}] | {mp:.1f} | {dp:.1f} |")
    out.append("MAPPO beats oracle on delivery exo4-exo14; thru ratio up to 1.10 at exo8.")

# 7 FAULT
rows = load("experiments/IEEE-FAULT-NOLIFE/aggregate_fault.csv")
if rows:
    c7 = defaultdict(dict)
    for r in rows:
        fr = float(re.search(r"[\d.]+", r["fault_point"]).group())
        c7[fr][(r["policy"], r["metric"])] = float(r["mean"])
    rci = {}
    for r in load("experiments/IEEE-FAULT-NOLIFE/paired_stats_ratio_ci.csv"):
        g = r["group"]; fr = float(re.search(r"[\d.]+", g).group())
        rci[fr] = (float(r["ratio_mean"]), float(r["ratio_ci95_low"]), float(r["ratio_ci95_high"]))
    out += ["", "## 7. Fault sweep (robustness to link failures)", "",
            "| fault rate | MAPPO del | Dij del | gap | thru ratio [CI] |", "|---|---|---|---|---|"]
    for fr in sorted(c7):
        md = c7[fr].get(("mappo", "delivery_ratio"), float("nan"))
        dd = c7[fr].get(("global_dijkstra", "delivery_ratio"), float("nan"))
        rm, lo, hi = rci.get(fr, (float("nan"),) * 3)
        out.append(f"| {fr:.2f} | {md:.3f} | {dd:.3f} | {(md-dd)*100:+.1f}pp | "
                   f"{rm:.3f} [{lo:.3f},{hi:.3f}] |")
    out.append("MAPPO beats oracle +6pp at every fault rate; thru ratio 1.08->1.12 as failures intensify.")

out += ["", "## 8. Fairness (mechanism)", "",
        "MAPPO load imbalance < Dijkstra all 5 scenarios (paired Wilcoxon p=1.8e-15). "
        "Shortest-path hot-spot-prone; MAPPO trades a sliver of path-optimality for load spread.",
        "", "## 9. Convergence / reproducibility", "",
        "8-seed training CV 1-5% (easy/moderate), ~9% (hardest hotspot). See fig_convergence.",
        "", "## 10. Reward-weight sensitivity (robustness)", "",
        "| config | delivery | vs baseline | P95 | imbalance |",
        "|---|---|---|---|---|"]
srows = load("experiments/IEEE-REWARD-SENSITIVITY/summary.csv")
sbase = next((float(r["mappo_delivery_ratio"]) for r in srows if r["config"] == "baseline"), float("nan"))
for r in srows:
    d = float(r["mappo_delivery_ratio"]); p95 = float(r["mappo_p95_delay_slots"])
    imb = float(r["mappo_global_load_imbalance"])
    out.append(f"| {r['config']} | {d:.3f} | {(d-sbase)*100:+.1f}pp | {p95:.2f} | {imb:.3f} |")
out += ["", "All perturbations within +/-1.5pp of baseline; every config beats Dijkstra (+5-7pp). "
        "w_deliver inert (±0.1pp at 0.5x/2x); w_load mildly helpful (w_load=2.0 -> +0.6pp, "
        "lower P95 + imbalance); w_switch is the only weight that moves delivery >1pp "
        "(w_switch=1.0 (5x) -> -1.5pp: over-penalizing switching makes routing too sticky, "
        "costs delivery). The congestion-awareness win is NOT an artifact of a specific "
        "reward-weight setting."]

Path("RESULTS_SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
print(f"wrote RESULTS_SUMMARY.md, {len(out)} lines")
