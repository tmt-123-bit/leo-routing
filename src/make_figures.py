#!/usr/bin/env python3
"""
make_figures.py — publication figures + LaTeX tables from aggregate_metrics.csv.

Generates the IEEE paper's core figures (F1–F3) and Table I directly from the
hierarchical-bootstrap aggregate produced by run_exp004_mappo.py. Data-driven:
swap the --input directory when a fresh experiment finishes and every figure
regenerates. Uses ONLY column names that already exist in the schema:

    scenario, policy, metric, n, mean, std, ci95_low, ci95_high, ci_method

Figures
  F2  delivery by scenario, MAPPO vs every baseline, 95% CI error bars   (fig_delivery)
  F3  P95 delay + load-imbalance under stress scenarios                   (fig_tail_balance)
  F1  delivery-vs-budget crossover, needs >=2 --input dirs                (fig_budget)
  T1  main-results LaTeX table                                            (table_main)

Usage
  # single experiment (F2, F3, T1):
  python make_figures.py --input experiments/IEEE-EXP-004-FULL
  # budget sweep (add F1): labels must match --input order
  python make_figures.py -i experiments/IEEE-BUDGET-quick -i experiments/IEEE-BUDGET-x2k \
                         -i experiments/IEEE-BUDGET-x10k -i experiments/IEEE-EXP-004-FULL \
                         --budget-labels quick,x2k,x10k,full
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Display maps — keep names stable; missing policies/scenarios are skipped.
# ---------------------------------------------------------------------------
POLICY_DISPLAY = {
    "mappo": "MAPPO",
    "global_dijkstra": "Dijkstra (oracle)",
    "ospf_ecmp": "OSPF/ECMP",
    "q_routing": "Q-routing",
    "full_heuristic": "Heuristic",
    "delay_only": "Delay-only",
    "random": "Random",
}
# ordered left→right in grouped bars; MAPPO first, oracles next, then weak baselines
POLICY_ORDER = [
    "mappo", "global_dijkstra", "ospf_ecmp",
    "q_routing", "full_heuristic", "delay_only", "random",
]
# (color, hatch) — colorblind-safe + distinguishable in B/W print via hatch
POLICY_STYLE = {
    "mappo":            ("#1f4e79", ""),       # deep blue, solid — the protagonist
    "global_dijkstra":  ("#2ca02c", "///"),    # green oracle
    "ospf_ecmp":        ("#17becf", "\\\\\\"), # teal oracle
    "q_routing":        ("#ff7f0e", "..."),    # orange
    "full_heuristic":   ("#9467bd", "xxx"),    # purple
    "delay_only":       ("#8c564b", "---"),    # brown
    "random":           ("#999999", "++"),     # grey
}

SCENARIO_DISPLAY = {
    "low_load": "Low load",
    "medium_load": "Medium load",
    "hotspot_high_load": "Hotspot",
    "frequent_break": "Frequent break",
    "fault_links": "Fault links",
}
SCENARIO_ORDER = [
    "low_load", "medium_load", "hotspot_high_load", "frequent_break", "fault_links",
]
STRESS_SCENARIOS = ["hotspot_high_load", "frequent_break", "fault_links"]

METRIC_DISPLAY = {
    "delivery_ratio": "Delivery ratio",
    "drop_rate": "Drop rate",
    "p95_delay_slots": "P95 delay (slots)",
    "average_delay_slots": "Mean delay (slots)",
    "mean_queue_packets": "Mean queue (pkts)",
    "global_load_imbalance": "Load imbalance",
    "routing_switches": "Routing switches",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_aggregate(path: Path) -> dict:
    """Return {(scenario, policy, metric): {mean, ci_low, ci_high, n, std}}."""
    data = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            key = (r["scenario"], r["policy"], r["metric"])
            try:
                data[key] = {
                    "mean": float(r["mean"]),
                    "ci_low": float(r["ci95_low"]),
                    "ci_high": float(r["ci95_high"]),
                    "n": int(float(r["n"])),
                    "std": float(r.get("std", "nan") or "nan"),
                }
            except (KeyError, ValueError):
                continue
    return data


def scenarios_present(data) -> list:
    return [s for s in SCENARIO_ORDER if any(k[0] == s for k in data)]


def policies_present(data) -> list:
    return [p for p in POLICY_ORDER if any(k[1] == p for k in data)]


def _errbar_extent(data, scen, policy, metric):
    """Return (mean, [low_err, high_err]) or (nan, [0,0])."""
    rec = data.get((scen, policy, metric))
    if not rec:
        return np.nan, [0.0, 0.0]
    return rec["mean"], [rec["mean"] - rec["ci_low"], rec["ci_high"] - rec["mean"]]


# ---------------------------------------------------------------------------
# F2 — delivery by scenario (the main result figure)
# ---------------------------------------------------------------------------
def fig_delivery(data, outdir: Path):
    scens = scenarios_present(data)
    pols = policies_present(data)
    if not scens or not pols:
        print("  [fig_delivery] skip: no data")
        return
    n_s, n_p = len(scens), len(pols)
    width = 0.80 / n_p
    x = np.arange(n_s)

    fig, ax = plt.subplots(figsize=(min(0.55 * n_p * n_s + 3.5, 8.5), 3.6))
    for i, pol in enumerate(pols):
        means, lo, hi = [], [], []
        for s in scens:
            m, e = _errbar_extent(data, s, pol, "delivery_ratio")
            means.append(m)
            lo.append(e[0])
            hi.append(e[1])
        color, hatch = POLICY_STYLE.get(pol, ("#444444", ""))
        ax.bar(
            x + i * width, means, width, yerr=[lo, hi],
            capsize=2.2, color=color, hatch=hatch,
            edgecolor="black", linewidth=0.5,
            label=POLICY_DISPLAY.get(pol, pol),
        )
    ax.set_xticks(x + width * (n_p - 1) / 2)
    ax.set_xticklabels([SCENARIO_DISPLAY.get(s, s) for s in scens], fontsize=9)
    ax.set_ylabel(METRIC_DISPLAY["delivery_ratio"], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, ncol=min(n_p, 4), loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig2_delivery_by_scenario.{ext}", dpi=300)
    plt.close(fig)
    print(f"  [F2] fig2_delivery_by_scenario  ({n_s} scenarios x {n_p} policies)")


# ---------------------------------------------------------------------------
# F3 — P95 delay + load-imbalance under stress (tail & balance claim C5)
# ---------------------------------------------------------------------------
def fig_tail_balance(data, outdir: Path):
    scens = [s for s in STRESS_SCENARIOS if any(k[0] == s for k in data)]
    pols = [p for p in POLICY_ORDER if any(k[1] == p for k in data)]
    if not scens or not pols:
        print("  [fig_tail_balance] skip: no stress-scenario data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    width = 0.80 / len(pols)
    x = np.arange(len(scens))

    for ax, metric, ylabel, fmt in (
        (ax1, "p95_delay_slots", METRIC_DISPLAY["p95_delay_slots"], "{:.0f}"),
        (ax2, "global_load_imbalance", METRIC_DISPLAY["global_load_imbalance"], "{:.2f}"),
    ):
        for i, pol in enumerate(pols):
            means, lo, hi = [], [], []
            for s in scens:
                m, e = _errbar_extent(data, s, pol, metric)
                means.append(m); lo.append(e[0]); hi.append(e[1])
            color, hatch = POLICY_STYLE.get(pol, ("#444", ""))
            ax.bar(x + i * width, means, width, yerr=[lo, hi], capsize=2,
                   color=color, hatch=hatch, edgecolor="black", linewidth=0.5,
                   label=POLICY_DISPLAY.get(pol, pol))
        ax.set_xticks(x + width * (len(pols) - 1) / 2)
        ax.set_xticklabels([SCENARIO_DISPLAY.get(s, s) for s in scens], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    ax2.legend(fontsize=7, ncol=min(len(pols), 3), loc="upper center",
               bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig3_tail_and_balance.{ext}", dpi=300)
    plt.close(fig)
    print(f"  [F3] fig3_tail_and_balance  (P95 + imbalance, {len(scens)} stress scenarios)")


# ---------------------------------------------------------------------------
# F1 — delivery vs training budget (crossover figure, needs >=2 dirs)
# ---------------------------------------------------------------------------
def fig_budget(dirs_labels, outdir: Path):
    """dirs_labels: list of (label, aggregate_dict). Plots MAPPO delivery vs
    budget step-count, with baseline crossovers drawn from the largest budget."""
    series = [(lbl, d) for lbl, d in dirs_labels if d]
    if len(series) < 2:
        print("  [fig_budget] skip: need >=2 --input dirs for the crossover figure")
        return
    # step counts for the standard presets
    STEP_FOR = {"quick": 300, "x2k": 2000, "x10k": 10000, "full": 50000}

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # MAPPO line
    xs, ys, lo, hi = [], [], [], []
    for lbl, d in series:
        scens = scenarios_present(d)
        if not scens:
            continue
        # average MAPPO delivery across scenarios at this budget
        vals = [d.get((s, "mappo", "delivery_ratio")) for s in scens]
        vals = [v for v in vals if v]
        if not vals:
            continue
        m = np.mean([v["mean"] for v in vals])
        xs.append(STEP_FOR.get(lbl, np.nan))
        ys.append(m)
        lo.append(m - np.mean([v["mean"] - v["ci_low"] for v in vals]))
        hi.append(m + np.mean([v["ci_high"] - v["mean"] for v in vals]))
    if xs:
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        lo = np.array(lo)[order]; hi = np.array(hi)[order]
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", color="#1f4e79",
                    linewidth=2, capsize=3, label="MAPPO")
    # baseline crossovers from the largest-budget dir
    biggest = max(series, key=lambda t: STEP_FOR.get(t[0], 0))[1]
    for pol in ("global_dijkstra", "ospf_ecmp", "q_routing", "full_heuristic"):
        recs = [biggest.get((s, pol, "delivery_ratio")) for s in scenarios_present(biggest)]
        recs = [r for r in recs if r]
        if not recs:
            continue
        m = np.mean([r["mean"] for r in recs])
        color, _ = POLICY_STYLE.get(pol, ("#444", ""))
        ax.axhline(m, color=color, linestyle="--", linewidth=1.2,
                   label=f"{POLICY_DISPLAY.get(pol, pol)} (full-budget)")
    ax.set_xscale("log")
    ax.set_xlabel("Training steps", fontsize=10)
    ax.set_ylabel("Delivery ratio (avg over scenarios)", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig1_delivery_vs_budget.{ext}", dpi=300)
    plt.close(fig)
    print(f"  [F1] fig1_delivery_vs_budget  ({len(series)} budget points)")


# ---------------------------------------------------------------------------
# T1 — main results LaTeX table (delivery, drop, p95, queue)
# ---------------------------------------------------------------------------
def _fmt(rec, pct=False):
    if not rec:
        return "—"
    m = rec["mean"]
    if pct:
        return f"{m*100:.1f}"
    return f"{m:.2f}"


def table_main(data, outdir: Path):
    scens = scenarios_present(data)
    pols = policies_present(data)
    if not scens or not pols:
        print("  [table_main] skip: no data")
        return
    cols = [("delivery_ratio", True), ("drop_rate", True),
            ("p95_delay_slots", False), ("mean_queue_packets", False)]
    path = outdir / "table1_main_results.tex"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by make_figures.py — do not edit by hand.\n")
        fh.write("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\hline\n")
        fh.write("Scenario & Policy & " + " & ".join(
            METRIC_DISPLAY.get(m, m).replace(" (slots)", "").replace(" (pkts)", "")
            for m, _ in cols) + " \\\\\n\\hline\n")
        for s in scens:
            first = True
            for pol in pols:
                cells = [_fmt(data.get((s, pol, m)), pct=pct) for m, pct in cols]
                if all(c == "—" for c in cells):
                    continue
                sname = SCENARIO_DISPLAY.get(s, s) if first else ""
                pname = POLICY_DISPLAY.get(pol, pol)
                if pol == "mappo":
                    pname = "\\textbf{" + pname + "}"
                fh.write(f"{sname} & {pname} & " + " & ".join(cells) + " \\\\\n")
                first = False
            if not first:
                fh.write("\\hline\n")
        fh.write("\\end{tabular}\n")
    print(f"  [T1] table1_main_results.tex  ({len(scens)} scenarios x {len(pols)} policies)")


# ---------------------------------------------------------------------------
# F4 — ablation effect on delivery (needs paired_ablation_effects.csv)
# ---------------------------------------------------------------------------
VARIANT_DISPLAY = {
    "no_credit": "no credit", "no_ppo_protection": "no PPO protect.",
    "flat_critic": "flat critic", "no_queue": "no queue",
    "no_lifetime": "no lifetime", "no_packet_context": "no pkt context",
}
VARIANT_ORDER = ["no_credit", "no_ppo_protection", "flat_critic",
                 "no_queue", "no_lifetime", "no_packet_context"]
VARIANT_COLOR = {"no_credit": "#d62728", "no_ppo_protection": "#9467bd",
                 "flat_critic": "#2ca02c", "no_queue": "#ff7f0e",
                 "no_lifetime": "#1f77b4", "no_packet_context": "#7f7f7f"}


def fig_ablation(effects_csv: Path, outdir: Path):
    """Delivery change (pp) from ablating each component vs full. Negative = the
    component is useful (removing it hurts). Star = Benjamini-Hochberg p<0.05."""
    import csv as _csv
    rows = list(_csv.DictReader(open(effects_csv, encoding="utf-8-sig")))
    scens = [s for s in SCENARIO_ORDER if any(r["scenario"] == s for r in rows)]
    variants = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in rows)]
    if not scens or not variants:
        print("  [fig_ablation] skip: no data")
        return

    def get(s, v):
        for r in rows:
            if r["scenario"] == s and r["variant"] == v and r["metric"] == "delivery_ratio":
                return float(r["ablated_minus_full"]) * 100, float(r["benjamini_hochberg_p"])
        return None, None

    width = 0.8 / len(variants)
    x = np.arange(len(scens))
    fig, ax = plt.subplots(figsize=(min(0.7 * len(variants) * len(scens) + 3, 8.5), 4.0))
    for i, v in enumerate(variants):
        vals, sigs = [], []
        for s in scens:
            d, p = get(s, v)
            vals.append(d if d is not None else 0.0)
            sigs.append(p is not None and p < 0.05)
        ax.bar(x + i * width, vals, width, color=VARIANT_COLOR.get(v, "#444"),
               edgecolor="black", linewidth=0.4, label=VARIANT_DISPLAY.get(v, v))
        for j, sg in enumerate(sigs):
            if sg:
                yv = vals[j]
                ax.text(x[j] + i * width + width / 2, yv + (0.5 if yv >= 0 else -0.5), "*",
                        ha="center", va="bottom" if yv >= 0 else "top",
                        fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([SCENARIO_DISPLAY.get(s, s) for s in scens], fontsize=9)
    ax.set_ylabel("Delivery change vs full MAPPO (pp)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, ncol=min(len(variants), 3), loc="best", frameon=False)
    fig.text(0.5, 0.005,
             "Negative (below 0) = removing the component hurts delivery = it is useful.  * = BH p<0.05",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig4_ablation_effect.{ext}", dpi=300)
    plt.close(fig)
    print(f"  [F4] fig4_ablation_effect  ({len(variants)} variants x {len(scens)} scenarios)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", action="append", default=[], required=True,
                    help="experiment dir containing aggregate_metrics.csv (repeatable)")
    ap.add_argument("--budget-labels",
                    help="comma-separated labels matching --input order, e.g. quick,x2k,x10k,full")
    ap.add_argument("--outdir", default="figures", help="output dir (default: figures)")
    ap.add_argument("--ablation", help="ablation dir with paired_ablation_effects.csv (adds F4)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for d in args.input:
        agg = Path(d) / "aggregate_metrics.csv"
        if not agg.exists():
            print(f"WARNING: {agg} not found, skipping", file=sys.stderr)
            loaded.append((Path(d).name, None))
            continue
        loaded.append((Path(d).name, load_aggregate(agg)))

    # primary = the last (usually full-budget) dir with data
    primary = next((data for _, data in reversed(loaded) if data), None)
    if primary is None:
        print("ERROR: no aggregate_metrics.csv could be loaded", file=sys.stderr)
        sys.exit(2)

    print(f"### figures from {len([d for _,d in loaded if d])} experiment dir(s) -> {outdir}")
    fig_delivery(primary, outdir)
    fig_tail_balance(primary, outdir)
    table_main(primary, outdir)

    # F4 ablation effect (optional, needs the ablation experiment dir)
    if args.ablation:
        eff = Path(args.ablation) / "paired_ablation_effects.csv"
        if eff.exists():
            fig_ablation(eff, outdir)
        else:
            print(f"WARNING: {eff} not found, skipping F4", file=sys.stderr)

    # F1 budget crossover: needs labelled multi-dir input
    if args.budget_labels and len([d for _, d in loaded if d]) >= 2:
        labels = [lbl.strip() for lbl in args.budget_labels.split(",")]
        if len(labels) != len(loaded):
            print(f"WARNING: {len(labels)} labels vs {len(loaded)} dirs; "
                  f"using dir-name labels for F1", file=sys.stderr)
            labels = [name for name, _ in loaded]
        fig_budget(list(zip(labels, [d for _, d in loaded])), outdir)

    print("### done.")


if __name__ == "__main__":
    main()
