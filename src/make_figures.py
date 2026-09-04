#!/usr/bin/env python3
"""
make_figures.py — publication figures + LaTeX tables from aggregate_metrics.csv.

Generates the core figures (F1–F3) and Table I directly from the
hierarchical-bootstrap aggregate produced by run_exp004_mappo.py. Data-driven:
swap the --input directory when a fresh experiment finishes and every figure
regenerates. F4 additionally consumes the complete confirmatory
paired_ablation_effects.csv. The main-result inputs use this schema:

    scenario, policy, metric, n, mean, std, ci95_low, ci95_high, ci_method

Figures
  F2  delivery by scenario, MAPPO vs every baseline, 95% CI error bars   (fig_delivery)
  F3  P95 delay + load-imbalance under stress scenarios                   (fig_tail_balance)
  F1  delivery-vs-budget crossover, needs >=2 --input dirs                (fig_budget)
  F4  planned delivery ablations, treatment-reference effects + 95% CI    (fig_ablation)
  T1  main-results LaTeX table                                            (table_main)

Usage
  # single experiment (F2, F3, T1):
  python make_figures.py --input experiments/archive/EXP-004-FULL
  # budget sweep (add F1): labels must match --input order
  python make_figures.py -i experiments/qrouting-budget-quick -i experiments/qrouting-budget-2k \
                         -i experiments/qrouting-budget-10k -i experiments/archive/EXP-004-FULL \
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

try:
    from variant_definitions import PLANNED_CONTRASTS
except ImportError:
    # Support both direct execution and ``python -m src.make_figures``.
    from .variant_definitions import PLANNED_CONTRASTS

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
# F4 — confirmatory delivery effects (needs paired_ablation_effects.csv)
# ---------------------------------------------------------------------------
PRIMARY_ABLATION_METRIC = "delivery_ratio"
PRIMARY_HOLM_FIELD = "confirmatory_holm_within_metric_p"
PRIMARY_ABLATION_FAMILY_SIZE = len(PLANNED_CONTRASTS) * len(SCENARIO_ORDER)

CONTRAST_DISPLAY = {
    "remove_queue_mechanism_package": "Remove queue mechanism",
    "remove_centered_local_credit": "Remove centered credit",
    "remove_packet_context": "Remove packet context",
    "replace_graph_critic_with_flat_critic": "Use flat critic",
    # Keep the display stable across the protocol-ID spelling correction.
    "remove_ppo_protection_package": "Remove PPO protection",
    "add_lifetime_feature": "Add lifetime feature (L1 - L0)",
    "add_lifetime_reward": "Add lifetime reward (L2 - L1)",
    "add_hard_lifetime_mask": "Add hard mask (L3 - L2)",
}

CONTRAST_STYLE = {
    "component_removal": ("#1f4e79", "o"),
    "training_safeguard_package": ("#7a5195", "s"),
    "lifetime_ladder": ("#d95f02", "D"),
}


def _finite_float(row, field, *, cell):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field!r} for ablation cell {cell!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"non-finite {field!r} for ablation cell {cell!r}")
    return value


def load_confirmatory_delivery_effects(effects_csv: Path) -> dict:
    """Load the frozen 8 x 5 primary ablation family without schema fallbacks."""

    required = {
        "scenario",
        "contrast",
        "contrast_family",
        "reference_variant",
        "treatment_variant",
        "metric",
        "treatment_minus_reference",
        "difference_ci95_low",
        "difference_ci95_high",
        PRIMARY_HOLM_FIELD,
        "within_metric_family_size",
        "multiplicity_role",
    }
    with open(effects_csv, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "paired ablation CSV is not the confirmatory schema; missing "
                + ", ".join(sorted(missing_columns))
            )
        delivery_rows = [
            row for row in reader if row["metric"] == PRIMARY_ABLATION_METRIC
        ]

    contrast_by_name = {contrast.name: contrast for contrast in PLANNED_CONTRASTS}
    expected_cells = {
        (scenario, contrast.name)
        for scenario in SCENARIO_ORDER
        for contrast in PLANNED_CONTRASTS
    }
    records = {}
    for row in delivery_rows:
        cell = (row["scenario"], row["contrast"])
        if cell not in expected_cells:
            raise ValueError(f"unexpected primary ablation cell {cell!r}")
        if cell in records:
            raise ValueError(f"duplicate primary ablation cell {cell!r}")

        contrast = contrast_by_name[row["contrast"]]
        if row["reference_variant"] != contrast.reference:
            raise ValueError(
                f"wrong reference for {cell!r}: {row['reference_variant']!r}; "
                f"expected {contrast.reference!r}"
            )
        if row["treatment_variant"] != contrast.treatment:
            raise ValueError(
                f"wrong treatment for {cell!r}: {row['treatment_variant']!r}; "
                f"expected {contrast.treatment!r}"
            )
        if row["contrast_family"] != contrast.family:
            raise ValueError(
                f"wrong contrast family for {cell!r}: {row['contrast_family']!r}; "
                f"expected {contrast.family!r}"
            )
        if row["multiplicity_role"] != "primary_confirmatory_family":
            raise ValueError(f"primary delivery cell {cell!r} has the wrong role")
        try:
            family_size = int(row["within_metric_family_size"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid Holm family size for {cell!r}") from error
        if family_size != PRIMARY_ABLATION_FAMILY_SIZE:
            raise ValueError(
                f"wrong Holm family size for {cell!r}: {family_size}; "
                f"expected {PRIMARY_ABLATION_FAMILY_SIZE}"
            )

        effect = _finite_float(row, "treatment_minus_reference", cell=cell)
        ci_low = _finite_float(row, "difference_ci95_low", cell=cell)
        ci_high = _finite_float(row, "difference_ci95_high", cell=cell)
        holm_p = _finite_float(row, PRIMARY_HOLM_FIELD, cell=cell)
        if ci_low > ci_high:
            raise ValueError(f"reversed confidence interval for {cell!r}")
        if not 0.0 <= holm_p <= 1.0:
            raise ValueError(f"Holm-adjusted p-value outside [0, 1] for {cell!r}")
        records[cell] = {
            "effect_pp": 100.0 * effect,
            "ci_low_pp": 100.0 * ci_low,
            "ci_high_pp": 100.0 * ci_high,
            "holm_p": holm_p,
            "significant": holm_p <= 0.05,
            "contrast_family": contrast.family,
        }

    missing_cells = expected_cells.difference(records)
    if missing_cells:
        examples = sorted(missing_cells)[:5]
        raise ValueError(
            f"incomplete primary ablation grid: observed {len(records)} of "
            f"{len(expected_cells)} cells; missing examples={examples!r}"
        )
    return records


def fig_ablation(effects_csv: Path, outdir: Path):
    """Draw all predeclared delivery contrasts as a five-panel forest matrix."""

    records = load_confirmatory_delivery_effects(effects_csv)
    contrast_names = [contrast.name for contrast in PLANNED_CONTRASTS]
    y_positions = np.arange(len(contrast_names))
    all_bounds = [
        abs(record[field])
        for record in records.values()
        for field in ("ci_low_pp", "ci_high_pp", "effect_pp")
    ]
    max_abs = max(all_bounds, default=0.0)
    axis_limit = max(1.0, np.ceil(max_abs * 1.10 * 2.0) / 2.0)

    fig, axes = plt.subplots(
        1,
        len(SCENARIO_ORDER),
        sharex=True,
        sharey=True,
        figsize=(7.2, 5.0),
    )
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        for y, contrast_name in zip(y_positions, contrast_names):
            record = records[(scenario, contrast_name)]
            color, marker = CONTRAST_STYLE.get(
                record["contrast_family"], ("#444444", "o")
            )
            ax.hlines(
                y,
                record["ci_low_pp"],
                record["ci_high_pp"],
                color=color,
                linewidth=1.25,
                zorder=2,
            )
            ax.scatter(
                record["effect_pp"],
                y,
                marker=marker,
                s=27,
                facecolor=color if record["significant"] else "white",
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        ax.axvline(0.0, color="black", linewidth=0.75, zorder=1)
        ax.axhline(4.5, color="#bdbdbd", linewidth=0.6, linestyle=":")
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_title(SCENARIO_DISPLAY.get(scenario, scenario), fontsize=8.5)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(
        [
            CONTRAST_DISPLAY.get(name, name.replace("_", " "))
            for name in contrast_names
        ],
        fontsize=7.2,
    )
    axes[0].invert_yaxis()
    fig.supxlabel(
        "Delivery effect: treatment - reference (percentage points)",
        fontsize=9,
        y=0.075,
    )
    fig.text(
        0.5,
        0.018,
        "Lines: 95% crossed-bootstrap CI. Filled markers: primary Holm-FWER "
        "adjusted p <= 0.05 (40 tests).",
        ha="center",
        fontsize=6.7,
    )
    fig.subplots_adjust(left=0.29, right=0.99, bottom=0.16, top=0.91, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig4_ablation_effect.{ext}", dpi=300)
    plt.close(fig)
    print(
        "  [F4] fig4_ablation_effect  "
        f"({len(contrast_names)} planned contrasts x {len(SCENARIO_ORDER)} scenarios)"
    )


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
