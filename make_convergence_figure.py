"""Training-convergence / stability figure (the standard 'is training reproducible?' panel).

Reads per-step training_metrics.jsonl from every seed of a (scenario, variant) run and plots,
across the 8 policy seeds:
  Left   -- episodic return (smoothed) with seed-95%CI band  -> converges & plateaus, tight band
  Center -- critic explained variance with CI band           -> rises to a healthy plateau
  Right  -- normalized policy entropy with CI band           -> falls from ~1 (uniform) to a
                                                              committed-but-exploratory floor

The tight across-seed CI bands are the reproducibility evidence; a healthy EV plateau and
falling entropy confirm stable CTDE/PPO optimization (no collapse, no runaway KL).

Usage:
  python make_convergence_figure.py \
      --source experiments/IEEE-NOLIFE-full/checkpoints \
      --scenario medium_load --variant no_lifetime --outdir figures
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fields of interest -> (jsonl key, y-label, smoothing window)
PANELS = [
    ("return_mean", "Episodic return", 25),
    ("explained_variance", "Critic explained variance", 15),
    ("normalized_entropy", "Policy entropy (normalized)", 15),
]


def smooth(y, w):
    if w <= 1:
        return np.asarray(y, float)
    y = np.asarray(y, float)
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def load_seed_curve(jsonl_path):
    steps, fields = [], {k: [] for k, _, _ in PANELS}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("record_type") != "training_update":
                continue
            steps.append(d["environment_steps"])
            for k, _, _ in PANELS:
                fields[k].append(d[k])
    return np.asarray(steps), fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="experiments/IEEE-NOLIFE-full/checkpoints")
    ap.add_argument("--scenario", default="medium_load")
    ap.add_argument("--variant", default="no_lifetime")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    base = Path(args.source) / args.scenario / args.variant
    seeds = sorted(base.glob("seed_*"))
    if not seeds:
        raise FileNotFoundError(f"no seed dirs under {base}")
    # load all seeds, align to the shortest length
    raw_steps, raw_fields = [], []
    for sd in seeds:
        jl = glob.glob(str(sd / "*/training_metrics.jsonl"))
        if not jl:
            continue
        steps, fields = load_seed_curve(jl[0])
        raw_steps.append(steps)
        raw_fields.append(fields)
    n_seeds = len(raw_fields)
    if n_seeds == 0:
        raise FileNotFoundError(f"no training_metrics.jsonl under {base}")
    min_len = min(len(s) for s in raw_steps)
    steps_common = raw_steps[0][:min_len]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for ax, (key, ylabel, w) in zip(axes, PANELS):
        # stack smoothed, truncated-to-common curves
        sm = np.vstack([smooth(f[key][:min_len], w) for f in raw_fields])  # (n_seeds, T')
        mean = sm.mean(axis=0)
        lo = np.percentile(sm, 2.5, axis=0)
        hi = np.percentile(sm, 97.5, axis=0)
        x = smooth(steps_common, w)
        ax.plot(x, mean, color="#1f4e79", linewidth=1.8, zorder=3)
        ax.fill_between(x, lo, hi, color="#1f4e79", alpha=0.18, zorder=2,
                        label=f"{n_seeds} seeds, 95% band")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=6.5, loc="best", frameon=False)
        # annotate start/end
        ax.annotate(f"{mean[0]:.2f}", (x[0], mean[0]), textcoords="offset points",
                    xytext=(4, -8), fontsize=6.5, color="#1f4e79")
        ax.annotate(f"{mean[-1]:.2f}", (x[-1], mean[-1]), textcoords="offset points",
                    xytext=(4, -8), fontsize=6.5, color="#1f4e79")
        if "variance" in key:
            ax.axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
            ax.set_ylim(-0.05, max(0.6, hi.max() + 0.05))
        if "entropy" in key:
            ax.set_ylim(0, 1.02)

    fig.suptitle(f"Training dynamics — {args.scenario} ({args.variant}, {n_seeds} seeds)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_convergence.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_convergence] {n_seeds} seeds, {args.scenario}/{args.variant} "
          f"-> {outdir}/fig_convergence.{{png,pdf}}")


if __name__ == "__main__":
    main()
