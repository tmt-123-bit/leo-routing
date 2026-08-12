#!/usr/bin/env bash
# =============================================================================
# IEEE experiment suite — reproduction check + full re-run with the optimized
# trainer/env/training config. Read OPTIMIZATION_CHANGES.md first.
#
# WHY THIS EXISTS: every env/reward change in the optimization pass INVALIDATES
# prior tables (EXP-004-FULL etc.). They must be regenerated as ONE batch.
# Also, run_exp004 defaults to --mode quick (300 steps) and the old README never
# documented --mode full, so the headline numbers were not reproducible from the
# documented path. This script fixes that.
#
# USAGE:  bash run_ieee_reproduction.sh           # full pipeline (incl. budget sweep)
#         bash run_ieee_reproduction.sh smoke     # 300-step sanity only
#         bash run_ieee_reproduction.sh repro     # repro-check only (see STEP 1)
#         bash run_ieee_reproduction.sh budget    # budget sweep only (x2k, x10k)
#         bash run_ieee_reproduction.sh mde       # MDE report on the FULL results
#
# Stages 1-2 are the gating checks; 3-5 are the publication suite. Each stage
# writes under experiments/IEEE-<tag>/ so old results are never overwritten.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
# Prefer the F: venv (CUDA torch) if it exists; else the system `py`.
if [[ -z "${PY:-}" ]]; then
  if [[ -x /f/leo-venv/Scripts/python.exe ]]; then
    PY=/f/leo-venv/Scripts/python.exe
  else
    PY=py
  fi
fi
STAGE=${1:-all}
PROJ="F:/leo-routing-preliminary-matlab"
CLEANMARL="F:/cleanmarl"
DEVICE=${DEVICE:-cpu}   # set DEVICE=cuda if a GPU is available -> large speedup

run_exp004() { $PY run_exp004_mappo.py --cleanmarl "$CLEANMARL" --project "$PROJ" --device "$DEVICE" "$@"; }

# -----------------------------------------------------------------------------
# STEP 0 — smoke: does the optimized code train at all? (~1 min)
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "smoke" || "$STAGE" == "all" ]]; then
  echo "### STEP 0 smoke (600 steps, medium_load)"
  rm -rf /tmp/ieee-smoke
  (cd "$CLEANMARL" && $PY cleanmarl/mappo.py --env-type leo_multi --env-name medium_load \
     --leo-project-path "$PROJ" --leo-variant full --seed 7 --batch-size 4 \
     --total-timesteps 600 --epochs 3 --num-minibatches 4 --eval-steps 100000 \
     --num-eval-ep 3 --save-every-steps 1000000 --checkpoint-dir /tmp/ieee-smoke \
     --run-tag SMOKE --train-seed-start 9001 --train-seed-count 200 \
     --validation-seed-start 10001 --device "$DEVICE")
  echo "### smoke done — inspect /tmp/ieee-smoke/*/training_metrics.jsonl"
  echo "    expect explained_variance rising past ~0.5 within a few hundred steps"
  [[ "$STAGE" == "smoke" ]] && exit 0
fi

# -----------------------------------------------------------------------------
# STEP 1 — REPRO CHECK: does the CURRENT code at FULL budget reproduce the ~60%
# delivery band (not the ~10% V2 band)? This distinguishes code drift from
# training budget as the cause of the QUICK-V2 contradiction.
#   fault_links MAPPO ~ 0.55-0.65  => old numbers reproduce (budget was the issue)
#   fault_links MAPPO ~ 0.05-0.15  => CODE REGRESSION (fix code before Step 2)
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "repro" || "$STAGE" == "all" ]]; then
  echo "### STEP 1 repro-check (FULL, 8 seeds) -> experiments/IEEE-REPRO-CHECK"
  run_exp004 --mode full --output experiments/IEEE-REPRO-CHECK
  echo "### checking fault_links MAPPO delivery band..."
  $PY - <<'EOF'
import csv,sys
try:
    rows=[r for r in csv.DictReader(open("experiments/IEEE-REPRO-CHECK/aggregate_metrics.csv",encoding="utf-8-sig"))
          if r["scenario"]=="fault_links" and r["policy"]=="mappo" and r["metric"]=="delivery_ratio"]
    d=float(rows[0]["mean"]) if rows else float("nan")
    band="~60% (REPRODUCES old FULL)" if d>0.45 else ("~10% (CODE REGRESSION — do NOT proceed)" if d<0.25 else "mid")
    print(f"fault_links MAPPO delivery = {d:.3f}  =>  {band}")
    sys.exit(0 if d>0.45 else 2)
except Exception as e:
    print("could not parse result:",e); sys.exit(2)
EOF
  [[ "$STAGE" == "repro" ]] && exit 0
fi

# -----------------------------------------------------------------------------
# STEP 2 — FULL headline suite: 5 scenarios x 8 policy seeds x 50K steps, with
# all baselines (Dijkstra/Q-routing/heuristics), held-out test, paired stats.
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "all" ]]; then
  echo "### STEP 2 FULL headline suite -> experiments/IEEE-EXP-004-FULL"
  run_exp004 --mode full --output experiments/IEEE-EXP-004-FULL
fi

# -----------------------------------------------------------------------------
# STEP 3 — ABLATION across ALL 5 scenarios (was only medium+frequent; two of four
# headline conclusions flipped/vanished in the 2nd scenario). Edit
# run_ablation_experiments.py SCENARIOS to all 5 if not already.
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "all" ]]; then
  echo "### STEP 3 ablation (all 5 scenarios, FULL) -> experiments/IEEE-ABLATION-FULL"
  $PY run_ablation_experiments.py --scenarios low_load medium_load hotspot_high_load frequent_break fault_links \
     --mode full --output experiments/IEEE-ABLATION-FULL --cleanmarl "$CLEANMARL" --project "$PROJ" --device "$DEVICE" || \
  echo "NOTE: if --scenarios flag differs, edit the command (run_ablation_experiments.py default = 2 scenarios)"
fi

# -----------------------------------------------------------------------------
# STEP 4 — budget-sensitivity sweep: re-run at {2K, 10K} steps (quick=300 and
# full=50K come from STEP 1/2) to plot delivery-vs-steps and mark the crossover
# where MAPPO overtakes each baseline. Converts the old QUICK-V2 contradiction
# into a characterized sample-efficiency figure.
#   bash run_ieee_reproduction.sh budget
# x2k/x10k presets now exist in mode_config() (run_exp004_mappo.py).
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "budget" || "$STAGE" == "all" ]]; then
  for M in x2k x10k; do
    echo "### STEP 4 budget sweep --mode $M -> experiments/IEEE-BUDGET-$M"
    run_exp004 --mode "$M" --output "experiments/IEEE-BUDGET-$M"
  done
  echo "### budget sweep done — plot delivery_ratio_mean vs {quick,x2k,x10k,full}"
  echo "    per scenario from each experiments/IEEE-BUDGET-*/aggregate_metrics.csv"
fi

# -----------------------------------------------------------------------------
# STEP 5 — analysis: minimum-detectable-effect (MDE) report on the FULL results.
# Every "no significant difference" claim (e.g. frequent_break MAPPO vs Dijkstra)
# must be stated with its MDE, or a reviewer rejects it as possibly underpowered.
#   bash run_ieee_reproduction.sh mde
# -----------------------------------------------------------------------------
if [[ "$STAGE" == "mde" || "$STAGE" == "all" ]]; then
  FULL="experiments/IEEE-EXP-004-FULL/episode_metrics.csv"
  if [[ -f "$FULL" ]]; then
    echo "### STEP 5 MDE report -> experiments/IEEE-EXP-004-FULL/mde_report.csv"
    $PY compute_mde.py --episodes "$FULL" || echo "NOTE: MDE step failed (non-fatal)"
  else
    echo "### STEP 5 skipped — $FULL not found (run STEP 2 full first)"
  fi
fi

echo "### DONE. Aggregate + paired-test CSVs are under each experiments/IEEE-* dir."
echo "### Report: policy-seed std (now 8 seeds), effect sizes + 95% CI,"
echo "###         MDE on every null (mde_report.csv from STEP 5)."
