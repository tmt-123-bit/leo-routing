#!/bin/bash
# Dev-only: train the remaining 7 formal seeds of the constrained arm in the
# purge environment (LEO_PURGE_INFEASIBLE=1). Sequential, ~21 min per seed.
cd /f/leo-routing-preliminary-matlab/src || exit 1
for SEED in 183895110 310818925 515636025 626669596 746965870 985998595 2136406109; do
  echo "=== training seed $SEED ==="
  LEO_PURGE_INFEASIBLE=1 /f/leo-venv/Scripts/python.exe /f/cleanmarl/cleanmarl/mappo.py \
    --env-type leo_multi --env-name hotspot_high_load \
    --leo-project-path F:/leo-routing-preliminary-matlab/src \
    --leo-variant qos_only --seed "$SEED" \
    --batch-size 4 --total-timesteps 50000 --epochs 3 --num-minibatches 4 \
    --eval-steps 40 --num-eval-ep 10 --save-every-steps 5000 \
    --checkpoint-dir "../outputs/dev-purge-train-20260916/seed_$SEED" \
    --run-tag DEV-PURGE-ENV-CONSTRAINED-v1 \
    --train-seed-start 76001 --train-seed-count 200 \
    --validation-seed-start 77001 \
    --validation-selection-mode avoidable_switch_budget_constrained \
    --device cuda \
    --avoidable-switch-constraint-enabled --avoidable-switch-budget 0.12 \
    --avoidable-switch-dual-learning-rate 0.05 --avoidable-switch-dual-initial 0.0 \
    --avoidable-switch-dual-max 5 --avoidable-switch-reduction rollout_micro_mean \
    || { echo "SEED $SEED FAILED"; exit 1; }
done
echo "ALL SEEDS COMPLETE"
