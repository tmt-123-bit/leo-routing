#!/bin/bash
# Dev-only pilot: train 3 seeds of the constrained arm in the purge+SRPF
# environment (LEO_PURGE_INFEASIBLE=2). Sequential, ~21 min per seed.
cd /f/leo-routing-preliminary-matlab/src || exit 1
for SEED in 179055553 183895110 310818925; do
  echo "=== training seed $SEED (purge+SRPF env) ==="
  LEO_PURGE_INFEASIBLE=2 /f/leo-venv/Scripts/python.exe /f/cleanmarl/cleanmarl/mappo.py \
    --env-type leo_multi --env-name hotspot_high_load \
    --leo-project-path F:/leo-routing-preliminary-matlab/src \
    --leo-variant qos_only --seed "$SEED" \
    --batch-size 4 --total-timesteps 50000 --epochs 3 --num-minibatches 4 \
    --eval-steps 40 --num-eval-ep 10 --save-every-steps 5000 \
    --checkpoint-dir "../outputs/dev-srpf-train-20260917/seed_$SEED" \
    --run-tag DEV-SRPF-ENV-CONSTRAINED-v1 \
    --train-seed-start 76001 --train-seed-count 200 \
    --validation-seed-start 77001 \
    --validation-selection-mode avoidable_switch_budget_constrained \
    --device cuda \
    --avoidable-switch-constraint-enabled --avoidable-switch-budget 0.12 \
    --avoidable-switch-dual-learning-rate 0.05 --avoidable-switch-dual-initial 0.0 \
    --avoidable-switch-dual-max 5 --avoidable-switch-reduction rollout_micro_mean \
    || { echo "SEED $SEED FAILED"; exit 1; }
done
echo "PILOT COMPLETE"
