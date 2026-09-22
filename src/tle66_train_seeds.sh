#!/bin/bash
# One-day 66-sat feasibility study: train 3 seeds of QoS MAPPO in the
# SRPF environment on the 66-satellite Starlink topology at load 8.
cd /f/leo-routing-preliminary-matlab/src || exit 1
for SEED in 179055553 183895110 310818925; do
  echo "=== tle66 training seed $SEED ==="
  LEO_TLE_N_PLANES=11 LEO_TLE_SATS_PER_PLANE=6 LEO_TLE_INITIAL=8 LEO_TLE_EXOGENOUS=8 \
  LEO_PURGE_INFEASIBLE=2 PYTHONIOENCODING=utf-8 \
  /f/leo-venv/Scripts/python.exe cleanmarl_mappo_tle.py \
    --env-type leo_tle --env-name hotspot_high_load \
    --leo-project-path F:/leo-routing-preliminary-matlab/src \
    --leo-topology-csv ../data/starlink_66_links.csv \
    --leo-variant qos_only --seed "$SEED" \
    --batch-size 4 --total-timesteps 50000 --epochs 3 --num-minibatches 4 \
    --eval-steps 40 --num-eval-ep 10 --save-every-steps 5000 \
    --checkpoint-dir "../outputs/tle66-train-20260922/seed_$SEED" \
    --run-tag TLE66-SRPF-ENV-v1 \
    --train-seed-start 76001 --train-seed-count 200 \
    --validation-seed-start 77001 \
    --device cuda \
    || { echo "SEED $SEED FAILED"; exit 1; }
done
echo "TLE66 TRAINING COMPLETE"
