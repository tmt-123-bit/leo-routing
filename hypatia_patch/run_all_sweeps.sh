#!/bin/bash
# Full migration sweep: rebuild with the class patch, then run the 10-arm
# family sweep on 24/66/156 (already-generated shells) AND 1008 (72x14).
set -u
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy.log 2>&1
if grep -qE 'error:|Build failed' /tmp/deploy.log; then
  echo BUILD_FAILED
  grep -E 'error:' /tmp/deploy.log | head -5
  exit 1
fi
echo BUILD_OK

cd $HOME/hypatia/paper/satellite_networks_state
# 1008 = 72x14 shell (generate if missing; ~15-30 min generation)
NAME1008="custom_72x14_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
if [ ! -d "gen_data/$NAME1008" ]; then
  echo "=== generating 72x14 (1008-sat) shell ==="
  python3 custom_pn.py 72 14 10 100 8 > /tmp/gen_72x14.log 2>&1 || { tail -3 /tmp/gen_72x14.log; exit 1; }
fi

for SPEC in "4 6" "11 6" "26 6" "72 14"; do
  set -- $SPEC
  P=$1; N=$2
  NAME="custom_${P}x${N}_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
  GSBASE=$((P * N))
  echo "=== sweeping ${P}x${N} ==="
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "$GSBASE"
done
echo ALL_SWEEPS_DONE
