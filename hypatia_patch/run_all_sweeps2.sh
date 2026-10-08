#!/bin/bash
# Physically valid shells at 550 km (sats-per-orbit >= 9 so intra-plane
# spacing stays below the max ISL length the official generator enforces):
#   27 = 3x9, 66 = 6x11, 156 = 12x13, 1008 = 72x14 (generated), 1584 = 72x22 (generated)
set -u
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy.log 2>&1
if grep -qE 'error:|Build failed' /tmp/deploy.log; then
  echo BUILD_FAILED; grep -E 'error:' /tmp/deploy.log | head -5; exit 1
fi
echo BUILD_OK

cd $HOME/hypatia/paper/satellite_networks_state
for SPEC in "3 9" "6 11" "12 13" "72 14"; do
  set -- $SPEC
  P=$1; N=$2
  NAME="custom_${P}x${N}_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
  if [ ! -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_0.txt" ]; then
    rm -rf "gen_data/$NAME"
    echo "=== generating ${P}x${N} shell ==="
    python3 custom_pn.py $P $N 10 100 8 > /tmp/gen_${P}x${N}.log 2>&1 || { tail -3 /tmp/gen_${P}x${N}.log; continue; }
    [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_0.txt" ] || { echo "GEN_INCOMPLETE ${P}x${N}"; tail -3 /tmp/gen_${P}x${N}.log; continue; }
  fi
  GSBASE=$((P * N))
  echo "=== sweeping ${P}x${N} ==="
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "$GSBASE"
done
echo ALL_SWEEPS_DONE
