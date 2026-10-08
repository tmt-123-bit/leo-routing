#!/bin/bash
# Generate the three missing shells (24, 66, 156) with the official generator,
# then run the full 10-arm family sweep on each.
set -u
cd $HOME/hypatia/paper/satellite_networks_state

run_shell () {
  local P=$1 N=$2
  local NAME="custom_${P}x${N}_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
  if [ ! -d "gen_data/$NAME" ]; then
    echo "=== generating ${P}x${N} shell ==="
    python3 custom_pn.py $P $N 10 100 8 > /tmp/gen_${P}x${N}.log 2>&1 || { tail -3 /tmp/gen_${P}x${N}.log; return 1; }
  fi
  local GSBASE=$((P * N))
  echo "=== sweeping ${P}x${N} (GS base $GSBASE) ==="
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "$GSBASE"
}

run_shell 4 6
run_shell 11 6
run_shell 26 6
echo ALL_SMALL_SHELLS_DONE
