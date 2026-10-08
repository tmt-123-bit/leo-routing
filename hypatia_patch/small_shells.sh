#!/bin/bash
# 24-sat (4x6) and 66-sat (11x6) official-Hypatia shells via kNN ISLs,
# then the full 10-arm family sweep on each.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
cp /mnt/f/LEO研究代码汇总/hypatia_patch/custom_knn.py .

for SPEC in "4 6" "11 6"; do
  set -- $SPEC
  P=$1; N=$2
  NAME="knn_${P}x${N}_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls"
  if [ ! -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
    rm -rf "gen_data/$NAME"
    echo "=== generating knn ${P}x${N} shell ==="
    python3 custom_knn.py $P $N 10 100 8 2>&1 | grep -E "alt=|chosen|DONE|NO_VALID" | tail -8
  fi
  if [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
    GSBASE=$((P * N))
    echo "=== sweeping knn ${P}x${N} (GS base $GSBASE) ==="
    bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "$GSBASE"
  else
    echo "KNN_GEN_FAILED_${P}x${N}"
  fi
done
echo SMALL_KNN_DONE
