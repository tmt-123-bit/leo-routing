#!/bin/bash
# Final 24-sat (4x6) generation with whole-horizon stable kNN ISLs + sweep.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
cp /mnt/f/LEO研究代码汇总/hypatia_patch/custom_knn.py .
NAME="knn_4x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls"
rm -rf "gen_data/$NAME" gen_data/_knn_probe
echo "=== generating knn 4x6 (stable) ==="
python3 custom_knn.py 4 6 10 100 8 2>&1 | grep -E "alt=|chosen|DONE|NO_VALID|Error" | tail -18
if [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
  echo "=== sweeping knn 4x6 ==="
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "24"
else
  echo KNN24_STILL_FAILED
fi
echo FINAL24_DONE
