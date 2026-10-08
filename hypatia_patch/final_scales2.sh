#!/bin/bash
# 12x10 = 120 sats (smallest valid attempt #2) + the full 1584 sweep.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
NAME="custom_12x10_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
if [ ! -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_0.txt" ]; then
  rm -rf "gen_data/$NAME"
  echo "=== generating 12x10 (120-sat) shell ==="
  python3 custom_pn.py 12 10 10 100 8 > /tmp/gen_12x10.log 2>&1 || echo "12x10 generation failed"
fi
if [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_0.txt" ]; then
  echo "=== sweeping 12x10 ==="
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "120"
fi

echo "=== sweeping starlink_550 (1584) ==="
bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh \
  "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls" "1584"
echo ALL_FINAL_DONE
