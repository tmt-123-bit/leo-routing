#!/bin/bash
# Final scales: smallest physically valid shell (12x9=108) + full 1584 sweep.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
NAME="custom_12x9_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
if [ ! -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_0.txt" ]; then
  rm -rf "gen_data/$NAME"
  echo "=== generating 12x9 (108-sat) shell ==="
  python3 custom_pn.py 12 9 10 100 8 > /tmp/gen_12x9.log 2>&1 || { tail -3 /tmp/gen_12x9.log; exit 1; }
fi
echo "=== sweeping 12x9 ==="
bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "108"

echo "=== sweeping starlink_550 (1584) ==="
bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh \
  "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls" "1584"
echo ALL_FINAL_DONE
