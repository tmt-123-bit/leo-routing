#!/bin/bash
# CMADR switch-budget routing variant on official Starlink-1584:
# fstate post-processor (budget-constrained route adaptation) + sweep.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
cp /mnt/f/LEO研究代码汇总/hypatia_patch/cmadr_postprocess.py .
SRC="gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
DST="gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_cmadr_budget"
if [ ! -f "$DST/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
  rm -rf "$DST"
  echo "=== post-processing fstates into CMADR budget routing ==="
  python3 cmadr_postprocess.py "$SRC" "$DST" | tail -1
fi
if [ -f "$DST/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
  echo "=== sweeping CMADR routing"
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh \
    "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_cmadr_budget" \
    "1584" droptail purge_edf
fi
echo CMADR_VARIANT_DONE
