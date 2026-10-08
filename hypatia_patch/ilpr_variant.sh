#!/bin/bash
# ILPR routing variant on the official Starlink-1584 constellation:
# post-process the official fstates into persistent feasible routing,
# then sweep {droptail, purge_edf}.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
cp /mnt/f/LEO研究代码汇总/hypatia_patch/ilpr_postprocess.py .
SRC="gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
DST="gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_ilpr_persistent"
if [ ! -f "$DST/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
  rm -rf "$DST"
  echo "=== post-processing fstates into ILPR persistent routing ==="
  python3 ilpr_postprocess.py "$SRC" "$DST" | tail -1
fi
if [ -f "$DST/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
  echo "=== sweeping ILPR routing"
  bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh \
    "starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_ilpr_persistent" \
    "1584" droptail purge_edf
fi
echo ILPR_VARIANT_DONE
