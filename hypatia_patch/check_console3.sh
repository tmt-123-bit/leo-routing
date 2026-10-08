#!/bin/bash
C=$HOME/hypatia/family_results/custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls/console_droptail.log
tail -6 "$C" 2>/dev/null
echo "--- finished flag:"
cat $HOME/hypatia/family_results/custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls/run_droptail/logs_ns3/finished.txt 2>/dev/null
