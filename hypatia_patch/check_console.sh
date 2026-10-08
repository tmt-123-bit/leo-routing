#!/bin/bash
C=$HOME/hypatia/family_results/custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls/console_droptail.log
tail -12 "$C"
echo "--- schedule head:"
head -3 $HOME/hypatia/family_results/custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls/deadline_udp_burst_schedule.csv
