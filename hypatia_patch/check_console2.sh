#!/bin/bash
O=$HOME/hypatia/family_results
C=$O/custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls/console_droptail.log
echo "--- 4x6 droptail console tail:"
tail -8 "$C" 2>/dev/null
echo "--- build check:"
tail -3 /tmp/deploy.log 2>/dev/null
grep -cE 'error' /tmp/deploy.log 2>/dev/null
