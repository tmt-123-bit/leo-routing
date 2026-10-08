#!/bin/bash
tail -8 /tmp/gen_4x6.log 2>/dev/null || echo "log gone"
echo "--- 4x6 gen_data contents:"
find $HOME/hypatia/paper/satellite_networks_state/gen_data/custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls -type f | head -8
