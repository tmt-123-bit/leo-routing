#!/bin/bash
G=$HOME/hypatia/paper/satellite_networks_state/gen_data
for NAME in custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls custom_11x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls custom_72x14_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls; do
  echo "=== $NAME"
  ls "$G/$NAME" | grep dynamic
  ls "$G/$NAME"/dynamic_state_*/ | head -4
done
