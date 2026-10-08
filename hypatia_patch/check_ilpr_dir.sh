#!/bin/bash
D=$HOME/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_ilpr_persistent/dynamic_state_100ms_for_10s
echo "fstate files: $(ls $D | grep -c '^fstate_')"
echo "gsl files: $(ls $D | grep -c '^gsl_if')"
echo "fstate_0 lines: $(wc -l < $D/fstate_0.txt)"
echo "fstate_100000000 lines: $(wc -l < $D/fstate_100000000.txt)"
