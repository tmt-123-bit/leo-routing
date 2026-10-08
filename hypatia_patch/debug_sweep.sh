#!/bin/bash
set -ux
NAME="custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
bash -x /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "24" droptail 2>&1 | tail -20
