#!/bin/bash
# Official routing-layer comparison on the Starlink-1584 constellation:
# Hypatia's three native routing algorithms x {droptail, purge_edf} queues.
#   1. free_one_only_over_isls   (already generated; baseline ISL-only shortest path)
#   2. free_one_only_gs_relays   (ground stations may relay traffic)
#   3. paired_many_only_over_isls (paired many: multiple GSL interfaces per satellite)
# Static files (TLEs/ISLs/GS) are reused from the existing starlink_550 dir;
# only gsl_interfaces_info (for paired_many) and the dynamic state are computed.
set -u
cd $HOME/hypatia/paper/satellite_networks_state
BASE="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
GS100=100

make_variant () {
  local ALGO=$1           # algorithm_free_one_only_gs_relays | algorithm_paired_many_only_over_isls
  local IFPER=$2          # GSL interfaces per satellite
  local TAG=$3
  local NAME="${BASE/algorithm_free_one_only_over_isls/$ALGO}"
  if [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
    echo "=== $TAG already generated"; return 0
  fi
  echo "=== generating $TAG dynamic state"
  mkdir -p "gen_data/$NAME"
  for F in tles.txt isls.txt ground_stations.txt; do
    cp "gen_data/$BASE/$F" "gen_data/$NAME/$F"
  done
  cp "gen_data/$BASE/description.txt" "gen_data/$NAME/description.txt"
  python3 - "$NAME" "$IFPER" << 'EOF'
import sys, os
sys.path.append(os.path.join(os.getcwd(), "..", "..", "satgenpy"))
import satgen
name, ifper = sys.argv[1], int(sys.argv[2])
satgen.generate_simple_gsl_interfaces_info(
    f"gen_data/{name}/gsl_interfaces_info.txt", 1584, 100, ifper, 1, 1, 1)
print("gsl_interfaces_info written, if/sat =", ifper)
EOF
  python3 - "$NAME" "$ALGO" << 'EOF'
import sys, os
sys.path.append(os.path.join(os.getcwd(), "..", "..", "satgenpy"))
import satgen
name, algo = sys.argv[1], sys.argv[2]
with open(f"gen_data/{name}/description.txt") as f:
    d = dict(l.split("=", 1) for l in f.read().strip().split("\n") if "=" in l)
satgen.help_dynamic_state(
    "gen_data", 8, name, 100, 10,
    float(d["max_gsl_length_m"]), float(d["max_isl_length_m"]),
    algo, True)
EOF
  [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ] \
    && echo "=== $TAG OK" || echo "=== $TAG FAILED"
}

make_variant algorithm_free_one_only_gs_relays 1 gs_relays
make_variant algorithm_paired_many_only_over_isls 100 paired_many

# Sweep: each routing x {droptail, purge_edf} on 1584
for VAR in gs_relays paired_many; do
  if [ "$VAR" = "gs_relays" ]; then
    NAME="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_gs_relays"
  else
    NAME="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_paired_many_only_over_isls"
  fi
  if [ -f "gen_data/$NAME/dynamic_state_100ms_for_10s/fstate_9900000000.txt" ]; then
    echo "=== sweeping routing=$VAR"
    bash /mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh "$NAME" "1584" droptail purge_edf
  fi
done
echo ROUTING_VARIANTS_DONE
