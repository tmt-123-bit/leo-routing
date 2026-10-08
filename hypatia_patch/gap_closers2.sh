#!/bin/bash
# Gap closers v2 (A/B/D; C already succeeded):
#   A) dropfront at 5 scales — each on ITS OWN constellation
#   B) deadline sweep at 1584 (fixed arg order)
#   D) traffic-seed statistical panel at 1584 (schedule regenerated per seed)
set -u
H=$HOME/hypatia
OUT=$H/gap_results
STAR="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
SCHED=$H/family_results/$STAR/deadline_udp_burst_schedule.csv

run_one () {  # run_one <run_dir> <gen_name> <queue_type> <seed> <min_hop_ns> [sched]
  local D=$1 GEN=$2 QT=$3 SD=$4 MH=$5 SC=${6:-$SCHED}
  rm -rf "$D"; mkdir -p "$D/logs_ns3"
  cp "$SC" "$D/deadline_udp_burst_schedule.csv"
  cat > "$D/config_ns3.properties" << EOF
simulation_end_time_ns=9000000000
simulation_seed=$SD
satellite_network_dir="../../paper/satellite_networks_state/gen_data/$GEN"
satellite_network_routes_dir="../../paper/satellite_networks_state/gen_data/$GEN/dynamic_state_100ms_for_10s"
dynamic_state_update_interval_ns=100000000
isl_data_rate_megabit_per_s=10.0
gsl_data_rate_megabit_per_s=10.0
isl_max_queue_size_pkts=100
gsl_max_queue_size_pkts=100
enable_isl_utilization_tracking=false
tcp_socket_type=TcpNewReno
enable_tcp_flow_scheduler=false
enable_udp_burst_scheduler=false
enable_pingmesh_scheduler=false
enable_deadline_udp_burst_scheduler=true
deadline_udp_burst_schedule_filename="deadline_udp_burst_schedule.csv"
isl_queue_type=$QT
purge_min_hop_latency_ns=$MH
EOF
  (cd $H/ns3-sat-sim/simulator && \
   ./waf --run="main_satnet --run_dir='$D'" > "$D/console.log" 2>&1)
  python3 - "$D" << 'PYEOF'
import csv, os, sys
d = os.path.join(sys.argv[1], "logs_ns3")
sent = arr = on = 0
try:
    for r in csv.DictReader(open(os.path.join(d, "deadline_udp_sent.csv"))):
        try: sent += int(r["sent_packets"])
        except (TypeError, ValueError): continue
except FileNotFoundError: pass
try:
    for r in csv.DictReader(open(os.path.join(d, "deadline_udp_packets.csv"))):
        try:
            on += int(r["on_time"]); arr += 1
        except (TypeError, ValueError): continue
except FileNotFoundError: pass
print(f"RESULT,{sent},{arr},{on},{on/sent if sent else 0:.6f}")
PYEOF
}

echo "===== A) dropfront at 5 scales ====="
for SPEC in "knn_4x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls 4x6" \
            "knn_11x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls 11x6" \
            "custom_12x13_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls 12x13" \
            "custom_72x14_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls 72x14" \
            "$STAR starlink"; do
  set -- $SPEC
  SC=$H/family_results/$1/deadline_udp_burst_schedule.csv
  R=$(run_one "$OUT/dropfront_$2" "$1" dropfront 123456789 3000000 "$SC" | tail -1)
  echo "dropfront_$2,$R"
done

echo "===== B) deadline sweep at 1584 ====="
for DL in 20 40 60 100 150; do
  SC=$OUT/sched2_dl_$DL.csv
  python3 - "$SCHED" "$SC" "$DL" << 'PYEOF'
import csv, sys
src, dst, dl = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.reader(open(src)))
out = [rows[0]]
for r in rows[1:]:
    r = list(r); r[6] = dl; out.append(r)
csv.writer(open(dst, "w", newline="")).writerows(out)
PYEOF
  [ -s "$SC" ] || { echo "sched gen failed for $DL"; continue; }
  for ARM in droptail purge_edf; do
    R=$(run_one "$OUT/dl_${DL}_${ARM}" "$STAR" $ARM 123456789 3000000 "$SC" | tail -1)
    echo "deadline_${DL}ms_${ARM},$R"
  done
done

echo "===== D) traffic-seed panel at 1584 ====="
for TS in 1 2 3 4 5; do
  SC=$OUT/sched2_seed_$TS.csv
  python3 - "$SC" "$TS" << 'PYEOF'
import random, sys
dst, seed = sys.argv[1], int(sys.argv[2])
rng = random.Random(20261007 + seed * 7919)
gsbase = 1584
hot = gsbase
rows = []
bid = 0
for i in range(20):
    src = rng.randrange(gsbase, gsbase + 100)
    while src == hot:
        src = rng.randrange(gsbase, gsbase + 100)
    dl, cls = rng.choice([(60, 2), (60, 2), (60, 2), (150, 0)])
    rows.append(f"{bid},1000000000,{src},{hot},2.5,3000000000,{dl},{cls}")
    bid += 1
for i in range(20):
    src = rng.randrange(gsbase, gsbase + 100)
    d2 = rng.randrange(gsbase, gsbase + 100)
    while d2 == src or src == hot or d2 == hot:
        src = rng.randrange(gsbase, gsbase + 100)
        d2 = rng.randrange(gsbase, gsbase + 100)
    dl, cls = rng.choice([(60, 2), (150, 0), (150, 1)])
    rows.append(f"{bid},1000000000,{src},{d2},2.5,3000000000,{dl},{cls}")
    bid += 1
with open(dst, "w") as f:
    f.write("burst_id,start_time_ns,from_node_id,to_node_id,target_rate_mbps,duration_ns,deadline_ms,traffic_class\n")
    f.write("\n".join(rows) + "\n")
PYEOF
  [ -s "$SC" ] || { echo "sched gen failed for seed $TS"; continue; }
  for ARM in droptail purge_edf; do
    R=$(run_one "$OUT/tseed_${TS}_${ARM}" "$STAR" $ARM 123456789 3000000 "$SC" | tail -1)
    echo "tseed_${TS}_${ARM},$R"
  done
done
echo GAP_CLOSERS_V2_DONE
