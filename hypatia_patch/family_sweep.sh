#!/bin/bash
# Full packet-management family sweep on an official-Hypatia shell.
# Usage: bash family_sweep.sh <gen_basename> <gs_base> [arms...]
#   gen_basename: e.g. custom_4x6_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls
#   gs_base: first ground-station node id (= number of satellites)
# Writes per-arm results and a summary CSV in ~/hypatia/family_results/<gen_basename>/.
set -u
GENBASE=$1
GSBASE=$2
shift 2
ARMS="${@:-droptail edf class_priority lcfs codel red purge purge_srpf purge_edf purge_lcfs}"

GEN=$HOME/hypatia/paper/satellite_networks_state/gen_data/$GENBASE
FSTATE=$(ls -d $GEN/dynamic_state_* | head -1)
OUT=$HOME/hypatia/family_results/$GENBASE
rm -rf "$OUT"; mkdir -p "$OUT"

# Traffic: identical pattern to the Starlink-1584 run (20 hotspot + 20 random,
# mixed deadlines 60/150 ms, three traffic classes), seeded deterministically.
python3 - "$GSBASE" > "$OUT/deadline_udp_burst_schedule.csv" << 'EOF'
import random, sys
gsbase = int(sys.argv[1])
rng = random.Random(20261007)
hot = gsbase
rows = []
bid = 0
for i in range(20):
    src = rng.randrange(gsbase, gsbase + 100)
    while src == hot:
        src = rng.randrange(gsbase, gsbase + 100)
    dl, cls = rng.choice([(60,2),(60,2),(60,2),(150,0)])
    rows.append(f"{bid},1000000000,{src},{hot},2.5,3000000000,{dl},{cls}")
    bid += 1
for i in range(20):
    src = rng.randrange(gsbase, gsbase + 100)
    dst = rng.randrange(gsbase, gsbase + 100)
    while dst == src or src == hot or dst == hot:
        src = rng.randrange(gsbase, gsbase + 100)
        dst = rng.randrange(gsbase, gsbase + 100)
    dl, cls = rng.choice([(60,2),(150,0),(150,1)])
    rows.append(f"{bid},1000000000,{src},{dst},2.5,3000000000,{dl},{cls}")
    bid += 1
print("burst_id,start_time_ns,from_node_id,to_node_id,target_rate_mbps,duration_ns,deadline_ms,traffic_class")
for r in rows:
    print(r)
EOF

echo "arm,sent,arrived,on_time,on_time_ratio,hot_on,rand_on,purged" > "$OUT/summary.csv"

cd $HOME/hypatia/ns3-sat-sim/simulator
for ARM in $ARMS; do
  D=$OUT/run_$ARM
  mkdir -p "$D/logs_ns3"
  cat > "$D/config_ns3.properties" << EOF
simulation_end_time_ns=9000000000
simulation_seed=123456789
satellite_network_dir="../../../paper/satellite_networks_state/gen_data/$GENBASE"
satellite_network_routes_dir="../../../paper/satellite_networks_state/gen_data/$GENBASE/$(basename $FSTATE)"
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
isl_queue_type=$ARM
purge_min_hop_latency_ns=3000000
EOF
  cp "$OUT/deadline_udp_burst_schedule.csv" "$D/"
  echo "=== [$GENBASE] arm: $ARM ==="
  ./waf --run="main_satnet --run_dir='$D'" > "$OUT/console_$ARM.log" 2>&1
  PT=$(grep -o 'PURGE_TOTAL,[0-9]*' "$OUT/console_$ARM.log" | cut -d, -f2)
  [ -z "$PT" ] && PT=0
  python3 - "$OUT" "$ARM" "$PT" >> "$OUT/summary.csv" << 'EOF'
import csv, os, sys
out, arm, pt = sys.argv[1], sys.argv[2], sys.argv[3]
d = os.path.join(out, "run_" + arm, "logs_ns3")
sent = 0
try:
    for row in csv.DictReader(open(os.path.join(d, "deadline_udp_sent.csv"))):
        try: sent += int(row["sent_packets"])
        except (TypeError, ValueError): continue
except FileNotFoundError: pass
arr = on = hot_on = rand_on = 0
try:
    for row in csv.DictReader(open(os.path.join(d, "deadline_udp_packets.csv"))):
        try:
            ot = int(row["on_time"]); b = int(row["burst_id"])
        except (TypeError, ValueError): continue
        arr += 1; on += ot
        if b < 20: hot_on += ot
        else: rand_on += ot
except FileNotFoundError: pass
ratio = on / sent if sent else 0.0
print(f"{arm},{sent},{arr},{on},{ratio:.4f},{hot_on},{rand_on},{pt}")
EOF
  tail -1 "$OUT/summary.csv"
done
echo "DONE $GENBASE"
