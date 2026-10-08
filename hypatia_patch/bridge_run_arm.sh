#!/bin/bash
# WSL-side single bridge arm. Usage: bridge_run_arm.sh <policy> <queue> <hostip> [gen]
set -u
POL=$1; QT=$2; HOSTIP=$3
GEN=${4:-knn_4x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls}
D=$HOME/hypatia/gap_results/bridge_${POL}_${QT}
rm -rf "$D"; mkdir -p "$D/logs_ns3"
cp $HOME/hypatia/family_results/$GEN/deadline_udp_burst_schedule.csv "$D/"
cat > "$D/config_ns3.properties" << EOF
simulation_end_time_ns=9000000000
simulation_seed=123456789
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
purge_min_hop_latency_ns=3000000
satellite_network_routing_mode=bridge
bridge_host=$HOSTIP
bridge_port=7350
bridge_policy=$POL
EOF
cd ~/hypatia/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='$D'" > "$D/console.log" 2>&1
echo "ends: $(grep -c 'SIMULATION END' "$D/console.log")"
grep -E 'BRIDGE_UPDATE' "$D/console.log" | tail -1
grep -E 'aborted|terminate' "$D/console.log" | head -2
python3 - << PYEOF
import csv, os
d = "$D/logs_ns3"
sent = arr = on = 0
try:
    for r in csv.DictReader(open(d + "/deadline_udp_sent.csv")):
        try: sent += int(r["sent_packets"])
        except (TypeError, ValueError): continue
except FileNotFoundError: pass
try:
    for r in csv.DictReader(open(d + "/deadline_udp_packets.csv")):
        try:
            on += int(r["on_time"]); arr += 1
        except (TypeError, ValueError): continue
except FileNotFoundError: pass
print(f"bridge_${POL}_${QT}: sent={sent} arr={arr} on={on} ratio={on/sent if sent else 0:.4f}")
PYEOF
