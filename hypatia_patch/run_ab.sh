#!/bin/bash
# Official-Hypatia A/B: isl_queue_type = droptail vs purge_srpf
# Uses the already-generated Kuiper-630 constellation from the integration
# test (Manila=630, Dalian=631 among 632 nodes) with saturating deadline
# UDP bursts in both directions (9 Mbit/s over 10 Mbit/s links, 75 ms
# deadline). On-time delivery ratio is computed from the receiver logs.
set -u
BASE=~/hypatia
ROOT=$BASE/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline

for ARM in base mech; do
  D=$ROOT/$ARM
  rm -rf "$D"; mkdir -p "$D/logs_ns3"
  if [ "$ARM" = "mech" ]; then QT="purge_srpf"; else QT="droptail"; fi
  cat > "$D/config_ns3.properties" << EOF
simulation_end_time_ns=5000000000
simulation_seed=123456789
satellite_network_dir="../../gen_data/reduced_kuiper_630_algorithm_free_one_only_over_isls"
satellite_network_routes_dir="../../gen_data/reduced_kuiper_630_algorithm_free_one_only_over_isls/dynamic_state_100ms_for_200s"
dynamic_state_update_interval_ns=100000000
isl_data_rate_megabit_per_s=10.0
gsl_data_rate_megabit_per_s=10.0
isl_max_queue_size_pkts=100
gsl_max_queue_size_pkts=100
enable_isl_utilization_tracking=true
isl_utilization_tracking_interval_ns=1000000000
tcp_socket_type=TcpNewReno
enable_tcp_flow_scheduler=false
enable_udp_burst_scheduler=false
enable_pingmesh_scheduler=false
enable_deadline_udp_burst_scheduler=true
deadline_udp_burst_schedule_filename="deadline_udp_burst_schedule.csv"
isl_queue_type=$QT
purge_min_hop_latency_ns=2500000
EOF
  cat > "$D/deadline_udp_burst_schedule.csv" << 'EOF'
burst_id,start_time_ns,from_node_id,to_node_id,target_rate_mbps,duration_ns,deadline_ms
0,1000000000,17,18,5.0,2000000000,22
1,1000000000,17,18,5.0,2000000000,22
2,1000000000,17,18,5.0,2000000000,60
3,1000000000,17,18,5.0,2000000000,60
4,1000000000,18,17,5.0,2000000000,22
5,1000000000,18,17,5.0,2000000000,22
6,1000000000,18,17,5.0,2000000000,60
7,1000000000,18,17,5.0,2000000000,60
EOF
  echo "=== running arm: $ARM (queue=$QT) ==="
  cd $BASE/ns3-sat-sim/simulator
  ./waf --run="main_satnet --run_dir='$D'" 2>&1 | tail -4
done

echo "=== ANALYSIS ==="
python3 - << 'EOF'
import csv, os
root = os.path.expanduser("~/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline")
for arm in ("base", "mech"):
    d = os.path.join(root, arm, "logs_ns3")
    sent = 0
    try:
        with open(os.path.join(d, "deadline_udp_sent.csv")) as f:
            for row in csv.DictReader(f):
                sent += int(row["sent_packets"])
    except FileNotFoundError:
        print(arm, "no sent log"); continue
    arrived = ontime = 0
    try:
        with open(os.path.join(d, "deadline_udp_packets.csv")) as f:
            for row in csv.DictReader(f):
                arrived += 1; ontime += int(row["on_time"])
    except FileNotFoundError:
        pass
    print(f"{arm}: sent={sent} arrived={arrived} on_time={ontime} "
          f"on_time_ratio={ontime/sent if sent else 0:.4f} arrival_ratio={arrived/sent if sent else 0:.4f}")
EOF
