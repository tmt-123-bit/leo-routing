#!/bin/bash
# Close the official-environment gaps:
#   A) dropfront arm at all 5 scales (the missing 11th method)
#   B) deadline sweep at 1584 (official load/deadline curve)  {droptail, purge_edf}
#   C) margin (conservativeness) sweep at 1584               {purge_edf}
#   D) 5-seed statistical panel at 1584                      {droptail, purge_edf}
set -u
H=$HOME/hypatia
SWEEP=/mnt/f/LEO研究代码汇总/hypatia_patch/family_sweep.sh
OUT=$H/gap_results
mkdir -p $OUT
STAR="starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
SCHED=$H/family_results/$STAR/deadline_udp_burst_schedule.csv

run_one () {  # run_one <run_dir> <queue_type> <seed> <min_hop_ns> [sched]
  local D=$1 QT=$2 SD=$3 MH=$4 SC=${5:-$SCHED}
  rm -rf "$D"; mkdir -p "$D/logs_ns3"
  cp "$SC" "$D/deadline_udp_burst_schedule.csv"
  cat > "$D/config_ns3.properties" << EOF
simulation_end_time_ns=9000000000
simulation_seed=$SD
satellite_network_dir="../../paper/satellite_networks_state/gen_data/$STAR"
satellite_network_routes_dir="../../paper/satellite_networks_state/gen_data/$STAR/dynamic_state_100ms_for_10s"
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
for SPEC in "knn_4x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls 24 4x6" \
            "knn_11x6_isls_knn_ground_stations_top_100_algorithm_free_one_only_over_isls 66 11x6" \
            "custom_12x13_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls 156 12x13" \
            "custom_72x14_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls 1008 72x14" \
            "$STAR 1584 starlink"; do
  set -- $SPEC
  R=$(run_one "$OUT/dropfront_$3" dropfront 123456789 3000000 \
      "$H/family_results/$1/deadline_udp_burst_schedule.csv" | tail -1)
  echo "dropfront_$3,$R"
done

echo "===== B) deadline sweep at 1584 ====="
for D in 20 40 60 100 150; do
  SC=$OUT/sched_dl_$D.csv
  python3 - "$SCHED" "$SC" "$D" << 'PYEOF'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
out = [rows[0]]
for r in rows[1:]:
    r = list(r); r[6] = sys.argv[2]; out.append(r)
w = csv.writer(open(sys.argv[3], "w", newline=""))
w.writerows(out)
PYEOF
  for ARM in droptail purge_edf; do
    R=$(run_one "$OUT/dl_${D}_${ARM}" $ARM 123456789 3000000 "$SC" | tail -1)
    echo "deadline_${D}ms_${ARM},$R"
  done
done

echo "===== C) margin sweep at 1584 (purge_edf) ====="
for M in 1500000 3000000 6000000 12000000; do
  R=$(run_one "$OUT/margin_$M" purge_edf 123456789 $M | tail -1)
  echo "margin_${M}ns_purge_edf,$R"
done

echo "===== D) 5-seed panel at 1584 ====="
for SD in 1 2 3 4 5; do
  for ARM in droptail purge_edf; do
    R=$(run_one "$OUT/seed_${SD}_${ARM}" $ARM $SD 3000000 | tail -1)
    echo "seed_${SD}_${ARM},$R"
  done
done
echo ALL_GAP_CLOSERS_DONE
