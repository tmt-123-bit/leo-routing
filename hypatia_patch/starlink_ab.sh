#!/bin/bash
# Starlink-1584 (72x22, official Hypatia state) mechanism A/B.
# Traffic: 40 deadline-UDP bursts among top-100 ground stations
# (node ids >= 1584). Half the flows target ONE popular ground station
# (hotspot pressure), the rest are random pairs; deadlines are mixed
# (60 ms tight / 150 ms loose) so that purged certainly-dead packets
# free transmission slots for still-viable ones.
set -u
GEN=$HOME/hypatia/paper/satellite_networks_state/gen_data
# discover the generated dir (base name starlink_550)
NETDIR=$(ls -d $GEN/*starlink* 2>/dev/null | head -1)
if [ -z "$NETDIR" ]; then echo "NO_GEN_DATA"; exit 1; fi
echo "constellation: $NETDIR"
FSTATE=$(ls -d $NETDIR/dynamic_state_* | head -1)
NGS=$(($(wc -l < "$NETDIR/ground_stations.txt")))
GSBASE=1584
echo "ground stations: $NGS (ids $GSBASE..$((GSBASE+NGS-1)))"

ROOT=$HOME/hypatia/runs_starlink
mkdir -p $ROOT
python3 - "$GSBASE" "$NGS" > /tmp/schedule_gen.log << 'EOF'
import random, sys
gsbase, ngs = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(20261007)
hot = gsbase  # first GS = the popular destination
rows = []
bid = 0
# 20 hotspot flows (various sources -> hot dst), mixed deadlines
for i in range(20):
    src = rng.randrange(gsbase, gsbase + ngs)
    while src == hot:
        src = rng.randrange(gsbase, gsbase + ngs)
    dl = rng.choice([60, 60, 60, 150])
    rows.append(f"{bid},1000000000,{src},{hot},2.5,3000000000,{dl}")
    bid += 1
# 20 random-pair flows, mixed deadlines
for i in range(20):
    src = rng.randrange(gsbase, gsbase + ngs)
    dst = rng.randrange(gsbase, gsbase + ngs)
    while dst == src or (src == hot) or (dst == hot):
        src = rng.randrange(gsbase, gsbase + ngs)
        dst = rng.randrange(gsbase, gsbase + ngs)
    dl = rng.choice([60, 150, 150])
    rows.append(f"{bid},1000000000,{src},{dst},2.5,3000000000,{dl}")
    bid += 1
print("burst_id,start_time_ns,from_node_id,to_node_id,target_rate_mbps,duration_ns,deadline_ms")
for r in rows:
    print(r)
EOF
cp /tmp/schedule_gen.log /tmp/deadline_udp_burst_schedule.csv
wc -l /tmp/deadline_udp_burst_schedule.csv

for ARM in base mech; do
  D=$ROOT/$ARM
  rm -rf "$D"; mkdir -p "$D/logs_ns3"
  if [ "$ARM" = "mech" ]; then QT="purge_srpf"; else QT="droptail"; fi
  cat > "$D/config_ns3.properties" << EOF
simulation_end_time_ns=9000000000
simulation_seed=123456789
satellite_network_dir="../../paper/satellite_networks_state/gen_data/$(basename $NETDIR)"
satellite_network_routes_dir="../../paper/satellite_networks_state/gen_data/$(basename $NETDIR)/$(basename $FSTATE)"
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
purge_min_hop_latency_ns=3000000
EOF
  cp /tmp/deadline_udp_burst_schedule.csv "$D/"
  echo "=== running arm: $ARM (queue=$QT) ==="
  cd $HOME/hypatia/ns3-sat-sim/simulator
  ./waf --run="main_satnet --run_dir='$D'" > /tmp/starlink_$ARM.log 2>&1
  echo "ends: $(grep -c 'SIMULATION END' /tmp/starlink_$ARM.log)"
  grep -E 'QUEUESTATS|PURGE_TOTAL|aborted' /tmp/starlink_$ARM.log | head -3
done

echo "=== ANALYSIS ==="
python3 - << 'EOF'
import csv, os
root = os.path.expanduser("~/hypatia/runs_starlink")
for arm in ("base", "mech"):
    d = os.path.join(root, arm, "logs_ns3")
    sent = 0
    try:
        with open(os.path.join(d, "deadline_udp_sent.csv")) as f:
            for row in csv.DictReader(f):
                try:
                    sent += int(row["sent_packets"])
                except (TypeError, ValueError):
                    continue
    except FileNotFoundError:
        print(arm, "no sent log"); continue
    arr = on = hot_arr = hot_on = rand_arr = rand_on = 0
    try:
        with open(os.path.join(d, "deadline_udp_packets.csv")) as f:
            for row in csv.DictReader(f):
                try:
                    ot = int(row["on_time"]); b = int(row["burst_id"])
                except (TypeError, ValueError):
                    continue
                arr += 1; on += ot
                if b < 20:
                    hot_arr += 1; hot_on += ot
                else:
                    rand_arr += 1; rand_on += ot
    except FileNotFoundError:
        pass
    print(f"{arm}: sent={sent} arrived={arr} on_time={on} "
          f"overall_on={on/sent if sent else 0:.4f} | "
          f"hot: arr={hot_arr} on={hot_on} | random: arr={rand_arr} on={rand_on}")
EOF
