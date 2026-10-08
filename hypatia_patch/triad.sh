#!/bin/bash
set -u
D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech
cd ~/hypatia/ns3-sat-sim/simulator
for MODE in inert purge_only full; do
  case $MODE in
    inert)     sed -i 's/^purge_enable=.*/purge_enable=false/; s/^purge_enable_srpf=.*/purge_enable_srpf=false/; s/^purge_debug_hops=.*/purge_debug_hops=false/' "$D/config_ns3.properties";;
    purge_only) sed -i 's/^purge_enable=.*/purge_enable=true/;  s/^purge_enable_srpf=.*/purge_enable_srpf=false/' "$D/config_ns3.properties";;
    full)      sed -i 's/^purge_enable=.*/purge_enable=true/;  s/^purge_enable_srpf=.*/purge_enable_srpf=true/'  "$D/config_ns3.properties";;
  esac
  rm -f "$D/logs_ns3/deadline_udp_sent.csv" "$D/logs_ns3/deadline_udp_packets.csv" "$D/logs_ns3/isl_utilization.csv"
  ./waf --run="main_satnet --run_dir='$D'" > /tmp/triad_$MODE.log 2>&1
  PT=$(grep -o 'PURGE_TOTAL,[0-9]*' /tmp/triad_$MODE.log | cut -d, -f2)
  ISLTOT=$(awk -F, 'NR>1{s+=$NF} END{printf "%d", int(s)}' "$D/logs_ns3/isl_utilization.csv" 2>/dev/null)
  python3 - "$MODE" "$PT" "$ISLTOT" << 'EOF'
import csv, os, sys
mode, pt, isltot = sys.argv[1], sys.argv[2], sys.argv[3]
d = os.path.expanduser("~/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech/logs_ns3")
arr = on = 0
try:
    for r in csv.DictReader(open(d + "/deadline_udp_packets.csv")):
        try:
            on += int(r["on_time"]); arr += 1
        except (TypeError, ValueError):
            continue
except FileNotFoundError:
    pass
print(f"{mode}: arrived={arr} on_time={on} purge_total={pt} isl_util_sum={isltot}")
EOF
done
