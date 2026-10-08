#!/bin/bash
set -u
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy.log 2>&1
tail -1 /tmp/deploy.log

D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech
sed -i '/^isl_queue_type=/a purge_enable_srpf=false' "$D/config_ns3.properties"
rm -f "$D/logs_ns3/deadline_udp_sent.csv" "$D/logs_ns3/deadline_udp_packets.csv"

cd ~/hypatia/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='$D'" > /tmp/mech_bisect.log 2>&1
echo "SIM_END_count: $(grep -c 'SIMULATION END' /tmp/mech_bisect.log)"
grep -E 'PURGE_TOTAL|srpf' /tmp/mech_bisect.log | head -2
python3 - << 'EOF'
import csv, os
d = os.path.expanduser("~/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech/logs_ns3")
arr = on = 0
for r in csv.DictReader(open(d + "/deadline_udp_packets.csv")):
    try:
        on += int(r["on_time"]); arr += 1
    except (TypeError, ValueError):
        continue
print(f"purge-only arm: arrived={arr} on_time={on}")
EOF
