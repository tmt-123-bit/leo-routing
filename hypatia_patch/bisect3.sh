#!/bin/bash
# Line-level bisect: chase runs but the drop decision is compiled out.
set -u
SN=$HOME/hypatia/ns3-sat-sim/simulator/contrib/satellite-network
sed -i 's/m_purgeActive (true)/m_purgeActive (false)/' "/mnt/f/LEO研究代码汇总/hypatia_patch/model/purge-srpf-queue.h"
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy.log 2>&1
tail -1 /tmp/deploy.log
D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech
rm -f "$D/logs_ns3/deadline_udp_packets.csv"
cd ~/hypatia/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='$D'" 2>&1 | grep -E 'QUEUESTATS|PURGE_TOTAL|aborted' | head -3
python3 - << 'EOF'
import csv, os
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
print(f"chase-only arm: arrived={arr} on_time={on}")
EOF
sed -i 's/m_purgeActive (false)/m_purgeActive (true)/' "/mnt/f/LEO研究代码汇总/hypatia_patch/model/purge-srpf-queue.h"

