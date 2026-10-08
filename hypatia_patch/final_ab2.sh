#!/bin/bash
set -u
R=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline
cd ~/hypatia/ns3-sat-sim/simulator
for ARM in base mech; do
  D=$R/$ARM
  rm -f "$D/logs_ns3/deadline_udp_sent.csv" "$D/logs_ns3/deadline_udp_packets.csv"
  echo "=== $ARM ==="
  ./waf --run="main_satnet --run_dir='$D'" > /tmp/final_$ARM.log 2>&1
  echo "ends: $(grep -c 'SIMULATION END' /tmp/final_$ARM.log)"
  grep -E 'aborted|Aborted|error' /tmp/final_$ARM.log | head -2
  grep -E 'QUEUESTATS|PURGE_TOTAL' /tmp/final_$ARM.log
done
bash /mnt/f/LEO研究代码汇总/hypatia_patch/check_ab.sh 2>&1 | tail -3
