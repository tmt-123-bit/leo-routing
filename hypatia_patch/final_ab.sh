#!/bin/bash
# Final A/B with dequeue-time purge (fixes the ns-3 UDP error-path stall).
set -u
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy.log 2>&1
D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline
# clean bisect flags from mech config
sed -i '/^purge_enable=/d; /^purge_enable_srpf=/d; /^purge_debug_hops=/d' "$D/mech/config_ns3.properties"
bash /mnt/f/LEO研究代码汇总/hypatia_patch/run_ab.sh > /tmp/ab.log 2>&1
grep -cE 'SIMULATION END' /tmp/ab.log
grep -E 'QUEUESTATS|PURGE_TOTAL' /tmp/ab.log | head -2
bash /mnt/f/LEO研究代码汇总/hypatia_patch/check_ab.sh 2>&1 | tail -3
