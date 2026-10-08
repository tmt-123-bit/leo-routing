#!/bin/bash
set -u
bash /mnt/f/LEO研究代码汇总/hypatia_patch/deploy.sh > /tmp/deploy.log 2>&1
tail -1 /tmp/deploy.log
D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech
sed -i 's/^purge_enable=false/purge_enable=true/; s/^purge_enable_srpf=false/purge_enable_srpf=true/' "$D/config_ns3.properties"
grep -q '^purge_debug_hops=' "$D/config_ns3.properties" || echo "purge_debug_hops=true" >> "$D/config_ns3.properties"
cd ~/hypatia/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='$D'" 2>&1 | grep -E 'HOPDEBUG,dst=17$|HOPDEBUG,dst=18$|HOPDEBUG,dst=0$|PURGE_TOTAL' | head -6
