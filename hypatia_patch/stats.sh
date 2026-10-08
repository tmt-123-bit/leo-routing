#!/bin/bash
set -u
tail -2 /tmp/deploy.log 2>/dev/null
D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech
cd ~/hypatia/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='$D'" 2>&1 | grep -E 'QUEUESTATS|PURGE_TOTAL|aborted'
