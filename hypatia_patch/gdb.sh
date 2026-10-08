#!/bin/bash
set -u
D=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline/mech
cd ~/hypatia/ns3-sat-sim/simulator
./waf --run="main_satnet --run_dir='$D'" --command-template="gdb -batch -ex run -ex 'bt 12' --args %s" 2>&1 | grep -A 14 'SIGSEGV\|Program received' | head -18
