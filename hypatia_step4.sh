#!/bin/bash
set -u
T="$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp"
P="$HOME/hypatia/ns3-sat-sim/simulator/contrib/basic-sim/tools/plotting/plot_tcp_flow"
cd "$P" || exit 1
for R in kuiper_630_isls_sat_one_17_to_18_with_TcpNewReno_at_10_Mbps kuiper_630_isls_sat_many_17_to_18_with_TcpNewReno_at_10_Mbps; do
  mkdir -p "$T/data/$R" "$T/pdf/$R"
  python3 plot_tcp_flow.py "$T/runs/$R/logs_ns3" "$T/data/$R" "$T/pdf/$R" 0 1000000000 >/dev/null 2>&1 || true
done
echo "data files:"
ls "$T/data/kuiper_630_isls_sat_one_17_to_18_with_TcpNewReno_at_10_Mbps/" | head -4
cd "$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper" || exit 1
python3 step_5_verify.py && echo FINAL_VERIFY_PASS
