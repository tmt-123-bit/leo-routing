#!/bin/bash
R=$HOME/hypatia/integration_tests/test_manila_dalian_over_kuiper/temp/runs_deadline
for ARM in base mech; do
  echo "=== $ARM isl_utilization (head+totals) ==="
  head -2 "$R/$ARM/logs_ns3/isl_utilization.csv"
  awk -F, 'NR>1{s+=$NF; n++} END{printf "rows=%d total_tx=%d\n", n, s}' "$R/$ARM/logs_ns3/isl_utilization.csv"
done
