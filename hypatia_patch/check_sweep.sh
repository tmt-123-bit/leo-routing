#!/bin/bash
ls ~/hypatia/family_results/ 2>/dev/null || echo NO_RESULTS_DIR
for D in ~/hypatia/family_results/*/; do
  echo "=== $(basename $D)"
  cat "$D/summary.csv" 2>/dev/null
done
echo "--- gen_data dirs:"
ls ~/hypatia/paper/satellite_networks_state/gen_data/ | head -8
