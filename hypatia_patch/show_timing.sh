#!/bin/bash
R=$HOME/hypatia/family_results/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls
echo "=== 每臂的真实墙钟起止时间（ns-3 自带时间戳）==="
for ARM in droptail purge_edf codel; do
  echo "--- $ARM:"
  grep -A 2 "BASIC SIMULATION END" $R/console_$ARM.log | grep -E "Date|Time" | head -2
done
echo ""
echo "=== 各臂结果文件的落盘时刻（证明顺序执行）==="
ls -la --time-style=full-iso $R/run_*/logs_ns3/finished.txt | awk '{print $6, $7, $9}' | sed 's|.*/run_||; s|/logs||'
echo ""
echo "=== ns-3 自己的计时文件 ==="
cat $R/run_purge_edf/logs_ns3/timing_results.txt 2>/dev/null | head -10
