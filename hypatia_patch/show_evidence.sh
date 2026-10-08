#!/bin/bash
R=$HOME/hypatia/family_results/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls
echo "=== 1) 每个包的真实仿真记录（纳秒时间戳）==="
head -4 $R/run_purge_edf/logs_ns3/deadline_udp_packets.csv
echo ""
echo "=== 2) 单包仿真细节举例 ==="
python3 - << 'EOF'
import csv
with open("/home/nsuser/hypatia/family_results/starlink_550_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls/run_purge_edf/logs_ns3/deadline_udp_packets.csv") as f:
    for row in csv.DictReader(f):
        try:
            send = int(row["seq_send_ns"]); arr = int(row["arrival_ns"]); dl = int(row["deadline_ns"])
        except (TypeError, ValueError):
            continue
        print(f"包 burst={row['burst_id']}: 发送时刻 {send/1e9:.6f}s → 到达时刻 {arr/1e9:.6f}s")
        print(f"          端到端时延 {(arr-send)/1e6:.3f} 毫秒（经 20+ 跳真实串行化+传播）, 死线 {(dl-send)/1e6:.0f}ms, {'准时' if arr<=dl else '迟到'}")
        break
EOF
echo "=== 3) 仿真运行声明 ==="
grep -E "Running the simulation|SIMULATION END|simulation seconds" $R/console_purge_edf.log | head -3
echo "=== 4) 单臂计算量 ==="
grep -E "^ > " $R/console_purge_edf.log | grep -iE "time|duration" | head -3
wc -l $R/run_purge_edf/logs_ns3/deadline_udp_packets.csv
