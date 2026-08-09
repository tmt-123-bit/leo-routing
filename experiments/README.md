# 实验结果说明

这里保存可复查的轻量结果，不包含体积较大的训练 checkpoint。

## EXP-004

当前 CSV 来自 quick 模式：3 个 policy seed、2 个场景、300 training steps、20 validation workload 和 20 held-out test workload。训练、验证和测试 seed 互不重叠。

| Scenario | Policy | Delivery ratio | Throughput (packet/slot) | Delivered-packet delay (slot) |
|---|---|---:|---:|---:|
| medium_load | delay-only | 0.183 | 1.168 | 10.007 |
| medium_load | full heuristic | 0.338 | 2.165 | 8.313 |
| medium_load | MAPPO quick | 0.151 | 0.966 | 7.742 |
| frequent_break | delay-only | 0.148 | 0.950 | 7.727 |
| frequent_break | full heuristic | 0.283 | 1.812 | 7.559 |
| frequent_break | MAPPO quick | 0.194 | 1.239 | 8.344 |

`episode_metrics.csv` 是原始 episode 结果，`aggregate_metrics.csv` 使用 policy/workload 两层 bootstrap，`paired_tests.csv` 是配对 Wilcoxon 和多重比较校正。

这组结果只验证实验闭环。MAPPO 没有超过 full heuristic；`medium_load` 下较低的已投递包时延还受到低投递率造成的 survivor bias，不能单独解释为性能更好。

正式配置入口：

```powershell
py run_exp004_mappo.py --mode full --output experiments\EXP-004-full
```

full 模式是 5 个场景、3 个 policy seed、每个 50,000 steps、50 validation 和 50 test workload，本轮 CPU 会话没有把它跑完。

## EXP-005

`run_diagnostics.csv` 只分析 `experiment_manifest.json` 指向的 6 个 selected runs。quick run 中：

- entropy `<0.01`：0/6；
- Actor gradient `<0.01`：0/6；
- Critic gradient 触发 0.5 clipping：6/6；
- validation tuple 完全无区分：0/6。

这些是训练诊断，不是收敛证明。

## TLE-REPLAY

拓扑由冻结 Starlink TLE 和 SGP4 生成，共 24 星、30 个连通时隙。位置和传播时延来自 TLE；ISL 构造、100 Mbps 容量和 0.995 reliability 是仿真假设。

当前结果是 heuristic 的 zero-shot topology-transfer 回放，不是真实 Starlink 实测，也不是在 TLE 拓扑上重新训练后的 MAPPO 结果。

