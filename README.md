# 低轨卫星网络分布式动态路由

暂定题目：**基于实时 Dec-POMDP 建模的队列感知低轨卫星网络分布式动态路由研究**

## 摘要

我研究的是低轨卫星网络中的分布式下一跳路由。每颗卫星只读取自身队列、一跳邻居和相邻链路状态，不依赖地面中心持续下发全局路径。训练采用 CTDE/MAPPO：24 颗卫星同时作为 Agent，共享同一个候选邻居 Actor；训练阶段由集中式图 Critic 和团队奖励学习协作，执行阶段每颗卫星独立选择下一跳。

当前代码、正式多 seed 实验、受控消融、TLE 跨星座测试和训练诊断已经跑完。结果说明 MAPPO 明显优于 Q-routing 和本地启发式，但没有全面超过拥有全局链路状态的 Dijkstra。队列感知和局部 credit assignment 得到消融支持；原来的链路寿命 mask 在当前场景中反而降低性能，原因是环境只缩短了 `T_rem`，并没有让链路真实断开。这一项不能作为论文贡献，需要放到 TLE/ns-3 真实断链环境中重新验证。

## 1. 具体问题

LEO 路由不是单纯求一条静态最短路，主要有三个问题：

1. 拓扑和业务状态变化快，全网状态同步和集中重算不适合作为每一跳的唯一依据；
2. 只看传播时延会忽略下一跳队列和链路负载，热点流量下容易积压；
3. 卫星各自追求局部最优时，可能把流量集中到相同链路，需要团队奖励和 credit assignment 协调。

当前研究问题收敛为：

> 在只允许使用本地和一跳邻居信息的条件下，能否用共享参数 MAPPO 学到一个分布式下一跳策略，使投递率、队列和尾时延接近全局 Dijkstra，并优于本地启发式和 Q-routing？

## 2. 当前设计

### 2.1 环境

- 24 颗卫星对应 24 个 Agent；
- 每个 slot 所有 Agent 读取同一份冻结拓扑和队列快照，再同时决策；
- 每个 active Agent 只处理本地 FIFO 队首包；
- 动作是 `NO_OP + 最多 6 个候选下一跳`；
- 环境统一批量解析容量争用、转发、投递、deadline、外生到达和队列更新；
- packet ID 守恒、单一归属、单 slot 单次发送、链路容量和动作 mask 都有自动测试。

### 2.2 Actor 与 Critic

Actor 对每个候选邻居使用同一个 scorer。每个候选有 26 维输入，包含：

- 自身和邻居队列；
- 传播时延、剩余带宽、负载、可靠性、`T_rem`；
- 当前节点、邻居和目的节点的位置编码；
- packet class、等待时间、TTL、visited、backtrack 和 route-switch context。

执行时 Actor 只读取本地候选，不读取全网队列或未来拓扑。Critic 只在训练阶段读取全局节点/边图状态，使用共享编码、边感知注意力和 permutation-invariant pooling。

### 2.3 团队奖励和 credit

```text
R_team =
    - 1.0 * C_delay
    - 0.8 * C_queue
    - 0.5 * C_imbalance
    - 0.2 * C_switch
    + 2.0 * G_throughput
    - 0.1 * C_control
    - 2.0 * C_drop
```

前六项对应端到端时延、全网队列、负载不均衡、路由切换、吞吐量和控制开销；`C_drop` 是可靠投递保护项。每个 active Agent 再得到零均值局部修正：

```text
credit_i = r_local_i - mean(r_local_active)
r_i = R_team + 0.25 * credit_i
```

credit 总和为 0，所以平均 Agent reward 仍等于团队奖励。

## 3. 正式实验

主实验使用 5 个场景、3 个 policy seed、每个 run 50,000 slots。训练、validation、test workload 分别为 `9001..9020`、`10001..10050`、`11001..11050`。最终得到 2,500 个 test episode、450 行聚合指标和 300 行配对检验，无缺失值。

| 场景 | MAPPO 投递率 | 全局 Dijkstra | Q-routing | full heuristic | MAPPO P95 | Dijkstra P95 |
|---|---:|---:|---:|---:|---:|---:|
| low_load | 0.9021 | 0.9070 | 0.6319 | 0.5576 | 6.686 | 6.395 |
| medium_load | 0.6890 | 0.7255 | 0.4258 | 0.3375 | 12.273 | 12.580 |
| hotspot_high_load | 0.2763 | 0.2948 | 0.2396 | 0.1534 | 20.204 | 21.014 |
| frequent_break | 0.4104 | 0.4096 | 0.2654 | 0.2705 | 16.712 | 16.858 |
| fault_links | 0.6173 | 0.6527 | 0.3748 | 0.3201 | 13.661 | 14.210 |

目前能下的结论：

- MAPPO 对 Q-routing 和 full heuristic 的投递率优势在五场景均显著；
- MAPPO 没有全面超过全局 Dijkstra；
- frequent 场景的 MAPPO 与 Dijkstra 投递率统计上无差异；
- hotspot 和 fault 中，MAPPO 的 P95 比 Dijkstra 分别低 0.810 和 0.549 slot；
- 当前方法的合理定位是“局部信息下接近全局路由，并明显优于分布式/本地基线”，不是“全面超过全局最短路”。

原始文件：

- [逐 episode 指标](experiments/EXP-004-FULL/episode_metrics.csv)
- [聚合指标和 95% CI](experiments/EXP-004-FULL/aggregate_metrics.csv)
- [配对检验](experiments/EXP-004-FULL/paired_tests.csv)

> **复现命令（重要）**：`run_exp004_mappo.py` 默认 `--mode quick`（300 步，仅供开发快速验证），**论文级结果必须用 `--mode full`（50,000 步）**。一键复现/重跑见 [`run_ieee_reproduction.sh`](run_ieee_reproduction.sh)（含区分"代码漂移 vs 训练预算"的复现核查）。本仓库经过一次系统优化（trainer Critic 根因修复 + 真实断链 + 拥塞激活等），所有环境/奖励改动会使旧实验表作废，详见 [`OPTIMIZATION_CHANGES.md`](OPTIMIZATION_CHANGES.md)。

## 4. 受控消融

消融使用 medium 和 frequent 两个场景、3 个 policy seed、每 run 5,000 slots、每组 50 个 test workload，共 2,100 episode。它只判断模块方向，不与 50,000-step 主表直接比较绝对值。

| 去掉的机制 | 主要变化 | 判断 |
|---|---|---|
| queue awareness | medium 投递率 -3.38 pp，P95 +1.142，平均队列 +0.0825 | 队列感知有支持 |
| local credit | medium 投递率 -9.39 pp；frequent -5.58 pp，P95 和队列同时变差 | credit 有明确支持 |
| packet context | 两场景主要指标无显著变化 | 当前预算不支持独立贡献 |
| graph critic | medium 投递率反而 +4.81 pp；frequent 无显著变化 | 不支持图 Critic 优于 flat critic |
| PPO protections | 投递率无显著变化；medium P95 变差 0.696 | 只看到部分稳定性作用 |
| lifetime mechanism | medium 投递率 +12.16 pp；frequent +41.90 pp | 当前寿命 mask 有害 |

寿命结果需要特别说明：代码中的 `frequent_break` 只把部分链路 `T_rem` 调低，链路仍为 `available=True`，并没有真实断开。full 策略过滤了仍可转发的链路，所以性能变差。这里应改称“短 `T_rem` 告警压力场景”，不能用来证明故障恢复或寿命约束有效。

- [消融逐 episode 指标](experiments/ABLATION-FULL/episode_metrics.csv)
- [消融配对效应](experiments/ABLATION-FULL/paired_ablation_effects.csv)

## 5. TLE 跨星座测试

我从冻结的 CelesTrak TLE 中分别生成 Starlink 和 OneWeb 的 24 星、30 时隙连通快照。位置和传播时延来自 SGP4；ISL 选择、100 Mbps 容量和 0.995 reliability 是仿真假设。

Starlink 上训练 3 个 MAPPO seed，每个 20,000 slots；同一 checkpoint 在 OneWeb 上 zero-shot 测试，不微调。

| 拓扑 | MAPPO 投递率 | 全局 Dijkstra | full heuristic | MAPPO P95 |
|---|---:|---:|---:|---:|
| Starlink | 0.3754 | 0.3983 | 0.1023 | 18.750 |
| OneWeb zero-shot | 0.2997 | 0.3175 | 0.2032 | 18.445 |

从 Starlink 到 OneWeb，MAPPO 投递率下降 7.57 个百分点。它能跨星座运行，但没有超过 Dijkstra，也不能等同真实运营网络实测。

- [TLE 聚合结果](experiments/TLE-STARLINK-FULL/aggregate_metrics.csv)
- [TLE 配对检验](experiments/TLE-STARLINK-FULL/paired_tests.csv)

## 6. 训练诊断

15 个正式 run 中：

- entropy collapse：0/15；
- tail Actor gradient `<0.01`：0/15；
- validation 完全不可区分：0/15；
- critic gradient clipping：15/15。

Critic 尾段原始梯度约为 7.87–283.76，说明裁剪是必要的，Critic 仍是后续需要改进的部分。

## 7. ns-3/Hypatia 接口

当前新增：

- `ns3_policy_protocol.schema.json`：版本化的 24-Agent slot-state 协议；
- `ns3_policy_bridge.py`：加载 MAPPO checkpoint，读取 JSONL 状态并返回下一跳动作。

使用正式 checkpoint 的 smoke test 中，bridge 与直接 Actor 的 24 个动作完全一致，全部满足 mask，其中 12 个为非 `NO_OP` 决策。

本机有 WSL2 Ubuntu 22.04，**Hypatia 已安装在 `F:/third_party/hypatia`**（更正：早期版本曾写"未安装"，与实际不符）。当前的 ns-3 集成有两个阶段：

1. **v1.0 控制桥（离线 JSONL）**：`ns3_policy_bridge.py` + `ns3_policy_protocol.schema.json`（24-Agent、26 维特征）。正式 checkpoint 的 smoke test 中，bridge 与直接 Actor 的 24 个动作完全一致，全部满足 mask，其中 12 个为非 `NO_OP` 决策。
2. **v3.0 真实 packet-level 闭环（TCP-jsonl，per-lookup policy 拦截）**：位于 `leo-routing-current-results/NS3-CLOSED-LOOP-33/`。已编译的 closed-loop 可执行文件以 exit 0 跑通，v3.0 在 3 个 seed × 2 个 variant（full vs no_lifetime）上运行了真实数据面仿真，使用 fixture 注入的 ISL 中断（`attempt5_overlay_success`）。

**当前 ns-3 验证的边界（诚实说明）**：
- fixture 是 33 星 Kuiper integration-test 子集（42 条 ISL），**与训练用的 Starlink 24 星（4×6）拓扑不同**——需生成匹配的 starlink_24 fixture 才能做干净的部署测试；
- 观测是 **domain-adapted**（队列占用由 NetDevice 实时队列归一化、负载由排队字节计算等，与训练 env 的逻辑队列/累计 used_rate 不同）——是部署验证，不是同语义迁移；
- 统计功效不足：仅 1 条 ISL 中断、2 个 UDP burst、5s，no_lifetime 仅 5 个 drop / 5004 包（差 0.1pp）。**扩到 100+ drop 才能做统计声明**（见 `run_ieee_reproduction.sh` 的 ns-3 阶段）。

后续计划仍是完整的动态星座闭环：

```text
Hypatia/轨道模块生成动态星座
    -> ns-3 维护链路、队列、业务流、真实断链和丢包
    -> 每个控制时隙发送局部候选状态
    -> Python MAPPO bridge 返回下一跳
    -> ns-3 执行动作并记录 flow/queue/overhead trace
```

## 8. 主要代码

- `leo_multiagent_env.py`：24-Agent 同步环境与团队奖励；
- `cleanmarl_leo_multiagent_wrapper.py`：CleanMARL 接口；
- `mappo_design.py`：共享候选 Actor、图 Critic、GAE 和 PPO 工具；
- `run_exp004_mappo.py`：正式训练、baseline、held-out 评测和统计；
- `run_ablation_experiments.py`：受控消融；
- `tle_topology_builder.py`：TLE/SGP4 拓扑生成；
- `run_tle_training_experiment.py`：TLE 训练和跨星座测试；
- `run_exp005_diagnostics.py`：训练诊断；
- `test_mappo_design.py`：28 项自动化测试；
- `MAPPO当前设计与问题审查.md`：完整设计、问题和实验记录。

运行测试：

```powershell
py -m unittest test_mappo_design.py
```

当前结果应按下面的边界使用：队列感知、局部 credit 和分布式 MAPPO 主体已有实验支持；链路寿命、图 Critic 优势和真实 packet-level 效果还没有被证明。

## 9. 参考入口

- [Distributed routing article, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1000936122001297)
- [Queue-aware LEO routing, arXiv:2306.01346](https://arxiv.org/abs/2306.01346)
- [Queue-aware multi-agent LEO routing, arXiv:2605.04448](https://arxiv.org/abs/2605.04448)
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Hypatia](https://github.com/snkas/hypatia)
- [ns-3](https://www.nsnam.org/)
