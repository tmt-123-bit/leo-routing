# LEO 路由研究过程原型：Phase 1-3

这个文件夹不是最终算法实现，而是论文前期“中间研究过程”的 MATLAB 原型。它现在覆盖从传统全局 Dijkstra baseline 到逐跳本地决策的过渡步骤。后续主线代码建议看 `leo-routing-preliminary-matlab`，因为那里进一步整理了局部观测、动作掩码、reward、控制开销和计算开销。

它对应后续研究路线里的前三块：

1. **动态拓扑 + Dijkstra 基线**
   - 对应时变图建模：`G(t) = (V, E(t), X(t))`
   - 作用：验证 LEO 拓扑会随时间变化，并建立传统最短路基线。
   - 注意：Dijkstra 不是本文最终方法，只是 baseline。

2. **加入队列和链路负载的 queue/load-aware baseline**
   - 对应变量：`q_i(t)`、`q_j(t)`、`delay_ij(t)`、`rho_ij(t)`、`T_rem_ij(t)`
   - 作用：证明只看传播时延的最短路径不一定是最低端到端时延路径。
   - 注意：这一块仍然不是最终 MAPPO/CTDE，只是用来铺垫为什么要做队列感知和负载感知。

3. **逐跳本地决策 + 动作掩码 + 防环**
   - 作用：把“全局一次性算完整路径”改成“当前卫星只根据邻居局部状态选择下一跳”。
   - 这是后续接 Dec-POMDP / MAPPO / 其他 MARL 算法前的中间层。
   - 注意：这里的本地策略仍然是手写启发式，不是最终学习算法。

## 文件

| 文件 | 作用 |
|---|---|
| `run_phase12.m` | MATLAB 主脚本，包含 Phase 1-3。 |
| `outputs/phase1_dynamic_topology_dijkstra.csv` | 动态拓扑下的全局 Dijkstra 路径记录。 |
| `outputs/phase2_policy_metrics.csv` | delay-only 和 queue/load-aware 两种全局策略的指标对比。 |
| `outputs/phase3_local_next_hop_metrics.csv` | 逐跳本地决策策略的指标。运行脚本后生成。 |
| `outputs/phase123_policy_metrics.csv` | Phase 1-3 策略汇总对比。运行脚本后生成。 |
| `outputs/phase123_policy_comparison.png` | Phase 1-3 指标对比图。运行脚本后生成。 |

## 怎么运行

在 MATLAB 里打开这个文件夹：

```matlab
cd('F:\leo-routing-phase12-matlab')
run_phase12
```

运行后会自动生成或更新 `outputs` 文件夹。

## 和论文建模的对应关系

脚本里的拓扑状态与论文变量对应如下：

```text
G(t) = (V, E(t), X(t))

x_i(t)  = [q_i(t), Q_i^max, service_i(t)]
x_ij(t) = [d_ij(t), C_ij(t), r_ij(t), rho_ij(t), p_ij^out(t), T_rem_ij(t)]
```

MATLAB 代码中的字段：

| MATLAB 字段 | 论文变量 | 含义 |
|---|---|---|
| `G.Nodes.Queue` | `q_i(t)` | 卫星当前队列长度 |
| `G.Nodes.QMax` | `Q_i^max` | 队列上限 |
| `G.Nodes.Service` | `service_i(t)` | 每个时隙服务能力 |
| `G.Edges.DelayMs` | `d_ij(t)` / `delay_ij(t)` | 链路传播时延 |
| `G.Edges.CapacityMbps` | `C_ij(t)` | 链路容量 |
| `G.Edges.UsedRateMbps` | `r_ij(t)` | 已占用速率 |
| `G.Edges.Rho` | `rho_ij(t)` | 链路负载率 |
| `G.Edges.Reliability` | `reliability_ij(t)` | 链路可靠性 |
| `G.Edges.POut` | `p_ij^out(t)` | 链路中断概率 |
| `G.Edges.TRem` | `T_rem_ij(t)` | 链路剩余可用时间 |

## Phase 1：动态拓扑 + Dijkstra

Phase 1 做的是：

```text
对每个时间片 t:
    构造 G(t)
    用 DelayMs 作为边权
    用 Dijkstra 求 src 到 dst 的全局最短传播时延路径
    记录边数、跳数、路径时延和路径
```

这一块的论文定位：

```text
Dijkstra 是传统最短路径基线，用来说明在动态 LEO 拓扑下，路径会随时间变化。
它不是本文最终方法。
```

## Phase 2：加入队列和链路负载

Phase 2 比较两种全局路径策略：

| 策略 | 边权 | 目的 |
|---|---|---|
| `B0_delay_only_dijkstra` | `delay_ij` | 只看传播时延的传统最短路。 |
| `B1_queue_load_dijkstra` | `delay_ij + queue_j + rho_ij` | 观察队列和链路负载是否能缓解拥塞。 |

queue/load-aware 边权写成：

```text
cost_ij =
    delay_ij
    + beta_q   * D_ref * (q_j / Q_max)
    + beta_rho * D_ref * rho_ij
```

其中：`delay_ij` 对应传播时延，`q_j / Q_max` 对应下一跳队列占用率，`rho_ij` 对应链路负载率，`beta_q` 和 `beta_rho` 是队列和负载项的权重。

这一块的论文定位：

```text
Phase 2 不是最终算法，而是验证“只按传播时延选路可能忽略拥塞”。
如果 queue/load-aware baseline 的最大队列和 P95 时延下降，就说明队列/负载状态确实值得进入后续 Dec-POMDP/MAPPO 的局部观测设计。
```

## Phase 3：逐跳本地决策 + 动作掩码 + 防环

Phase 3 做的是把全局 Dijkstra 路由改成更接近强化学习环境的形式：

```text
对每个数据包:
    current = src
    visited = {src}
    while current != dst:
        构造当前时刻 G(t)
        只读取 current 的邻居状态
        action mask 去掉已经访问过的节点
        在剩余邻居里选择下一跳
        如果没有可选动作，则丢包并记为 loop/mask drop
        如果超过最大跳数 TTL，则丢包并记为 ttl drop
```

本地下一跳代价现在写成：

```text
local_cost =
    delay_ij
    + beta_q   * D_ref * (q_j / Q_max)
    + beta_rho * D_ref * rho_ij
    + beta_progress * D_ref * topo_distance(j, dst)
    + beta_Trem * D_ref / max(1, T_rem_ij)
```

这里的意思很简单：下一跳不能只看当前链路时延，还要看下一跳队列、链路负载、离目的节点的拓扑距离，以及链路剩余可用时间。`visited` 和 `maxLocalHops` 是手写防环机制。以后接 MAPPO 时，`visited` 和邻居可达性就可以变成动作掩码。

这一块的论文定位：

```text
Phase 3 是从传统路由到学习型路由的过渡。
它不再假设每个节点都知道全局最短路，而是模拟“卫星根据本地观测选择下一跳”。
这为后续 Dec-POMDP 建模、动作空间定义、动作掩码和防环约束打基础。
```

## 后续怎么接到最终方法

老师的建议是：后续不再继续深挖 Dijkstra。Dijkstra 保留为 baseline。主线应转向逐跳本地决策、真实星座、学习型算法和通信开销约束。

后续路线建议如下：

```text
Phase 1: 动态拓扑 + Dijkstra baseline
Phase 2: queue/load-aware global baseline
Phase 3: local next-hop + action mask + loop prevention
Phase 4: 封装成 Dec-POMDP / MARL 环境
Phase 5: 接入 MAPPO 或其他现成 MARL 算法库
Phase 6: 加入真实星座拓扑和学习型对比方法
Phase 7: 加入通信开销预算约束
```

所以向老师解释时可以说：

```text
前两块不是最终方法，而是为了验证仿真环境和问题动机。
Phase 3 把全局路径规划改成逐跳本地决策，为 Dec-POMDP/MARL 做接口准备。
最终方法仍然是“实时 Dec-POMDP + 学习型多智能体路由 + 队列/链路寿命/通信开销约束”。
```

## 参考资料

1. **MAPPO official implementation**
   - GitHub: https://github.com/marlbenchmark/on-policy
   - 用途：老师给的 MAPPO 官方实现，后续接 MAPPO 时优先参考。

2. **CleanMARL**
   - GitHub: https://github.com/AmineAndam04/cleanmarl
   - 用途：单文件 MARL 实现，更适合理解算法和快速改环境接口。

3. **BenchMARL**
   - GitHub: https://github.com/facebookresearch/BenchMARL#algorithm
   - 用途：较完整的 MARL benchmark 框架，比较重，后期作为算法组织参考。

4. **MA-DRL satellite routing simulator**
   - GitHub: https://github.com/SatCom-TELMA/MA-DRL_Routing_Simulator
   - 用途：参考卫星路由仿真、Dijkstra/Q-routing/MA-DRL baseline 的组织方式。

5. **Hypatia: LEO satellite network simulator**
   - GitHub: https://github.com/snkas/hypatia
   - 用途：参考动态 LEO 拓扑、时间片链路变化和路径计算。

6. **LEOPath**
   - GitHub: https://github.com/Fundacio-i2CAT/LEOPath
   - 用途：参考 Python 版 LEO 路由仿真框架、拓扑与路由评估。

7. **Queue-Aware and Resilient Routing in LEO Satellite Networks Using Multi-Agent Reinforcement Learning**
   - arXiv: https://arxiv.org/abs/2605.04448
   - 用途：支撑队列积压和实时流量变化会影响 LEO 路由决策。

8. **Real-Time Routing Design for LEO Satellite Networks: An Enhanced Multi-Agent DRL Approach**
   - IEEE Xplore: https://ieeexplore.ieee.org/document/10693714/
   - 用途：支撑 RTMDP/实时路由建模思路。

9. **NetworkX shortest path / Dijkstra documentation**
   - https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
   - 用途：虽然本代码是 MATLAB，但 Dijkstra baseline 的实验逻辑和常见实现方式可参考该文档。
