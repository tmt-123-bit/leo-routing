# next_hop_v2 这块在做什么

老师问的核心其实是：后面不能只停在“全局 Dijkstra 改几个权重”，要说明学习策略怎么接上去。

我现在把后续路线先收成这几步：

1. `global_dijkstra` 只保留成 baseline  
   它用全局链路状态算完整路径，后面论文里用来说明“全局最短路在动态拥塞下不一定稳”。

2. `local_delay_noloop` 做逐跳本地决策  
   每颗卫星只看当前邻居，选下一跳；数据包里带 visited 集合，已经走过的节点不再选，用来先解决防环问题。

3. `local_q_life_noloop` 加队列、负载、可靠性和链路剩余寿命  
   这块对应论文里的主要想法：下一跳不是只看传播时延，还要看邻居堵不堵、链路忙不忙、链路是不是快断。

4. `local_q_life_export` 导出 MAPPO 后面要用的数据  
   这版还没有训练 MAPPO，但是会导出类似 rollout 的表：当前节点、动作、候选动作数量、队列、链路负载、可靠性、剩余寿命、即时奖励等。

生成文件在：

```text
try_matlab/tmp_out_v2/stat_v2.csv
try_matlab/tmp_out_v2/slot_log_v2.csv
try_matlab/tmp_out_v2/mappo_rollout_like_v2.csv
try_matlab/tmp_out_v2/quick_plot_v2.png
```

运行方式：

```matlab
cd try_matlab
try_next_hop_v2
```

## 后面接 MAPPO 时不自己硬写算法

老师截图里提到的参考库可以这样用：

- `marlbenchmark/on-policy`：官方 MAPPO 实现，后面如果正式训练，优先参考这个。
- `AmineAndam04/cleanmarl`：单文件实现比较清楚，适合先看 MAPPO 的数据流。
- `facebookresearch/BenchMARL`：比较重，更适合后面做规范 benchmark，不适合一开始就改。

所以我的策略是：

```text
MATLAB 先把环境、局部决策、防环、指标跑通
↓
导出 obs/action/reward/next_obs/done 这样的轨迹
↓
用 Python/MAPPO 库接训练
↓
训练好后把 Actor 的打分逻辑换回逐跳下一跳选择
```

## 现在这块和论文思路的关系

这一版对应的是论文方法的前半段：

```text
全局 Dijkstra baseline
    ↓
逐跳本地决策
    ↓
visited 防环
    ↓
队列 + 负载 + 链路寿命约束
    ↓
MAPPO 环境接口
```

还没做完的部分：

- 真实星座轨迹：后面接 Hypatia、LEOPath 或 TLE 数据生成动态拓扑，不在这里手编假数据冒充真实星座。
- 学习型对比方法：后面至少加 Q-routing / DQN 类 baseline，再和 MAPPO 比。
- 通信开销预算：这一版已经统计 `ctrlRatio = control bytes / data bytes`，后面再做 Hello 周期、状态字段裁剪和预算约束。

## 这次跑出来的现象

当前只是小规模 toy 拓扑，不能当论文结论，但它能说明下一步该往哪里调：

```text
global_dijkstra:
  dropRate = 0
  avgDelay = 29.23
  p95Delay = 49.99
  maxQ = 31
  Jain = 0.467
  badFirstHop = 26
  ctrlRatio = 0.036

local_q_life_noloop:
  dropRate = 0.0017
  avgDelay = 36.73
  p95Delay = 82.01
  maxQ = 20
  Jain = 0.631
  badFirstHop = 0
  ctrlRatio = 0.064
```

我的理解是：

- 全局 Dijkstra 在这个小场景里平均时延最低，所以它可以继续当强 baseline；
- 本地队列/寿命策略的平均时延更高，因为逐跳决策会绕路；
- 但是它把风险首跳从 26 次压到 0 次，最大队列也从 31 降到 20，负载均衡明显更好；
- `ctrlRatio` 超过了先设的 5% 预算，所以后面必须把 Hello 字段、Hello 周期和通信预算一起做实验。

所以这块不是要证明本地启发式已经赢了，而是说明：只靠手工权重很难同时兼顾时延、队列、链路寿命和控制开销，这正好引出后面用 MAPPO/CTDE 学权重和策略。

## 这块能跟老师怎么说

我现在不再把 Dijkstra 当成后续主方法，而是把它固定成全局最短路 baseline。后续方法改成逐跳本地下一跳决策，每个数据包维护已访问节点集合来防止环路，并在候选动作过滤和代价中加入队列、链路负载、可靠性和链路剩余寿命。这样可以先验证“局部状态是否真的有用”。同时，代码开始导出 MAPPO 所需的交互轨迹，后面不准备从零实现 MAPPO，而是基于官方 `on-policy` 或 `cleanmarl` 做环境适配。
