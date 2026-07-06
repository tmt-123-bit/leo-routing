# LEO 分布式动态路由前期研究代码说明

这份代码是论文前期研究原型，不是最终 MAPPO/CTDE 训练代码。它的作用是把论文里提出的研究思路先落到可运行仿真中，验证几个基础判断：

1. LEO 网络应建成时变图 `G(t)=(V,E(t),X(t))`；
2. 只用 Dijkstra 最短传播时延路径，可能忽略队列和链路负载；
3. 加入 `q_j` 和 `rho_ij` 后，是否能降低最大队列和 P95 时延；
4. 加入 `T_rem_ij` 动作掩码后，是否能减少选择快断链路；
5. 在比较路由性能时，也要同时统计控制开销和估算计算开销。

## 文件

| 文件 | 作用 |
|---|---|
| `run_preliminary_leo_routing.m` | MATLAB 前期仿真主脚本，负责 baseline、逐跳本地策略和消融对比。 |
| `leo_marl_env.py` | Python MARL 环境接口，提供 `reset/step/observation/action_mask/reward/info`，后续接 MAPPO、CleanMARL 或其他成熟算法库。 |
| `outputs/policy_metrics.csv` | 不同策略的总体指标。 |
| `outputs/time_slot_logs.csv` | 每个时间片的中间日志。 |
| `outputs/policy_comparison.png` | 指标对比图。 |

## 运行方式

MATLAB 前期仿真和消融对比：

```matlab
cd('F:\leo-routing-preliminary-matlab')
run_preliminary_leo_routing
```

Python 环境接口 smoke test：

```bash
python leo_marl_env.py
```

Python 文件不是新的独立算法，而是后续接 MAPPO/CTDE 的环境层。

## 和论文研究思路的对应关系

论文里的建模：

```text
G(t) = (V, E(t), X(t))

x_i(t)  = [q_i(t), Q_i^max, service_i(t)]
x_ij(t) = [d_ij(t), C_ij(t), r_ij(t), rho_ij(t), p_ij^out(t), T_rem_ij(t)]
```

代码中的对应字段：

| 代码字段 | 论文变量 | 含义 |
|---|---|---|
| `G.Nodes.Queue` | `q_i(t)` | 节点队列 |
| `G.Nodes.QMax` | `Q_i^max` | 队列上限 |
| `G.Nodes.Service` | `service_i(t)` | 节点服务能力 |
| `G.Edges.DelayMs` | `d_ij(t)` | 链路传播时延 |
| `G.Edges.CapacityMbps` | `C_ij(t)` | 链路容量 |
| `G.Edges.UsedRateMbps` | `r_ij(t)` | 已占用速率 |
| `G.Edges.Rho` | `rho_ij(t)` | 链路负载率 |
| `G.Edges.Reliability` | `reliability_ij(t)` | 链路可靠性 |
| `G.Edges.POut` | `p_ij^out(t)` | 链路中断概率 |
| `G.Edges.TRem` | `T_rem_ij(t)` | 链路剩余可用时间 |
| `G.Edges.AgeTo` | `age_j(t)` | 邻居状态信息年龄 |

## 已实现的前期策略

| 策略 | 含义 | 论文定位 |
|---|---|---|
| `P0_delay_only` | 只按传播时延做 Dijkstra | 传统最短路基线 |
| `P1_queue_load` | 代价中加入队列和负载 | 验证队列/负载是否有用 |
| `P2_queue_load_lifetime_mask` | 在 P1 基础上加入 `T_rem_ij > T_safe` 等动作掩码 | 验证链路寿命约束是否有用 |
| `P3_local_full` | 完整逐跳本地策略：队列/负载 + 链路寿命 mask + 可靠性风险 + 目的方向进展 + visited 防环 + TTL | 最接近后续 Actor + action mask 的完整启发式原型 |
| `A1_local_no_queue_load` | 在 P3 基础上去掉队列和负载项 | 消融队列/负载感知是否缓解拥塞 |
| `A2_local_no_lifetime_mask` | 在 P3 基础上去掉链路寿命和带宽/可靠性硬过滤 | 消融动作 mask 与链路寿命约束是否减少风险链路 |
| `A3_local_no_reliability_risk` | 在 P3 基础上去掉可靠性风险代价 | 消融 reliability/risk 奖励项是否影响丢包和风险选择 |
| `A4_local_no_progress` | 在 P3 基础上去掉目的方向进展项 | 消融目的方向进展是否减少绕路和 TTL 超时 |

## 局部观测 z_ij

代码里已经实现了 `localObservationZ()`，对应论文中的：

```text
z_ij(t) = [
  q_j(t) / Q_j^max,
  delay_ij(t) / D_ref,
  rho_ij(t),
  reliability_ij(t),
  T_rem_ij(t) / T_safe,
  age_j(t) / T_hello,
  progress_ij(t)
]
```

这一步是为后面接 MAPPO/CTDE 做准备。当前代码还没有训练 Actor，只是把 Actor 需要的局部观测先构造出来。

## 动作掩码

代码里 `P2` 和 `P3` 使用动作掩码：

```text
T_rem_ij < T_safe           -> mask
reliability_ij < R_min      -> mask
remaining bandwidth < B_min -> mask
q_j >= Q_max                -> mask
```

其中 `P3_local_full` 还额外维护当前数据包的 `visited` 节点集合。如果候选下一跳已经访问过，就直接 mask 掉，避免形成路由环路。如果所有邻居都被 mask 掉，则记为 `loopDrops`；如果超过 `maxLocalHops` 还没到目的节点，则记为 `ttlDrops`。

这不是主要创新点，而是工程约束，用来避免策略选择明显不可用、快断、高风险或会形成环路的链路。

## 局部奖励 r_local

代码中实现了一个局部即时奖励原型：

```text
r_local =
  - delay_norm
  - queue_norm
  - risk
  - lifetime_penalty
  + progress_reward
```

它对应论文里“这一跳到底选哪个邻居”的局部评价。后面做 MAPPO/CTDE 时，可以把它作为 reward shaping 或分析指标。

## 输出指标

`policy_metrics.csv` 包含：

| 指标 | 含义 |
|---|---|
| `avgDelayMs` | 平均端到端时延 |
| `p95DelayMs` | 95% 分位时延 |
| `maxQueue` | 最大队列长度 |
| `avgJainLoad` | 平均 Jain 负载公平性指数 |
| `rerouteRiskEvents` | 选择快断/低可靠链路的风险次数 |
| `controlBytes` | Hello 状态交换开销 |
| `estimatedFLOPs` | 估算的 Actor-like 打分计算量 |
| `invalidActionRatio` | 被动作掩码过滤的动作比例 |
| `avgLocalReward` | 平均局部奖励 |

这些指标可以支撑论文前期问题：

```text
只看平均时延不够，还要看最大队列、P95 时延、链路风险、控制开销和计算开销。
```

## 论文里怎么描述这份代码

可以这样写：

```text
在正式训练 MAPPO/CTDE 策略之前，本文先实现一个前期仿真与环境接口原型。
MATLAB 部分将 LEO 网络建模为时变图，并实现 delay-only Dijkstra、queue/load-aware baseline、lifetime-constrained action mask、逐跳本地决策和消融策略。
Python 部分进一步封装实时 Dec-POMDP/MARL 环境接口，提供局部观测、动作掩码、局部奖励、集中式 Critic 状态、Hello 控制开销和真实星座拓扑输入占位。
该阶段不作为最终训练算法，而作为后续接 MAPPO、CleanMARL 或其他成熟 MARL 算法库的建模基础。
```

## Python 环境接口与文稿对齐

`leo_marl_env.py` 是后续接 MAPPO/CTDE 的环境层，不是新的独立算法。它对应文稿里的“实时 Dec-POMDP + CTDE/MAPPO”路线：执行时只输出当前卫星的一跳邻居观测和动作掩码，训练时通过 `global_state_for_critic` 提供集中式 Critic 可用的全局摘要。

| 文稿模块 | Python 代码对应 |
|---|---|
| 低/中/高负载、热点、快断、故障场景 | `SCENARIOS` 和 `ScenarioConfig` |
| 局部观测 `o_i` / `z_ij` | `neighbor_features`，包含队列、时延、剩余带宽、负载、可靠性、`T_rem`、AoI、progress、SHORT 启发 `(U,W)` |
| 动作过滤 / action mask | `action_mask` 和 `_action_feasible()` |
| 防环 | `PacketState.visited` |
| TTL | `max_local_hops` |
| 局部即时奖励 | `_local_reward()` |
| 集中式 Critic 状态 | `global_state_for_critic` 和 `as_mappo_inputs()` 里的 `critic_state` |
| Hello 控制开销 | `hello_fields`、`hello_period_slots`、`estimate_control_overhead_bytes()` |
| 控制预算约束 | `control_budget_ratio` 和 `w_overhead` |
| AoI 信息时效 | `age` 和观测中的 `age_j / T_hello` |
| 真实/准真实星座接口 | `topology_provider` |
| MAPPO/CleanMARL 适配 | `as_mappo_inputs(obs)` 返回 `actor_obs`、`critic_state`、`action_mask` |

Python smoke test 会依次跑 `low_load`、`medium_load`、`hotspot_high_load`、`frequent_break`、`fault_links` 五种场景。当前它只验证环境接口可运行，不代表最终训练结果。

## 真实星座和 SHORT 分层路由参考

用户上传的 `Stable Hierarchical Routing for Operational LEO Networks` / SHORT 笔记对本项目主要有四点启发：第一，真实 LEO 网络不能只看扁平全局路由，控制开销和路由更新次数很关键；第二，可以引入 shell/orbit/geographic 层次，把快速变化的卫星运动和相对稳定的地理目标解耦；第三，每跳决策应尽量保持近 O(1)，只比较少量邻居；第四，实验指标除了平均时延，还要看控制开销、可用性、重路由/切换、CPU/内存或推理开销。

因此 `leo_marl_env.py` 里预留了 `orbital_geodetic_coord()`。它现在只是 toy 版本的 `(U,W)` 特征占位，不是完整 TLE/SGP4。后续如果接 Hypatia、StarryNet、TLE 或其他真实星座数据，应通过 `topology_provider(t, env)` 替换 `_build_topology()`，并把真实的 shell/orbit/geographic 坐标填入观测。

## 重点参考仓库

后续接学习算法时，优先参考这些成熟实现，避免从零填算法坑：MAPPO 官方实现 `marlbenchmark/on-policy`，轻量实现 `AmineAndam04/cleanmarl`，较重但完整的 `facebookresearch/BenchMARL`。用户还给出 `tmt-123-bit/leo-routing` 作为 PS-MAPPO queue-aware 和 link-lifetime constrained LEO routing 的研究笔记参考；当前环境无法直接访问 GitHub 页面，所以这里只记录为重点参考，不臆造仓库内部结构。

## Python 批量评测入口

`run_python_experiments.py` 是论文级实验前的统一评测入口。它不是最终 MAPPO 训练脚本，而是先把环境、场景、策略和指标导出流程固定下来，方便后续把训练好的 MAPPO Actor 接入同一个评测接口。

运行方式：

```bash
python run_python_experiments.py
```

当前评测场景包括：`low_load`、`medium_load`、`hotspot_high_load`、`frequent_break`、`fault_links`。当前启发式策略包括：`random_feasible`、`delay_only`、`queue_load`、`full_masked_heuristic`。输出文件为：

```text
outputs/python_policy_eval_metrics.csv
```

该 CSV 包含每个场景和策略的 `deliveryRatio`、`dropRate`、`avgDelayMs`、`p95DelayMs`、`avgHops`、`avgReward`、`avgControlOverheadRatio` 和 `avgDecisionFLOPs`。这些指标对应文稿里的时延、吞吐/投递、拥塞、稳定性和开销五类评价指标。

需要注意：当前 `python_policy_eval_metrics.csv` 仍然是启发式策略和轻量学习策略的评测结果，不是最终 MAPPO 训练结果，也不是最终 IEEE 投稿实验。它的作用是先把评测闭环打通。后续要达到论文级结果，还需要接入 MAPPO/CleanMARL 训练、真实/准真实星座拓扑、Q-routing/普通 MAPPO 等学习型对比方法，并扩大 episode 数和随机种子数量。

## 轻量学习型 baseline

`train_linear_policy.py` 是一个无 PyTorch 依赖的轻量学习型 baseline。当前环境不能直接使用 PyTorch/Gym，也不能稳定联网安装依赖，所以这里先用 numpy 实现 masked linear softmax policy，并用 REINFORCE-style 更新跑通“学习型策略训练 -> 保存权重 -> 统一评测”的闭环。

运行方式：

```bash
python train_linear_policy.py
python run_python_experiments.py
```

训练输出：

```text
models/linear_policy_weights.npz
outputs/linear_policy_training_log.csv
```

统一评测脚本 `run_python_experiments.py` 已加入 `linear_policy`。如果 `models/linear_policy_weights.npz` 存在，就加载训练后的线性策略；如果不存在，就回退到 `full_masked_heuristic`。

需要注意：`linear_policy` 不是最终 MAPPO，也不是 IEEE 级最终算法。它的作用是证明当前环境已经支持学习型策略闭环，并为后续接 `marlbenchmark/on-policy`、`cleanmarl` 或 `BenchMARL` 提供接口参照。最终论文实验仍应接入正式 MAPPO/CTDE 或其他成熟 MARL 实现，并扩大随机种子、episode 数、星座规模和真实拓扑。

## CleanMARL/MAPPO 接口适配

已经读取本地 `F:\cleanmarl\cleanmarl\mappo.py` 的训练接口。CleanMARL 的 MAPPO 代码要求环境提供：`n_agents`、`reset()`、`step(actions)`、`get_avail_actions()`、`get_state()`、`get_obs_size()`、`get_state_size()` 和 `get_action_size()`。

本项目新增：

```text
cleanmarl_leo_wrapper.py
```

该文件把 `LeoRoutingEnv` 包装成 CleanMARL 风格环境。当前先按单包逐跳路由建成 `n_agents = 1` 的任务，输出维度如下：

```text
obs_shape   = (1, 66)   # 6 个候选邻居 × 11 维邻居特征
action_size = 6         # 最多 6 个候选下一跳
state_size  = 9         # 集中式 Critic 使用的全局摘要
```

运行 smoke test：

```bash
python cleanmarl_leo_wrapper.py
```

当前选择先适配 CleanMARL，而不是直接改 `marlbenchmark/on-policy` 或 BenchMARL，原因是：CleanMARL 单文件结构最清楚，环境接口要求也最容易对齐；`marlbenchmark/on-policy` 更适合后续正式 MAPPO 复现实验；BenchMARL 较重，适合后期做标准化 benchmark 和多算法对比。

需要注意：目前 wrapper 只是接口适配，还没有直接运行 CleanMARL 的 PyTorch MAPPO 训练，因为当前环境没有安装 PyTorch。后续在有 PyTorch 的环境里，可以把 `CleanMARLLeoWrapper` 接入 cleanmarl 的 `environment()` 分支，或者写一个 `env/leo_wrapper.py`，再运行 MAPPO 训练。

## 算法仓库接入建议

当前已经定位到 `F:\cleanmarl`、`F:\on-policy`、`F:\BenchMARL` 三个算法仓库。综合代码复杂度、依赖和当前环境约束，现阶段最适合先接的是 `cleanmarl`，因为它的 `cleanmarl/mappo.py` 环境接口最直接，当前项目已经新增了 `cleanmarl_leo_wrapper.py` 来对齐它需要的 `reset/step/get_avail_actions/get_state/get_obs_size/get_state_size/get_action_size/n_agents` 接口。

为什么当前先选 `cleanmarl`：第一，它是单文件结构，便于快速理解和修改；第二，它要求的环
## CleanMARL 风格训练占位入口

当前已经新增：

```text
train_cleanmarl_style_stub.py
outputs/cleanmarl_rollout_preview.csv
```

这个脚本不会真正训练 PyTorch MAPPO，但会尽量模仿 `F:\cleanmarl\cleanmarl\mappo.py` 的 rollout 数据采集方式，验证 `cleanmarl_leo_wrapper.py` 输出的 `obs/actions/log_prob/reward/states/done/avail_actions` 数据结构是否合理。它的作用是：在当前没有 torch 的环境里，先把 CleanMARL 正式接入前最容易出问题的 rollout 结构验证掉。

## on-policy 官方 MAPPO 升级计划

当前还新增了：

```text
ON_POLICY_INTEGRATION_PLAN.md
```

这个文件说明了为什么 `F:\on-policy` 适合放在第二阶段升级、它最值得参考的结构是什么、以及当前为什么先走 `cleanmarl -> on-policy` 这条路线，而不是直接硬改官方 MAPPO。

## CleanMARL 环境注册模板与运行模板

当前还额外新增了两个面向正式接入的辅助文件：

```text
cleanmarl_env_registration_example.py
RUN_CLEANMARL_LEO.md
```

其中：

- `cleanmarl_env_registration_example.py` 展示如果后续修改 `F:\cleanmarl\cleanmarl\mappo.py`，应该怎样新增 `env_type="leo"` 的环境分支；
- `RUN_CLEANMARL_LEO.md` 给出在有 PyTorch 环境时如何启动 CleanMARL + LEO 路由的最小命令模板，以及训练后如何回接到 `run_python_experiments.py` 统一评测。

## 论文推进清单

如果后续需要快速判断“当前已经做到哪一步、还差什么、下一步先补什么”，可以直接看：

```text
PAPER_PROGRESS_CHECKLIST.md
```

这个文件把当前项目已经完成的 MATLAB/Python/评测/学习型 baseline/cleanmarl 桥接工作，和仍未完成的正式 MAPPO 训练、真实星座、论文级完整对比，整理成了一份清单。

## 真实星座拓扑模板与接入计划

为了后续把 toy topology 升级到真实/准真实星座拓扑，当前还新增了：

```text
real_topology_provider_example.py
REAL_TOPOLOGY_INTEGRATION_PLAN.md
```

其中：

- `real_topology_provider_example.py` 定义了 `topology_provider(t, env)` 需要返回的 `LinkState` 字典格式；
- `REAL_TOPOLOGY_INTEGRATION_PLAN.md` 说明了后续如何把 Hypatia、StarryNet、TLE/SGP4 等外部拓扑数据映射到当前环境，而不重写训练和评测逻辑。

## Hypatia 风格拓扑 stub 与数据格式说明

当前还新增了：

```text
hypatia_topology_provider_stub.py
REAL_TOPOLOGY_DATA_FORMAT.md
```

其中：

- `hypatia_topology_provider_stub.py` 演示如何从外部 CSV 快照读取时间片链路数据，并转换成 `topology_provider(t, env)` 需要的 `LinkState` 字典；
- `REAL_TOPOLOGY_DATA_FORMAT.md` 说明真实/准真实拓扑数据文件最少需要哪些列，以及如何从 Hypatia、StarryNet、TLE/SGP4 等来源先统一转成 CSV。

## tmt 主参考仓库后续并入计划

当前已经补了一份：

```text
TMT_REPO_MERGE_PLAN.md
```

它的作用是：如果后续定位到 `tmt-123-bit/leo-routing` 本地仓库，不需要推翻当前项目，而是按“主参考设计源”的方式，把它的状态设计、reward、mask、训练脚本思路和实验组织方式逐项并入当前工程实现。
