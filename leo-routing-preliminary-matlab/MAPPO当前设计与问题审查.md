# 当前 MAPPO 设计与问题审查

> 文档状态：当前实现基线说明与问题清单
> 审查时间：2026-07-11
> 适用代码：EXP-004 / EXP-005 所使用的 CleanMARL MAPPO 与 LEO wrapper
> 结论摘要：当前实现是单智能体 masked PPO 与非对称 critic 的组合，并非具有并发卫星协作语义的 MAPPO。环境本身已通过 H-004 v3 的有限范围有效性门，但 actor/critic 表示存在状态混叠，actor 不具备候选排列等变性，PPO 训练配置和诊断能力不足。后续不应继续直接调参或扩大训练，而应先完成 MAPPO 设计修订和工程有效性测试。

---

## 1. 文档目的

本文回答两个问题：

1. 当前代码中的“MAPPO”究竟是怎样设计和运行的？
2. 该设计已经确认存在哪些问题，哪些只是待验证的可能原因？

本文只描述当前实现和已验证问题，不提出未经实验支持的性能承诺，也不把 EXP-004 的负结果归因于某个单一因素。

---

## 2. 代码与证据范围

### 2.1 核心代码

| 模块 | 路径 | 作用 |
|---|---|---|
| MAPPO/PPO trainer | `/home/anyflow/projects/cleanmarl/cleanmarl/mappo.py` | Actor、critic、rollout buffer、TD(λ) return、PPO update、checkpoint 和 validation |
| CleanMARL LEO wrapper | `leo-routing-preliminary-matlab/cleanmarl_leo_wrapper.py` | 将 LEO 单包逐跳环境适配成 CleanMARL 输入接口 |
| LEO routing environment | `leo-routing-preliminary-matlab/leo_marl_env.py` | 动态拓扑、background packet flow、局部观测、动作 mask、reward 和逐跳转移 |
| EXP-004 runner | `scripts/research_pipeline/run_exp004_ordinary_mappo.py` | 固定训练/验证/测试 split，运行多 seed MAPPO，执行 held-out evaluation |
| EXP-005 diagnostics | `scripts/research_pipeline/run_exp005_diagnostics.py` | 只读提取训练曲线、动作分布、checkpoint 和策略行为诊断 |

### 2.2 本文绑定的源代码哈希

| 文件 | SHA-256 |
|---|---|
| `mappo.py` | `593ced0642305f5ca81959f72875afb2211a0b7fc2a8cf3292c22d0370db4e3b` |
| `cleanmarl_leo_wrapper.py` | `c8eae18356fcb9500f6ab4ebe72cb98784a680d2f95446a77a2e85318b25d458` |
| `leo_marl_env.py` | `d81f15a574c8c9d8920e1f10b4f464dbb13b2e777664689513c71c1ddb08ea88` |

### 2.3 直接证据

- EXP-004：`experiments/EXP-004/`
- EXP-004 独立审计：`experiments/EXP-004/independent_audit.json`
- EXP-005：`experiments/EXP-005/`
- EXP-005 独立审计：`experiments/EXP-005/independent_audit.json`
- EXP-005 实验卡：`experiments/experiment_cards/EXP-005_mappo-failure-diagnosis.md`

---

## 3. 当前系统的整体结构

当前数据流如下：

```text
LeoRoutingEnv
  │
  │ observation = 当前节点的候选邻居特征 + action mask
  │ state       = 9 维网络聚合摘要
  ▼
CleanMARLLeoWrapper
  │
  │ n_agents = 1
  │ actor obs  : [1, 66] = 6 个候选槽位 × 11 维特征
  │ critic state: [9]
  │ action mask : [1, 6]
  ▼
CleanMARL mappo.py
  ├── Actor: 66 → 32 → 32 → 6 logits
  ├── Critic: 9 → 64 → 64 → 1 value
  ├── Categorical masked action sampling
  ├── batch_size=4 个完整 episode rollout
  ├── TD(λ) return
  └── 3 epochs PPO full-batch update
```

训练时 actor 采样动作；validation 和最终 evaluation 使用 masked logits 的 greedy argmax。

---

## 4. 环境和决策语义

### 4.1 决策实体

当前每个 episode 只有一个 foreground packet。每一步由一个 active decision point 为该 packet 选择下一跳。

Wrapper 明确设置：

```text
n_agents = 1
```

因此，当前任务没有：

- 多个卫星同时行动；
- 多个 packet agent 并发决策；
- agent 间联合动作；
- 多智能体 credit assignment；
- 多 agent shared reward 下的协作学习。

当前实现虽然复用了 MAPPO 的张量和代码接口，但从决策过程看，本质上是：

```text
single-agent masked PPO + asymmetric/aggregate critic
```

它不能作为多智能体协作或真正 CTDE 有效性的证据。

### 4.2 动作空间

动作是当前节点邻居列表中的索引，而不是全局 satellite ID：

```text
action ∈ {0, 1, ..., 5}
```

环境实际邻居通常不超过 4 个，其余槽位为 padding。动作 mask 排除：

- 已访问节点；
- 不可用链路；
- 剩余容量不足；
- queue 达到上限；
- reliability 低于阈值；
- link remaining lifetime 低于安全阈值。

Mask 在 rollout、PPO log-prob 重算和 greedy evaluation 中均被使用。独立审计确认 action masking 本身实现一致，没有发现策略选择被 mask 禁止动作的通用错误。

---

## 5. Actor 的当前设计

### 5.1 每个候选邻居的 11 维特征

对当前节点的每个邻居，actor 接收：

1. 当前节点归一化 queue；
2. 邻居节点归一化 queue；
3. 归一化 link delay；
4. 归一化 remaining bandwidth；
5. link utilization `rho`；
6. link reliability；
7. 归一化 remaining lifetime；
8. 邻居信息 age；
9. 相对 destination 的 progress；
10. 邻居轨道坐标 `u_coord`；
11. 邻居轨道坐标 `w_coord`。

最多 6 个候选，padding 后形成：

```text
candidate_features: [6, 11]
```

Wrapper 将其直接 flatten：

```text
actor_obs: [66]
```

### 5.2 Actor 网络

当前 actor 是普通 MLP：

```text
Linear(66, 32)
ReLU
Linear(32, 32)
ReLU
Linear(32, 6)
```

输出 6 个 slot-specific logits，然后使用 action mask 将不可行动作 logit 设置为 `-1e9`。

训练时：

```text
Categorical(logits).sample()
```

评测时：

```text
argmax(masked logits)
```

### 5.3 Actor 没有显式输入的状态

当前 actor 看不到：

- packet 当前 hop count；
- remaining hop budget；
- `last_next_hop`；
- 是否会触发 switch penalty；
- 完整 visited/path context；
- 当前节点或 destination 的明确身份/编码；
- topology phase/time slot 的显式表示。

Destination 只通过每个候选的单个 `progress` 特征间接进入观测；visited 信息只通过当前 action mask 间接体现。

---

## 6. Critic 的当前设计

### 6.1 Critic 输入

Wrapper 的 critic state 为 9 维网络摘要：

1. average queue；
2. maximum queue；
3. average rho；
4. Jain load index；
5. available-link ratio；
6. control-overhead ratio；
7. delivery ratio；
8. drop rate；
9. switch count。

Critic 网络：

```text
Linear(9, 64)
ReLU
Linear(64, 64)
ReLU
Linear(64, 1)
```

### 6.2 Critic 没有输入的回报相关状态

Critic 看不到：

- packet 当前节点；
- packet destination；
- 当前候选邻居及 action mask；
- hop count / remaining hop budget；
- last next hop；
- visited/path；
- topology phase；
- actor 当前局部观测。

因此，它不是 packet-conditioned value function。

---

## 7. Reward 设计

每次合法转发的局部 reward 为：

```text
r_local =
    - w_delay    × normalized_delay
    - w_queue    × normalized_neighbor_queue
    - w_load     × rho
    - w_risk     × (1 - reliability)
    - w_lifetime × lifetime_cost
    + w_progress × destination_progress
```

当前权重：

| 项 | 权重 |
|---|---:|
| delivery bonus | `+2.0` |
| delay | `-1.5` |
| queue | `-1.0` |
| load | `-1.0` |
| risk | `-2.0` |
| lifetime | `-1.0` |
| loop / TTL | `-3.0` |
| invalid action | `-2.0` |
| progress | `+0.5` |
| control overhead | `-0.2` |
| switch | `-0.2` |

额外规则：

- 到达 destination：加 `w_deliver`；
- hop count 达到上限：减 `w_loop`，episode 以 `ttl_exceeded` 结束；
- 与上一跳选择不同：减 `w_switch`；
- 无可行动作时 runner 传入 `-1`，环境记录为 `invalid_action` 并终止。

注意：EXP-005 已确认，所有 `invalid_action` 结果都发生在 `feasibleCount=0`，不是 actor 违反 action mask。

---

## 8. PPO 训练设计

### 8.1 EXP-004 使用的关键配置

| 参数 | 当前值 |
|---|---:|
| training seeds | `7, 42, 1024` |
| train workload seeds | `9001..9020` |
| validation seeds | `10001..10050` |
| test seeds | `11001..11050` |
| configured timesteps | `50,000` |
| batch size | `4 episodes` |
| PPO epochs | `3` |
| actor learning rate | `8e-4` |
| critic learning rate | `8e-4` |
| gamma | `0.99` |
| TD lambda | `0.95` |
| PPO clip | `0.2` |
| entropy coefficient | `0.001` |
| advantage normalization | 关闭 |
| return normalization | 关闭 |
| reward normalization | 关闭 |
| gradient clipping | 关闭 |

### 8.2 Rollout

每轮收集 4 个完整 episode。Rollout buffer 保存：

- observation；
- sampled action；
- old log probability；
- reward；
- critic state；
- available-action mask；
- done；
- valid-transition mask。

不同长度 episode 使用 padding，PPO loss 只在 valid-transition mask 上计算。

### 8.3 Return 和 advantage

当前使用 forward TD(λ) return：

```text
G_t^λ = r_t + γ[λG_(t+1)^λ + (1-λ)V(s_(t+1))]
A_t   = G_t^λ - V(s_t)
```

对当前完整 terminal/drop episode，最后一步 bootstrap 为 0。独立审计认为当前用法成立；代码中的 `last_advantage` 是未使用变量，不构成当前 GAE 错误。

但 `done` 被收集后没有参与 return 递推。对于当前被视为吸收 packet drop 的 TTL 截断，这不影响当前结果；若未来出现普通 time-limit truncation 或 partial rollout，该实现会产生错误 bootstrap 语义。

### 8.4 PPO update

Actor 使用标准 clipped surrogate：

```text
ratio = exp(logπ_new - logπ_old)
L_actor = -min(ratio × A, clip(ratio) × A) - entropy_coef × entropy
```

Critic 使用 return target 的 MSE。

独立审计确认：

- surrogate 符号正确；
- clipping 形式正确；
- entropy bonus 符号正确；
- padding mask 和 action mask 使用正确。

---

## 9. Checkpoint 与评测设计

训练每约 5,000 environment steps 保存 periodic checkpoint，共 10 个候选。

每个候选在固定 validation seeds `10001..10050` 上 greedy evaluation，按以下顺序选择：

1. delivery ratio 最大；
2. average reward 最大；
3. delivered-packet delay 最小。

选出的 checkpoint 在 test seeds `11001..11050` 上与 frozen linear 和 full heuristic 比较。

EXP-005 又重放了 180 个 actor：

```text
15 runs × (10 periodic + validation-best + final)
```

结论：checkpoint selection 存在 50-episode 粗粒度不足，但只是次要问题。Final checkpoint 在两个 run 上优于 selected、零 run 更差，但 aggregate delivery 仍远低于 linear，不能解释 EXP-004 的主要失败。

---

## 10. 已确认的问题

### P0-1：Actor observation 不是 reward/transition-sufficient state

EXP-005 构造了两个直接反例。

#### Switch reward alias

两个状态的 actor input 完全一致，仅 `last_next_hop` 不同。对同一个 action：

```text
reward A = -1.535
reward B = -1.735
```

差值恰好为 switch penalty `0.2`。

#### TTL transition alias

两个状态的 actor input 完全一致，仅 hop count 不同：

```text
hop_count = 0  → forwarded,    reward = -1.535
hop_count = 11 → ttl_exceeded, reward = -4.535
```

结论：当前 feed-forward actor 将 reward 和 transition 不同的状态映射为相同输入。继续调 PPO 超参数无法消除该结构问题。

### P0-2：Critic state 严重混叠

同一个 9 维网络摘要可对应不同：

- source/destination；
- 当前 packet 位置；
- 剩余路由预算；
- visited path；
- action mask；
- 后续可达性与 return。

此外，100-episode probe 观察到 critic 各维尺度从约 `0.000632` 到 `36`，两个维度在保存的非终止训练状态中为零方差。当前又没有 state/return normalization 和 critic gradient clipping。

这使 value regression 同时面临状态别名、无效维度和尺度条件问题。

### P0-3：Actor 不具备候选排列等变性

下一跳候选本质上是一个集合，但当前网络将排序后的 6 个槽位整体 flatten，并使用不同输出神经元对应不同槽位。

729 个候选交换状态上的直接审计：

| 指标 | 结果 |
|---|---:|
| argmax permutation equivariance | `63.9%` |
| median max-logit error | `3.69` |
| P95 max-logit error | `10.06` |

这证明当前 actor 可以依赖候选排序或 satellite ID 排序产生 slot shortcut。需要强调：该结果证明 inductive bias 不匹配，但尚未证明它单独导致 EXP-004 的 delivery deficit。

### P0-4：当前没有真正的 MAPPO / CTDE 多智能体语义

`n_agents=1` 时：

- actor 不存在 agent 间参数共享带来的协作问题；
- critic 不处理 joint observation/action；
- 不存在 multi-agent credit assignment；
- 不存在 concurrently acting satellites。

因此，当前代码只能用于建立单 active decision routing-policy baseline，不能支撑多智能体贡献。

### P1-1：PPO 优化保护和诊断不足

EXP-005 观察：

- `10/15` runs 后期 entropy 持续低于 `0.01`；
- `8/15` runs 后期 actor gradient 持续低于 `0.01`；
- critic gradient 在 final 10% window 的 run-median 范围约 `22.01..166.20`；
- 仅约 `1.6%` logged updates 出现非零 clip fraction；
- 仅约 `0.84%` updates 的 KL 高于 `0.001`；
- `8/15` runs 的十个 checkpoint validation aggregate 完全不变。

这说明 actor 更新活动低、actor/critic 优化存在明显失衡信号。但这些是关联证据，不足以确定 batch size、normalization、entropy 或 clipping 中哪一项是根因。

另一个重要限制是：TensorBoard 的 final collapse 指标与 validation-selected actor 经常不是同一个 checkpoint。不能直接将 final collapse 归因于被评测策略。15 个 run 上 collapse rate 与 delivery 的 Spearman 相关仅 `-0.080`，且受场景混杂影响。

### P1-2：Normalization 可选路径不能直接打开

虽然 EXP-004 没有启用 normalization，但当前可选实现存在潜在问题：

- advantage/return normalization 的标准差除法缺少 epsilon；
- return normalization 在 advantage 已按原 value scale 计算后改变 critic target scale；
- reward normalization 使用单个 rollout 统计量，而不是稳定的 running statistics。

因此，不能把 `normalize_*` 开关直接全部打开作为修复。必须先通过零方差、尺度一致性和手算 trajectory 单元测试。

### P1-3：终止与截断语义不通用

`done` 被存储但没有用于 return recursion；所有 episode 最后状态一律 zero-bootstrap。

对当前 packet delivered/drop/TTL-as-absorbing-drop 设计可以成立，但不适用于：

- 普通 time-limit truncation；
- partial rollout；
- 需要从非终止边界 bootstrap 的任务。

未来若扩展多 agent 或固定 rollout horizon，需要重构该部分。

### P1-4：Validation 分辨率有限

八个 run 的十个 checkpoint 具有不同 state_dict hash，却得到完全相同的 50-episode aggregate validation tuple。

这并不意味着 checkpoint function 完全相同，只说明当前 validation 指标和样本量无法区分其功能差异。全 checkpoint replay 发现 final 在两个 run 上有 `+8 pp` 和 `+14 pp` improvement，但仍不改变 H-005 失败。

---

## 11. 已排除或弱化的问题

当前证据不支持把以下项目视为主要实现 bug：

| 项目 | 审计结论 |
|---|---|
| PPO surrogate 符号 | 正确 |
| PPO ratio clipping 公式 | 正确 |
| entropy bonus 符号 | 正确，优化方向鼓励探索 |
| action masking | rollout、update、evaluation 一致 |
| padding transition mask | 正确排除 padding |
| 当前完整 terminal/drop episode 的 TD(λ) 递推 | 成立 |
| one-agent tensor shape | 内部一致 |
| flattened MLP 的绝对表达能力 | 未证明不足；问题是 inductive bias 和 equivariance |
| checkpoint selection 作为唯一失败原因 | 已弱化，不能解释主要 deficit |
| actor 普遍选择 mask 禁止动作 | 已排除；`invalid_action` 均来自无可行动作状态 |

---

## 12. EXP-004/EXP-005 对当前设计的总体判断

### 12.1 性能事实

EXP-004 validation-best MAPPO delivery：

| Scenario | MAPPO | Linear | Full heuristic |
|---|---:|---:|---:|
| low_load | `0.5000` | `1.00` | `0.66` |
| medium_load | `0.5467` | `1.00` | `0.64` |
| hotspot_high_load | `0.5800` | `1.00` | `0.64` |
| frequent_break | `0.3400` | `0.62` | `0.46` |
| fault_links | `0.6800` | `0.96` | `0.68` |

H-005 v2 的三个 gate 均为 `0/5`，已被拒绝。

### 12.2 设计判断

可以确认：

- 当前 actor/critic 状态设计不充分；
- 当前 candidate actor 架构缺少正确的集合归纳偏置；
- 当前 critic 表示与尺度条件不适合作为稳定 value baseline；
- 当前训练缺少必要的 PPO 数值保护和可诊断性；
- 当前任务不是多智能体合作任务。

不能确认：

- 某一个问题单独导致全部性能 deficit；
- 提高 entropy coefficient 一定有效；
- 打开 normalization 一定有效；
- 换 shared candidate scorer 一定提高 delivery；
- 增加训练步数能够解决问题；
- MAPPO/PPO 一般性不适合 LEO routing。

独立审计对“EXP-004 总体根因”的置信度为低；对 critic representation/scaling 有贡献的置信度为中等。

---

## 13. 后续 MAPPO 设计必须满足的原则

本文不展开新版网络的完整方案，但后续设计至少必须满足：

1. **Actor 状态完整性**
   Actor 必须区分 remaining hop budget、previous next hop/switch context，以及其他可观测且影响 reward/transition 的 packet context。

2. **Packet-conditioned critic**
   Critic 必须包含当前 packet 的位置、destination、路由预算、path context 和局部候选状态，而不能只使用网络聚合摘要。

3. **Candidate permutation equivariance**
   每个邻居应通过 shared candidate encoder/scorer 处理；交换候选输入后，输出 logits 和动作必须严格对应交换。

4. **PPO 数值有效性**
   在启用 normalization、gradient clipping 或新的 entropy 方案前，先验证 return、advantage、terminal/truncation、零方差和 loss scaling。

5. **可诊断训练**
   至少记录 advantage/return 分布、critic explained variance、pre/post clipping gradient、feasible-action normalized entropy、per-epoch KL、clip fraction 和 action-slot 频率。

6. **单智能体与多智能体分阶段**
   先建立可信的单 active packet PPO routing baseline；真正 MAPPO/CTDE 需要另行定义并发 agent、joint state、shared/global reward 和 credit assignment。

7. **受控实验而非打包修改**
   表示修复、critic 修复、PPO stability 修复和 reward 修改不能一次性打包。每个变化必须对应可证伪的控制实验。

---

## 14. 当前决策

当前 MAPPO 实现不应继续直接用于：

- 大规模超参数 sweep；
- 延长训练步数以追赶 baseline；
- 风险/负载/寿命机制消融；
- 多智能体合作 claim；
- 论文主结果。

下一阶段应先形成新版 MAPPO 设计规范，并以工程有效性门验证：

```text
状态可区分性
→ candidate permutation equivariance
→ actor/critic shape 与 mask
→ return/loss 数值测试
→ supervised policy capacity control
→ 小规模受控 PPO 实验
→ 新的预注册性能实验
```

在这些门通过之前，H-005 的负结果保持有效，C-001 和候选贡献保持 unsupported/blocked。

---

## 15. 已确认的新设计决策：卫星级共享参数多 Agent

经讨论，后续 MAPPO 的基本 Agent 语义确定为：

```text
每颗卫星 = 一个逻辑 Agent
所有卫星共享一套 Actor/PPO 参数
每颗卫星根据自己的局部观测独立决策
训练阶段使用 centralized critic
执行阶段只使用本地可获得信息
```

该决策区分了“Agent 独立性”和“网络参数独立性”：

- 24 星 toy topology 应暴露 `n_agents=24`；
- 每颗卫星有独立 observation、action mask、action 和局部 packet/queue state；
- Actor 参数 `θ` 在卫星之间共享，即 `a_i ~ π_θ(a_i | o_i, e_i)`；
- `e_i` 优先使用物理/拓扑位置编码，而不是只使用任意 satellite ID one-hot；
- 完全独立的 24 套 PPO 参数只作为可选 IPPO 对照，不作为默认主设计。

这一设计无法通过修改现有 wrapper 的 shape 完成。当前 `LeoRoutingEnv` 是单 foreground packet 串行逐跳环境，同一 Agent 槽位随 packet 在不同卫星间移动。真正的卫星级 MAPPO 需要专门的同步多 Agent 卫星网络环境。

### 15.1 专用环境的最低语义

第一版建议固定 FIFO/HOL 规则，避免同时引入 packet scheduling 学习问题：

- 每颗卫星维护真实 packet queue；
- 每个 slot，每颗卫星最多为一个 head-of-line packet 选择下一跳；
- 无 packet 时由环境强制 `NO_OP`，该样本不进入 routing policy loss；
- active HOL packet 必须在当前可用邻居中选择下一跳；无可行候选时记录显式 `no_route/drop`；
- 第一版 action 是环境强制 inactive `NO_OP` + active next-hop，不学习 hold/scheduling；
- 所有 24 个 Agent 基于同一个 slot snapshot 同时决策；
- 环境统一解析链路容量、冲突、失败和 queue admission；
- packet transmission 以 batch 方式提交，禁止按 satellite ID 顺序逐个修改环境。

推荐的 transition 边界为：

```text
slot-t topology/queue snapshot
→ construct 24 local observations and masks
→ 24 agents choose independently
→ validate joint actions
→ resolve capacity/contention from the frozen snapshot
→ batch packet transmission and ownership transfer
→ service/admit exogenous arrivals according to one frozen event order
→ advance topology/time
→ construct slot-(t+1) state
```

具体 arrival/service/forwarding 的先后顺序必须在实现前冻结，并用 packet identity ledger 验证，不能依赖 Python 循环顺序。

### 15.2 每个卫星 Agent 的局部观测

至少包括：

- 自身 queue 长度、HOL packet 等待时间和 packet class；
- HOL destination 的相对轨道/拓扑编码；
- remaining hop/TTL budget；
- previous next hop 或 route-switch context；
- 自身 plane、in-plane position、orbital phase 等物理位置编码；
- 每个候选邻居的 queue、delay、remaining bandwidth、rho、reliability、remaining lifetime 和 destination progress；
- 可本地维护的 visited/loop-avoidance 摘要；
- action mask 和 `NO_OP` 可用性。

局部 Actor 不应读取全局 queue matrix、其他卫星不可观测 packet 信息或未来 topology，从而保持 decentralized execution 的信息约束。

### 15.3 Centralized critic state

训练时 critic 至少需要：

- 全部卫星的 queue/HOL packet context；
- 当前动态 topology 和链路容量/可靠性/寿命；
- active packet 的 source/destination/current owner/TTL；
- 当前 slot/orbital phase；
- 全部 Agent 的 action masks；
- 必要时的 previous joint action 或 contention state。

不再使用当前 9 维 aggregate summary 作为唯一 critic 输入。实现上可以使用 permutation-aware graph/attention critic，避免把 24 个 Agent 的 joint state 无结构地 flatten。

### 15.4 环境必须通过的硬门

专用环境在进入 MAPPO 训练前必须验证：

1. **Packet identity conservation**：generated packet IDs 等于 delivered、dropped 与 backlog 的不交并集。
2. **Single ownership**：任一 packet 在任一边界只属于一个 queue、in-flight 集合或 terminal 集合。
3. **One transmission per packet per slot**：禁止同一 packet 被多个 Agent 或多条链路重复发送。
4. **Capacity conservation**：每条 directed link 的 batch transmission 不超过该 slot 的有效容量。
5. **Order invariance**：重排 Agent iteration order 后，joint transition 结果不变。
6. **Deterministic replay**：固定 environment/workload seed 和 joint actions 时，raw trace hash 完全一致。
7. **Satellite relabeling consistency**：同构重编号后，state/action/transition 按映射对应。
8. **Mask correctness**：所有可行、不可行和 `NO_OP` 原因有显式 ledger，不能只统计 feasible 子集。
9. **No hidden global leakage**：Actor observation 只包含执行阶段可获得信息。
10. **Concurrent action evidence**：至少两个卫星在同一 slot 实际执行非 `NO_OP` 决策，才能称为 multi-agent run。

只有这些环境语义通过后，shared-parameter Actor、centralized critic 和 MAPPO loss 的实验才具有多卫星 CTDE 含义。

---

## 16. 2026-07-14 修订结果

这次已经按审查结论修改本机现有代码。审查中提到的 Linux EXP-004/EXP-005 runner 和 `/home/anyflow/...` 目录不在当前机器上，因此没有伪造或重建那些实验结果；修改范围是 GitHub/F 盘现有的 LEO 环境、CleanMARL wrapper 和 `F:\cleanmarl\cleanmarl\mappo.py`。

### 16.1 已修改

1. 单包 PPO baseline
   - candidate feature 从 11 维扩为 20 维；
   - 增加 remaining hop budget、hop count、candidate-specific switch context、current/destination 位置、visited ratio 和 topology phase；
   - TTL exhaustion 改为吸收式 terminal drop；
   - critic 改为网络摘要 + packet context + candidates + action mask，维度从 9 变为 146；
   - wrapper 继续明确声明 `n_agents=1`，只作为单 active packet PPO baseline。

2. 候选 Actor 与 PPO 数值保护
   - 使用 shared candidate encoder/scorer 和 masked pooled context；
   - 候选交换后 logits 严格随之交换；
   - reward normalization 改为 running statistics；
   - advantage normalization 增加 epsilon；
   - return normalization 只在 critic loss 内保持同尺度变换；
   - GAE 显式区分 terminal 和 time-limit truncation；
   - inactive `NO_OP` Agent 不进入 routing policy loss；
   - 增加 0.5 gradient clipping、target KL、feasible-action normalized entropy、explained variance、pre/post clip gradient、return/advantage 分布和 action-slot 频率；
   - 增加 periodic/final checkpoint 和环境 schema 元数据。

3. 卫星级多 Agent 环境
   - 新增 `SynchronousLeoMultiAgentEnv`，24 颗卫星对应 24 个 Agent；
   - FIFO/HOL，每个 active Agent 每 slot 最多处理一个 packet；
   - inactive Agent 强制 `NO_OP`；
   - 所有 Agent 基于同一 frozen snapshot 决策，之后批量解析 transmission、TTL、delivery 和 queue admission；
   - 增加 packet identity、single ownership、one-transmission、capacity 和 mask reason ledger；
   - CleanMARL `leo_multi` wrapper 输出 `(24,140)` actor observation、`(24,7)` action mask 和 3073 维 centralized state。

### 16.2 工程测试结果

`test_mappo_design.py` 共 13 项测试，全部通过，包括：

- switch/TTL 状态可区分；
- packet-conditioned critic；
- candidate permutation equivariance；
- zero-variance normalization；
- terminal/truncation 手算 GAE；
- supervised candidate capacity；
- packet conservation 和 single ownership；
- 同 slot 多 Agent 并发动作；
- Agent iteration order invariance；
- deterministic replay；
- inactive `NO_OP` 与 active policy mask；
- Actor 无显式 global-state leakage；
- 24 Agent 单次 PPO update loss/gradient 为有限值。

同步环境 smoke test 第一 slot 有 8 个卫星执行非 `NO_OP` 动作，满足 concurrent-action evidence。

CleanMARL `leo_multi` 64-timestep smoke training 完整结束并保存 periodic/final checkpoint。最后记录值：

```text
actor_loss                 = -0.4808
critic_loss                = 74.7235
normalized_entropy         = 0.4413
actor_gradient_pre_clip    = 0.02075
critic_gradient_pre_clip   = 193.3326
critic_gradient_post_clip  = 0.5000
explained_variance         = 0.0416
advantage_mean             ≈ 0
advantage_std              = 1.0
```

critic 原始梯度很大，说明原审查指出的 critic 优化失衡在 smoke run 中确实出现；修订后的 gradient clipping 将其限制到 0.5，且没有 NaN/Inf。

### 16.3 更新后的轻量 baseline

20 维 observation 下重新训练 linear baseline，并在 5 个 toy 场景、每场景 50 episode 上统一评测：

| Policy | 平均投递率 | 平均丢包率 | 平均时延 | 平均 P95 |
|---|---:|---:|---:|---:|
| delay-only | 0.452 | 0.548 | 58.90 ms | 108.73 ms |
| full heuristic | 0.560 | 0.440 | 51.53 ms | 85.10 ms |
| retrained linear | 0.884 | 0.116 | 33.64 ms | 63.49 ms |

linear 相对 delay-only 的 toy 结果为：投递率增加 43.2 个百分点、丢包率下降 78.8%、平均时延下降 42.9%、P95 下降 41.6%。这仍只是轻量 baseline，不是新版 MAPPO 结果。

### 16.4 还没有完成

- 没有重新运行 EXP-004/EXP-005，因为本机缺少对应 runner、split 和实验目录；
- 没有完成新版 MAPPO 的多 seed、held-out、预注册性能实验；
- centralized critic 目前仍是固定顺序 flat state + MLP baseline，graph/attention critic 还没有实现；
- satellite relabeling consistency 还没有形成完整自动化 gate；
- 真实星座/TLE/Hypatia、多流并发压力和统计显著性实验还没有完成。

因此现在可以说“结构修订和工程有效性门已通过现有 13 项测试”，不能说“新版 MAPPO 已经提高多少性能”。旧 H-005 负结果仍然有效，直到新的预注册实验完成。
