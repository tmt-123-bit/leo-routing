# LEO Routing with Constraint-Aware MAPPO

这个仓库研究动态 LEO 星座中的分布式下一跳路由。每颗卫星作为一个 agent，使用共享参数的候选邻居 Actor 做本地决策；训练阶段使用图 Critic，部署阶段只需要本地包状态、缓存下一跳和一跳链路/队列信息。

当前主线不是继续堆奖励项，而是直接约束“本来可以不换路、但策略仍然换了下一跳”的决策。代码、训练协议、经典基线和一次性 sealed test 都已经冻结。

## 项目是怎么走到这一版的

这个项目先后解决了两个不同问题。

第一阶段关注投递性能。早期策略把剩余链路寿命作为特征、奖励和硬动作 mask，希望提前避开即将断开的链路。实际观察到的副作用是策略会过早放弃仍然可用的链路，增加绕路和队列压力。后续采用 `no_lifetime` 配置：物理链路仍然会随拓扑变化而断开，但 Actor 不再使用预测寿命特征、寿命奖励或寿命硬 mask。

历史名称容易混淆，代码中的对应关系是：

| 历史名称 | 当前名称 | Lifetime feature | Lifetime reward | Hard mask |
|---|---|---:|---:|---:|
| `no_lifetime` | `proposed` / L0 | 关闭 | 关闭 | 关闭 |
| - | `with_lifetime_feature` / L1 | 开启 | 关闭 | 关闭 |
| - | `with_lifetime_reward` / L2 | 开启 | 开启 | 关闭 |
| `full` | `with_hard_lifetime_mask` / L3 | 开启 | 开启 | 开启 |

旧的 5k-step 消融支持“关闭 lifetime 更合适”这一工程判断，但它只有正式训练预算的 10%，因此这里只把它当作探索性证据。计划中的完整 50k-step 组件消融尚未完成，不能据此声称已经严格证明每个 lifetime 组件都会导致性能下降。

第二阶段关注路由决策本身。关闭 lifetime 后，QoS-only MAPPO 仍可能因为候选分数的小幅波动，反复替换一个仍然可用的缓存下一跳。因此当前版本没有恢复 lifetime，也没有继续调整一个难解释的换路惩罚系数，而是定义“可避免切换”，并直接约束它在所有有效机会中的比例。

```text
带 lifetime 的策略
        ↓ 发现提前避障会造成过度绕路
关闭 lifetime 的 QoS-only MAPPO
        ↓ 投递性能改善，但非必要换路仍没有明确上限
Constraint-aware MAPPO
        ↓ 用 12% 决策级预算直接管理可避免切换
冻结训练、独立 gate、一次性 sealed test
```

## 两套结果不要混用

仓库保留了两套用途不同的证据。

### 历史五场景结果：关闭 lifetime 后的 QoS-MAPPO

这一组回答“关闭 lifetime 后，MAPPO 的投递率相对传统路由如何”。修正后的 `eval-main` 使用 8 个独立 policy seed 和每个 seed 50 个共同 workload。相对 Global Dijkstra 的投递率差值为：

| 场景 | MAPPO - Dijkstra |
|---|---:|
| `low_load` | +0.22 pp |
| `medium_load` | +5.31 pp |
| `hotspot_high_load` | -0.90 pp |
| `frequent_break` | +5.15 pp |
| `fault_links` | +4.89 pp |

这些数字来自修正后的 evaluator，因此与早期汇报中的约 `+4.81 / +2.42 / +4.31 / -1.12 pp` 不完全相同。当前仓库以 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) 和 `experiments/legacy-reanalysis/eval-main/` 为准。这套结果是对已完成 checkpoint 的回顾性重分析，不是新一轮独立重训练，也不是当前 constraint 方法的 sealed result。

### 当前正式结果：显式约束是否有效

正式实验包含两个 24 星场景、3 个 MAPPO 方法、3 个经典方法、8 个独立 policy seed 和 50 个未见 workload。sealed panel 共 4,100 行，没有缺失或重复。

主比较是 `qos_only_constrained` 相对同奖励、同结构的 `qos_only_baseline`：

| 场景 | 投递率差值 | 95% CI | 可避免切换率差值 | 95% CI |
|---|---:|---:|---:|---:|
| `medium_load` | -0.958 pp | [-1.587, -0.182] pp | -15.219 pp | [-19.739, -10.357] pp |
| `hotspot_high_load` | +0.772 pp | [+0.353, +1.188] pp | -36.245 pp | [-40.872, -31.305] pp |

两个场景都通过了预先写定的四个门限：

- 投递率单侧 95% 下界不低于 -2 pp；
- 可避免切换率差值的单侧上界低于 0；
- constrained 策略的切换率单侧上界不高于 12%；
- 每个 policy seed 的切换率都不高于 12%。

中负载并不是“无损提升”：投递率下降约 0.96 pp，只是仍在预设的 2 pp non-inferiority 容差内。热点场景下 constrained MAPPO 的投递率也低于 Q-routing（0.2945 对 0.3114）。仓库和论文都保留这两个结果。

![Sealed-test effects](paper/icc2027/fig_sealed_results.png)

## 和传统路由思路相比

这里的改进不是把最短路换成一个黑盒策略，而是把动态路由里的两个目标拆开处理：QoS 由 MAPPO 学习，可避免的下一跳切换由显式约束控制。所有方法在相同场景和 sealed workload 上评估；不同方法可用的信息并不完全相同，因此下表是端到端方案比较，不是同信息条件下的理论优越性证明。

| 方法 | 基本思路 | 这个实现补了什么 |
|---|---|---|
| Global Dijkstra | 根据当前全局链路代价重算单条最短路 | 部署时不需要集中式全图计算，并能利用本地队列、包和拥塞状态 |
| OSPF-ECMP | 在等价最短路之间分流 | 候选集 Actor 不限于等价最短路，并在采样前屏蔽失效链路 |
| Q-routing | 用下游反馈更新逐目的地 Q 值 | 共享参数适配动态候选邻居，同时直接约束仍可沿用缓存路由时的非必要切换 |
| QoS-only MAPPO | 用奖励同时表达投递、时延和开销 | 增加独立的 12% 切换率预算，避免切换惩罚被其他奖励尺度淹没 |
| Reward-shaped MAPPO | 在标量奖励中加入换路惩罚 | 约束值有直接的运行含义，不必把固定惩罚系数解释成切换率保证 |

sealed test 中，各方法的投递率 / 可避免切换率如下：

| 方法 | `medium_load` | `hotspot_high_load` |
|---|---:|---:|
| Constraint-aware MAPPO | 0.7829 / 0.0568 | 0.2945 / 0.0194 |
| QoS-only MAPPO | 0.7925 / 0.2090 | 0.2868 / 0.3818 |
| Reward-shaped MAPPO | 0.7925 / 0.1463 | 0.2885 / 0.2427 |
| Q-routing | 0.7875 / 0.2605 | 0.3114 / 0.3960 |
| OSPF-ECMP | 0.7386 / 0.1875 | 0.3012 / 0.1982 |
| Global Dijkstra | 0.7381 / 0.1886 | 0.3011 / 0.1985 |

相对同结构的 QoS-only MAPPO，约束方法把可避免切换率分别降低约 72.8% 和 94.9%。相对 Q-routing，切换率分别降低约 78.2% 和 95.1%，但投递率分别低约 0.46 pp 和 1.69 pp。也就是说，当前证据最稳妥的结论是“用很小或受控的投递率变化换取明显更少的非必要换路”，而不是全面击败传统路由。

### 相比上一版，具体提升在哪里

| 维度 | 上一版 `no_lifetime` MAPPO | 当前 constraint-aware MAPPO |
|---|---|---|
| 主要问题 | 提高投递率、减少 lifetime 引起的绕路 | 控制仍可沿用缓存路由时的非必要切换 |
| 换路处理 | 换路开销混在标量奖励中 | 独立的机会条件切换率和 12% 预算 |
| 对照方法 | 重点与 Dijkstra、Q-routing 比投递率 | 增加同结构 QoS-only MAPPO 和 reward-shaped control |
| 统计单位 | 后来修正为 policy seed | 从协议开始就以 policy seed 为独立单位 |
| 数据使用 | 已有 checkpoint 的回顾性重分析 | selection、independent gate 和 sealed workload 完全分离 |
| 可审计性 | 有训练与评估 manifest | 协议、checkpoint、Q 表和最终结果逐层 hash 冻结 |
| 支持的结论 | 多数场景投递率优于 Dijkstra，热点失败 | 两场景显著减少可避免切换，并通过预设投递率容差 |

因此，这一版最主要的提升不是多得到几个百分点的投递率，而是把“路由不要无意义地来回换”从一个模糊的奖励偏好，变成可以定义、训练、检验和复现的约束目标。代价也写得很清楚：中负载下付出了约 0.96 pp 的投递率，热点下虽然相对普通 MAPPO 有所改善，但仍没有超过 Q-routing、OSPF-ECMP 和 Dijkstra。

## 方法

### 决策级可避免切换

对每个 contention 之前的有效路由决策：

```text
o = 1  当缓存下一跳和至少一个替代下一跳都可行
c = 1  当 o = 1 且策略选择了不同下一跳
R = sum(c) / sum(o)
```

首次选路、缓存链路已经失效后的强制切换、`NO_OP` 和 padding 都不进入分子。计数发生在 contention 之前，因此不会因为后续链路竞争失败而漏掉策略已经提出的切换。

### Candidate-set MAPPO

- 26 维候选特征，覆盖队列、链路状态、几何进展、包上下文和缓存信息；
- 共享候选编码器和对称池化，候选顺序变化只会重排 logits；
- 不可行动作在采样前 mask；
- 图 Critic 只在 centralized training 中使用；
- team reward 加零均值 local credit；
- PPO 使用 GAE、value clipping、可行动作归一化 entropy 和 KL 约束。

约束臂的 Actor loss 为：

```text
L_actor = L_PPO + lambda * C_surrogate

lambda_next = clip(lambda + 0.05 * (R_rollout - 0.12), 0, 5)
```

`lambda` 每个完整 rollout 更新一次。它是经验约束控制器，不是 CPO，也不提供逐轨迹的硬保证。

## 实验设计

| 项目 | 设置 |
|---|---|
| 星座 | 4 planes x 6 satellites |
| 场景 | `medium_load`, `hotspot_high_load` |
| MAPPO arms | QoS baseline, constrained, reward-shaped control |
| Policy seeds | 每个 arm/scenario 8 个 |
| 训练预算 | 50,000 target steps，实际完整 rollout 边界 50,040 |
| 训练 workloads | 76001--76200 |
| checkpoint selection | 77001--77010 |
| independent gate | 77011--77020 |
| sealed test | 78001--78050，只实例化一次 |
| 经典基线 | Q-routing, OSPF-ECMP, Global Dijkstra |

Q-routing 每个场景、每个 policy identity 重新训练 500 episodes。OSPF-ECMP 使用 8 个路由随机身份；Global Dijkstra 是确定性的，只使用一个 sentinel identity。

统计推断以 policy seed 为独立单位，不把 50 个 workload 当成 50 次独立训练。置信区间使用 5,000 次 crossed bootstrap，同时重采样 seed 行和 workload 列；敏感性检验枚举 `2^8` 个 sign flips，并对四个主检验做 Holm 校正。

## 仓库结构

```text
src/          环境、MAPPO、基线、统计、实验 runner 和测试
docs/         预注册协议、修订记录和 claim boundary
experiments/  紧凑结果、冻结文件和公开的 sealed rows
figures/      历史实验图表
paper/        ICC 2027 草稿、图和参考文献
data/         TLE 与导出的 24 星拓扑
```

这次正式结果对应：

```text
experiments/avoidable-switch-constraint-formal-v1-r2/
experiments/avoidable-switch-classical-baselines-formal-v1-r3/
experiments/avoidable-switch-joint-sealed-test-v1/
```

前两个目录在 GitHub 中只保留顶层 preregistration、aggregate rows、statistics 和 freeze。训练 checkpoint、逐 job 状态、重复 JSONL 和运行日志留在本地，不进入 Git。sealed test 的公开数据以 `sealed_test_rows.csv` 为准。

## 安装

Python 3.10 或 3.11 均可。PyTorch/CUDA 版本请按本机驱动选择，其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MAPPO trainer 目前通过 CleanMARL 兼容入口运行。正式训练 runner 会记录源码、依赖、CUDA 设备和 checkpoint hash；不要直接修改已冻结目录继续训练。

## 测试

在仓库根目录执行：

```bash
cd src
python -m unittest discover -p "test_*.py"
```

只检查本次 constraint/sealed-test 链路：

```bash
cd src
python -m unittest \
  test_avoidable_switch_constraint \
  test_avoidable_switch_constraint_formal \
  test_formal_avoidable_switch_statistics \
  test_avoidable_switch_classical_baselines_formal \
  test_avoidable_switch_joint_sealed_test
```

sealed workloads 已经按协议使用过一次。不要删除输出目录后重新跑 test panel，也不要用 test 结果重新选 seed、checkpoint 或场景。公开结果的只读入口是：

```text
experiments/avoidable-switch-joint-sealed-test-v1/sealed_test_statistics.json
experiments/avoidable-switch-joint-sealed-test-v1/sealed_test_rows.csv
```

## 冻结记录

```text
MAPPO training freeze
153b5cd85305f3f4c9a03fd549eea3e06785a0d1a447b96ec9005098c7d25742

MAPPO validation freeze
d3c039657802e9c784c82cec31024e5098b31e5a3e217e69bbc64dad5a8469d6

Classical training freeze
df7cf104d30360df898ef8239b64789c16e0baa9e87c78c52c08a676ecb62eba

Classical gate freeze
36bac779a6961ce38eb6872ec0f9cfa3953898ceb00d9b5bd7f10ce80fbc3a1e

Joint sealed-test authorization
9e2c173b0399adf52adeb575ffb11742d2c3452eb0d50bb6bdccdac22ceed653

Final sealed-test freeze
c5cc82c7edb96add6e8da1275924e83cf159e65a2062184d52f3364ad8e508a9
```

协议细节见：

- [`docs/AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md`](docs/AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md)
- [`docs/AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md`](docs/AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md)
- [`docs/AVOIDABLE_SWITCH_JOINT_SEALED_TEST_V1.md`](docs/AVOIDABLE_SWITCH_JOINT_SEALED_TEST_V1.md)

## 结果边界

目前证据支持的说法很具体：在这个 24 星 slot simulator 的两个已测试负载场景中，显式 decision-level constraint 相对匹配的 QoS-only MAPPO 大幅降低了可避免切换率，并通过预设的投递率 non-inferiority 门限。

它还不能说明：

- 对任意 LEO 星座和流量都有效；
- 优于所有经典路由方法；
- 已经验证控制面收敛或真实信令开销；
- 能零样本扩展到大星座；
- 已达到工程部署条件。

下一步更有价值的是在 TLE/SGP4 或 Hypatia/ns-3 环境中做独立外部验证，而不是继续调整这次 sealed result。
