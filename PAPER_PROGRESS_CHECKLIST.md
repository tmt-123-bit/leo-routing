# 论文实验推进清单（现在代码进度汇总）

这里主要记：当前 `F:\leo-routing-preliminary-matlab` 已经做到哪一步、还差哪一步、下一步该怎么补，写成一个可以直接对照论文思路推进的清单。

---

## 1. 已经做完的部分

### 1.1 MATLAB 前期仿真与消融

已经做了：

- `run_preliminary_leo_routing.m`
- Dijkstra baseline
- queue/load-aware baseline
- link-lifetime action mask
- local next-hop routing
- visited 防环
- TTL 限制
- 局部 reward 原型
- 多组消融策略：
  - `A1_local_no_queue_load`
  - `A2_local_no_lifetime_mask`
  - `A3_local_no_reliability_risk`
  - `A4_local_no_progress`

用处：

- 用来支撑论文里“为什么 Dijkstra 只是 baseline”；
- 用来支撑“队列/负载/链路寿命/防环”这些模块的消融验证；
- 用来生成前期结果图和解释性指标。

### 1.2 Python MARL 环境层

已经做了：

- `leo_marl_env.py`
- 场景配置 `SCENARIOS`
- 局部观测 `neighbor_features`
- `action_mask`
- `PacketState.visited`
- TTL / `max_local_hops`
- `_local_reward()`
- `global_state_for_critic`
- `as_mappo_inputs()`
- `hello_fields` / `estimate_control_overhead_bytes()`
- `control_budget_ratio`
- `AoI/age`
- `orbital_geodetic_coord()` 占位
- `topology_provider` 占位

用处：

- 用来支撑论文里“实时 Dec-POMDP + CTDE/MAPPO 环境层”的代码映射；
- 用来作为后续 cleanmarl / on-policy / BenchMARL 训练的核心环境。

### 1.3 统一评测入口

已经做了：

- `run_python_experiments.py`
- 批量运行多个场景：
  - `low_load`
  - `medium_load`
  - `hotspot_high_load`
  - `frequent_break`
  - `fault_links`
- 输出：
  - `outputs/python_policy_eval_metrics.csv`

当前支持策略：

- `random_feasible`
- `delay_only`
- `queue_load`
- `full_masked_heuristic`
- `linear_policy`

用处：

- 固定统一的评测流程；
- 之后训练出的 MAPPO policy 也应该通过这个入口做统一对比，而不是另起一套评测逻辑。

### 1.4 轻量学习型 baseline

已经做了：

- `train_linear_policy.py`
- `models/linear_policy_weights.npz`
- `outputs/linear_policy_training_log.csv`

用处：

- 证明现在环境已经支持学习型策略闭环；
- 还不是最后的 MAPPO，只是过渡性学习 baseline。

### 1.5 CleanMARL 接入桥接层

已经做了：

- `cleanmarl_leo_wrapper.py`
- `train_cleanmarl_style_stub.py`
- `outputs/cleanmarl_rollout_preview.csv`
- `TRAINING_NEXT_STEPS.md`
- `CLEANMARL_INTEGRATION_PLAN.md`
- `cleanmarl_env_registration_example.py`
- `RUN_CLEANMARL_LEO.md`
- `ON_POLICY_INTEGRATION_PLAN.md`

用处：

- 已经把现在环境和 cleanmarl 的接口对齐；
- 已经把后续 cleanmarl / on-policy 的接入路线写清楚；
- 已经验证了 rollout 数据结构不是拍脑袋写的。

---

## 2. 还没做完的部分

### 2.1 工程 smoke training 已跑，正式性能实验还没跑

已经跑通修订版 CleanMARL `leo_multi` 的小规模 PPO smoke training，并保存 periodic/final checkpoint；还没有完成：

- `F:\cleanmarl\cleanmarl\mappo.py`
- `F:\on-policy` 的官方 MAPPO

原因：

- 当前这个会话环境没有 `torch`
- 所以只能把环境、wrapper、模板、说明和评测准备好，不能直接跑正式 PyTorch 训练

### 2.2 真实星座拓扑还没接

当前只是：

- `topology_provider` 占位
- `orbital_geodetic_coord()` 占位

还没有真正接入：

- TLE / SGP4
- Hypatia
- StarryNet
- 真实/准真实星座轨迹与链路数据

### 2.3 论文级完整对比还没做

还缺：

- 正式 MAPPO policy
- Q-routing / QRLSN 风格对比
- 普通 MAPPO 与本文完整方法对比
- 更大规模星座
- 多随机种子结果
- 更完整曲线和显著性/稳定性分析

### 2.4 tmt-123-bit/leo-routing 本地仓库还没定位到

现在已经找到：

- `F:\cleanmarl`
- `F:\on-policy`
- `F:\BenchMARL`

但 `tmt-123-bit/leo-routing` 本地仓库路径还没找到，因此目前只能把它作为研究方向参考，不能按它的代码结构直接对齐。

---

## 3. 如果目标是“继续往 IEEE 级别推进”，优先顺序应该是什么

### 第一优先级

在有 `torch` 的环境里，把：

```text
CleanMARLLeoWrapper
 -> 接入 F:\cleanmarl\cleanmarl\mappo.py
```

先跑通真正的 MAPPO。

为什么第一优先做这个：

- 这是从“环境原型”变成“正式训练代码”的分水岭；
- 前面的桥接工作基本都是为这一步准备的；
- cleanmarl 是三个算法仓库里当前最轻、最稳、最容易先落地的。

### 第二优先级

把训练好的策略接回：

```text
run_python_experiments.py
```

新增一个 `mappo_policy`，和以下策略统一对比：

- `delay_only`
- `queue_load`
- `full_masked_heuristic`
- `linear_policy`
- `mappo_policy`

### 第三优先级

接真实/准真实星座拓扑：

- TLE / SGP4
- Hypatia
- StarryNet
- 其他轨道/链路数据

### 第四优先级

再升级到：

- `F:\on-policy` 官方 MAPPO
- 必要时再参考 `F:\BenchMARL`

---

## 4. 当前已有结果可以怎么用于论文

### 已经可以支撑前期说明的部分

当前已经可以用于写论文前几节或前期实验说明的部分包括：

1. **为什么 Dijkstra 不够**
   - 通过 MATLAB baseline / Python heuristics 可以说明只看传播时延会忽略队列、负载和链路寿命。

2. **为什么要逐跳本地决策**
   - 当前已有 local next-hop routing + visited + TTL + action mask。

3. **为什么环境已经对齐文稿思路**
   - `leo_marl_env.py` 和 README 已经能一一对应到文稿中的状态、动作、动作掩码、AoI、控制预算、critic state。

4. **为什么当前已经能支持学习型策略**
   - `linear_policy` 已证明环境可以训练并统一评测。

### 还不能作为最终论文结论的部分

现在还不能直接作为最终论文结论的包括：

- “MAPPO 比 baseline 明显更好”
- “真实星座下表现稳定”
- “对比现有 MARL/路由算法全面领先”

因为正式 MAPPO、真实拓扑、多种子和更完整对比还没做完。

---

## 5. 建议你后面继续推进时的实际操作顺序

### 方案 A：你本机有 torch

直接优先做：

1. 按 `RUN_CLEANMARL_LEO.md` 和 `CLEANMARL_INTEGRATION_PLAN.md`
2. 接 `F:\cleanmarl\cleanmarl\mappo.py`
3. 训练 `mappo_policy`
4. 回接 `run_python_experiments.py`
5. 导出统一评测结果

### 方案 B：你本机暂时也没 torch

那就先继续扩：

1. 真实星座拓扑输入
2. 更细的实验场景参数
3. 更完整的 README / 论文映射
4. 更清楚的对比实验脚本

---

## 6. 我现在的判断

现在这版已经完成到：

```text
研究思路 -> 前期仿真 -> MARL环境 -> 统一评测 -> 轻量学习闭环 -> CleanMARL桥接
```

还没完成的是：

```text
正式MAPPO训练 -> 真实星座拓扑 -> 论文级完整对比实验
```

所以它已经不是“只有想法”，但也还不是“最终 IEEE 投稿实验代码”。它已经是一个能继续往正式论文代码推进的完整中间阶段。
