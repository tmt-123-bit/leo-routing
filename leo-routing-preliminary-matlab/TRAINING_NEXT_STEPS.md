# LEO 路由训练接入说明

这里不重复论文思路，主要把当前 `F:\leo-routing-preliminary-matlab` 里的环境、评测和训练脚本，和我已经 clone 到 F 盘的三个算法仓库对应起来，说明下一步最适合怎么接。

## 1. 当前我们已经有什么

这个文件夹里已经有四类东西：

1. `run_preliminary_leo_routing.m`
   - MATLAB 前期仿真与消融脚本。
   - 用处：验证 Dijkstra baseline、queue/load-aware baseline、链路寿命约束、逐跳本地决策、防环、TTL、局部 reward 和消融策略。
   - 它不是训练代码，而是论文前期验证和消融实验代码。

2. `leo_marl_env.py`
   - Python 环境层。
   - 用处：把论文里的“实时 Dec-POMDP + CTDE/MAPPO”思路转成可训练环境接口。
   - 已包含：场景配置、局部观测、action mask、防环、TTL、AoI/Hello、控制预算、真实星座占位、critic state、MAPPO 风格输入。

3. `run_python_experiments.py`
   - 统一评测入口。
   - 用处：固定多场景、多策略、多指标导出流程。
   - 当前支持：`random_feasible`、`delay_only`、`queue_load`、`full_masked_heuristic`、`linear_policy`。

4. `train_linear_policy.py`
   - 轻量学习型 baseline。
   - 用处：保留一个简单学习 baseline，用来检查环境和统一评测闭环。
   - 它还不是最后的 MAPPO。

此外还有：

5. `cleanmarl_leo_wrapper.py`
   - 这是当前最重要的桥接文件。
   - 它把 `LeoRoutingEnv` 包装成 cleanmarl `mappo.py` 期望的接口：
     - `reset()`
     - `step(actions)`
     - `get_avail_actions()`
     - `get_state()`
     - `get_obs_size()`
     - `get_state_size()`
     - `get_action_size()`
     - `n_agents`

## 2. 三个算法仓库怎么选

我已经 clone 到 F 盘的三个算法仓库：

- `F:\cleanmarl`
- `F:\on-policy`
- `F:\BenchMARL`

### 2.1 为什么当前先选 cleanmarl

现在更适合先接的是 `cleanmarl`，原因是：

第一，它最轻。`cleanmarl/cleanmarl/mappo.py` 是单文件实现，环境接口清晰，最容易对齐当前 `LeoRoutingEnv`。

第二，它要求的环境 API 很直接。读取 `cleanmarl/cleanmarl/mappo.py` 后，可以确认环境至少要提供：

- `reset()`
- `step(actions)`
- `get_avail_actions()`
- `get_state()`
- `get_obs_size()`
- `get_state_size()`
- `get_action_size()`
- `n_agents`

当前 `cleanmarl_leo_wrapper.py` 已经把这些都补齐了。

第三，它更适合作为“第一条正式 MAPPO 训练线”。也就是说，先用 cleanmarl 跑通真正的 MAPPO/IPPO 风格训练，比一开始直接硬接更重的官方 MAPPO 或 BenchMARL 更稳。

### 2.2 on-policy 适合什么时候接

`F:\on-policy` 是 MAPPO 官方实现。它更适合：

- 后面要写“我们使用官方 MAPPO 实现”时；
- 需要更正式地对齐论文里的 MAPPO/CTDE 训练流程时；
- 在已有 cleanmarl 接入经验后，把环境进一步包装成官方 MAPPO 风格时。

所以它更像第二阶段目标，而不是第一阶段目标。

### 2.3 BenchMARL 适合什么时候接

`F:\BenchMARL` 更像 benchmark 框架，不适合现在优先接。原因是：

- 它更重；
- 更依赖 TorchRL/Hydra；
- 本地已有 torch，CleanMARL `leo_multi` smoke training 已通过；
- 现在最急的是把训练线跑通，而不是先做大而全 benchmark。

因此当前对 BenchMARL 的使用建议是：

- 参考它的 benchmark 组织方式；
- 参考它如何规范记录实验；
- 不作为第一条训练接入线。

## 3. 当前 cleanmarl wrapper 的定位

`cleanmarl_leo_wrapper.py` 当前把 LEO 路由任务先包装成 `n_agents = 1` 的最小兼容形式。

这不是说论文最终就变成单智能体，而是一个工程过渡：

- 现在任务本质上是“当前持包卫星根据局部观测选择下一跳”；
- 因此先把“当前活跃决策点”包装成一个 agent，最容易跑通训练；
- 之后如果要更接近严格的多智能体表述，可以把“多包并发”或“多卫星并发决策”扩成多 agent 版本。

这种做法的好处是：

- 不破坏论文主线；
- 不会为了强行多 agent 而先把环境写乱；
- 可以先验证学习型策略是否优于启发式和 Dijkstra baseline；
- 后续再升级为多 agent 共享参数版本。

## 4. 当前我更想先走的下一步

### Step 1
先保留现在的 `train_linear_policy.py` 和 `run_python_experiments.py`，作为“环境可训练 + 指标可导出”的底线。

### Step 2
基于 `cleanmarl_leo_wrapper.py`，新增一个“如果本地将来装好 torch，就如何把 cleanmarl 的 mappo.py 接到本环境”的训练入口。

### Step 3
等训练线跑通后，把训练好的策略接回 `run_python_experiments.py`，与 `delay_only`、`queue_load`、`full_masked_heuristic` 做统一评测。

### Step 4
训练线稳定后，再考虑两条升级路线：

- 升级到 `F:\on-policy` 的官方 MAPPO 实现；
- 接真实/准真实星座拓扑数据，替换 `topology_provider`。

## 5. 为什么现在还不能说已经是最终 IEEE 训练代码

目前还不能这么说，原因有四个：

1. 只跑了 CleanMARL 工程 smoke training，还没有多 seed、预注册的正式性能实验，也没有跑官方 on-policy MAPPO。
2. 真实星座/TLE/Hypatia/StarryNet 还没接入，只是有接口占位。
3. 对比方法还不完整，尤其缺少正式学习型对比和更大规模种子实验。
4. 当前轻量 `linear_policy` 只是学习型 baseline，还不是最后的 MAPPO/CTDE 结果。

所以当前最准确的说法是：

> 现在已经具备“论文前期原型 + MARL 环境接口 + 轻量学习闭环 + cleanmarl 风格适配器”的基础，下一步可以正式接 cleanmarl / on-policy 做训练。

## 6. 当前文件推荐使用顺序

如果后续继续推进，建议按这个顺序看和用：

1. `README_前期代码说明.md`
2. `run_preliminary_leo_routing.m`
3. `leo_marl_env.py`
4. `run_python_experiments.py`
5. `train_linear_policy.py`
6. `cleanmarl_leo_wrapper.py`
7. `F:\cleanmarl\cleanmarl\mappo.py`
8. `F:\on-policy\README.md`

## 7. 我现在的判断

在现阶段，最适合当前 LEO 环境接入的实现方式是：

- 论文主线仍然按我现在的文稿；
- 算法接入优先使用 `cleanmarl` 作为第一正式训练目标；
- `on-policy` 作为第二阶段官方 MAPPO 对齐目标；
- `BenchMARL` 暂时只参考结构，不优先接；
- 当前 `cleanmarl_leo_wrapper.py` 是从“研究原型”过渡到“正式算法仓库训练”的关键桥梁。
