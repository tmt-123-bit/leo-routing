# on-policy 官方 MAPPO 接入计划

这份说明文件用于第二阶段升级：当 `cleanmarl` 路线跑通之后，如何把当前 LEO 路由环境继续对齐到 `F:\on-policy` 的官方 MAPPO 实现。

## 1. 为什么 on-policy 是第二阶段，而不是第一阶段

当前不直接优先接 `F:\on-policy`，主要有三个原因：

1. 当前本地环境没有 PyTorch；
2. `on-policy` 的环境包装和训练入口更重；
3. 我们已经用 `cleanmarl_leo_wrapper.py` 把最关键的环境接口桥接跑通，先走这条更稳。

所以推荐路线是：

```text
先 cleanmarl 跑通第一条正式 MAPPO 训练线
再对齐 on-policy 官方 MAPPO
```

## 2. on-policy 当前最值得参考的部分

根据已读取的 `F:\on-policy\README.md`，这个仓库最值得当前项目参考的点是：

- 共享策略（shared policy）作为默认训练方式；
- 清晰的 `envs/`、`runner/`、`scripts/` 结构；
- 官方 MAPPO 的超参数组织方式；
- 更适合后续论文里写“采用官方 MAPPO 实现并做环境适配”。

## 3. 当前 LEO 环境要对齐的核心对象

从我们的 `leo_marl_env.py` 和 `cleanmarl_leo_wrapper.py` 出发，后续接 on-policy 时需要准备的核心对象有：

1. **Actor 输入**
   - 当前已经通过 `as_mappo_inputs(obs)` 给出 `actor_obs`。
   - 后续可直接作为 on-policy policy 网络的局部观测输入。

2. **Centralized critic state**
   - 当前 `global_state_for_critic` 和 `critic_state` 已经给出。
   - 后续可以作为官方 MAPPO centralized value function 的状态输入。

3. **Action mask**
   - 当前环境已经实现 `action_mask`。
   - 后续要在 on-policy 的动作采样处接入 mask，避免选择非法下一跳。

4. **Shared policy**
   - 当前环境本质上是“当前持包卫星进行一跳决策”，和共享策略假设相容。
   - 后续扩成多 agent 版本时，也仍然可以保持参数共享。

## 4. 最推荐的升级顺序

### Step 1
保持当前项目目录不变，继续把环境和评测逻辑放在：

```text
F:\leo-routing-preliminary-matlab
```

### Step 2
在有 PyTorch 的环境中，先跑：

- `cleanmarl_leo_wrapper.py`
- `train_cleanmarl_style_stub.py`
- `train_linear_policy.py`

确保 rollout / obs / state / avail_actions 的逻辑都稳定。

### Step 3
再在 `F:\on-policy` 仓库里新增一个 LEO 环境 wrapper，思路是：

- 把 `leo_marl_env.py` 作为环境核心；
- 把 `as_mappo_inputs()` 映射到官方 MAPPO 期望的局部观测；
- 把 `critic_state` 映射到 centralized value input；
- 把 `action_mask` 接到 policy sampling。

### Step 4
先在最小场景上跑：

- `medium_load`
- 小 episode length
- 小 rollout threads
- 小 total timesteps

先验证训练能走通，再扩大规模。

### Step 5
训练跑通之后，把训练好的 Actor 回接：

```text
run_python_experiments.py
```

新增例如：

- `mappo_policy`

然后统一与：

- `delay_only`
- `queue_load`
- `full_masked_heuristic`
- `linear_policy`

做同一套指标对比。

## 5. 当前结论

当前最务实的工程路线不是“立刻硬改 on-policy”，而是：

```text
LeoRoutingEnv
 -> CleanMARLLeoWrapper
 -> cleanmarl first
 -> official on-policy second
```

这样做的好处是：

- 风险最低；
- 跟你当前文稿最一致；
- 能先把环境和指标验证好；
- 后续升级到官方 MAPPO 时改动更可控。
