# CleanMARL 正式接入计划

这里记录：当后续在有 PyTorch 的环境里继续推进时，如何把当前 `F:\leo-routing-preliminary-matlab` 里的 LEO 路由环境正式接到 `F:\cleanmarl\cleanmarl\mappo.py`。

已经做完的桥接文件是：

```text
cleanmarl_leo_wrapper.py
```

它已经把我们的环境包装成 cleanmarl `mappo.py` 需要的 API 风格。

---

## 1. 当前 cleanmarl 需要的环境接口

根据 `F:\cleanmarl\cleanmarl\mappo.py`，环境至少要提供：

- `n_agents`
- `reset()` -> `obs, info`
- `step(actions)` -> `next_obs, reward, done, truncated, infos`
- `get_avail_actions()`
- `get_state()`
- `get_obs_size()`
- `get_state_size()`
- `get_action_size()`

当前 `CleanMARLLeoWrapper` 已经全部满足。

---

## 2. 这个 wrapper 的输入输出定义

这个 wrapper 把任务先包装成单智能体 cooperative 形式：

```text
n_agents    = 1
obs_shape   = (1, 66)
action_size = 6
state_size  = 9
```

含义：

- `obs_shape=(1,66)`：1 个 agent，最多 6 个候选邻居，每个邻居 11 维特征；flatten 后是 66。
- `action_size=6`：最多 6 个可选下一跳。
- `state_size=9`：集中式 Critic 用的全局摘要（队列、负载、公平性、控制开销等）。

这还不是最后的多 agent 扩展版，而是当前最稳的 MAPPO 接入起点。

---

## 3. 在有 PyTorch 环境时怎么接 cleanmarl

推荐方式有两种。

### 方式 A：在 cleanmarl 仓库内新增一个环境分支

在 `F:\cleanmarl\cleanmarl\mappo.py` 里，`environment()` 函数目前支持：

- `pz`
- `smaclite`
- `lbf`

后续可以新增：

```python
elif env_type == "leo":
    from cleanmarl_leo_wrapper import CleanMARLLeoWrapper
    env = CleanMARLLeoWrapper(scenario=env_name)
```

然后命令行这样运行：

```bash
python cleanmarl/mappo.py --env_type="leo" --env_name="medium_load"
```

优点：

- 最贴近 cleanmarl 现有结构；
- 改动小；
- 最适合先跑通第一版正式 MAPPO。

### 方式 B：把 wrapper 移进 cleanmarl 的 env 目录

也可以在 `F:\cleanmarl` 里新增一个文件，例如：

```text
cleanmarl/env/leo_wrapper.py
```

再在 `mappo.py` 中 import 它。

优点：

- 结构更整齐；
- 后续 IPPO、QMIX、COMA 等算法复用更方便。

缺点：

- 比方式 A 多一步搬运。

当前建议优先方式 A，先把训练线跑通。

---

## 4. 为什么当前不直接改 on-policy

`F:\on-policy` 是官方 MAPPO 实现，更适合后续正式论文版本。

但当前先不直接改它，原因是：

1. 它环境接入更重；
2. 当前机器没有 torch，暂时跑不起来；
3. cleanmarl 更适合作为第一条最短训练链。

因此推荐顺序是：

```text
先 cleanmarl
再 on-policy
最后如果需要，再对接 BenchMARL
```

---

## 5. 为什么当前不直接改 BenchMARL

`F:\BenchMARL` 更适合做标准 benchmark，但当前不适合作为第一接入目标：

- 它依赖更重；
- 配置系统更复杂；
- 当前最需要的是先跑通一条正式训练线，而不是马上做多算法 benchmark 平台。

所以 BenchMARL 当前的作用主要是：

- 参考 benchmark 组织方式；
- 参考结果汇报规范；
- 未来在真实训练线稳定后再接。

---

## 6. 当前继续推进时我更想先走的实际顺序

### Step 1
保留当前本地文件：

- `leo_marl_env.py`
- `cleanmarl_leo_wrapper.py`
- `run_python_experiments.py`
- `train_linear_policy.py`

### Step 2
在有 torch 的环境中，把 `CleanMARLLeoWrapper` 接进 `F:\cleanmarl\cleanmarl\mappo.py`。

### Step 3
先跑最小配置：

- `env_type=leo`
- `env_name=medium_load`
- 小 batch
- 小 total_timesteps

只验证训练流程通。

### Step 4
训练好的策略再回接到：

```text
run_python_experiments.py
```

跟以下策略统一对比：

- `delay_only`
- `queue_load`
- `full_masked_heuristic`
- `linear_policy`
- `mappo_policy`（后续新增）

### Step 5
训练线稳定后，再考虑：

- 接 `F:\on-policy` 官方 MAPPO；
- 接真实星座拓扑；
- 扩大种子和场景；
- 做 IEEE 级完整对比实验。

---

## 7. 现在的判断

当前最合理的正式接入路径是：

```text
LeoRoutingEnv
   -> CleanMARLLeoWrapper
   -> cleanmarl/mappo.py
   -> 训练得到 mappo_policy
   -> run_python_experiments.py 统一评测
```

这条线是目前最轻、最稳、最贴近你现有环境和论文思路的路线。
