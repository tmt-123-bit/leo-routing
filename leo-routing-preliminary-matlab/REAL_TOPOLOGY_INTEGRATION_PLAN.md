# 真实/准真实星座拓扑接入计划

这份文件说明：后续如果要把 `leo_marl_env.py` 从 toy topology 升级到真实或准真实星座拓扑，应该怎么做，最小需要哪些字段，以及当前项目里哪些文件已经为这件事做好准备。

当前已经准备好的接口是：

```text
EnvConfig.topology_provider
real_topology_provider_example.py
leo_marl_env.py 里的 orbital_geodetic_coord()
```

---

## 1. 当前环境已经支持什么

`leo_marl_env.py` 里，`EnvConfig` 已经有：

```python
topology_provider: Optional[TopologyProvider] = None
```

环境的行为是：

- 如果 `topology_provider is None`，就使用当前 toy topology；
- 如果传入了 `topology_provider(t, env)`，就直接用外部提供的链路快照替换 `_build_topology()`。

所以后续接真实拓扑时，最重要的不是重写整个环境，而是：

```text
提供一个正确的 topology_provider
```

---

## 2. topology_provider 需要返回什么

返回值格式已经在：

```text
real_topology_provider_example.py
```

里固定下来。要求是：

```python
Dict[(src, dst), LinkState]
```

其中每条有向边要尽量提供这些字段：

- `delay_ms`
- `capacity_mbps`
- `used_rate_mbps`
- `rho`
- `reliability`
- `p_out`
- `t_rem`
- `available`
- `is_cross`（可选）
- `shell_src`（可选）
- `shell_dst`（可选）

如果某条边不存在，就不要放进字典里。环境会自动把它看成不可用。

---

## 3. 三种最现实的接法

### 方式 A：接 Hypatia

如果后续使用 Hypatia，最推荐的做法是：

1. 先让 Hypatia 生成时间片级的链路快照；
2. 把每个时间片的拓扑转成：

```text
(src, dst) -> delay / availability / optional capacity
```

3. 在 `topology_provider(t, env)` 里按当前时间片读取对应快照；
4. 结合环境当前 `used_rate` 计算 `rho`；
5. 如果 Hypatia 本身不给 `reliability` / `t_rem`，则按你论文里的规则额外估计。

适合原因：

- Hypatia 很适合提供准真实的 LEO 链路可见性和时延；
- 它比直接手写轨道传播更稳。

### 方式 B：接 StarryNet

如果后续使用 StarryNet，思路类似：

1. 从 StarryNet 导出轨迹 / 链路状态 / 时延数据；
2. 用中间脚本转成 `LinkState` 字典；
3. 在 `topology_provider` 里按时间片喂给环境。

适合原因：

- StarryNet 更偏网络仿真系统视角；
- 如果你后面要做更真实的网络层实验，它会更方便。

### 方式 C：接 TLE / SGP4

如果后续自己接 TLE/SGP4，建议顺序是：

1. 用 TLE/SGP4 算出每个时间片的卫星位置；
2. 判断哪些卫星间链路可见；
3. 计算传播距离和时延；
4. 根据链路可见窗口估计 `t_rem`；
5. 再把这些结果转成 `LinkState`。

适合原因：

- 论文里更容易解释“真实/准真实星座拓扑来自 TLE”；
- 可以更灵活地控制场景。

缺点是：

- 实现成本比 Hypatia / StarryNet 高；
- 你得自己负责更多轨道与链路转换逻辑。

---

## 4. reliability 和 t_rem 怎么落地

真实拓扑接入时，最容易卡住的通常不是 delay，而是：

- `reliability`
- `t_rem`

### 4.1 reliability

如果外部数据没有直接给链路可靠性，建议先分层处理：

第一层：最小可运行版本
- 先用一个规则近似，例如随 `rho` 或故障场景下调。

第二层：论文更合理版本
- 根据滑窗成功率 / ETX 风格统计估计 `reliability_ij`。

第三层：真实实验版本
- 如果仿真平台本身能给链路中断率或信道质量，就直接映射。

### 4.2 t_rem

`t_rem` 最重要，因为你的文稿里链路寿命约束是主创新点之一。

如果外部拓扑没有直接给：

- 可以根据下一次链路不可见时刻减当前时刻得到；
- 或者根据时间片级链路可用窗口长度估计；
- 再换算成秒或 slot-equivalent。

原则是：

```text
只要 t_rem 的来源在实验中前后一致，就可以先作为有效的工程近似
```

---

## 5. 当前最推荐的接入顺序

### Step 1
保持当前 `leo_marl_env.py` 不动，只新增一个真正的 provider 文件。

例如未来新增：

```text
hypatia_topology_provider.py
```
或
```text
tle_topology_provider.py
```

### Step 2
在里面实现：

```python
from leo_marl_env import LinkState

def topology_provider(t, env):
    ...
    return graph_dict
```

### Step 3
在构造环境时传入：

```python
cfg = EnvConfig(topology_provider=topology_provider)
env = LeoRoutingEnv(cfg)
```

### Step 4
保持其余训练/评测代码不变：

- `run_python_experiments.py`
- `train_linear_policy.py`
- `cleanmarl_leo_wrapper.py`

这样你换真实拓扑时，不需要把训练和评测逻辑全部重写。

---

## 6. 当前结论

当前项目已经把真实/准真实星座接入最关键的一步准备好了：

```text
环境接口已经支持外部 topology_provider
```

所以后面真正升级到真实星座时，重点工作不是推倒重来，而是：

```text
写一个把 Hypatia / StarryNet / TLE 数据映射到 LinkState 的 provider
```

这会比重写环境轻很多，也更适合论文逐步推进。
