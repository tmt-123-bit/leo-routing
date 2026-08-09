# 六目标全局奖励和 credit assignment

这份说明对应 `leo_multiagent_env.py` 当前实际代码，不是后续设想。

## 1. Team reward

每个 slot 先完成所有 Agent 的联合动作解析，再计算一次团队奖励：

```text
R_team =
    - 1.0 * C_delay
    - 0.8 * C_queue
    - 0.5 * C_imbalance
    - 0.2 * C_switch
    + 2.0 * G_throughput
    - 0.1 * C_control
    - 2.0 * C_drop
```

`C_drop` 是可靠投递约束的附加项。前六项对应论文原来列出的端到端时延、全网队列、负载不均衡、路由切换、吞吐量和控制开销。

## 2. 每项怎样计算

| 项 | 当前定义 | 变大说明什么 | 对策略的直接影响 |
|---|---|---|---|
| `C_delay` | 本 slot 有投递时，使用归一化累计链路时延和业务等待时间；没有投递时使用平均单跳传播时延 | 路径更慢或包等待更久 | 奖励下降，偏向短路径和及时投递 |
| `C_queue` | slot 结束后全网 backlog / 全网总队列容量 | 积压更严重 | 奖励下降，鼓励把包从拥塞区域疏散 |
| `C_imbalance` | `1 - Jain(link utilization)` | 流量更集中在少量链路 | 奖励下降，鼓励分散链路负载 |
| `C_switch` | 本 slot 发生 route switch 的 accepted action / active Agent | 下一跳策略更不稳定 | 奖励下降，减少频繁切换 |
| `G_throughput` | 本 slot delivered packet / active Agent | 同一时隙完成投递更多 | 奖励上升 |
| `C_control` | Hello bytes / (Hello bytes + 成功转发数据 bytes) | 控制面占比更高 | 奖励下降，避免低数据效率 |
| `C_drop` | 本 slot drop / active Agent | 无路由、TTL、队列溢出或 deadline drop 更多 | 奖励明显下降 |

所有原始分量都会写入每一步的 `info['reward_components']`，正式评测 CSV 记录 episode 内各分量平均值，因此可以检查某个性能变化到底来自时延、队列还是吞吐。

## 3. Agent credit

只给所有卫星同一个 `R_team` 时，很难判断某个卫星的动作是否有效。当前使用零均值局部修正：

```text
credit_i = r_local_i - mean(r_local_active)
r_i = R_team + beta * credit_i
beta = 0.25
```

只有存在可行 routing action 的 active Agent 获得局部修正；inactive/no-route Agent 的修正为零。由于 active Agent 的 `credit_i` 总和严格为零，24 个 Agent reward 的平均值仍等于 `R_team`。这样 Actor 能区分本地动作质量，Critic 仍然学习团队目标。

`no_credit` 消融把 `beta` 设为 0，其余环境、Actor、Critic、workload 和训练预算保持不变。

## 4. 不能提前下的结论

- 权重是当前预注册起点，不是文献给出的通用最优值。
- 某项权重更大不代表对应指标一定单调改善，因为六项目标会相互竞争。
- 平均已投递包时延下降但投递率同时下降时，可能是 survivor bias，不能直接解释成时延性能提升。
- 最终影响大小必须使用同一 test workload 上的配对消融结果报告。

