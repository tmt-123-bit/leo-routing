# CTDE 具体模型：PS-MAPPO

## 采用哪一个 CTDE 模型

本文采用：

**PS-MAPPO = Parameter Sharing Multi-Agent Proximal Policy Optimization**

中文可写作：

**参数共享的多智能体近端策略优化模型**

模型结构为：

```text
Centralized Critic + Decentralized Actor + Parameter Sharing + Action Mask
```

也就是：

- 训练阶段：集中式 Critic 可以看到全局网络状态。
- 执行阶段：每颗卫星只部署共享 Actor，根据局部观测独立选择下一跳。
- 参数共享：所有卫星使用同一个 Actor 网络，降低星上存储和训练复杂度。
- 动作掩码：动态过滤不可用、即将断开、过载或不可靠的邻居链路。

## 为什么选择 MAPPO

CTDE 是训练执行范式，不是具体算法。LEO 分布式路由需要选择一个可落地的 CTDE 算法。

本文选择 MAPPO 的原因：

- 支持集中训练、分布式执行。
- 支持离散动作，适合“选择下一跳邻居”。
- 容易加入 action mask，适合动态邻居集合。
- 可以使用参数共享，适合大量同构卫星。
- 训练稳定性通常优于许多复杂 off-policy 多智能体方法。

相比 QMIX，MAPPO 更适合本文场景，因为 LEO 路由中每颗卫星的候选邻居会随时间变化，动作空间需要根据链路可用性动态掩码。MAPPO 可以较自然地处理带掩码的离散动作策略。

## 权威模型依据

核心参考文献：

**Yu et al., 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**

- arXiv: https://arxiv.org/abs/2103.01955
- NeurIPS 页面: https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html
- NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf

这篇文章系统研究了 MAPPO 在合作多智能体任务中的效果，证明了经过合理实现和调参后，PPO 类 on-policy 方法在 cooperative MARL 中可以取得很强的性能。本文可引用它作为 PS-MAPPO 模型选择依据。

可写入论文的话：

> Yu 等提出的 MAPPO 表明，基于 PPO 的多智能体策略梯度方法在合作多智能体任务中具有较好的稳定性和性能。由于本文 LEO 路由任务具有智能体同构、动作离散、训练时可获得全局状态、执行时需独立决策等特征，因此采用参数共享 MAPPO 作为 CTDE 的具体实现模型。

## 智能体定义

每颗卫星是一个智能体：

```text
agent_i = satellite_i
```

在每个数据包转发时刻，卫星 `i` 根据局部观测选择下一跳。

## Actor 输入

执行阶段 Actor 只能看到局部观测：

```text
o_i(t) = [self_feature_i, neighbor_features_ij, packet_feature]
```

具体包括：

```text
o_i(t) = {q_i, dest, q_j, R_ij, d_ij, rho_ij, P_out_ij, T_rem_ij}
```

含义：

- `q_i`：当前卫星队列长度。
- `dest`：目的卫星、目的地面站或目的区域编码。
- `q_j`：邻居卫星队列长度。
- `R_ij`：链路剩余带宽。
- `d_ij`：链路传播距离或传播时延。
- `rho_ij`：链路负载率。
- `P_out_ij`：链路中断概率。
- `T_rem_ij`：链路剩余可用时间。

## Actor 输出

Actor 输出下一跳动作：

```text
a_i(t) in N_i(t)
```

即从卫星 `i` 当前邻居集合中选择一个邻居作为下一跳。

由于不同卫星邻居数量可能变化，实践中可以设置最大邻居数 `K_max`，对不足部分 padding，并对无效邻居使用 action mask。

## 动作掩码

动作掩码用于过滤明显不应选择的链路：

```text
A_i(t) = {j in N_i(t) |
          R_ij > R_min,
          q_j < Q_max,
          P_out_ij < P_max,
          T_rem_ij > T_safe}
```

如果链路满足以下任一条件，则被 mask：

- 剩余带宽不足。
- 下一跳队列过长。
- 链路中断概率过高。
- 链路剩余可用时间过短。
- 链路当前不可用。
- 邻居已经在当前数据包访问路径中，可能形成环路。

动作掩码的意义：

**先用网络约束过滤明显错误动作，再让 MAPPO 在可行动作中学习最优选择。**

## Critic 输入

训练阶段集中式 Critic 可以看到全局状态：

```text
s(t) = {Q(t), R(t), D(t), rho(t), P_out(t), T_rem(t), F(t)}
```

其中：

- `Q(t)`：全网卫星队列状态。
- `R(t)`：全网链路带宽状态。
- `D(t)`：全网链路传播时延。
- `rho(t)`：全网链路负载。
- `P_out(t)`：全网链路中断概率。
- `T_rem(t)`：全网链路剩余可用时间。
- `F(t)`：全网业务流量分布。

Critic 只在训练阶段使用，不部署到星上。

## 奖励函数

全局协同奖励：

```text
R_g(t) =
- w1 * average_end_to_end_delay
- w2 * total_queue_backlog
- w3 * link_load_variance
- w4 * packet_loss_rate
- w5 * route_switch_count
+ w6 * throughput
```

局部即时奖励：

```text
r_i(t) =
- alpha * D_ij
- beta  * q_j
- gamma * rho_ij
- delta * P_out_ij
- lambda / (T_rem_ij + epsilon)
- mu    * I_loop
- nu    * I_switch
+ xi    * I_deliver
```

建议表述：

**训练时以全局协同奖励为主，局部即时奖励为辅助。全局奖励负责多智能体协作目标，局部奖励负责引导单跳转发行为并加快收敛。**

## PPO 损失

对每个智能体采样到的轨迹，使用 PPO clipped objective：

```text
L_actor(theta) =
E[min(r_t(theta) * A_t,
      clip(r_t(theta), 1 - eps, 1 + eps) * A_t)]
```

其中：

```text
r_t(theta) = pi_theta(a_t | o_t) / pi_theta_old(a_t | o_t)
```

Critic 损失：

```text
L_critic(phi) = E[(V_phi(s_t) - R_t)^2]
```

总损失可以写为：

```text
L = -L_actor + c1 * L_critic - c2 * entropy
```

## 训练与执行流程

训练阶段：

1. 构建 LEO 时变拓扑仿真环境。
2. 初始化共享 Actor `pi_theta` 和集中式 Critic `V_phi`。
3. 每颗卫星根据局部观测和 action mask 选择下一跳。
4. 环境返回全局状态、局部奖励、全局奖励和性能指标。
5. 使用全局状态训练 Critic，使用 PPO 更新共享 Actor。
6. 重复训练直到收敛。

执行阶段：

1. 每颗卫星部署共享 Actor。
2. 通过 Hello 包获取邻居队列、负载、可用状态和链路剩余时间。
3. 对不可用链路执行 action mask。
4. Actor 在候选邻居中选择下一跳。
5. 不需要全局拓扑，不需要集中式控制器。

## 可直接回答老师的话

本文采用的 CTDE 具体模型是 PS-MAPPO，即参数共享的 Multi-Agent PPO。训练阶段在地面仿真平台使用集中式 Critic 获取全局状态，并通过全局协同奖励优化全网端到端时延、队列积压、负载均衡、丢包率、路由切换次数和吞吐量。执行阶段每颗卫星只部署共享 Actor，根据自身和一跳邻居状态独立选择下一跳。为了适应 LEO 动态拓扑，本文在 Actor 输出前引入 action mask，过滤即将断开、可靠性低、带宽不足或队列过长的链路。
