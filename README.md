# 基于 PS-MAPPO 的队列感知与链路寿命约束低轨卫星网络分布式动态路由研究

## 1. 研究问题

本文面向高动态低轨卫星网络中的分布式路由问题。LEO 卫星高速运动，星间链路频繁变化，若依赖地面中心实时收集全网状态并计算路由，会带来较高控制开销和较慢收敛。同时，传统最短路径路由主要关注传播时延或跳数，容易忽略节点队列积压、链路拥塞以及链路即将断开等问题。

因此，本文拟研究：

**在每颗卫星仅能获取自身和一跳邻居状态的条件下，如何设计一种同时感知队列拥塞与链路剩余可用时间的分布式动态路由机制，使卫星能够自主选择低时延、低拥塞且更稳定的下一跳。**

推荐题目：

**基于 PS-MAPPO 的队列感知与链路寿命约束低轨卫星网络分布式动态路由研究**

## 2. 具体痛点

本文主要针对以下三个问题：

1. **局部信息下的快速路由决策问题**  
   大规模 LEO 网络中，全局链路状态更新开销高，集中式路由难以及时适应拓扑变化。需要每颗卫星依赖局部观测和邻居信息完成下一跳选择。

2. **拥塞感知不足问题**  
   最短路径不一定是实际低时延路径。在高负载场景下，节点队列积压和链路负载可能成为主要时延来源，若路由仍只关注传播时延，容易导致热点拥塞。

3. **链路动态失效问题**  
   LEO 星间链路当前可用并不代表可以稳定维持。如果数据包被转发到即将断开的链路，可能引起重路由、丢包和额外时延。因此需要在决策中考虑链路剩余可用时间。

## 3. 方案设计

本文拟采用 **PS-MAPPO** 作为具体 CTDE 模型，即参数共享的多智能体近端策略优化方法：

**集中训练、分布式执行、参数共享、动作掩码。**

整体设计思路如下：

1. **局部观测**  
   每颗卫星只观测自身队列状态、一跳邻居状态以及相邻链路状态。

2. **邻居信息交换**  
   卫星通过轻量级 Hello 包与一跳邻居交换队列长度、负载水平、链路可用状态等信息，不进行全网状态泛洪。

3. **动作过滤**  
   在选择下一跳前，先过滤不可用链路、带宽不足链路、队列过长邻居、可靠性低链路和即将断开的链路。

4. **PS-MAPPO 路由决策**  
   训练阶段使用集中式 Critic 获取全局状态并优化全网目标；执行阶段每颗卫星只部署共享 Actor，根据局部观测独立选择下一跳。

5. **队列与链路寿命约束**  
   路由决策同时考虑传播时延、下一跳队列长度、链路负载、链路可靠性和链路剩余可用时间，使路由不只追求当前最短，而是兼顾拥塞和稳定性。

## 4. 为什么选择 PS-MAPPO

CTDE 是训练与执行范式，不是具体算法。本文选择 PS-MAPPO 的原因是：

- LEO 路由是多智能体协作问题，每颗卫星的转发决策会影响其他节点的队列和链路负载。
- 训练阶段可以在地面仿真平台获得全局状态，适合使用集中式 Critic。
- 执行阶段必须星上自治，每颗卫星只能根据局部观测独立决策。
- 卫星节点具有较强同构性，适合共享同一个 Actor 网络，降低星上存储和计算开销。
- MAPPO 支持离散动作和动作掩码，适合“从邻居中选择下一跳”的路由场景。

可参考文献：

**Yu et al., 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**  
https://arxiv.org/abs/2103.01955

## 5. 预期创新点

1. **队列感知的分布式路由决策**  
   将下一跳队列长度和链路负载引入路由选择，避免流量过度集中到局部最短路径。

2. **链路寿命约束的动作过滤机制**  
   在策略决策前过滤即将断开或可靠性较低的链路，减少路径中断和重路由。

3. **基于 PS-MAPPO 的协同路由框架**  
   训练阶段利用全局状态和全局协同奖励引导多智能体学习，执行阶段保持完全分布式。

4. **面向局部信息约束的协同优化目标**  
   在不依赖全局实时控制的前提下，使局部下一跳决策尽可能服务于全网低时延、低拥塞和高稳定性目标。

## 6. 实验验证思路

实验拟从整体性能和模块贡献两个层面展开。

对比算法：

- Dijkstra 最短路径。
- 虚拟拓扑路由。
- Q-routing 或 QRLSN。
- 普通 MAPPO。
- 本文完整方法。

实验场景：

- 低负载场景：验证本文方法不会明显劣于最短路径。
- 高负载热点场景：验证队列感知能否缓解拥塞。
- 链路频繁断开场景：验证链路寿命约束能否降低丢包和重路由。
- 节点或链路故障场景：验证鲁棒性。

核心指标：

- 平均端到端时延。
- 吞吐量。
- 投递成功率。
- 丢包率。
- 平均队列长度。
- 链路负载均衡度。
- 路由切换次数。
- 控制消息开销。

消融实验：

- 去掉队列感知。
- 去掉链路负载项。
- 去掉链路剩余可用时间。
- 去掉动作掩码。
- 去掉全局协同奖励。

通过消融实验验证各模块对时延、拥塞、丢包率和路由稳定性的具体影响。

## 7. 相关文献依据

1. **强化学习分布式路由基础**  
   Boyan and Littman, *Packet Routing in Dynamically Changing Networks: A Reinforcement Learning Approach*  
   https://proceedings.neurips.cc/paper_files/paper/1993/file/4ea06fbc83cdd0a06020c35d50e1e89a-Paper.pdf

2. **LEO 强化学习分布式路由**  
   *Reinforcement Learning Based Dynamic Distributed Routing Scheme for Mega LEO Satellite Networks*  
   https://www.sciencedirect.com/science/article/pii/S1000936122001297

3. **队列状态与拥塞控制理论**  
   Tassiulas and Ephremides, *Stability Properties of Constrained Queueing Systems and Scheduling Policies for Maximum Throughput in Multihop Radio Networks*  
   https://drum.lib.umd.edu/items/571fda52-aefb-4497-9a2d-69d8c7c907b9

4. **LEO 负载感知路由**  
   Papapetrou and Pavlidou, *Distributed Load-Aware Routing in LEO Satellite Networks*  
   https://www.cs.uoi.gr/~epap/papers/2008_c01_globecom.pdf

5. **链路失效与可靠性约束**  
   Zhao et al., *A Routing Optimization Method for LEO Satellite Networks with Stochastic Link Failure*  
   https://www.mdpi.com/2226-4310/9/6/322

6. **空间网络链路接触窗口思想**  
   Burleigh, *Dynamic Routing for Delay-Tolerant Networking in Space Flight Operations*  
   https://ntrs.nasa.gov/citations/20150014735

7. **MAPPO/CTDE 依据**  
   Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*  
   https://arxiv.org/abs/2103.01955

## 8. 可直接用于开题汇报的总结

本文拟研究高动态 LEO 卫星网络中的分布式动态路由问题。针对传统集中式路由控制开销高、最短路径路由缺乏拥塞感知、以及链路频繁变化导致路径失效等问题，本文提出一种基于 PS-MAPPO 的队列感知与链路寿命约束路由机制。该机制在训练阶段利用全局状态和协同奖励学习路由策略，在执行阶段每颗卫星仅依赖自身和一跳邻居状态独立选择下一跳。同时，通过动作掩码过滤不可用、过载或即将断开的链路，以降低端到端时延、缓解热点拥塞并提升动态拓扑下的路由稳定性。
