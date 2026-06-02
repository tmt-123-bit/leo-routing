# 权威文献佐证

本文件整理可用于支撑本文研究设计的代表性文献。每篇都附原文链接，并说明能支撑本文中的哪一类说法。

## 1. MAPPO 作为 CTDE 具体模型

**Yu et al., 2022, The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**

- arXiv: https://arxiv.org/abs/2103.01955
- NeurIPS 页面: https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html
- NeurIPS PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf

支撑点：

- MAPPO 是 cooperative MARL 中常用且有效的 CTDE 算法。
- 适合集中训练、分布式执行。
- 适合多智能体协作任务。

论文可写：

> Yu 等提出的 MAPPO 表明，经过合理实现和调参后，基于 PPO 的多智能体策略梯度方法在合作多智能体任务中具有较强性能。由于本文 LEO 路由任务具有智能体同构、动作离散、训练时可获得全局状态、执行时需独立决策等特征，因此采用参数共享 MAPPO 作为 CTDE 的具体实现模型。

## 2. 局部信息下的强化学习路由

**Boyan and Littman, 1994, Packet Routing in Dynamically Changing Networks: A Reinforcement Learning Approach**

- PDF: https://proceedings.neurips.cc/paper_files/paper/1993/file/4ea06fbc83cdd0a06020c35d50e1e89a-Paper.pdf
- CMU 页面: https://publications.ri.cmu.edu/packet-routing-in-dynamically-changing-networks-a-reinforcement-learning-approach/

支撑点：

- 路由可以建模为强化学习问题。
- 节点可以基于局部信息进行在线下一跳学习。
- 在动态网络中，Q-routing 可以根据延迟反馈自适应调整路由。

论文可写：

> Boyan 和 Littman 提出的 Q-routing 表明，网络节点可以在动态变化网络中利用局部信息进行在线路由学习，并通过最小化数据包总投递时间来适应链路负载变化。这为本文采用分布式强化学习路由提供了基础依据。

## 3. LEO 动态拓扑和虚拟拓扑思想

**Werner, 1997, A Dynamic Routing Concept for ATM-Based Satellite Personal Communication Networks**

- DOI: https://doi.org/10.1109/49.634801
- IEEE 页面: https://ieeexplore.ieee.org/document/634801

支撑点：

- LEO/MEO 卫星网络拓扑是周期性时变的。
- 传统方案常采用虚拟拓扑或时间片方式处理动态拓扑。
- 说明 LEO 路由天然是时变拓扑路由问题。

论文可写：

> Werner 早期提出的动态虚拟拓扑路由思想通过时间片和预计算路径来处理 LEO 卫星网络周期性变化，说明动态拓扑是 LEO 路由设计中的核心问题。但该类方法依赖预计算和周期更新，难以充分应对实时流量变化与突发拥塞。

## 4. 大规模 LEO 中集中式路由开销

**Page et al., 2022, Distributed Probabilistic Congestion Control in LEO Satellite Networks**

- arXiv: https://arxiv.org/abs/2209.08565

支撑点：

- 密集 LEO 星座中，集中式最小时延路由会产生显著信令和计算开销。
- 分布式方案可以利用邻居交换的最新流量信息进行拥塞控制。

论文可写：

> Page 等指出，在密集 LEO 星座中采用集中式最小时延路由会产生较高的信令和计算开销，因此他们基于邻居流量信息提出分布式拥塞控制机制。这与本文仅依赖邻居状态进行分布式路由决策的思想一致。

## 5. 队列状态与拥塞控制

**Tassiulas and Ephremides, 1992, Stability Properties of Constrained Queueing Systems and Scheduling Policies for Maximum Throughput in Multihop Radio Networks**

- University of Maryland 页面: https://drum.lib.umd.edu/items/571fda52-aefb-4497-9a2d-69d8c7c907b9
- PDF: https://drum.lib.umd.edu/bitstreams/fe3d18b7-29e1-4bb1-a15e-dd54a25f6d0a/download

支撑点：

- 多跳网络中的调度和路由可以依据队列长度进行。
- 队列稳定性和吞吐量密切相关。
- 为队列感知路由和 backpressure routing 提供理论基础。

论文可写：

> Tassiulas 和 Ephremides 的队列稳定性理论表明，在多跳网络中，基于队列状态的调度策略能够扩大网络稳定区域并提升吞吐性能。因此，将下一跳队列长度作为 LEO 路由决策输入具有理论依据。

## 6. LEO 分布式负载感知路由

**Papapetrou and Pavlidou, 2008, Distributed Load-Aware Routing in LEO Satellite Networks**

- PDF: https://www.cs.uoi.gr/~epap/papers/2008_c01_globecom.pdf
- 页面: https://www.sciweavers.org/publications/distributed-load-aware-routing-leo-satellite-networks

支撑点：

- LEO 卫星网络中存在负载不均衡和局部拥塞。
- 分布式负载感知路由可以缓解拥塞。
- 链路负载状态应作为路由决策因素。

论文可写：

> Papapetrou 和 Pavlidou 提出的 DLAR 采用分布式方式进行负载感知，并通过逐跳流量分散缓解 LEO 极区拥塞问题。这说明链路负载和局部拥塞状态应作为下一跳选择的重要因素。

## 7. LEO 热点流量和负载均衡

**Liu, Tao and Liu, 2019, Load-Balancing Routing Algorithm Based on Segment Routing for Traffic Return in LEO Satellite Networks**

- DOI: https://doi.org/10.1109/ACCESS.2019.2934932
- DOAJ: https://doaj.org/article/2f79ae9d1e014f728cc9c6bb590a4b14

支撑点：

- 地面网关集中会造成 LEO 网络拥塞。
- 负载均衡路由可改善吞吐量、最大链路利用率和平均时延。

论文可写：

> Liu 等针对 LEO 卫星网络中地面网关集中引起的拥塞问题设计负载均衡路由，并通过仿真证明其可改善吞吐量、链路利用率和平均时延。这可用于支撑本文引入链路负载率和负载均衡奖励项。

## 8. 队列感知的适用边界

**Jurski and Wozniak, 2009, Routing Decisions Independent of Queuing Delays in Broadband LEO Networks**

- DOI: https://doi.org/10.1109/GLOCOM.2009.5425658
- ResearchGate: https://www.researchgate.net/publication/224121371_Routing_Decisions_Independent_of_Queuing_Delays_in_Broadband_LEO_Networks

支撑点：

- 在宽带 ISL 和合理负载条件下，队列时延可能不总是主导因素。
- 队列感知需要通过低负载和高负载场景分别验证。

论文可写：

> Jurski 和 Wozniak 指出，在宽带 LEO ISL 的合理工作条件下，队列时延可能并非总是主导因素。因此本文将分别设置低负载和高负载实验场景，以验证队列感知机制主要在拥塞或热点流量场景下发挥作用。

## 9. 链路随机失效和可靠性约束

**Zhao et al., 2022, A Routing Optimization Method for LEO Satellite Networks with Stochastic Link Failure**

- MDPI: https://www.mdpi.com/2226-4310/9/6/322
- DOI: https://doi.org/10.3390/aerospace9060322

支撑点：

- LEO 星间链路存在随机失效。
- 路由优化需要考虑链路失败、切换次数和路由成本。

论文可写：

> Zhao 等针对存在随机链路失效的 LEO 卫星网络建立路由优化模型，并将链路失效、切换次数和路由成本纳入优化目标。这为本文在动作过滤和奖励函数中加入链路可靠性约束提供了依据。

## 10. 链路剩余可用时间和空间网络接触窗口

**Burleigh, 2015, Dynamic Routing for Delay-Tolerant Networking in Space Flight Operations**

- NASA NTRS: https://ntrs.nasa.gov/citations/20150014735

支撑点：

- 空间网络可以建模为由有限通信接触窗口组成的时变拓扑。
- 路由不应只看当前链路可用性，还应考虑未来通信机会。

论文可写：

> NASA 的 Contact Graph Routing 将空间网络建模为由计划通信接触窗口组成的时变拓扑，并利用有边界的通信机会进行路径计算。这说明在空间网络中，链路剩余可用时间或接触窗口长度是影响路由可靠性和路径稳定性的关键因素。

## 11. Contact Graph Routing 教程

**Fraire, De Jonckere and Burleigh, 2021, Routing in the Space Internet: A Contact Graph Routing Tutorial**

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S1084804520303489
- CONICET: https://ri.conicet.gov.ar/handle/11336/150016

支撑点：

- Space Internet 面临延迟、中断和时间动态调度问题。
- CGR 结合时间动态调度和图模型。
- 可作为链路寿命约束的理论旁证。

论文可写：

> Fraire 等系统总结了 CGR 在空间互联网中的应用，强调空间网络路由需要处理延迟、中断和时间动态调度问题。因此，本文在 LEO 路由中加入链路剩余可用时间约束，是对空间网络时间动态特性的合理建模。

## 12. MADDPG 与 CTDE

**Lowe et al., 2017, Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**

- arXiv: https://arxiv.org/abs/1706.02275
- OpenAI 页面: https://openai.com/index/learning-to-cooperate-compete-and-communicate/

支撑点：

- 提出集中式训练和分布式执行的多智能体 actor-critic 思路。
- 训练阶段可利用其他智能体信息，执行阶段每个智能体独立行动。

论文可写：

> MADDPG 提出集中式训练、分布式执行思想，在训练阶段利用其他智能体信息提升学习稳定性，而执行阶段各智能体基于自身观测独立决策。这与本文地面集中训练、星上分布式执行的部署模式一致。

## 13. QMIX 与全局价值分解

**Rashid et al., 2018, QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning**

- arXiv: https://arxiv.org/abs/1803.11485

支撑点：

- 通过单调价值分解连接全局联合动作价值和局部动作价值。
- 支撑 cooperative MARL 中全局目标和局部执行的一致性。

论文可写：

> QMIX 通过单调价值分解将全局联合动作价值与各智能体局部动作价值关联起来，使集中训练得到的协同策略能够在分布式执行阶段由各智能体独立选择动作。这可用于支撑本文采用全局协同奖励引导局部路由决策。

## 14. COMA 与多智能体信用分配

**Foerster et al., 2018, Counterfactual Multi-Agent Policy Gradients**

- arXiv: https://arxiv.org/abs/1705.08926
- AAAI PDF: https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653

支撑点：

- 协作多智能体任务中需要解决信用分配。
- 可使用集中 critic、分布式 actor。
- 网络路由可以自然建模为合作多智能体问题。

论文可写：

> COMA 指出，网络数据包路由可以自然建模为合作多智能体问题，并采用集中式 critic 与分布式 actor 解决多智能体信用分配问题。这为本文使用全局奖励训练分布式卫星路由策略提供了方法论依据。

## 15. LEO 强化学习分布式路由

**Reinforcement Learning Based Dynamic Distributed Routing Scheme for Mega LEO Satellite Networks, 2023, Chinese Journal of Aeronautics**

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S1000936122001297
- DOI: https://doi.org/10.1016/j.cja.2022.06.021

支撑点：

- 巨型 LEO 星座面临路由计算和维护挑战。
- Q-learning 可用于巨型 LEO 卫星网络动态分布式路由。
- 强化学习路由可降低端到端时延和网络开销。

论文可写：

> QRLSN 证明了强化学习可用于巨型 LEO 卫星网络的动态分布式路由，并能降低端到端时延和网络开销。本文在此基础上进一步引入队列感知、链路寿命约束和多智能体协同奖励，以增强拥塞规避和拓扑高动态下的稳定性。

## 文献与本文设计因素对应关系

| 设计因素 | 支撑文献 | 说明 |
|---|---|---|
| PS-MAPPO 模型 | Yu et al. 2022 | MAPPO 是合作 MARL 中有效 CTDE 方法 |
| 局部信息路由 | Boyan and Littman 1994; QRLSN 2023 | 节点可基于局部信息在线学习下一跳 |
| 队列长度 | Tassiulas and Ephremides 1992; Boyan and Littman 1994 | 队列影响拥塞、吞吐和投递时间 |
| 链路负载 | Papapetrou 2008; Liu 2019 | 负载均衡可降低拥塞和平均时延 |
| 链路剩余时间 | Burleigh 2015; Fraire 2021 | 空间链路具有时间窗口，路由应考虑可用时间 |
| 链路可靠性 | Zhao 2022 | 随机链路失败会影响路由成本和切换次数 |
| CTDE | Lowe 2017; Yu 2022; QMIX 2018; COMA 2018 | 训练可用全局信息，执行只用局部观测 |
| 全局协同奖励 | QMIX 2018; COMA 2018 | 解决局部最优和多智能体协作问题 |
| 消融实验必要性 | Jurski 2009 | 队列项不一定所有场景都有效，需要低/高负载对比 |
