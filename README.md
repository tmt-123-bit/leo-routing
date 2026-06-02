# LEO Satellite Networks Distributed Routing Research

本地 Git 仓库主题：

**基于 PS-MAPPO 的队列感知与链路寿命约束低轨卫星网络分布式动态路由研究**

本仓库用于整理研究问题、九步法分析、CTDE 具体模型、实验因素影响、权威文献佐证和实验设计。

## 研究主线

本文研究高动态低轨卫星网络中的分布式动态路由问题。针对传统集中式路由在大规模星座中存在全局链路状态更新开销高、拓扑变化下收敛慢、拥塞状态感知不足、链路即将断开但仍被选中等问题，拟设计一种基于集中训练分布式执行的多智能体强化学习路由机制。

本文不只是泛泛采用 CTDE，而是明确采用：

**PS-MAPPO = Parameter Sharing Multi-Agent Proximal Policy Optimization**

也就是：

**参数共享 Actor + 集中式 Critic + 分布式执行 + 动作掩码**

训练阶段在地面仿真平台使用全局网络状态和全局协同奖励；执行阶段每颗卫星仅依赖自身状态、一跳邻居状态和相邻链路状态独立选择下一跳。

## 文件说明

- [research-nine-steps.md](./research-nine-steps.md)：按“研究思维画布九步法”回答你的九个问题。
- [design-thinking-and-steps.md](./design-thinking-and-steps.md)：详细说明设计思路、设计步骤、为什么这么设计，以及可直接写进论文的方法设计段落。
- [ctde-mappo-model.md](./ctde-mappo-model.md)：具体说明为什么选择 PS-MAPPO，以及 Actor、Critic、动作掩码和奖励函数。
- [factor-impact-analysis.md](./factor-impact-analysis.md)：说明队列、负载、带宽、链路寿命等因素分别影响哪些实验结果，以及如何影响。
- [literature-evidence.md](./literature-evidence.md)：整理权威文献、原文链接和每篇文献能支撑你论文里的哪句话。
- [experiment-design.md](./experiment-design.md)：给出实验场景、对比算法、消融实验和评价指标。

## 一句话版本

本文解决的是：

**在 LEO 卫星高速运动、拓扑频繁变化、链路负载不均衡的条件下，每颗卫星如何仅依赖局部观测和邻居信息，快速选择低时延、低拥塞且不易失效的下一跳。**

本文方法是：

**队列感知 + 链路寿命约束 + PS-MAPPO-CTDE + 动作掩码 + 全局协同奖励。**
