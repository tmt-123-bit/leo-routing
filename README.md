# 低轨卫星网络分布式动态路由(MAPPO)

> **基于实时 Dec-POMDP 建模的队列感知低轨卫星网络分布式动态路由研究**

每颗卫星只读取**自身队列与一跳邻居/链路状态**,不依赖地面中心持续下发全局路径,用共享参数 MAPPO(CTDE)学习下一跳策略。24 颗卫星 = 24 个 Agent,共享一个候选邻居 Actor;训练用集中式 Critic + 团队奖励,执行时每颗卫星独立决策。

## 核心结论(采用 `no_lifetime` 变体后的最终结果)

去中心化、只用**一跳局部信息**的 MAPPO,在 24 星座 5 个场景上:

- **显著反超看全图的集中式最短路 oracle(Dijkstra/ECMP)**:中负载 **+4.8 pp**、故障 **+4.3 pp**、频繁断链 **+2.5 pp**(均 p<1e-6);轻负载持平,仅在高负载饱和场景落后 1.1 pp。
- **碾压分布式基线**:对 Q-routing 和队列感知启发式领先 **14–43 pp**(p<1e-9)。
- 零样本泛化到 **5.5× 规模**和**真实 Starlink/OneWeb 拓扑**,对奖励权重扰动和 8 个训练种子都鲁棒。
- **数据包级验证**:ns-3 重放(静态 + 动态断链拓扑)确认投递率优势在包级仿真中成立(详见 §8)。

> 这是从早期"接近但未超过全局最短路"定位的实质提升 —— 关键转折是发现并移除了净有害的链路寿命子系统(见消融 §5)。

## 1. 要解决的问题

LEO 路由不是求一条静态最短路,有三个本质难点:

1. **拓扑和业务变化快**,全网状态同步和集中重算不适合作为每一跳的唯一依据;
2. **只看传播时延会忽略队列和负载**,热点流量下最短路反而积压;
3. **卫星各自局部最优会撞车**,把流量集中到相同链路,需要团队奖励 + credit assignment 协调。

研究问题收敛为:

> 在只允许使用本地和一跳邻居信息的条件下,能否用共享参数 MAPPO 学到一个分布式下一跳策略,使投递率、队列和尾时延**匹敌甚至超过**全局 Dijkstra,并显著优于分布式/本地基线?

## 2. 方法

### 2.1 环境

- 24 颗卫星 = 24 个 Agent(4 平面 × 6 星,环面网格);
- 每个 slot 所有 Agent 读取同一份冻结拓扑与队列快照,同时决策;
- 每个 active Agent 只处理本地 FIFO 队首包;动作 = `NO_OP + 最多 6 个候选下一跳`;
- 环境统一批量解析容量争用、转发、投递、deadline、外生到达和队列更新;
- packet ID 守恒、单一归属、单 slot 单次发送、链路容量与动作 mask 均有自动测试(28 项全绿)。

### 2.2 Actor 与 Critic

Actor 对每个候选邻居用**同一个 scorer**(DeepSets 风格的置换等变打分),每个候选 26 维特征:自身/邻居队列、传播时延、剩余带宽、负载、可靠性、`T_rem`、位置编码、packet class、等待时间、TTL、visited、backtrack、route-switch context。执行时 Actor 只读本地候选。Critic 仅训练阶段读取全局节点/边图状态(共享编码 + 边感知注意力 + permutation-invariant pooling);消融表明 **flat critic 已足够**,默认采用 flat。

### 2.3 团队奖励 + 零均值局部 credit

```
R_team = +2.0·throughput - 2.0·drop - 1.0·delay - 0.8·queue - 0.1·imbalance - 0.2·switch - 0.1·control
r_i    = R_team + 0.25·(r_local_i - mean(r_local_active))
```

credit 项零均值,锐化个体信号而不偏移团队目标。消融证明这是**最大的单一架构因子**(去掉掉 −8.8 pp)。

## 3. 主要结果(5 场景,8 seed × 50k 步,`no_lifetime`)

| 场景 | MAPPO(局部) | Dijkstra(全局 oracle) | 差距 | 配对统计 |
|---|---:|---:|---:|---|
| low_load | 0.904 | 0.907 | −0.3 pp | 持平(p=0.051) |
| medium_load | **0.774** | 0.726 | **+4.8 pp** | dz=2.84, p=1.1e-9 |
| hotspot_high_load | 0.284 | 0.295 | −1.1 pp | 唯一落后场景 |
| frequent_break | **0.439** | 0.414 | **+2.5 pp** | p=1.1e-6 |
| fault_links | **0.747** | 0.704 | **+4.3 pp** | dz=1.94, p=1.2e-9 |

对分布式基线:Q-routing 投递率 0.237–0.640,启发式 0.145–0.592 —— MAPPO 领先 14–43 pp,全部 p<1e-9。

**机制(C5 尾部与均衡)**:MAPPO 匹配或超过 oracle 投递率的同时,负载不均衡(`global_load_imbalance`)在全部 5 场景显著低于 Dijkstra(p=1.8e-15),P95 延迟与队列更低 —— 即用一点点路径最优性换更好的负载分散。

统计方法:分层 bootstrap(5000 重采样,独立重采样策略 seed 与 workload seed),配对 Wilcoxon + Benjamini-Hochberg 校正,配对 Cohen's dz,所有"无显著差异"声明附 MDE(α=0.05, power=0.8)。

数据:[`experiments/IEEE-NOLIFE-full/`](experiments/IEEE-NOLIFE-full/)(`aggregate_metrics.csv`、`paired_tests.csv`、`episode_metrics.csv`)。

## 4. 复现核查(代码漂移 vs 训练预算)

复现"失败"曾是一个 3-bug 的 BOM/heredoc 问题(utf-8→utf-8-sig + 列名 + 缺失指标过滤),数据本身正确。核查确认:**无代码回归**,故障场景 MAPPO=0.747 可稳定复现。

> ⚠️ `run_exp004_mappo.py` 默认 `--mode quick`(300 步,仅供开发);**论文级结果必须用 `--mode full`(50,000 步)**。一键复现见 [`run_ieee_reproduction.sh`](run_ieee_reproduction.sh)。

## 5. 受控消融(7 变体 × 5 场景 × 8 seed,5k 步)

| 去掉的机制 | 效应 | 判断 |
|---|---|---|
| local credit | −8.81 pp(p<1e-15) | **最大架构因子**,保留 |
| **lifetime 子系统** | 移除后 +12.4 pp(全 6 指标 × 全 5 场景) | **净有害,已删除** |
| graph critic | flat 反而 +1.07 pp | 采用 flat 为默认 |
| queue awareness | −3.4 pp(medium) | 保留 |
| PPO 保护 | 去掉后训练不稳 | 保留 |

**lifetime 负面结果(诚实报告)**:基于直觉设计的"预测链路即将断裂并规避"机制,在受控消融中全面拖累性能 —— 它把流量逼到更长更贵的路径,时延/队列代价超过了避免的故障。已从最终架构移除。在安全关键的网络控制中,直觉合理的组件仍需经验验证。

数据:[`experiments/IEEE-ABLATION-FULL/`](experiments/IEEE-ABLATION-FULL/)。

## 6. 泛化能力(零样本迁移)

| 实验 | 设置 | 结果 |
|---|---|---|
| **星座规模** | n24 训练 → n24/n66/n110/n132 零样本 | 吞吐比 **1.074/1.192/1.055/0.945**,前三个规模显著击败 oracle |
| **真实拓扑** | 合成 4×6 → Starlink-24/OneWeb-24(冻结 TLE) | 吞吐比 0.928/0.935,P95 延迟 MAPPO 更低 |
| **负载扫描** | medium 训练 → exo2..28 | exo4–exo14 全程显著击败 oracle,峰值吞吐比 **1.101**(exo8) |
| **故障率扫描** | fault_link_ratio 0..0.20 | **每个**故障率领先 ~+6 pp,故障越重优势越大 |

> 真实拓扑上 oracle 在原始投递率上仍领先(真实星间几何直径更长 → 更多在途包,这是 episode 长度的尺度塌缩伪影,非策略失败);**吞吐比(尺度不变)是诚实指标**,~93–94%,且 MAPPO 延迟与公平性显著更优。

## 7. 鲁棒性

- **收敛/可复现**:8 个训练 seed,最终回报 CV 1–9%;价值函数 EV 0.25–0.99。见 `figures/fig_convergence`。
- **奖励权重敏感**:7 个 OAT 配置(扰动 w_deliver/w_load/w_switch)全在 baseline ±1.5 pp 内,每个仍领先 Dijkstra 5–7 pp —— 拥塞感知优势不是某个特定权重调出来的。
- **跨星座训练(C3)**:在真实 Starlink 上训练**并不**优于零样本迁移(迁移就够了);跨星座 Starlink→OneWeb 保持 0.960。

## 8. ns-3 数据包级验证(静态 + 动态断链拓扑)

在 ns-3.48 中重放每条策略的**逐包源路由路径**,通过真实 FIFO 丢尾队列、字节带宽、传播时延的 P2P 环面网格。两策略跑在**完全相同**的物理模型与流量上,唯一变量是路径选择;负载通过 slot 时间压缩(1×–8×)扫描。

> **修正说明(2026-08-14)**:早期版本把 env 中被丢弃包的**部分路径**终点误计为"交付"(Dijkstra 被丢弃包更多、被抬高也更多),由此得出的"静态重放中 MAPPO 优势消失"结论是该 harness bug 的伪影。修复后静态与动态重放结果如下。

### 8.1 静态拓扑(medium_load,5 episodes)

| | env(slot 同步) | ns-3 @1× | @2× | @4× | @8× |
|---|---|---|---|---|---|
| MAPPO | **0.783** [0.769, 0.798] | **0.854** [0.846, 0.864] | **0.811** [0.797, 0.829] | **0.609** [0.550, 0.684] | **0.502** [0.442, 0.586] |
| Dijkstra | 0.719 [0.686, 0.752] | 0.782 [0.753, 0.812] | 0.760 [0.727, 0.794] | 0.565 [0.514, 0.643] | 0.476 [0.422, 0.543] |

MAPPO 在静态重放的**每个负载点**都 ≥ Dijkstra(1×/2× CI 分离;4×/8× 饱和区 CI 重叠,统计平局),差距随拥塞收窄(+7.2 → +2.6 pp),与 env 的 +6.4 pp 一致。见 `figures/fig_ns3_validation`。

### 8.2 动态断链拓扑(frequent_break,16 episodes)

链路按 env **真实的逐 slot 断链调度**通断(`links_<policy>.csv`,两策略完全相同);断链时在途包在节点等待、下个 slot 边界重试,超过业务等级 deadline 判丢,投递仅在 deadline 内有效。

| | env(slot 同步) | ns-3 @1× | @2× | @4× | @8× |
|---|---|---|---|---|---|
| MAPPO | **0.765** [0.752, 0.777] | **0.830** [0.818, 0.841] | **0.830** [0.818, 0.841] | **0.759** [0.739, 0.777] | **0.345** [0.319, 0.377] |
| Dijkstra | 0.429 [0.390, 0.465] | 0.483 [0.441, 0.524] | 0.483 [0.441, 0.524] | 0.463 [0.421, 0.504] | 0.235 [0.206, 0.265] |

**MAPPO 的优势完整迁移到数据包级动态重放**:每个负载点 CI 全部分离,差距 +34.6 / +34.6 / +29.6 / +11.0 pp,配对 Wilcoxon p ≤ 9.2e-5(16 episodes;1×/2× 无丢包故结果重合)。8× 压缩下两策略同时退化(deadline 丢包为主),排序不翻转。@1× 时 ns-3 投递数与 env **逐包完全一致**(2349 = 2349),是 harness 与 env 语义对齐的 sanity check。见 `figures/fig_ns3_dynamic`;统计 [`experiments/IEEE-NS3-DYN/ns3_dynamic_stats.csv`](experiments/IEEE-NS3-DYN/ns3_dynamic_stats.csv)。

## 9. 主要代码

所有源代码均在 [`src/`](src/) 目录下。脚本以 `python src/<脚本>.py` 运行(Python 会自动把 `src/` 加入路径);测试在 `src/` 内用 `unittest` 运行。

| 文件 | 作用 |
|---|---|
| `leo_multiagent_env.py` | 24-Agent 同步环境与团队奖励 |
| `leo_marl_env.py` | 基础环境/场景/链路模型 |
| `cleanmarl_leo_multiagent_wrapper.py` | CleanMARL 接口封装 |
| `mappo_design.py` / `mappo_evaluation.py` | 共享候选 Actor、Critic、GAE/PPO 工具、基线策略(Dijkstra/ECMP/Q-routing) |
| `run_exp004_mappo.py` | 正式训练 + 基线 + held-out 评测 + 统计 |
| `run_ablation_experiments.py` | 受控消融 |
| `run_scale_experiment.py` / `run_realism_transfer.py` / `run_load_sweep.py` / `run_fault_sweep.py` | 泛化实验 |
| `run_reward_sensitivity.py` | 奖励权重敏感 |
| `run_tle_training_experiment.py` + `tle_topology_builder.py` | TLE/SGP4 真实拓扑训练 |
| `ns3_trace_extractor.py` + `ns3_leo_validation.cc` | ns-3 逐包验证 harness |
| `make_*.py` | 全部图表生成 |
| `test_mappo_design.py` | 28 项自动化测试 |

运行测试:
```bash
cd src && python -m unittest test_mappo_design -v      # 28 项自动化测试
```

## 10. 快速开始 / 复现

```bash
# 环境(精简依赖见 requirements.txt;启用 GPU 需安装对应 CUDA 版本的 PyTorch)
python -m venv leo-venv && leo-venv/Scripts/pip install -r requirements.txt

# 一键复现 headline(自动检测 GPU,约 9h)
bash run_ieee_reproduction.sh full

# 生成全部图表
python src/make_figures.py -i experiments/IEEE-NOLIFE-x2k -i experiments/IEEE-NOLIFE-x10k -i experiments/IEEE-NOLIFE-full \
    --ablation experiments/IEEE-ABLATION-FULL --outdir figures
```

复现清单(代码源文件 SHA、git ref、pip freeze、fixture SHA)见 [`experiments/REPRO_MANIFEST.json`](experiments/REPRO_MANIFEST.json)。

## 11. 参考

- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)(MAPPO)
- [Queue-aware LEO routing, arXiv:2306.01346](https://arxiv.org/abs/2306.01346)
- [Boyan & Littman, Q-routing, NeurIPS 1994](https://arxiv.org/abs/cs/9609110)
- [Deep Sets, NeurIPS 2017](https://arxiv.org/abs/1703.06114)
- [Hypatia](https://github.com/snkas/hypatia)、[ns-3](https://www.nsnam.org/)

---

*训练依赖 [`cleanmarl`](https://github.com/AmineAndam04/cleanmarl) 的 MAPPO trainer(本地定制版)。ns-3.48 © UCL/Nsnam,CC-BY-SA 3.0。*
