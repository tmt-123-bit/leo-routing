# LEO 星座 MAPPO 分布式路由

24 星 Walker-Delta LEO 星座上的分布式下一跳路由研究:每颗卫星是一个 agent,共享参数的候选-邻居 Actor(MAPPO,CTDE),团队奖励 + 零均值局部 credit。对照基线包括集中式 SPF/ECMP、表格 Q-routing、逐节点 DQN、陈旧链路状态 SPF。验证链路三级:自建 slot 环境 → ns-3 包级重放(路径验证)→ ns-3 闭环(策略在 ns-3 事件循环内基于真实队列状态决策)。

## 目录结构

```
src/          全部 Python/C++ 源码(环境、训练、评估、基线、ns-3、图表)
experiments/  实验输出(CSV 数据 + manifest;checkpoint 被 .gitignore 排除,仅存本地)
figures/      出版级图(每个 .png 配 .pdf)
data/         真实 TLE 与导出拓扑(Starlink / OneWeb,2026-07-15)
run_preliminary_leo_routing.m   早期 MATLAB 原型(项目起点,保留存档)
setup_venv_f.sh / requirements*.txt   环境与依赖
run_reproduction.sh   全管线复现入口(smoke / repro / budget / mde)
```

## src/ 源码一览

**环境与核心**
| 文件 | 说明 |
|---|---|
| `leo_marl_env.py` | 单 agent 底层路由环境(拓扑、链路、包生命周期) |
| `leo_multiagent_env.py` | 多 agent 环境:24 星、slot 时钟、队列/链路容量/到达过程、26 维候选特征、动作屏蔽 |
| `mappo_design.py` | MAPPO 设计层:动作屏蔽、零均值局部 credit、团队奖励 |
| `mappo_evaluation.py` | 评估与基线策略(GlobalDijkstra / OSPF-ECMP / Q-routing / heuristic) |
| `cleanmarl_leo_multiagent_wrapper.py` | CleanMARL 训练接口的向量化 wrapper |
| `tle_topology_builder.py` + `data/` | 从真实 TLE 构建星间拓扑 |

**训练入口**
| 文件 | 说明 |
|---|---|
| `run_exp004_mappo.py` | 主训练入口(`--mode quick|full`,`--scenario`,经 `/f/cleanmarl` 的 MAPPO trainer 执行) |
| `run_full_training_matrix.py` | 5 场景 × 8 seed 全矩阵训练 |
| `run_ablation_experiments.py` / `run_ablation_training_shard.py` | 消融(no_credit / flat_critic / lifetime 等) |
| `run_tle_training_experiment.py` | 真实 TLE 拓扑上训练 |
| `run_exp005_diagnostics.py` | 训练诊断 |

**基线与评估实验**
| 文件 | 说明 |
|---|---|
| `run_dqn_baseline.py` + `dqn_baseline.py` | 逐节点 DQN(神经 Q-routing)基线 |
| `run_stale_baseline.py` | 陈旧链路状态 SPF(K=1/3/5/10) |
| `run_fault_sweep.py` / `run_load_sweep.py` | 故障率 / 外生负载扫描 |
| `run_scale_experiment.py` / `run_qrouting_scale_experiment.py` | 星座规模迁移(n24→n132)与 Q 表规模对照 |
| `run_realism_transfer.py` | 真实 TLE(Starlink/OneWeb)零样本迁移 |
| `run_reward_sensitivity.py` | 奖励权重敏感性 |
| `compute_mde.py` / `analyze_transfer_stats.py` | 统计功效与配对检验 |

**ns-3 验证(WSL2 Ubuntu-22.04,ns-3.48,用户 nsuser,`~/ns-3.48`)**
| 文件 | 说明 |
|---|---|
| `ns3_trace_extractor.py` | 从 slot 环境导出包级 trace(episode/包/链路 CSV)供 ns-3 读取 |
| `ns3_leo_validation.cc` | ns-3 包级重放:预计算路径在 ns-3 事件循环里执行 |
| `ns3_closed_loop.cc` | ns-3 闭环:每 slot 上报自身队列/HOL 包/逐链路 TX,等待策略决策后执行 |
| `ns3_closed_loop_server.py` | Windows 侧策略服务器:用 env 同一套特征代码从 ns-3 状态重建 26 维候选特征 |
| `ns3_policy_bridge.py` + `ns3_policy_protocol.schema.json` | 检查点加载与 slot 级决策协议 |
| `run_ns3_closed_loop.py` | 编排:构建 scratch 程序、起服务器、经 WSL NAT 网关连 ns-3 |
| `run_ns3_sweep.sh` / `run_ns3_dyn_sweep.sh` / `dbg_ns3.sh` | 负载/动态场景批量重放 |

**图表生成**
| 文件 | 输出 |
|---|---|
| `make_figures.py` | fig1–fig4(主结果/分场景/尾延迟与均衡/消融)+ Table I |
| `make_convergence_figure.py` / `make_fairness_figure.py` | 收敛曲线 / 公平性 |
| `make_{fault,load,reward}_sweep_figure.py` | 各 sweep 图 |
| `make_scale_figure.py` / `make_realism_figure.py` / `make_tle_figure.py` | 规模迁移 / 真实拓扑 |
| `make_qscale_figure.py` | Q 表规模崩溃 vs MAPPO 零样本 |
| `make_ns3_figure.py` / `make_ns3_dynamic_figure.py` / `make_ns3_closedloop_figure.py` | ns-3 三级验证图 |
| `make_deployment_cost.py` / `analyze_hotspot.py` / `make_results_summary.py` | 部署代价 / hotspot 机制分解 / 汇总表 |

**测试**:`cd src && python -m unittest test_mappo_design`(28 个用例)

## experiments/ 目录说明

**主结果**
| 目录 | 内容 | 生成入口 |
|---|---|---|
| `train-main/` | 主训练(5 场景 × 8 seed,`no_lifetime` 变体) | `run_full_training_matrix.py` |
| `eval-main/` | 主评估:5 场景 × 8 seed × 50 held-out episodes,全策略同信息集 | `run_exp004_mappo.py` 评估段 |
| `ablation/` | 消融(no_credit −8.8pp 等) | `run_ablation_experiments.py` |
| `train-budget-*` / `qrouting-budget-*` / `eval-budget-*` | 训练预算 x2k/x10k 的 MAPPO 与平价重训 Q-routing | `run_reproduction.sh budget` |

**泛化与机制**
| 目录 | 内容 | 生成入口 |
|---|---|---|
| `sweep-fault/` `sweep-load/` `sweep-reward/` | 故障率 / 外生负载 / 奖励权重扫描 | `run_fault_sweep.py` 等 |
| `sweep-scale/` | 星座规模 n24→n132 零样本迁移 | `run_scale_experiment.py` |
| `sweep-realism/` | 真实 TLE(Starlink/OneWeb)拓扑迁移 | `run_realism_transfer.py` |
| `qscale-transfer/` | Q 表跨规模部署崩溃 vs MAPPO 零样本对照 | `run_qrouting_scale_experiment.py` |
| `dqn-baseline/` | 逐节点 DQN 五场景结果 | `run_dqn_baseline.py --scenario <name>` |
| `stale-spf/` | 陈旧链路 SPF(K=1/3/5/10) | `run_stale_baseline.py` |
| `deployment-cost/` | 参数量/内存/MACs/决策时延 | `make_deployment_cost.py` |
| `hotspot-mechanism/` | hotspot 场景丢包机制分解 | `analyze_hotspot.py` |

**ns-3 验证**
| 目录 | 内容 | 生成入口 |
|---|---|---|
| `ns3-trace16/` | 16 episode 包级 trace(闭环输入) | `ns3_trace_extractor.py` |
| `ns3-replay/` `ns3-replay-dyn/` | ns-3 包级重放(静态负载 / 动态断链) | `run_ns3_sweep.sh` 等 |
| `ns3-closedloop/` | ns-3 闭环 16 episodes 正式结果 | `run_ns3_closed_loop.py` |
| `ns3-closedloop-5ep/` | 闭环 5 episode 冒烟测试 | 同上 |
| `archive/` | 历史代次数据(早期 EXP-004/005、修正前 sweep、repro-check 等),仅存档 | — |

## 快速开始

```bash
# 环境(Windows + F 盘 venv;CUDA torch 在 RTX 3070 上训练)
bash setup_venv_f.sh            # 或直接用 /f/leo-venv/Scripts/python.exe
pip install -r requirements.txt

# 冒烟训练(~1 分钟)
/f/leo-venv/Scripts/python.exe src/run_exp004_mappo.py --cleanmarl F:/cleanmarl \
    --project F:/leo-routing-preliminary-matlab/src --mode quick --scenario medium_load

# 全管线复现(smoke / repro / budget / mde 四个子命令)
bash run_reproduction.sh

# 测试
cd src && /f/leo-venv/Scripts/python.exe -m unittest test_mappo_design
```

ns-3 验证需要 WSL2(用户 `nsuser`,`~/ns-3.48`):

```bash
/f/leo-venv/Scripts/python.exe src/run_ns3_closed_loop.py \
    --scenario medium_load --workload-seeds 21001,...,21016 --policies mappo,dijkstra
```

闭环架构:ns-3 每 slot 边界把自身原始状态(队列长度、HOL 包字段、到达数、逐有向链路传输量)经 TCP 发给策略服务器;服务器用环境同一套 `_candidate_features`/`_mask_reason` 代码重建候选特征(几何项来自环境模型,队列/带宽/竞争项来自 ns-3 实际数据面),MAPPO 或 Dijkstra 决策后由 ns-3 真实 FIFO 数据面执行——策略条件化的正是 ns-3 自己的队列,环路闭合。

## 关键结果速览

主评估(`eval-main/`,全策略同信息集,投递率):MAPPO 0.913/0.788/0.290/0.759/0.763(low/medium/hotspot/frequent/fault),对集中式 SPF/ECMP 领先 +4.9~+5.3pp(p≤1.2e-9);表格 Q-routing 平价重训后与 MAPPO 打平(±0.6pp)——MAPPO 的差异化在零样本迁移:Q 表跨星座部署崩溃(n132 仅 9.2MB 表也训不满),MAPPO 零样本保持 ≥oracle 至 4.6× 规模。ns-3 三级验证一致:重放 +6.1pp、动态 +4.8pp、闭环 +7.4pp(16/16 episodes,Wilcoxon p=4.3e-4)。

所有比较均无变体不对称(基线与 MAPPO 使用同一动作屏蔽信息集);历史修正前的数据保存在 `archive/` 与各目录 `superseded_*` 子目录中,可复查。
