# LEO 路由与包管理：约束 MAPPO（第一层） + 可行性清除机制（第二层）

这个仓库现在包含一个两层系统，对应论文的两个部分：

**第一层（路由，前期工作）**：动态 LEO 星座中的分布式下一跳路由。每颗卫星独立选下一跳，Actor 参数共享；图 Critic 只在训练时使用。核心约束是"可避免切换"——缓存下一跳仍可用时不必要换路，用拉格朗日乘子压进 12% 预算。sealed test 换路率 1.94%（Q-routing 为 39.60%），投递率通过预设 non-inferiority 门限。

**第二层（包管理，核心贡献）**：诊断发现瓶颈不在选路而在目的端服务容量——问题从"怎么选路"重构为"哪些包值得占用服务轮次"。机制分两步：① **必死包清除**：BFS 乐观下界（当前可用链路最少跳数 × 每跳最小时延）超过死线余量 +1 的包必然超时，提前清除腾出服务轮次，**不误杀是可证明的定理**（乐观下界=必要条件），不是经验调参；② **临门包排序**：队列按剩余跳数（SRPF）/ 死线（EDF）/ 新鲜度（LCFS）重排。四个性质：可证安全、策略正交（不换路由不重训）、免调参（保守性扫描）、本地微秒级（零泛洪）。

第一层训练出的 MAPPO checkpoint 已零样本桥接进官方 Hypatia 路由仲裁器（见 `hypatia_patch/`），两层在同一官方环境中完成端到端验证。

## 研究脉络

方法演化只有一条主线：

1. **Lifetime 消融**：早期版本使用剩余链路寿命特征，实验发现策略过早绕路。关闭后保留真实断链，不再使用寿命特征/奖励/硬 mask。5k-step 消融只解释方案选择，50k 全量消融未跑完。
2. **可避免切换约束**：关闭 lifetime 后 QoS-only MAPPO 反复替换仍可用的下一跳。加入 decision-level 约束，sealed test 切换率下降 72.8%~94.9%，投递率过 non-inferiority 门限（详见下文"第一层"）。
3. **问题重构（当前主线）**：切换率压下来以后，诊断显示损失大头是目的端服务容量（积压+超时），不是选路。研究重心从路由层移到**包管理层**——ISL 队列里的清除与排序决策。这一层先在自研时隙仿真器上做机制发现与预注册面板，再整体迁移到官方 Hypatia/ns-3 环境做外部验证（`hypatia_patch/`）。

## 第一层：约束 MAPPO sealed test（24 星，历史正式结果）

当前正式实验只做中负载和热点高负载两个场景。比较对象包括 3 个 MAPPO 版本和 3 个经典方法；每个 MAPPO 版本使用 8 个独立训练 seed，测试集是 50 个此前未使用的 workload。

主比较是 `qos_only_constrained` 相对同奖励、同结构的 `qos_only_baseline`：

| 场景 | 投递率差值 | 95% CI | 可避免切换率差值 | 95% CI |
|---|---:|---:|---:|---:|
| `medium_load` | -0.958 pp | [-1.587, -0.182] pp | -15.219 pp | [-19.739, -10.357] pp |
| `hotspot_high_load` | +0.772 pp | [+0.353, +1.188] pp | -36.245 pp | [-40.872, -31.305] pp |

实验前写了四个通过条件（投递率单侧下界 ≥ −2 pp；切换率差值上界 < 0；constrained 切换率上界 ≤ 12%；每个 seed 切换率 ≤ 12%），全部通过。中负载少投递约 0.96 pp 是预设门限内的代价，不是无损改进。

各方法投递率 / 可避免切换率：

| 方法 | `medium_load` | `hotspot_high_load` |
|---|---:|---:|
| Constraint-aware MAPPO | 0.7829 / 0.0568 | 0.2945 / 0.0194 |
| QoS-only MAPPO | 0.7925 / 0.2090 | 0.2868 / 0.3818 |
| Reward-shaped MAPPO | 0.7925 / 0.1463 | 0.2885 / 0.2427 |
| Q-routing | 0.7875 / 0.2605 | 0.3114 / 0.3960 |
| OSPF-ECMP | 0.7386 / 0.1875 | 0.3012 / 0.1982 |
| Global Dijkstra | 0.7381 / 0.1886 | 0.3011 / 0.1985 |

这版结果不能解释成全面领先传统路由：热点场景投递率仍低于 Q-routing。五场景早期结果的修正版统计见 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)。

### 决策级可避免切换

```text
o = 1  当缓存下一跳和至少一个替代下一跳都可行
c = 1  当 o = 1 且策略选择了不同下一跳
R = sum(c) / sum(o)
```

首次选路、缓存链路失效后的强制切换、`NO_OP` 和 padding 不进入分子。计数发生在 contention 之前。

### Candidate-set MAPPO

- 26 维候选特征，覆盖队列、链路状态、几何进展、包上下文和缓存信息；
- 共享候选编码器和对称池化，候选顺序变化只重排 logits；
- 不可行动作采样前 mask；图 Critic 仅 centralized training 使用；
- team reward 加零均值 local credit；PPO 使用 GAE、value clipping、归一化 entropy 和 KL 约束。

约束臂 Actor loss：`L_actor = L_PPO + lambda * C_surrogate`，`lambda_next = clip(lambda + 0.05 * (R_rollout - 0.12), 0, 5)`。它是经验约束控制器，不是 CPO，不提供逐轨迹硬保证。

## 第二层：包管理方法族（核心贡献）

环境：hotspot 高负载 ｜ 固定路由（缓存 Dijkstra）｜ 只变包管理臂 ｜ 丢弃族参数经调参冻结为各家族最优。指标：投递率（送达/生成），Δ = 相对 FIFO 基线。

**11 种方法**（自研环境与官方环境同一清单）：FIFO（基线）、EDF 最早死线优先（实时调度/RC-EDF）、类别严格优先级（QoS 分级/CBQ）、LCFS 新包优先（AoI 文献）、CoDel 滞留丢弃（互联网 AQM）、RED 随机早丢（AQM 经典）、Drop-front 队首压力丢（DTN drop-oldest, RFC 6693）、**清除 Purge**（本工作：BFS 乐观下界证明必然超时→提前清除，可证不误杀）、**清除+SRPF**（本工作主栈）、**清除+EDF**（交叉消融）、**清除+LCFS**（本工作，千星新冠军）。

### 自研时隙仿真器主要结果

**66 星深饱和**（负载12，40 workload，base=0.2753）：清除+SRPF 0.3190（+15.9%，单侧下界+14.7%）> 清除+EDF +13.7% > 清除 +10.8% > 类别优先级 +3.9% > EDF/RED/Drop-front ≈0 > LCFS −2.5% > CoDel −4.2%。预注册正式面板（独立新 workload）：清除+SRPF **+16.63%（下界+15.50%，40/40 正，p=9.1e-13）**，P1/S1/S2 三判定全过。

**千星 1008**（负载12，20 workload，base=0.2405）：清除+LCFS 0.2539（+5.6%，下界+4.6%）🥇 > 清除+EDF +3.4% > 清除 +2.5% > 清除+SRPF +2.1%；LCFS 单独 +4.0%（无安全性质，下界不可比）；类别优先级 −2.2%；CoDel −6.6%；RED/Drop-front 0 次触发。

**跨规模倒 U 包络**：24星 +7.80%（8/8 正，p=0.0078）→ 66星 +16.63%（40/40 正）→ 156星 +31.6%（甜点区）→ 1008星 +2~6%。时标×2 因果实验确认千星端崩塌是损失模式迁移（传递不可达→目的端瓶颈排队），不是死线太短。

### 官方 Hypatia/ns-3 环境验证（hypatia_patch/）

应要求，全部仿真从自研仿真器迁移到官方 Hypatia（github.com/snkas/hypatia，IMC'20）+ ns-3.31 包级数据面 + satgenpy 官方星座管线重跑。机制以源码扩展形式并入官方代码库（"implemented as an extension of Hypatia"）。

**11 方法 × 5 规模 = 55 次仿真**（指标=准时投递率；统一 40 条死线 UDP 流、25000 包/臂）：

| 方法 | 24星(4×6 kNN) | 66星(11×6 kNN) | 156星 | 1008星 | 1584星(完整星链) |
|---|---:|---:|---:|---:|---:|
| FIFO（基线） | 12.50 | 2.50 | 20.22 | 49.69 | 59.97 |
| EDF | 12.50 | 2.50 | 20.22 | 47.54 | 56.85 |
| 类别严格优先级 | 12.50 | 2.50 | 20.22 | 50.80 | 59.85 |
| LCFS | 12.50 | 2.50 | 20.22 | 48.16 | 59.97 |
| CoDel | 5.00 | 0.00 | 7.50 | 5.00 | 7.50 |
| RED | 12.50 | 2.50 | 20.21 | 51.01 | 59.86 |
| Drop-front | 12.50 | 2.50 | 20.19 | 46.93 | 55.64 |
| **清除 Purge** | 12.50 | 2.50 | 20.24 | **53.31** | 62.03 |
| **清除+SRPF** | 12.50 | 2.50 | 20.24 | **53.31** | 60.00 |
| **清除+EDF** | 12.50 | 2.50 | 20.22 | 52.80 | **64.08** |
| **清除+LCFS** | 12.50 | 2.50 | 20.24 | 52.74 | 60.73 |

读表：1584 星清除+EDF 夺冠（+6.8%，热点流准时 +54%）；1008 星清除族包揽前四（+6.1~+7.3%，热点 +51%）；24/66/156 星全族持平（官方环境独立复现倒 U 边界行为）；CoDel 五档全部灾难。

**路由层正交性（1584 星，每算法跑"原始 + 叠加机制"）**：最短路（Hypatia 官方算法1）59.97→64.08%（+6.8%）；配对多路径（官方算法3）10.84→15.01%（**+38.5%**，路由越弱杠杆越大）；ILPR 持久化（忠实移植）≡ 最短路；CMADR 预算约束（忠实移植）≡ 最短路；POMAP 式队列感知（桥接在线运行）≡ 最短路；约束 MAPPO checkpoint 零样本桥接（退化档持平）。地面中继（官方算法2）为无 ISL 架构不可比；MATMR 原文无公开源码未移植。

**统一发现（四重独立证据）**：官方 100ms 粒度路由是准静态的（每窗全网仅 ~20 条变更）且 ISL 队列浅而弥散——路由层一切适应性结构性无操作空间，性能分化的唯一舞台在包管理层，正是本机制所在层。机制叠加在每一种可比路由上全部为正。

**分析面板**：死线扫描复现倒 U 包络（40ms 档 +22.0%）；margin 扫描证明免调参（1.5~6ms 平台期，4× 过度保守杀可行包）；5 流量种子统计面板 +7.55pp（相对 +13.9%，配对 t=4.84，p<0.01，5/5 种子为正）。

注意：两套环境指标口径不同（官方=准时投递率，自研=投递率），数字不可互换；各工况的归属在实验记录中分别标注。

## 仓库结构

```text
src/          自研时隙仿真器：环境、MAPPO、基线、包管理机制、统计、实验 runner 和测试
hypatia_patch/  官方 Hypatia/ns-3 扩展（C++ 机制 + Python 驱动，deploy.sh 一键部署）
docs/         预注册协议、修订记录和 claim boundary
experiments/  紧凑结果、冻结文件和公开的 sealed rows
figures/      历史实验图表
data/         TLE 与导出的拓扑
```

`hypatia_patch/` 内容：

```text
model/    PurgeSrpfQueue（11 模式 ISL 队列，isl_queue_type 一键切换）、DeadlineTag、
          DeadlineUdpApplication、SatnetHopOracle（实时跳数 oracle）
helper/   DeadlineUdpScheduler（可选第 8 类流量列）
edit/     官方源码接入点：拓扑队列接线、main_satnet、激光设备空指针保护、
          桥接仲裁器 helper（TCP 上报→外部策略服务→五元组热装）
bridge_server.py   qaware / MAPPO 零样本策略服务（Python 策略服务 ↔ WSL ns-3）
custom_knn.py / custom_pn.py   kNN / +Grid 星座壳层生成器（含三时点稳定性校验）
ilpr_postprocess.py / cmadr_postprocess.py   SOTA 路由忠实移植后处理器
family_sweep.sh 等驱动   任意壳层 × 任意方法组合的扫描；deploy.sh 一键部署重建
```

两个已知的 ns-3 集成坑（实现备注）：① ns-3.31 UDP 发送错误路径会让整条流停滞，判死必须放在出队时 `DropAfterDequeue` 而非入队拒绝；② 激光设备 Send() 假设"入队后队列非空"，清掉唯一包会 SIGSEGV，需要空指针保护。

## 安装

自研仿真器：Python 3.10 或 3.11，PyTorch/CUDA 按本机驱动选择。

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

官方 Hypatia 扩展：先按官方流程在 WSL Ubuntu 克隆构建 Hypatia（ns-3.31 构建需 `./waf configure --build-profile=optimized --disable-werror`），再运行 `bash hypatia_patch/deploy.sh` 拷贝补丁并重建。MAPPO 桥接臂另需 PyTorch 策略服务（`bridge_server.py`）。

## 测试

自研仿真器（仓库根目录）：

```bash
cd src
python -m unittest discover -p "test_*.py"
```

只检查 constraint/sealed-test 链路：

```bash
cd src
python -m unittest \
  test_avoidable_switch_constraint \
  test_avoidable_switch_constraint_formal \
  test_formal_avoidable_switch_statistics \
  test_avoidable_switch_classical_baselines_formal \
  test_avoidable_switch_joint_sealed_test
```

sealed workloads 已经按协议使用过一次。不要删除输出目录后重新跑 test panel，也不要用 test 结果重新选 seed、checkpoint 或场景。公开结果的只读入口是：

```text
experiments/avoidable-switch-joint-sealed-test-v1/sealed_test_statistics.json
experiments/avoidable-switch-joint-sealed-test-v1/sealed_test_rows.csv
```

## 冻结记录

```text
MAPPO training freeze
153b5cd85305f3f4c9a03fd549eea3e06785a0d1a447b96ec9005098c7d25742

MAPPO validation freeze
d3c039657802e9c784c82cec31024e5098b31e5a3e217e69bbc64dad5a8469d6

Classical training freeze
df7cf104d30360df898ef8239b64789c16e0baa9e87c78c52c08a676ecb62eba

Classical gate freeze
36bac779a6961ce38eb6872ec0f9cfa3953898ceb00d9b5bd7f10ce80fbc3a1e

Joint sealed-test authorization
9e2c173b0399adf52adeb575ffb11742d2c3452eb0d50bb6bdccdac22ceed653

Final sealed-test freeze
c5cc82c7edb96add6e8da1275924e83cf159e65a2062184d52f3364ad8e508a9
```

协议细节见：

- [`docs/AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md`](docs/AVOIDABLE_SWITCH_CONSTRAINT_FORMAL_V1.md)
- [`docs/AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md`](docs/AVOIDABLE_SWITCH_CLASSICAL_BASELINES_FORMAL_V1.md)
- [`docs/AVOIDABLE_SWITCH_JOINT_SEALED_TEST_V1.md`](docs/AVOIDABLE_SWITCH_JOINT_SEALED_TEST_V1.md)

## 结果边界

目前证据支持的说法很具体：

- 24 星自研环境中，显式 decision-level 约束相对匹配的 QoS-only MAPPO 大幅降低可避免切换率，并通过预设投递率 non-inferiority 门限；
- 66 星深饱和与千星自研环境中，可行性清除族是唯一全规模正增益的包管理家族，清除+SRPF 通过预注册正式面板（+16.63%，下界+15.50%）；
- 官方 Hypatia/ns-3 环境中（24→1584 星五档），清除族是唯一全规模非负且大规模显著为正的方法（1584 星 +6.8%，热点流 +54%），且叠加在每一种可比路由上全部为正；官方环境的准静态路由层使包管理层成为巨型星座性能分化的唯一舞台。

它还不能说明：

- 机制在目的端瓶颈排队型损失下有效（千星崩塌边界已定位为损失模式迁移，需网关调度/接入控制，future work）；
- 优于所有经典路由方法（第一层热点场景投递率仍低于 Q-routing）；
- 已经验证控制面收敛或真实信令开销；
- 能达到工程部署条件。
