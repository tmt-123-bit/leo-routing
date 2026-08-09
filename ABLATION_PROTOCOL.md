# 受控消融实验登记

原则：每次只关闭一个机制，其他网络结构、训练 workload、validation/test split、优化预算和随机种子保持不变。

| Variant | 唯一改变 | 要回答的问题 | 主要观察指标 |
|---|---|---|---|
| `full` | 不关闭任何机制 | 参照组 | 全部指标 |
| `no_queue` | Actor/Critic 队列输入置零，同时关闭 queue reward shaping | 队列感知整体是否减少积压和丢包 | mean queue、delivery、P95 |
| `no_lifetime` | 寿命输入置零、取消寿命 mask 和寿命局部代价 | 提前规避快断链路是否有效 | frequent-break delivery/drop |
| `no_credit` | `beta=0`，所有 Agent 只拿 team reward | 局部 credit 是否改善协作学习 | delivery、梯度、收敛 |
| `no_packet_context` | 去掉 TTL、switch、visited、class、waiting 等 packet context | 状态完整性修复是否必要 | delivery、TTL/deadline drop |
| `flat_critic` | 图 Critic 换为相同 state 的 flat MLP | 图归纳偏置是否有贡献 | explained variance、delivery |
| `no_ppo_protection` | 关闭 advantage normalization、gradient clipping 和 target KL | 数值保护是否影响稳定性 | NaN、gradient、KL、delivery |

## 固定设置

- 场景：`medium_load`、`frequent_break`；
- policy seed：7、42、1024；
- train workload：9001..9020；
- validation：从 10001 开始；
- test：从 13001 开始；
- full ablation budget：每个 run 5,000 environment slots；
- 所有比较按相同 `(policy seed, workload seed)` 配对；
- Wilcoxon 检验后做 Benjamini-Hochberg 校正。

## 解释规则

- 消融变差支持该机制有用；消融不变或变好则不支持贡献声明。
- 只看平均已投递包时延不够，必须同时看 delivery 和 drop。
- quick 600-step 只检查实验链路；5,000-step 消融用于模块筛选，不替代 50,000-step 主实验。

