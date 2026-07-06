# 真实拓扑数据文件格式说明

这份文件的目的，是把后续接真实/准真实星座拓扑时需要的最小数据格式固定下来。这样不管数据来自：

- Hypatia
- StarryNet
- TLE / SGP4
- 你自己整理的链路快照

最后都可以先转成统一 CSV，再由 `hypatia_topology_provider_stub.py` 或后续 provider 读取。

---

## 1. 最小必需列

至少需要下面 5 列：

```text
time_slot, src, dst, delay_ms, available
```

含义：

- `time_slot`：当前链路快照属于第几个时间片
- `src`：源卫星编号
- `dst`：目的卫星编号
- `delay_ms`：该有向链路的传播时延（或传播+传输近似）
- `available`：该时间片这条链路是否存在

如果只有这 5 列，环境仍然可以跑，但：

- `capacity_mbps` 会用默认值；
- `reliability` 会用默认值；
- `t_rem` 会用默认值；
- 这样只适合做“最小可运行”版本。

---

## 2. 推荐列（论文更合理）

为了更接近你文稿里的建模，推荐至少补到下面这些列：

```text
time_slot, src, dst, delay_ms, available,
capacity_mbps, reliability, t_rem,
is_cross, shell_src, shell_dst
```

这些字段分别对应：

- `capacity_mbps`：链路容量
- `reliability`：链路可靠性
- `t_rem`：链路剩余可用时间
- `is_cross`：是否为跨轨/跨层链路
- `shell_src`：源卫星所在壳层
- `shell_dst`：目的卫星所在壳层

这套字段已经能比较好地覆盖你文稿里“队列感知 + 链路寿命约束 + 分层参考”的需求。

---

## 3. 一个最小 CSV 示例

```csv
time_slot,src,dst,delay_ms,available,capacity_mbps,reliability,t_rem,is_cross,shell_src,shell_dst
1,1,2,8.2,True,100.0,0.98,999.0,False,1,1
1,2,1,8.2,True,100.0,0.98,999.0,False,1,1
1,2,3,8.5,True,100.0,0.97,999.0,False,1,1
1,3,2,8.5,True,100.0,0.97,999.0,False,1,1
1,3,4,12.0,True,100.0,0.94,4.0,True,1,1
1,4,3,12.0,True,100.0,0.94,4.0,True,1,1
```

当前项目里运行：

```bash
python hypatia_topology_provider_stub.py
```

会自动生成一个示例文件：

```text
outputs/hypatia_topology_demo.csv
```

---

## 4. 从外部系统映射时最重要的两点

### 4.1 有向边

环境要求的是：

```text
(src, dst) -> LinkState
```

所以即使外部系统给的是无向链路，也最好在导入时显式展开成两条有向边：

- `(u, v)`
- `(v, u)`

这样和当前环境内部的转发逻辑最一致。

### 4.2 时间片一致性

同一个 `time_slot` 下，最好保证所有链路都来自同一次快照。

不要出现：

- 一半链路来自时刻 `t`
- 另一半链路来自时刻 `t+1`

否则 `t_rem`、delay、availability 的语义会乱掉。

---

## 5. reliability 和 t_rem 如果外部没有怎么办

### reliability

如果外部数据没有给可靠性，建议：

- 最小可运行版本：先统一给一个默认值，比如 `0.98`
- 更合理版本：根据故障率、链路类型、历史成功率、ETX 或 outage 统计估计

### t_rem

如果外部数据没有直接给 `t_rem`，建议：

- 用“当前时刻到下一次链路断开时刻”的差值近似；
- 或者用可见窗口长度减去已过去时间；
- 如果只是最小可运行版本，可以先给跨轨链路一个较小值、同轨链路一个很大值。

---

## 6. 当前项目与该格式的关系

当前项目里已经有三层支持：

1. `leo_marl_env.py`
   - 提供 `topology_provider` 接口

2. `real_topology_provider_example.py`
   - 说明 provider 返回的 Python 数据结构

3. `hypatia_topology_provider_stub.py`
   - 说明如何从 CSV 快照读入并转成 provider

所以后续如果拿到真实/准真实拓扑，最推荐做法是：

```text
先转成统一 CSV
再写 provider 读取它
最后传给 EnvConfig(topology_provider=...)
```

这样不会破坏现有训练和评测链条。
