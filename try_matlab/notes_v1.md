# try_matlab 这一版是干嘛的

这不是最终算法，就是先把论文里几个想法跑一下。

目前试了四个版本：

| 名字 | 干嘛 |
|---|---|
| `dijkstra_delay` | 最普通的最短传播时延路由 |
| `add_q_load` | 在代价里加下一跳队列和链路负载 |
| `add_life_mask` | 再加 `T_rem`、可靠性、剩余带宽过滤 |
| `try_risk_prog` | 又试了一点可靠性风险和目的方向 progress |

这里面 Dijkstra 不是我的方法，只是先做 baseline。真正论文后面还是要接：

```text
Dec-POMDP + RTMDP + MAPPO/CTDE
```

现在这版主要想看三件事：

1. 只看传播时延会不会把队列挤起来；
2. 加 `q_j` 和 `rho_ij` 后，最大队列和 P95 时延会不会下降；
3. 加 `T_rem` 过滤后，是不是能少选快断链路，但可能绕一点路。

运行：

```matlab
cd try_matlab
try_route_v1
```

会生成：

```text
tmp_out/stat.csv
tmp_out/slot_log.csv
tmp_out/quick_plot.png
```

这版里面的变量和 README 里的对应关系大概是：

| 代码里 | README 里 |
|---|---|
| `q` | `q_i(t)` / `q_j(t)` |
| `X.d` | `delay_ij(t)` |
| `X.rho` | `rho_ij(t)` |
| `X.rel` | `reliability_ij(t)` |
| `X.trem` | `T_rem_ij(t)` |
| `obsZ()` | `z_ij(t)` |
| `localR()` | 局部即时奖励雏形 |

参考过的东西主要是：

- MA-DRL_Routing_Simulator: https://github.com/SatCom-TELMA/MA-DRL_Routing_Simulator
- Hypatia: https://github.com/snkas/hypatia
- LEOPath: https://github.com/Fundacio-i2CAT/LEOPath
- Queue-Aware LEO MARL: https://arxiv.org/abs/2605.04448
- RTMDP routing: https://ieeexplore.ieee.org/document/10693714/

后面如果要继续写，就在这个基础上加：

```text
visited nodes 防环路
更真实的星座轨迹
地面站
Q-routing baseline
MAPPO 环境接口
```

