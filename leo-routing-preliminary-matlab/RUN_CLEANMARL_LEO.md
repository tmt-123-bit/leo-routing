# 在有 PyTorch 环境时运行 CleanMARL + LEO 路由的最小模板

这份文件是给未来正式运行 cleanmarl MAPPO 用的。

当前限制：
- 这个会话里的 Python 环境没有 `torch`
- 所以这里不直接运行 `F:\cleanmarl\cleanmarl\mappo.py`
- 但可以先把命令和接入方式固定下来

---

## 1. 需要的前置条件

至少需要：

- Python 环境里安装 `torch`
- `F:\cleanmarl` 可运行
- 当前项目目录加入 `PYTHONPATH`，或者把 `cleanmarl_leo_wrapper.py` 复制进 cleanmarl 仓库

推荐做法是：

1. 保持当前项目目录不变：

```text
F:\leo-routing-preliminary-matlab
```

2. 把下面这些文件作为当前项目环境侧实现：

```text
leo_marl_env.py
cleanmarl_leo_wrapper.py
run_python_experiments.py
```

3. 在 `F:\cleanmarl\cleanmarl\mappo.py` 里加 `env_type == "leo"` 的环境分支。

---

## 2. 最小接入命令模板

如果你已经按 `cleanmarl_env_registration_example.py` 里的方式完成环境注册，最小运行命令可以从下面这个模板开始：

```bash
cd /d F:\cleanmarl
python cleanmarl\mappo.py --env_type="leo" --env_name="medium_load" --batch_size=4 --total_timesteps=50000 --learning_rate_actor=0.0008 --learning_rate_critic=0.0008 --device="cpu"
```

说明：

- `env_type="leo"`：这是后续在 cleanmarl 里新增的环境类型。
- `env_name="medium_load"`：这里复用我们 `SCENARIOS` 里的场景名。
- `batch_size=4`：先小一点，只验证能不能训练通。
- `total_timesteps=50000`：先跑最小闭环。
- `device="cpu"`：如果没有 GPU，就先 CPU。

---

## 3. 推荐先跑的场景顺序

建议先按这个顺序：

1. `medium_load`
2. `low_load`
3. `frequent_break`
4. `fault_links`
5. `hotspot_high_load`

原因：

- `medium_load` 最平衡，适合先查训练是不是稳定；
- `low_load` 用来看学到的策略会不会无意义绕路；
- `frequent_break` 和 `fault_links` 用来看链路寿命/可靠性约束有没有帮助；
- `hotspot_high_load` 最后再上，因为最难。

---

## 4. 训练跑通后怎么接回当前评测入口

训练跑通后，建议不要新写一套评测逻辑，而是把训练好的策略接回：

```text
F:\leo-routing-preliminary-matlab\run_python_experiments.py
```

然后在里面新增一个策略名，比如：

```text
mappo_policy
```

这样就能和当前已有的：

- `delay_only`
- `queue_load`
- `full_masked_heuristic`
- `linear_policy`

统一输出同一套指标 CSV。

---

## 5. 推荐的最小结果要求

如果 cleanmarl 的第一版 MAPPO 接入成功，至少要先检查这些：

- 能否稳定跑完 rollout 和 update
- reward 是否有上升趋势
- delivery ratio 是否不低于 `full_masked_heuristic`
- 在 `frequent_break` / `fault_links` 场景下，是否比 `delay_only` 更稳
- 控制开销比例是否没有明显恶化

如果这几项都成立，就说明正式 MAPPO 接入已经可用了。

---

## 6. 这一阶段的定位

这一阶段的目标不是直接出最终 IEEE 表格，而是：

```text
把 CleanMARL 的 MAPPO 真正接进当前 LEO 环境
```

一旦这条线跑通，后面就可以继续：

- 扩大 timesteps
- 扩大 seeds
- 增加真实拓扑
- 再升级到 `F:\on-policy` 官方 MAPPO
