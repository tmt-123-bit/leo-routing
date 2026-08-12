# GPU 设置（一次性，约 15 分钟）

你的机器有一张 **RTX 3070 Laptop GPU (8GB)**，但目前 PyTorch 是 CPU 版，显卡在空转。
本指南把它打开。**重的东西都装在 F 盘**（C 盘快满了）。

> 我已经在 F 盘建好 venv 并装 CUDA torch（`F:\leo-venv`，脚本 `setup_venv_f.sh`）。
> 你只需要做**第 1 步**（装驱动+重启），第 2 步验证。

---

## 第 1 步：更新 NVIDIA 驱动（绕不开，Windows 装驱动必须重启）

你现在驱动 `512.36`（2022 年）只支持 CUDA 11.6，装不了现代 CUDA torch。最新驱动 `610.88` 支持 CUDA 12.x。

**下载（二选一）：**

- 官方（推荐）：打开 https://www.nvidia.com/Download/index.aspx ，按下面选：
  - Product Type: **GeForce**
  - Product Series: **GeForce RTX 30 Series (Notebooks)** ← 必须是 Notebooks/笔记本
  - Product: **GeForce RTX 3070 Laptop GPU**
  - Operating System: **Windows 11**
  - Download Type: **Game Ready Driver (GRD)**
  - → Search → Download

- 镜像（官方慢时）：TechPowerUp 搜 "NVIDIA GeForce Graphics Driver 610.88"，选 **Notebook** 版。

**安装：** 双击下载的 exe → 选 "Express" → 装完**重启电脑**。

重启后打开终端验证驱动：
```bash
nvidia-smi | head -3
# 头部 "CUDA Version:" 应变成 12.x（之前是 11.6）
```

---

## 第 2 步：验证 GPU + torch（重启后）

venv 已在 F 盘建好。验证 CUDA 真的可用：
```bash
/f/leo-venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 必须输出: True NVIDIA GeForce RTX 3070 Laptop GPU
```

如果上面报 `False`，多半是 venv 里的 torch CUDA 还没装好（下载慢），跑一次：
```bash
bash /f/leo-routing-preliminary-matlab/setup_venv_f.sh
```

然后跑一个 GPU 冒烟（几十秒，看 EV 是否在涨）：
```bash
cd /f/leo-routing-preliminary-matlab
DEVICE=cuda bash run_ieee_reproduction.sh smoke
```
`run_ieee_reproduction.sh` 会**自动用 F 盘的 venv**（已配好），不用手动传 `PY=`。

---

## 第 3 步：真正开跑

GPU 验证通过后，先跑**复现核查**（最关键的一步，判断是代码漂移还是训练预算）：
```bash
DEVICE=cuda bash run_ieee_reproduction.sh repro
```
看 `fault_links` 的 MAPPO 投递率：落在 **~0.55–0.65** = 旧 FULL 数字可复现（之前是预算问题）；
落在 **~0.05–0.15** = 代码回归（告诉我，我查）。

确认无误后跑全量：
```bash
DEVICE=cuda bash run_ieee_reproduction.sh     # FULL + 全5场景消融 + budget + MDE
```

---

## 速度预期

- CPU（现在）：50K×8seed×5场景 ≈ **数天**（不可行）。
- RTX 3070：同样的量 ≈ **几十分钟到几小时**（取决于场景）。batch_size=4 对这张卡很轻，GPU 利用率不会满，但比 CPU 快 10–50×。

## 如果 venv 装太慢

网络慢时 `setup_venv_f.sh` 拉 torch（~2.5GB）会很久。可以：
- 换国内 PyPI 镜像加速 199 个普通包：编辑脚本，加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- torch 仍需从 `download.pytorch.org` 下（无国内镜像），耐心等或用代理。

## 回退

所有改动都在 F 盘 venv，**没动系统 Python313**。原来 `py ...` 的跑法完全不受影响。
删 venv 即可彻底回退：`rm -rf /f/leo-venv /f/leo-pip-cache`。
