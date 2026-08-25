#!/usr/bin/env bash
# =============================================================================
# One-time setup: F: venv with CUDA torch for the LEO MAPPO project.
# Honors "heavy stuff on F:" (C: is ~full). Uses CHINA MIRRORS for speed:
#   - Tsinghua PyPI for the 199 locked deps
#   - Aliyun pytorch-wheels for the exact CUDA torch wheel (direct URL)
# torchvision is NOT installed (unused by the project — verified).
#
# After this + NVIDIA driver update + reboot, run on GPU via:
#   DEVICE=cuda bash run_ieee_reproduction.sh smoke
# =============================================================================
set -uo pipefail
VENV=/f/leo-venv
PY="$VENV/Scripts/python.exe"
CACHE=/f/leo-pip-cache
REQ=/f/leo-routing-preliminary-matlab/requirements_locked.txt
TSINGHUA=https://pypi.tuna.tsinghua.edu.cn/simple
# Exact wheel: torch 2.11.0, CUDA 12.8, Python 3.13, Windows (verified 200 OK on Aliyun)
TORCH_WHEEL="https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl"

echo "### [1/4] upgrade pip"
"$PY" -m pip install --cache-dir "$CACHE" --upgrade pip

echo "### [2/4] install 199 locked deps from Tsinghua mirror (streaming)..."
"$PY" -m pip install --cache-dir "$CACHE" -r "$REQ" -i "$TSINGHUA" || \
  echo "WARNING: some frozen pkgs failed (often unrelated junk) — continuing if core deps are present"

echo "### core runtime deps (ensure present regardless of above)"
"$PY" -m pip install --cache-dir "$CACHE" -i "$TSINGHUA" numpy scipy tyro tensorboard

echo "### [3/4] install CUDA torch 2.11.0+cu128 from Aliyun (deps resolved via Tsinghua)"
"$PY" -m pip install --cache-dir "$CACHE" "$TORCH_WHEEL" -i "$TSINGHUA" || {
  echo "Aliyun direct-URL failed; falling back to pytorch.org cu128/cu126/cu124..."
  for IDX in cu128 cu126 cu124; do
    "$PY" -m pip install --cache-dir "$CACHE" "torch==2.11.0" \
      --index-url "https://download.pytorch.org/whl/$IDX" && break
  done
}

echo "### [4/4] verify"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| cuda_available', torch.cuda.is_available(), '| gpu_count', torch.cuda.device_count())"
echo "### project imports + 28 tests"
cd /f/leo-routing-preliminary-matlab
(cd src && "$PY" -c "import cleanmarl_leo_multiagent_wrapper, mappo_design, mappo_evaluation; print('project imports OK')")
(cd src && "$PY" -m unittest test_mappo_design) 2>&1 | tail -5
echo "### DONE. cuda_available flips True after the NVIDIA driver update (512.36 -> 610.88) + reboot."
