#!/usr/bin/env bash
set -e
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=4
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0

ENV=/kaggle/working/im_env
PY=$ENV/bin/python
LOG=/kaggle/working/setup_env.log
export PATH=$ENV/bin:$PATH
: > $LOG
trap 'echo "FAILED near line $LINENO. Last 40 log lines:"; tail -40 $LOG' ERR

printf '%s\n' "torch==2.1.0" "torchvision==0.16.0" "torchaudio==2.1.0" "xformers==0.0.22.post7" "numpy<2" "setuptools==80.10.2" > /kaggle/working/constraints.txt
printf '%s\n' "pytorch-lightning==2.1.2" "transformers==4.34.1" "diffusers==0.20.2" "huggingface-hub==0.17.3" "accelerate==0.24.1" "torchmetrics==1.2.1" "omegaconf==2.3.0" "einops" "webdataset" "tensorboard" "PyMCubes" "trimesh<5" "rembg" "onnxruntime" "imageio[ffmpeg]" "xatlas" "plyfile" > /kaggle/working/req_pinned.txt

echo "[1/6] Creating fresh Python 3.10 environment"
rm -rf $ENV
uv python install 3.10 >> $LOG 2>&1
uv venv $ENV --python 3.10 >> $LOG 2>&1

echo "[2/6] Installing PyTorch 2.1.0 (CUDA 12.1) and xformers"
uv pip install --python $PY torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 >> $LOG 2>&1
uv pip install --python $PY "numpy<2" xformers==0.0.22.post7 >> $LOG 2>&1

echo "[3/6] Building nvdiffrast (2 to 4 min)"
[ -d /kaggle/working/nvdiffrast_src ] || git clone -q https://github.com/NVlabs/nvdiffrast /kaggle/working/nvdiffrast_src
uv pip install --python $PY -c /kaggle/working/constraints.txt "setuptools==80.10.2" wheel ninja >> $LOG 2>&1
uv pip install --python $PY --no-build-isolation -c /kaggle/working/constraints.txt /kaggle/working/nvdiffrast_src >> $LOG 2>&1

echo "[4/6] Installing InstantMesh requirements (pinned, torch locked)"
uv pip install --python $PY -c /kaggle/working/constraints.txt -r /kaggle/working/req_pinned.txt >> $LOG 2>&1

echo "[5/6] Verifying"
$PY -W ignore /kaggle/working/verify_env.py 2>&1

echo "[6/6] Dependency conflict check and disk"
uv pip check --python $PY 2>&1 || true
df -h /kaggle/working | tail -1
echo "SCRIPT FINISHED"
