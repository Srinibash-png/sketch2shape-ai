#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
PY=/kaggle/working/im_env/bin/python

echo "[A] Cloning InstantMesh"
cd /kaggle/working
[ -d InstantMesh ] || git clone -q https://github.com/TencentARC/InstantMesh.git
mkdir -p InstantMesh/ckpts
cd InstantMesh

echo "[B] Downloading checkpoints"
$PY -W ignore -c "from huggingface_hub import hf_hub_download; [print(hf_hub_download(repo_id='TencentARC/InstantMesh', filename=f, local_dir='ckpts', local_dir_use_symlinks=False)) for f in ['diffusion_pytorch_model.bin', 'instant_mesh_base.ckpt']]"
ls -la ckpts
df -h /kaggle/working | tail -1
echo "MODELS FINISHED"
