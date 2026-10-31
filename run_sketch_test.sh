#!/usr/bin/env bash
cd /kaggle/working/InstantMesh
export CUDA_VISIBLE_DEVICES=0
PY=/kaggle/working/im_env/bin/python
SK=/kaggle/working/my_sketches
S=$(date +%s)
$PY -W ignore run.py configs/instant-mesh-base.yaml $SK --output_path outputs_sk_rembg
M=$(date +%s)
$PY -W ignore run.py configs/instant-mesh-base.yaml $SK --output_path outputs_sk_norembg --no_rembg
E=$(date +%s)
echo "TIMING rembg run: $((M-S)) s | no_rembg run: $((E-M)) s"
echo "JOB DONE"
