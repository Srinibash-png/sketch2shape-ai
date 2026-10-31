#!/usr/bin/env bash
cd /kaggle/working/InstantMesh
export CUDA_VISIBLE_DEVICES=0
PY=/kaggle/working/im_env/bin/python
SK=/kaggle/working/my_sketches_filled
$PY -W ignore run.py configs/instant-mesh-base.yaml $SK --output_path outputs_fill_rembg
$PY -W ignore run.py configs/instant-mesh-base.yaml $SK --output_path outputs_fill_norembg --no_rembg
echo "JOB DONE"
