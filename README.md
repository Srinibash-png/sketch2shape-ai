# Sketch2Shape AI: From a 2D Sketch to a 3D Asset

**TechZephyr 4.0 | IIT Bhubaneswar | 24-Hour Live Hackathon (Computer Vision & Generative AI)**

Sketch2Shape AI takes a single-view 2D sketch (drawn in the app or uploaded) plus an optional text prompt, and produces a real 3D mesh exported as `.obj` and `.glb`. It chains pretrained models: Stable Diffusion 1.5 with ControlNet (scribble), `rembg`, and InstantMesh. No model was trained or fine-tuned by us.

- Demo video (2 to 3 min): `<ADD LINK>`
- Team: `<ADD TEAM NAME AND MEMBERS>`

---

## Table of Contents

1. [Objective](#1-objective)
2. [Architecture](#2-architecture)
3. [Model Choices and Attribution](#3-model-choices-and-attribution)
4. [Hardware and Computational Resources](#4-hardware-and-computational-resources)
5. [Time Required to Generate a 3D Model](#5-time-required-to-generate-a-3d-model)
6. [Dependencies](#6-dependencies)
7. [Installation](#7-installation)
8. [Execution](#8-execution)
9. [Expected Inputs and Outputs](#9-expected-inputs-and-outputs)
10. [Results on Test Inputs](#10-results-on-test-inputs)
11. [Limitations and Known Issues](#11-limitations-and-known-issues)
12. [Troubleshooting](#12-troubleshooting)
13. [Repository Structure](#13-repository-structure)
14. [Datasets](#14-datasets)
15. [Licenses](#15-licenses)

---

## 1. Objective

A hand-drawn sketch shows little about depth, volume, hidden surfaces or materials. The goal is a working end-to-end application that accepts one sketch, keeps its recognizable shape, and outputs a usable 3D file that opens in standard 3D software.

Implemented:
- Input by **drawing** in the app or **uploading** an image (png, jpg, jpeg, webp, bmp).
- **Optional** text prompt (object, material, colour, style). It is never required.
- Preprocessing: resizing to 512x512, background flattening, auto-contrast, and either thresholding (line art) or edge detection (photos).
- Sketch-guided image generation (Stable Diffusion 1.5 + ControlNet scribble).
- Background removal and image-to-3D reconstruction (InstantMesh).
- Mesh validation (`trimesh`) and export to `.obj` and `.glb`.
- A Gradio web interface with an interactive 3D viewer, downloads, and a per-run report.

Not implemented: UV texture maps, an in-app eraser we control (the canvas uses Gradio's built-in sketch tools), quantitative sketch-fidelity metrics.

---

## 2. Architecture

```text
Sketch (draw / upload) + optional prompt
        |
        v
[1] Preprocessing (sketch_to_image.py)
    - flatten transparency to white, auto-contrast, fit into 512x512
    - auto-detect input type:
        line art -> threshold (dark pixels become white lines on black)
        photo    -> gradient edges (top 8% strongest edges)
        |
        v
[2] Stable Diffusion 1.5 + ControlNet scribble (fp16, 25 steps, UniPC scheduler)
    -> product-style image of one object on a plain background
        |
        v
[3] InstantMesh (instant-mesh-base): rembg background removal,
    multi-view generation, sparse-view 3D reconstruction -> .obj
        |
        v
[4] Validation and export (trimesh): non-empty vertices and faces,
    non-zero extent on all axes, watertight flag -> .glb + report
        |
        v
Outputs: .obj, .glb, generated image, control image, six views, JSON report
```

How the files fit together:

| File | Role |
|---|---|
| `pipeline.py` | Orchestrator. Runs stages 1 to 4 as separate subprocesses, so GPU memory is freed between stages, and times each stage. |
| `sketch_to_image.py` | Stages 1 and 2 (preprocessing and ControlNet generation). |
| `InstantMesh/run.py` | Stage 3 (cloned by `setup_models.sh`, not part of this repo). |
| `inspect_mesh.py` | Stand-alone mesh check and `.obj` to `.glb` export. |
| `app.py` | Gradio interface. Calls `pipeline.py` for every request. |

The app processes **one job at a time** (queue concurrency of 1), because the pipeline fills most of a 15 GB GPU.

---

## 3. Model Choices and Attribution

| Component | Model / library | Why |
|---|---|---|
| Base image model | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Common base for ControlNet, fits on a T4 in fp16. |
| Sketch conditioning | `lllyasviel/sd-controlnet-scribble` | Keeps the generated image tied to the sketch shape. |
| Image to 3D | InstantMesh (`TencentARC/InstantMesh`, `instant_mesh_base.ckpt` plus `diffusion_pytorch_model.bin`) | Pretrained single-image to mesh model, suggested by the problem statement. |
| Background removal | `rembg` (used inside InstantMesh) | Isolates the object before reconstruction. |
| Rasterization | NVIDIA `nvdiffrast` (compiled from source) | Required by InstantMesh. |
| Mesh checks and export | `trimesh` | Validation and `.glb` export. |
| Interface | Gradio | Canvas, upload, 3D viewer, downloads. |

Pretrained weights are **not** stored in this repository. They download automatically (see Installation). External APIs: none, apart from Hugging Face downloads of weights.

Two deliberate settings you should know about:
- The Stable Diffusion **safety checker is disabled** (`safety_checker=None`) in `sketch_to_image.py`. Re-enable it before exposing the app publicly.
- The seed is fixed at 0 by default, so repeated runs on the same input are reproducible on the same setup.

---

## 4. Hardware and Computational Resources

All measurements were made on **Kaggle**:

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla T4, about 15 GB memory |
| OS | Ubuntu-based Linux (Kaggle image) |
| Disk | 20 GB working volume; about 8.4 GB used after environment plus InstantMesh checkpoints, and several GB more after Stable Diffusion and ControlNet weights download |
| Precision | fp16 for ControlNet, as configured by the scripts |

Peak GPU memory was **not measured**. We only know the full pipeline ran on a 15 GB card. Running two GPU jobs at once on one T4 caused a CUDA out-of-memory error during development.

Minimum assumption for judges: Linux, an NVIDIA GPU with about 15 GB memory, CUDA 12.1 toolkit (with `nvcc`), `gcc`/`g++`, `git`, internet access, and at least 20 GB free disk. Native Windows is not supported by the setup scripts. Windows users could try WSL2 with an NVIDIA GPU, which we have not tested.

---

## 5. Time Required to Generate a 3D Model

Measured on the Tesla T4, one run per input, including model loading (models reload on every run):

| Stage | Time |
|---|---|
| ControlNet (preprocessing and image generation) | 14.4 to 15.9 s |
| InstantMesh (background removal, reconstruction, mesh) | 68.1 to 72.5 s |
| **Total per model** | **82.5 to 87.6 s (about 87 s on average)** |

First run only: Stable Diffusion 1.5, ControlNet and rembg weights download, which adds several GB of transfer. One-time environment build (`setup_env.sh`, including compiling nvdiffrast) took roughly 10 to 15 minutes in our Kaggle rebuild; this is an estimate, not a stopwatch measurement.

---

## 6. Dependencies

Two Python environments are used.

**Pipeline environment `im_env`** (Python 3.10, built by `setup_env.sh` with `uv`):

| Package | Version |
|---|---|
| torch / torchvision / torchaudio | 2.1.0 / 0.16.0 / 2.1.0 (CUDA 12.1 wheels) |
| xformers | 0.0.22.post7 |
| numpy | below 2 (1.26.4 observed) |
| transformers | 4.34.1 |
| diffusers | 0.20.2 |
| huggingface-hub | 0.17.3 |
| accelerate | 0.24.1 |
| pytorch-lightning / torchmetrics | 2.1.2 / 1.2.1 |
| trimesh | below 5 (4.12.2 observed) |
| others | omegaconf, einops, webdataset, tensorboard, PyMCubes, rembg, onnxruntime, imageio[ffmpeg], xatlas, plyfile, nvdiffrast (source build) |

Exact pins are in `constraints.txt` and `req_pinned.txt`.

**App environment** (system Python 3.10 or newer, outside `im_env`): `gradio`, `pillow`, `numpy`. Tested with Gradio 6.26.0 on Python 3.13. The Gradio 6 canvas API differs from older versions, so use a recent Gradio.

**System:** `git`, `gcc`/`g++`, CUDA 12.1 toolkit with `nvcc` at `/usr/local/cuda` (override with `CUDA_HOME`), `uv` (installed by the script if missing).

---

## 7. Installation

```bash
git clone https://github.com/Srinibash-png/sketch2shape-ai.git
cd sketch2shape-ai

# 1. Build the pipeline environment (about 10 to 15 minutes)
bash setup_env.sh          # expect "SCRIPT FINISHED"

# 2. Clone InstantMesh and download its checkpoints (about 3 GB)
bash setup_models.sh       # expect "MODELS FINISHED"

# 3. Install the app dependencies
pip install gradio pillow numpy
```

Notes:
- `setup_env.sh` compiles nvdiffrast for the T4 (`TORCH_CUDA_ARCH_LIST=7.5`). For another GPU, pass its compute capability, for example `TORCH_CUDA_ARCH_LIST=8.6 bash setup_env.sh`.
- Scripts locate their own folder automatically. Set `S2S_HOME` to use a different working folder.
- Internet is required for the first run (weights from Hugging Face).
- `setup_env.sh` prints a verification block (torch, CUDA, versions, an nvdiffrast render test). You want `all imports OK`.

---

## 8. Execution

### Web app (localhost)

```bash
python app.py
```

Open `http://localhost:7860`. In `app.py` the launch line uses `share=True`, which also prints a temporary public `gradio.live` link; set `share=False` for localhost only. Then:

1. **Draw** tab: draw with the black brush, or **Upload** tab: choose an image.
2. Optionally type a prompt, for example `a red wooden chair`.
3. Choose the input type (`auto`, `sketch` or `photo`).
4. Click the generate button and wait about 90 seconds.
5. Rotate the 3D preview, check the six views and report, and download the `.obj` or `.glb`.

### Command line (single image)

Use the Python from `im_env`:

```bash
im_env/bin/python pipeline.py \
  --input path/to/sketch.png \
  --output results/ \
  --prompt "a medieval sword" \
  --mode auto            # auto | sketch | photo
```

If `--prompt` is omitted, the file name is used as the subject. Results appear in `results/`.

### Check an exported mesh

```bash
im_env/bin/python inspect_mesh.py results/sword.obj
```

Prints vertex and face counts, extents, watertightness and colour type, then writes a `.glb` next to the `.obj`.

---

## 9. Expected Inputs and Outputs

**Inputs**
- One image of one object: png, jpg, jpeg, webp or bmp (transparent backgrounds are flattened to white), or a drawing made in the canvas.
- Best results: a clear dark outline of a single object on a light background.
- Photos also work through the edge-detection path, with lower fidelity.
- Prompt: optional; the app keeps only letters, digits, spaces, commas and hyphens (max 150 characters). Empty prompt becomes `a single object`.
- An almost empty canvas is rejected with an error message.

**Outputs** (per input `NAME`)

| File | Content |
|---|---|
| `NAME.obj`, `NAME.glb` | 3D mesh with vertex colours |
| `NAME_control.png` | Preprocessed edge/line map sent to ControlNet |
| `NAME_generated.png` | Image produced by Stable Diffusion + ControlNet |
| `NAME_views.png` | Six generated views used for reconstruction |
| `NAME_report.json` | Validation result, vertex/face counts, extents, watertight flag, stage timings |

Mesh validation: the report marks `valid: true` when vertices and faces exist and every bounding-box extent is above 0.001. `valid` does **not** measure how closely the model matches the sketch.

---

## 10. Results on Test Inputs

Six test inputs, run on the Tesla T4 (`examples/` contains these outputs):

| Input | Valid | Watertight | Vertices | Faces | ControlNet | InstantMesh | Total |
|---|---|---|---|---|---|---|---|
| bird | yes | yes | 19,538 | 39,072 | 14.4 s | 68.1 s | 82.5 s |
| chair | yes | yes | 55,508 | 110,984 | 15.1 s | 72.5 s | 87.6 s |
| chair_armed | yes | yes | 20,426 | 40,812 | 14.7 s | 72.2 s | 87.0 s |
| chair_wood | yes | yes | 24,748 | 49,496 | 14.8 s | 72.1 s | 86.9 s |
| fruit | yes | yes | 33,640 | 67,272 | 14.7 s | 72.1 s | 86.8 s |
| sword | yes | yes | 6,594 | 13,192 | 15.9 s | 70.2 s | 86.2 s |

All six runs completed and exported a non-empty, watertight mesh. The photo-type inputs (`bird`, `chair_wood`, `fruit`) were run with `--mode photo`.

Quality observations (from the sword and fruit runs we inspected):
- The sword's six views were consistent with each other and the background removal was clean, but the hilt came out spiky and asymmetric because the line map merged detail.
- The fruit input drifted from the original (colours and shape changed, and two objects appeared) in an early run with a vague prompt.
- Quality was judged by eye on a small sample. We did not compute similarity scores such as CLIP or silhouette IoU.

`<ADD A SHORT QUALITATIVE NOTE FOR bird, chair, chair_armed AND chair_wood AFTER VIEWING THEIR _views.png>`

---

## 11. Limitations and Known Issues

- **Vertex colours only.** Meshes have per-vertex colour, no UV texture map.
- **Hidden geometry is guessed.** A single view cannot specify the back or underside; shapes are plausible, not exact.
- **Fidelity varies.** Generated images can drift from the sketch (colour, extra objects, merged details), especially for photos and cluttered inputs. A specific prompt helps.
- **Input-type detection is a heuristic.** `auto` mode measures how many mid-grey pixels an image has. The `chair_wood` photo was misrouted to the line-art path, so use `--mode photo` (or the app's dropdown) for photos.
- **Each run reloads all models**, so every generation takes about 87 s. There is no warm cache.
- **One generation at a time**, and a second GPU job on the same card can run out of memory.
- **GPU required.** Linux with an NVIDIA GPU of about 15 GB. Not tested on other GPUs, on CPU, or on native Windows.
- **Nvdiffrast is built for compute capability 7.5 by default**; other cards need `TORCH_CUDA_ARCH_LIST` set.
- **Safety checker disabled** in the Stable Diffusion stage.
- **Small evaluation.** Six test images, validity and timing only; no quantitative fidelity metric.
- **No guarantee** that every sketch yields a recognizable model.
- **Development issue:** Kaggle sessions shut down when idle and wipe the working folder, so the environment had to be rebuilt from these scripts. That rebuild worked and reproduced the results above.

---

## 12. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `uv: command not found` | Run `pip install uv`, then rerun `setup_env.sh`. |
| `nvcc` or `CUDA_HOME` error while building nvdiffrast | Install the CUDA 12.1 toolkit; set `CUDA_HOME` to its path. |
| nvdiffrast fails at run time on a non-T4 GPU | Rebuild with `TORCH_CUDA_ARCH_LIST` set to your card, for example `8.6`. |
| `No module named 'rembg'` | The wrong Python was used. Run pipeline scripts with `im_env/bin/python`. |
| `CUDA out of memory` | Another GPU job is running. Stop it (`pkill -f run.py`) and rerun one job at a time. |
| `ControlNet stage failed, see .../s1.log` | Read that log. Usually missing internet for the first weight download. |
| `InstantMesh stage failed, see .../s2.log` | Read that log. Check that `InstantMesh/ckpts/` holds both checkpoint files. |
| App says the canvas looks empty | Draw thicker, darker strokes, or use the Upload tab. |
| No `gradio.live` link | Internet is off, or use `http://localhost:7860`. |
| Generated model does not match the sketch | Set the input type manually, use a more specific prompt, or draw a clearer closed outline. |

---

## 13. Repository Structure

```text
sketch2shape-ai/
|-- app.py                # Gradio interface
|-- pipeline.py           # end-to-end orchestrator with stage timing
|-- sketch_to_image.py    # preprocessing + Stable Diffusion/ControlNet
|-- inspect_mesh.py       # mesh check and .glb export
|-- verify_env.py         # environment verification used by setup_env.sh
|-- setup_env.sh          # builds im_env (Python 3.10, torch, nvdiffrast, ...)
|-- setup_models.sh       # clones InstantMesh, downloads its checkpoints
|-- constraints.txt       # pinned torch/xformers/numpy/setuptools
|-- req_pinned.txt        # pinned InstantMesh requirements
|-- examples/             # sample outputs (.obj, .glb, images, reports)
|-- README.md
`-- LICENSE
```

Files such as `run_filled.sh`, `run_sketch_test.sh` and `req_filtered.txt` are leftovers from development experiments and are not needed to run the app.

---

## 14. Datasets

No model was trained, so there is no training dataset. Test inputs:

| Input | Type | Source |
|---|---|---|
| bird, chair_wood, fruit | photographs | `<ADD SOURCE AND LICENSE>` |
| chair, chair_armed, sword | line drawings | `<ADD SOURCE AND LICENSE>` |

Pretrained models were trained by their authors on their own data (see each model card).

---

## 15. Licenses

Code in this repository: MIT License. Pretrained models are not included and keep their own licenses. Check the current license terms of Stable Diffusion 1.5, ControlNet, InstantMesh, rembg and nvdiffrast before reuse, especially for commercial use.
