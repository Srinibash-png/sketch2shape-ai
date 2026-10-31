import os, re, json, uuid, shutil, subprocess, glob
import numpy as np
import gradio as gr
from PIL import Image

W = "/kaggle/working"
PY = f"{W}/im_env/bin/python"
RUNS = f"{W}/app_runs"
os.makedirs(RUNS, exist_ok=True)

def to_pil(x):
    if x is None: return None
    if isinstance(x, dict):
        x = x.get("composite") if x.get("composite") is not None else x.get("background")
        if x is None: return None
    if isinstance(x, str): x = Image.open(x)
    if isinstance(x, np.ndarray): x = Image.fromarray(x.astype(np.uint8))
    if x.mode in ("RGBA", "LA", "P"):
        x = x.convert("RGBA"); bg = Image.new("RGBA", x.size, (255, 255, 255, 255))
        bg.alpha_composite(x); x = bg
    return x.convert("RGB")

def clean_prompt(p):
    # the pipeline puts the prompt inside a shell command, so allow only safe characters
    return re.sub(r"[^A-Za-z0-9 ,\-]", "", p or "").strip()[:150]

def run(img, prompt, mode, source):
    pil = to_pil(img)
    if pil is None:
        raise gr.Error("No image. Draw a sketch or upload one first.")
    g = np.array(pil.convert("L"))
    if (g < 128).mean() < 0.002:
        raise gr.Error("The canvas looks empty. Draw something with a dark brush.")
    run_id = uuid.uuid4().hex[:8]
    d = f"{RUNS}/{run_id}"; os.makedirs(d)
    src = f"{d}/in_{run_id}.png"; pil.save(src)
    p = clean_prompt(prompt) or "a single object"
    out = f"{d}/out"
    cmd = (f'cd {W} && {PY} pipeline.py --input {src} --output {out} --mode {mode} '
           f'--prompt "{p}"')
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise gr.Error("Generation timed out after 15 minutes.")
    reps = glob.glob(f"{out}/*_report.json")
    if r.returncode != 0 or not reps:
        raise gr.Error("Pipeline failed: " + (r.stdout + r.stderr)[-600:])
    rep = json.load(open(reps[0]))
    stem = os.path.basename(reps[0]).replace("_report.json", "")
    f = lambda s: (f"{out}/{stem}{s}" if os.path.exists(f"{out}/{stem}{s}") else None)
    status = (f"valid mesh: {rep['valid']} | watertight: {rep['watertight']} | "
              f"{rep['vertices']} vertices, {rep['faces']} faces | "
              f"time: {rep['times']['total_s']} s "
              f"(ControlNet {rep['times']['controlnet_s']} s, InstantMesh {rep['times']['instantmesh_s']} s)")
    return (status, f("_control.png"), f("_generated.png"), f("_views.png"),
            f(".glb"), [x for x in (f(".obj"), f(".glb")) if x], rep)

with gr.Blocks(title="Sketch2Shape AI") as demo:
    gr.Markdown("# Sketch2Shape AI\nDraw or upload a sketch, add an optional prompt, and get a 3D model "
                "(.glb and .obj). One generation takes about 1.5 minutes.")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                canvas = gr.Sketchpad(label="Draw here (black brush)", type="numpy",
                                      brush=gr.Brush(colors=["#000000"], default_color="#000000", default_size=4),
                                      canvas_size=(512, 512))
                b_draw = gr.Button("Generate 3D from drawing", variant="primary")
            with gr.Tab("Upload"):
                up = gr.Image(label="Upload a sketch image", type="filepath", sources=["upload"])
                b_up = gr.Button("Generate 3D from upload", variant="primary")
            prompt = gr.Textbox(label="Optional prompt (type, material, color, style)",
                                placeholder="e.g. a red wooden chair")
            mode = gr.Dropdown(["auto", "sketch", "photo"], value="auto",
                               label="Input type (auto detects line drawing vs photo)")
        with gr.Column():
            status = gr.Textbox(label="Result", interactive=False)
            model = gr.Model3D(label="3D preview (drag to rotate)")
            files = gr.File(label="Download", file_count="multiple")
    with gr.Row():
        o1 = gr.Image(label="Control (edges)"); o2 = gr.Image(label="Generated image"); o3 = gr.Image(label="Six views")
    rep = gr.JSON(label="Report")
    outs = [status, o1, o2, o3, model, files, rep]
    b_draw.click(lambda i, p, m: run(i, p, m, "draw"), [canvas, prompt, mode], outs)
    b_up.click(lambda i, p, m: run(i, p, m, "upload"), [up, prompt, mode], outs)

demo.queue(default_concurrency_limit=1)   # one GPU job at a time
demo.launch(share=True, allowed_paths=[RUNS], server_name="0.0.0.0")
