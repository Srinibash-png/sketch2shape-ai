import argparse, os, shutil, subprocess, time, json, glob, tempfile

PY = "/kaggle/working/im_env/bin/python"
W = "/kaggle/working"

def run(cmd, log):
    t = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--mode", default="auto", choices=["auto", "sketch", "photo"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.input))[0]
    work = tempfile.mkdtemp(prefix="p_", dir=W)
    inp, gen = f"{work}/in", f"{work}/gen"
    os.makedirs(inp); shutil.copy(a.input, inp)
    times, env = {}, "export CUDA_VISIBLE_DEVICES=0 &&"

    # Stage 1: sketch -> image (ControlNet)
    p = f'--prompt "{a.prompt}"' if a.prompt else "--name_prompt"
    rc, times["controlnet_s"] = run(f"cd {W} && {env} {PY} -W ignore sketch_to_image.py --input {inp} --output {gen} {p} --mode {a.mode} --seed {a.seed}", f"{work}/s1.log")
    if rc or not os.path.exists(f"{gen}/{stem}.png"):
        raise SystemExit(f"ControlNet stage failed, see {work}/s1.log")

    # Stage 2: image -> 3D (InstantMesh, rembg on)
    rc, times["instantmesh_s"] = run(f"cd {W}/InstantMesh && {env} {PY} -W ignore run.py configs/instant-mesh-base.yaml {gen} --output_path {work}/im", f"{work}/s2.log")
    obj = f"{work}/im/instant-mesh-base/meshes/{stem}.obj"
    if rc or not os.path.exists(obj):
        raise SystemExit(f"InstantMesh stage failed, see {work}/s2.log")

    # Stage 3: validate + export glb
    import trimesh
    m = trimesh.load(obj, force="mesh", process=False)
    ok = len(m.vertices) > 0 and len(m.faces) > 0 and min(m.extents) > 1e-3
    glb = f"{work}/im/instant-mesh-base/meshes/{stem}.glb"; m.export(glb)

    for src, dst in [(obj, f"{a.output}/{stem}.obj"), (glb, f"{a.output}/{stem}.glb"),
                     (f"{gen}/{stem}.png", f"{a.output}/{stem}_generated.png"),
                     (f"{gen}_ctrl/{stem}_control.png", f"{a.output}/{stem}_control.png")]:
        if os.path.exists(src): shutil.copy(src, dst)
    views = f"{work}/im/instant-mesh-base/images/{stem}.png"
    if os.path.exists(views): shutil.copy(views, f"{a.output}/{stem}_views.png")

    times["total_s"] = sum(times.values())
    rep = {"input": a.input, "prompt": a.prompt, "vertices": len(m.vertices), "faces": len(m.faces),
           "extents": [round(float(x), 3) for x in m.extents], "watertight": bool(m.is_watertight),
           "valid": bool(ok), "times": {k: round(v, 1) for k, v in times.items()}}
    json.dump(rep, open(f"{a.output}/{stem}_report.json", "w"), indent=2)
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(rep, indent=2))

main()
