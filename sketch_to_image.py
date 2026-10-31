import argparse, os, time, gc
import numpy as np, torch
from PIL import Image, ImageOps, ImageFilter
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

BASES = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]
CN = "lllyasviel/sd-controlnet-scribble"
EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def load_gray(path, size=512):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im); im = bg
    g = ImageOps.autocontrast(im.convert("L")); g.thumbnail((size, size))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(g, ((size - g.width) // 2, (size - g.height) // 2))
    return canvas

def is_photo(g):
    a = np.array(g)
    return ((a > 60) & (a < 200)).mean() > 0.25      # many mid-tones => photo, not line art

def prep(path, mode="auto", size=512):
    g = load_gray(path, size); a = np.array(g)
    photo = (mode == "photo") or (mode == "auto" and is_photo(g))
    if photo:
        b = np.array(g.filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)
        gy, gx = np.gradient(b); mag = np.hypot(gx, gy)
        lines = (mag > np.percentile(mag, 92)) * 255
    else:
        lines = (a < 128) * 255
    im = Image.fromarray(lines.astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    return im.convert("RGB"), ("photo/edges" if photo else "sketch/threshold")

def load_pipe():
    cn = ControlNetModel.from_pretrained(CN, torch_dtype=torch.float16)
    last = None
    for base in BASES:
        try:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base, controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
            print("base model:", base); break
        except Exception as e:
            last = e; print("failed base", base, "->", repr(e)[:150])
    else:
        raise last
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to("cuda")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", default=""); ap.add_argument("--name_prompt", action="store_true")
    ap.add_argument("--mode", default="auto", choices=["auto", "sketch", "photo"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    ctrl_dir = a.output.rstrip("/") + "_ctrl"; os.makedirs(ctrl_dir, exist_ok=True)
    pipe = load_pipe()
    neg = ("multiple objects, two objects, text, watermark, cropped, blurry, low quality, people, hands, "
           "wall, floor, room, shadow, backdrop, texture background, clutter")
    for f in sorted(os.listdir(a.input)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in EXT: continue
        subject = a.prompt or (stem.replace("_", " ") if a.name_prompt else "a single object")
        prompt = (f"{subject}, 3D render product photo, single object centered, isolated on a plain white "
                  "background, no shadow, soft even studio lighting, high detail")
        t = time.time()
        ctrl, kind = prep(os.path.join(a.input, f), a.mode)
        gen = torch.Generator("cuda").manual_seed(a.seed)
        out = pipe(prompt, image=ctrl, negative_prompt=neg, num_inference_steps=a.steps,
                   controlnet_conditioning_scale=1.0, generator=gen).images[0]
        out.save(os.path.join(a.output, stem + ".png")); ctrl.save(os.path.join(ctrl_dir, stem + "_control.png"))
        print(f"{f}: {time.time()-t:.1f}s  [{kind}]  prompt='{subject}'")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    print("CN DONE")
