import torch, numpy, xformers, transformers, diffusers, pytorch_lightning, huggingface_hub, accelerate
import trimesh, rembg, xatlas, mcubes, imageio, omegaconf, einops
import nvdiffrast.torch as dr
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("numpy:", numpy.__version__)
print("xformers:", xformers.__version__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("accelerate:", accelerate.__version__)
print("pytorch_lightning:", pytorch_lightning.__version__)
print("trimesh:", trimesh.__version__)
glctx = dr.RasterizeCudaContext()
pos = torch.tensor([[[-0.8,-0.8,0,1],[0.8,-0.8,0,1],[0,0.8,0,1]]], dtype=torch.float32, device="cuda")
tri = torch.tensor([[0,1,2]], dtype=torch.int32, device="cuda")
rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256,256])
print("nvdiffrast pixels covered:", int((rast[...,3] > 0).sum().item()))
print("all imports OK")
