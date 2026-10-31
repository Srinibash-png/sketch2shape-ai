import sys
import trimesh

path = sys.argv[1]
mesh = trimesh.load(path, force="mesh", process=False)

print("type:", type(mesh).__name__)
print("vertices:", len(mesh.vertices))
print("faces:", len(mesh.faces))
print("extents (x,y,z):", mesh.extents.round(3))
print("watertight:", mesh.is_watertight)
print("visual kind:", mesh.visual.kind)

out = path.replace(".obj", ".glb")
mesh.export(out)
print("exported:", out)
