import os
from PIL import Image

for i in range(512):
    ssaa_path = f"./renders/ssaa4x/demo_scene_frame{i:04d}_ssaa4x.png"
    save_path = f"./renders/antialias/demo_scene_frame{i:04d}_antialias.png"
    if not os.path.exists(ssaa_path):
        continue
    img = Image.open(ssaa_path)
    img_resized = img.resize((1920, 1080), Image.LANCZOS)
    img_resized.save(save_path)
    print("Resized Image " + str(i))