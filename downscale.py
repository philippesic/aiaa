import os
from PIL import Image

for i in range(1024):
    ssaa_path = f"./renders/540/ssaa16x/demo_scene_frame{i:04d}_ssaa16x.png"
    save_path = f"./renders/540/antialias/demo_scene_frame{i:04d}_antialias.png"
    if not os.path.exists(ssaa_path):
        continue
    img = Image.open(ssaa_path)
    img_resized = img.resize((960, 540), Image.LANCZOS)
    img_resized.save(save_path)
    print("Resized Image " + str(i))