import os, sys
from PIL import Image

def Downscale(BASE_W, BASE_H, SS_RES):

    num_files = len([f for f in os.listdir(f"./renders/{BASE_H}/ssaa{SS_RES}x/") if os.path.isfile(os.path.join(f"./renders/{BASE_H}/ssaa{SS_RES}x/", f))])

    for i in range(num_files):
        ssaa_path = f"./renders/{BASE_H}/ssaa{SS_RES}x/demo_scene_frame{i:04d}_ssaa{SS_RES}x.png"
        save_path = f"./renders/{BASE_H}/antialias/demo_scene_frame{i:04d}_antialias.png"
        if not os.path.exists(ssaa_path):
            continue
        img = Image.open(ssaa_path)
        img_resized = img.resize((int(BASE_W), int(BASE_H)), Image.LANCZOS)
        os.makedirs(f"./renders/{BASE_H}/antialias/", exist_ok=True)
        img_resized.save(save_path)
        print("Resized Image " + str(i))

if __name__ == "__main__":
    import sys
    Downscale(sys.argv[1], sys.argv[2], sys.argv[3])