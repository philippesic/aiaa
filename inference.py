import torch
from model import AntiAliasingNetwork
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import time
import json
from torch.amp import autocast

def Infer(width, height):
    TOTAL_FRAMETIME = 0

    config_path = f"./checkpoints/{height}/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AntiAliasingNetwork(config["layers"], config["channels"]).to(device, memory_format=torch.channels_last)
    model.eval()

    model = torch.compile(model)
    model.load_state_dict(torch.load(config["model"], map_location=device))

    print(f"Using model from: " + config["model"])

    os.makedirs(f"./output/{height}/result", exist_ok=True)

    img = Image.open(f"./output/{height}/test/test.png").convert("RGB")
    transform = T.Compose([
        T.Resize((int(height), int(width))),
        T.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device, non_blocking=True, memory_format=torch.channels_last)

    for i in range(1000):
        with torch.inference_mode():
            start_time = time.perf_counter()

            with autocast("cuda"):
                output = model(input_tensor).clamp(0, 1)

            torch.cuda.synchronize()
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000

        print(f"Pass: {i+1}")
        if (i==0):
            print(" (Compilation Pass)")
        else:
            TOTAL_FRAMETIME += elapsed_time_ms
        print(f"Inference Time: {elapsed_time_ms:.4f} ms")
        print(f"Maximum Possible Framerate: {(1000 / elapsed_time_ms):.2f} fps\n")
        

    print(f"Average Framerate: {(1000/(TOTAL_FRAMETIME/(i))):.2f} fps\n")

    out_img = TF.to_pil_image(output.squeeze(0).float().cpu())
    out_img.save(f"./output/{height}/result/result.png")
    print("Saved: result.png\n")

if __name__ == "__main__":
    import sys
    Infer(sys.argv[1], sys.argv[2])
