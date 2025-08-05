import torch
from model import AntiAliasingNetwork
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AntiAliasingNetwork().to(device)
    model.load_state_dict(torch.load("./checkpoints/model_epoch19.pth", map_location=device))
    model.eval()

    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    ensure_dir("./output/test/")
    ensure_dir("./output/result/")

    img = Image.open("./output/test/test.png").convert("RGB")
    transform = T.Compose([
        T.Resize((1080, 1920)),
        T.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()

        output = model(input_tensor).clamp(0, 1)

        torch.cuda.synchronize()
        elapsed_time_ms = (time.time() - start_time) * 1000

    print(f"Inference Time: {elapsed_time_ms:.2f} ms")
    print(f"Maximum Possible Framerate: {(1000/elapsed_time_ms):.1f} fps")

    out_img = TF.to_pil_image(output.squeeze().cpu())
    out_img.save("./output/result/result.png")
    print("Saved: result.png")

if __name__ == "__main__":
    main()