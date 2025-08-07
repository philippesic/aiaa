import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.amp import autocast, GradScaler

class AntiAliasingDataset(Dataset):
    def __init__(self, alias_dir, ssaa_dir, transform=None):
        self.alias_paths = sorted(glob.glob(os.path.join(alias_dir, "*.png")))
        self.ssaa_paths = sorted(glob.glob(os.path.join(ssaa_dir, "*.png")))
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.alias_paths)

    def __getitem__(self, idx):
        alias = Image.open(self.alias_paths[idx]).convert("RGB")
        ssaa  = Image.open(self.ssaa_paths[idx]).convert("RGB")
        return self.transform(alias), self.transform(ssaa)

class AntiAliasingNetwork(nn.Module):
    def __init__(self):
        super(AntiAliasingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x) + x
def train():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    dataset = AntiAliasingDataset(
        alias_dir="./renders/alias",
        ssaa_dir="./renders/antialias",
        transform=T.Compose([
            T.Resize((1080, 1920)),
            T.ToTensor()
        ])
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

    model = AntiAliasingNetwork().to(device)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    scaler = GradScaler("cuda") 
    EPOCHS = 20
    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0

        for aliased, target in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            aliased, target = aliased.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda"):
                output = model(aliased)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            with torch.no_grad():
                pred_gray   = output.mean(dim=1)
                target_gray = target.mean(dim=1)
                diff = torch.abs(pred_gray - target_gray)

                threshold = 0.05
                tp = ((diff < threshold) & (target_gray > 0.1)).sum().item()
                fp = ((diff < threshold) & (target_gray <= 0.1)).sum().item()
                fn = ((diff >= threshold) & (target_gray > 0.1)).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall    = total_tp / (total_tp + total_fn + 1e-8)

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(loader):.6f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        torch.save(model.state_dict(), f"./checkpoints/model_epoch{epoch+1}.pth")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()