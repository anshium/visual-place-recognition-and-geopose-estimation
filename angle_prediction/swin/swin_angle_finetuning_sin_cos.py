import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, SwinModel
from tqdm import tqdm

# --------------------- Dataset ---------------------
class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor

        valid_rows = []
        for _, row in dataframe.iterrows():
            angle = row['angle']
            if pd.isna(angle) or not np.isfinite(angle):
                continue  # skip invalid entries

            image_path = os.path.join(self.image_dir, row['filename'])
            if os.path.isfile(image_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Image file not found: {image_path}")
        
        self.df = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        angle_deg = float(row['angle']) % 360  # ensure in [0, 360)
        angle_rad = np.deg2rad(angle_deg)
        target = torch.tensor([np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32)

        return inputs, target

# --------------------- Model ---------------------
class SwinSinCosRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)  # Predict [sin, cos]

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        out = self.regressor(pooled_output)
        return F.normalize(out, dim=1, p=2, eps=1e-6)  # prevent divide-by-zero

# --------------------- Loss ---------------------
def angular_loss(preds, targets):
    cosine_sim = (preds * targets).sum(dim=1)
    cosine_sim = torch.clamp(cosine_sim, -0.999999, 0.999999)  # avoid NaNs
    angle_diff = torch.acos(cosine_sim)
    return torch.mean(torch.rad2deg(angle_diff))

# --------------------- Evaluation Metric ---------------------
def compute_angle_error(preds, targets):
    pred_angles = torch.atan2(preds[:, 0], preds[:, 1])  # sin, cos
    true_angles = torch.atan2(targets[:, 0], targets[:, 1])
    diff = torch.rad2deg(torch.abs(pred_angles - true_angles)) % 360
    return torch.mean(torch.minimum(diff, 360 - diff))

# --------------------- Load Data ---------------------
train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True)

train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", image_processor)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", image_processor)

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=48)

# --------------------- Training ---------------------
model = SwinSinCosRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"/home/skills/ansh/delme/angle_prediction/swin/training_{date_time}"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(50):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(train_loader):
        pixel_values = inputs["pixel_values"].cuda()
        targets = targets.cuda()

        preds = model(pixel_values)
        loss = angular_loss(preds, targets)

        if torch.isnan(loss):
            print("NaN loss detected! Skipping this batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1} - Train Loss (MAAE°): {total_loss / len(train_loader):.2f}")

    # Save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'{save_dir}/checkpoint_{epoch}.pth')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            pixel_values = inputs["pixel_values"].cuda()
            targets = targets.cuda()

            preds = model(pixel_values)
            error = compute_angle_error(preds, targets)

            if torch.isnan(error):
                print("NaN in validation! Skipping.")
                continue

            val_loss += error.item()

    print(f"Epoch {epoch+1} - Val MAAE: {val_loss / len(val_loader):.2f}°")
