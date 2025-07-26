import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

# ---------------- Dataset ----------------
class AngleDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor):
        self.df = dataframe
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        angle = float(row['angle']) % 360  # Wrap into [0, 360)
        target = torch.tensor(angle, dtype=torch.float32)
        return inputs, target

# ---------------- Model ----------------
class DinoV2AngleRegressor(nn.Module):
    def __init__(self, backbone_name="facebook/dinov2-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        angle = self.head(pooled).squeeze(-1)
        return angle % 360  # output angle in [0, 360)

# ---------------- Loss Function ----------------
def angular_regression_loss(preds, targets):
    diff = torch.abs(preds - targets) % 360
    return torch.mean(torch.minimum(diff, 360 - diff))

# ---------------- Evaluation ----------------
def mean_absolute_angular_error(preds, targets):
    diff = torch.abs(preds - targets) % 360
    return torch.mean(torch.minimum(diff, 360 - diff))

# ---------------- Load Data ----------------
train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

image_dir_train = "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train"
image_dir_val = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

train_dataset = AngleDataset(train_df, image_dir_train, processor)
val_dataset = AngleDataset(val_df, image_dir_val, processor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ---------------- Training Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DinoV2AngleRegressor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

save_dir = f"./dinov2_angle_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

# ---------------- Training Loop ----------------
for epoch in range(30):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = batch
        pixel_values = inputs["pixel_values"].to(device)
        targets = targets.to(device)

        preds = model(pixel_values)
        loss = angular_regression_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train MAAE: {avg_train_loss:.2f}°")

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            pixel_values = inputs["pixel_values"].to(device)
            targets = targets.to(device)

            preds = model(pixel_values)
            val_loss += mean_absolute_angular_error(preds, targets).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} - Val MAAE: {avg_val_loss:.2f}°")

    # Save checkpoint
    torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pth")
