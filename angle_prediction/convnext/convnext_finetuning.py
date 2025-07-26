import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from transformers import AutoImageProcessor, ConvNextModel

# ===============================
# Dataset Definition
# ===============================

class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform

        valid_rows = []
        for _, row in dataframe.iterrows():
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

        target = torch.tensor(row['angle'], dtype=torch.float32)

        return inputs, target

# ===============================
# ConvNeXt Model for Regression
# ===============================

class ConvNextRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNextModel.from_pretrained("facebook/convnext-base-224")
        self.regressor = nn.Linear(self.backbone.config.hidden_sizes[-1], 1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        return self.regressor(pooled_output).squeeze(-1)

# ===============================
# Loss Function
# ===============================

def mean_absolute_angular_error(preds, targets):
    diff = torch.abs(preds - targets)
    return torch.mean(torch.minimum(diff, 360 - diff))

# ===============================
# Load Data
# ===============================

train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224", use_fast=True)

train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", image_processor)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", image_processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ===============================
# Training Setup
# ===============================

model = ConvNextRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"/home/skills/ansh/delme/angle_prediction/convnext/training_{date_time}"
os.mkdir(save_dir)

# ===============================
# Training Loop
# ===============================

for epoch in range(100):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader):
        pixel_values = inputs["pixel_values"].cuda()
        targets = targets.cuda()

        preds = model(pixel_values)
        preds = preds % 360
        loss = mean_absolute_angular_error(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Train MAAE: {total_loss / len(train_loader):.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'{save_dir}/checkpoint_{epoch}_.pth')

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            pixel_values = inputs["pixel_values"].cuda()
            targets = targets.cuda()

            preds = model(pixel_values)
            preds = preds % 360
            loss = mean_absolute_angular_error(preds, targets)
            val_loss += loss.item()

    print(f"Epoch {epoch+1} - Val MAAE: {val_loss / len(val_loader):.4f}")
