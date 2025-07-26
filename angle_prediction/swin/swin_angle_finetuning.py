import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
import torch

import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import os

from sklearn.preprocessing import StandardScaler
import numpy as np


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

class SwinRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin_base_patch4_window12_384")
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 1)  # Predict a single angle

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output).squeeze(-1)  # Shape: (batch_size,)

def mean_absolute_angular_error(preds, targets):
    diff = torch.abs(preds - targets)
    return torch.mean(torch.minimum(diff, 360 - diff))

train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin_base_patch4_window12_384", use_fast=True)

train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", image_processor)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", image_processor)

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=48)

model = SwinRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

from datetime import datetime

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')

save_dir = f"/home/skills/ansh/delme/angle_prediction/swin/training_{date_time}"

os.mkdir(save_dir)

for epoch in range(50):
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

    # Save model checkpoint
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
