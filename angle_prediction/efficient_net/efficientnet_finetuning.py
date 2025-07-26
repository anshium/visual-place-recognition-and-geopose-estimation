import os
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ===============================
# Dataset Definition
# ===============================

class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.image_dir = image_dir
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

        image = self.transform(image)

        target = torch.tensor(row['angle'], dtype=torch.float32)

        return {"pixel_values": image}, target

# ===============================
# EfficientNet Model for Regression
# ===============================

class EfficientNetRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.backbone.classifier[1].in_features, 1)
        )

    def forward(self, pixel_values):
        return self.backbone(pixel_values).squeeze(-1)

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

# Use pretrained weights and associated transforms
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
transform = weights.transforms()

train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", transform)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ===============================
# Training Setup
# ===============================

model = EfficientNetRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"/home/skills/ansh/delme/angle_prediction/efficient_net/training_{date_time}"
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

# import os
# import pandas as pd
# from PIL import Image
# from datetime import datetime
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# # ===============================
# # Dataset Definition
# # ===============================

# class CampusDataset(Dataset):
#     def __init__(self, dataframe, image_dir, transform):
#         self.image_dir = image_dir
#         self.transform = transform

#         valid_rows = []
#         for _, row in dataframe.iterrows():
#             image_path = os.path.join(self.image_dir, row['filename'])
#             if os.path.isfile(image_path):
#                 valid_rows.append(row)
#             else:
#                 print(f"Warning: Image file not found: {image_path}")
        
#         self.df = pd.DataFrame(valid_rows)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image_path = os.path.join(self.image_dir, row['filename'])
#         image = Image.open(image_path).convert("RGB")

#         image = self.transform(image)

#         target = torch.tensor(row['angle'], dtype=torch.float32)

#         return {"pixel_values": image}, target

# # ===============================
# # EfficientNet Model for Regression
# # ===============================

# class EfficientNetRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
#         self.backbone.classifier = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),
#             nn.Linear(self.backbone.classifier[1].in_features, 1)
#         )

#     def forward(self, pixel_values):
#         return self.backbone(pixel_values).squeeze(-1)

# # ===============================
# # Loss Function
# # ===============================

# def mean_absolute_angular_error(preds, targets):
#     diff = torch.abs(preds - targets)
#     return torch.mean(torch.minimum(diff, 360 - diff))

# # ===============================
# # Load Data
# # ===============================

# train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
# val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

# # Use pretrained weights and associated transforms
# weights = EfficientNet_B0_Weights.IMAGENET1K_V1
# transform = weights.transforms()

# train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", transform)
# val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)

# # ===============================
# # Training Setup
# # ===============================

# model = EfficientNetRegressionModel().cuda()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 * 4)

# # Scheduler: Reduce LR on Plateau (based on validation loss)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
# )

# date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# save_dir = f"/home/skills/ansh/delme/angle_prediction/efficient_net/training_{date_time}"
# os.mkdir(save_dir)

# # ===============================
# # Training Loop
# # ===============================

# for epoch in range(100):
#     model.train()
#     total_loss = 0
#     for inputs, targets in tqdm(train_loader):
#         pixel_values = inputs["pixel_values"].cuda()
#         targets = targets.cuda()

#         preds = model(pixel_values)
#         preds = preds % 360
#         loss = mean_absolute_angular_error(preds, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     train_maae = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1} - Train MAAE: {train_maae:.4f}")

#     # Save checkpoint
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }, f'{save_dir}/checkpoint_{epoch}_.pth')

#     # Validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             pixel_values = inputs["pixel_values"].cuda()
#             targets = targets.cuda()

#             preds = model(pixel_values)
#             preds = preds % 360
#             loss = mean_absolute_angular_error(preds, targets)
#             val_loss += loss.item()

#     val_maae = val_loss / len(val_loader)
#     print(f"Epoch {epoch+1} - Val MAAE: {val_maae:.4f}")

#     # Step the LR scheduler based on validation loss
#     scheduler.step(val_maae)

#     # Print current LR
#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Current Learning Rate: {current_lr:.6f}")
