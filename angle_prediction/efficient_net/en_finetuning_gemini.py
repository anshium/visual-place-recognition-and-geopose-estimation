import os
import pandas as pd
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for potential normalization if added later
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ===============================
# Dataset Definition (Outputs cos/sin)
# ===============================

class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform

        valid_rows = []
        print("Verifying image files...")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Checking files"):
            image_path = os.path.join(self.image_dir, row['filename'])
            if os.path.isfile(image_path):
                valid_rows.append(row)
            # else: # Keep console cleaner, only print if debugging needed
                # print(f"Warning: Image file not found: {image_path}")

        self.df = pd.DataFrame(valid_rows)
        if len(self.df) < len(dataframe):
            print(f"Warning: Kept {len(self.df)} out of {len(dataframe)} rows due to missing files.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy item or raise an error, depending on desired behavior
            # For simplicity, returning the first item if error occurs (not ideal)
            if idx == 0: raise e # Raise if first item fails
            return self.__getitem__(0)

        image = self.transform(image)

        # Target angle in degrees
        angle_deg = row['angle']
        # Convert target angle to radians
        angle_rad = np.deg2rad(angle_deg)
        # Target is now a tensor [cos(angle), sin(angle)]
        target = torch.tensor([np.cos(angle_rad), np.sin(angle_rad)], dtype=torch.float32)

        return {"pixel_values": image}, target

# ===============================
# EfficientNet Model for Sin/Cos Regression
# ===============================

class EfficientNetSinCosModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)

        num_features = self.backbone.classifier[1].in_features
        # Modify the classifier head to output 2 values (cos, sin)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Slightly increased dropout
            nn.Linear(num_features, 2) # Output dimension is 2
        )

    def forward(self, pixel_values):
        # Output will be (batch_size, 2)
        preds_cos_sin = self.backbone(pixel_values)
        # Optional: Normalize the output vector to lie on the unit circle.
        # This can sometimes help stability, especially if using MSE loss.
        # preds_cos_sin = F.normalize(preds_cos_sin, p=2, dim=1)
        return preds_cos_sin

# ===============================
# Angular Loss Function (for validation reporting ONLY)
# ===============================

def mean_absolute_angular_error(preds_deg, targets_deg):
    """Calculates the MAAE, handling the circular nature of angles (0-360)."""
    diff = torch.abs(preds_deg - targets_deg)
    # minimum(abs(pred-target), 360 - abs(pred-target))
    return torch.mean(torch.minimum(diff, 360.0 - diff))

# ===============================
# Data Transforms
# ===============================

# Get normalization stats from pretrained weights
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
imagenet_stats = weights.transforms().mean, weights.transforms().std
input_size = 224 # Standard for EfficientNet B0

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.0)), # More robust cropping
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Color augmentation
    # transforms.RandomHorizontalFlip(), # Add ONLY if your angle definition allows/is adjusted for flips
    # transforms.RandomRotation(degrees=15), # Add ONLY if you adjust the target angle accordingly
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
])

val_transform = transforms.Compose([
    transforms.Resize(256), # Standard practice: Resize slightly larger than crop size
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
])


# ===============================
# Load Data
# ===============================
print("Loading data...")
# Adjust paths as needed
TRAIN_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv"
VAL_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
TRAIN_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train"
VAL_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"
SAVE_DIR_BASE = "/home/skills/ansh/delme/angle_prediction/efficientnet/sincos"

train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)

train_dataset = CampusDataset(train_df, TRAIN_IMG_DIR, train_transform)
val_dataset = CampusDataset(val_df, VAL_IMG_DIR, val_transform)

# Consider adding num_workers for faster data loading
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
print("Data loaded.")

# ===============================
# Training Setup
# ===============================
print("Setting up training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EfficientNetSinCosModel().to(device)

# Use standard MSELoss for training sin/cos outputs
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01) # AdamW is good, added small weight decay

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True) # Reduces LR if val MAAE plateaus

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"{SAVE_DIR_BASE}/training_{date_time}"
os.makedirs(save_dir, exist_ok=True) # Use makedirs to create parent dirs if needed
print(f"Checkpoints will be saved in: {save_dir}")

# ===============================
# Training Loop
# ===============================
NUM_EPOCHS = 120
best_val_maae = float('inf') # Track best MAAE for saving best model

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0 # This will be MSE loss
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

    for batch in train_pbar:
        inputs = batch[0] # Dataset returns tuple: (dict, tensor)
        targets_cos_sin = batch[1].to(device) # Targets are [cos, sin]
        pixel_values = inputs["pixel_values"].to(device)

        # Forward pass
        preds_cos_sin = model(pixel_values)

        # Calculate training loss (MSE between predicted and target cos/sin)
        loss = criterion(preds_cos_sin, targets_cos_sin)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_pbar.set_postfix(mse_loss=loss.item()) # Show current batch MSE

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train MSE Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval()
    total_val_maae = 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")

    with torch.no_grad():
        for batch in val_pbar:
            inputs = batch[0]
            targets_cos_sin = batch[1].to(device) # Targets are [cos, sin]
            pixel_values = inputs["pixel_values"].to(device)

            # Get model predictions (cos, sin)
            preds_cos_sin = model(pixel_values) # Shape: [batch, 2]

            # --- Convert predictions and targets back to angles (degrees) for MAAE ---
            # Predictions
            pred_angle_rad = torch.atan2(preds_cos_sin[:, 1], preds_cos_sin[:, 0]) # Output is [-pi, pi]
            pred_angle_deg = torch.rad2deg(pred_angle_rad) # Output is [-180, 180]
            pred_angle_deg = (pred_angle_deg + 360.0) % 360.0 # Map to [0, 360]

            # Targets
            target_angle_rad = torch.atan2(targets_cos_sin[:, 1], targets_cos_sin[:, 0]) # Output is [-pi, pi]
            target_angle_deg = torch.rad2deg(target_angle_rad) # Output is [-180, 180]
            target_angle_deg = (target_angle_deg + 360.0) % 360.0 # Map to [0, 360]

            # Calculate MAAE using the original function
            angular_error = mean_absolute_angular_error(pred_angle_deg, target_angle_deg)
            total_val_maae += angular_error.item()
            val_pbar.set_postfix(maae=angular_error.item()) # Show current batch MAAE

    avg_val_maae = total_val_maae / len(val_loader)
    print(f"Epoch {epoch+1} - Val MAAE: {avg_val_maae:.4f}")

    # --- Checkpointing and Scheduler ---
    # Save checkpoint for the current epoch
    checkpoint_path = f'{save_dir}/checkpoint_{epoch+1:03d}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_maae': avg_val_maae,
        'train_mse_loss': avg_train_loss,
    }, checkpoint_path)
    # print(f"Checkpoint saved to {checkpoint_path}") # Optional: print save path

    # Save the best model based on validation MAAE
    if avg_val_maae < best_val_maae:
        best_val_maae = avg_val_maae
        best_model_path = f'{save_dir}/best_model.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"*** New best model saved with Val MAAE: {best_val_maae:.4f} at {best_model_path} ***")

    # Step the scheduler based on validation MAAE
    scheduler.step(avg_val_maae)

print("Training finished.")
print(f"Best Validation MAAE achieved: {best_val_maae:.4f}")