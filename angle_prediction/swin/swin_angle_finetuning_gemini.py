import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
from transformers import AutoImageProcessor, SwinModel
from datetime import datetime
import numpy as np
import math # Import math for sin/cos and atan2
import torchvision.transforms as transforms # Import torchvision transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- Configuration ---
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
IMAGE_DIR_TRAIN = "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train"
IMAGE_DIR_VAL = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"
TRAIN_CSV = "/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv"
VAL_CSV = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
BATCH_SIZE = 32 # Keep batch size, experiment if needed
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100 # Increased epochs
SAVE_DIR_BASE = "/home/skills/ansh/delme/angle_prediction/swin/training"
GRAD_CLIP_NORM = 1.0 # Gradient clipping norm
DROPOUT_RATE = 0.3 # Dropout rate

# --- Dataset Class with Augmentation and Sin/Cos Target ---
class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform # torchvision transforms
        self.df = self._filter_valid_files(dataframe, image_dir)

    def _filter_valid_files(self, dataframe, image_dir):
        valid_rows = []
        for _, row in dataframe.iterrows():
            image_path = os.path.join(image_dir, row['filename'])
            if os.path.isfile(image_path):
                valid_rows.append(row)
            else:
                # print(f"Warning: Image file not found: {image_path}") # Keep this check or remove for cleaner output
                pass # Suppress warnings during large dataset loading
        return pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")
        original_angle_deg = row['angle']

        # Apply torchvision transforms
        if self.transform:
            # Need to check for horizontal flip and adjust angle
            for t in self.transform.transforms:
                 # Apply transform to image
                 image = t(image)
                 # If horizontal flip, adjust angle. Note: This is an approximation assuming
                 # horizontal flip maps angle theta to (180 - theta) or similar depending
                 # on convention. Adjust if your angle definition is different.
                 if isinstance(t, transforms.RandomHorizontalFlip):
                     if torch.rand(1) < t.p: # Check if flip happens based on probability
                         original_angle_deg = (180 - original_angle_deg) % 360
                 # For other transforms like RandomRotation, you'd need to adjust the target
                 # angle accordingly if you want the model to predict the *original* orientation's
                 # angle from the rotated image. If you just want the model to be robust to
                 # rotations without changing the target, then no angle adjustment is needed here.
                 # For simplicity, only adjusting for Horizontal Flip for now.
                 # RandomResizedCrop might also slightly affect perspective and effective angle,
                 # but adjusting target for it is complex.

        # Process image with Swin processor
        # Swin processor expects a list of images, even for a single image
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()} # Remove batch dim added by processor

        # Convert angle to radians and then to sin/cos
        angle_rad = math.radians(original_angle_deg)
        target_sin = math.sin(angle_rad)
        target_cos = math.cos(angle_rad)
        target = torch.tensor([target_sin, target_cos], dtype=torch.float32)

        return inputs, target

# --- Model Class Modified for Sin/Cos Output ---
class SwinRegressionModel(nn.Module):
    def __init__(self, backbone_model=MODEL_NAME, dropout_rate=0.0):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(backbone_model)
        # Disable gradient calculation for backbone initially (optional, can fine-tune all later)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), # Add an intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Add Dropout
            nn.Linear(hidden_size // 2, 2) # Predict 2 values (sin, cos)
        )

        # Re-enable gradients for the backbone after initial setup if fine-tuning all
        for param in self.backbone.parameters():
             param.requires_grad = True


    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # Use the pooled output for classification/regression tasks
        # Note: SwinModel's pooled_output might be from the last hidden state through a pooling layer
        # Check model documentation if needing a different feature representation
        pooled_output = outputs.pooler_output
        if pooled_output is None:
             # Fallback if pooler_output is not available for this model variant/version
             # Use mean pooling over spatial dimensions of the last hidden state
             last_hidden_state = outputs.last_hidden_state
             # Assuming shape is (batch_size, sequence_length, hidden_size)
             # We need to average over sequence_length which includes patch tokens + CLS token if any
             # For Swin, the first token is often the CLS token, but pooling over all spatial tokens is common
             pooled_output = last_hidden_state.mean(dim=1) # Simple mean pooling

        return self.regressor(pooled_output) # Output shape: (batch_size, 2)

# --- Evaluation Metric: Mean Absolute Angular Error (MAAE) ---
def mean_absolute_angular_error(preds_sin_cos, targets_sin_cos):
    # Convert predicted sin/cos to angles in degrees [0, 360)
    # atan2 returns angle in radians [-pi, pi]
    preds_rad = torch.atan2(preds_sin_cos[:, 0], preds_sin_cos[:, 1])
    preds_deg = torch.rad2deg(preds_rad)
    preds_deg = (preds_deg + 360) % 360 # Map to [0, 360)

    # Convert target sin/cos to angles in degrees [0, 360)
    targets_rad = torch.atan2(targets_sin_cos[:, 0], targets_sin_cos[:, 1])
    targets_deg = torch.rad2deg(targets_rad)
    targets_deg = (targets_deg + 360) % 360 # Map to [0, 360)

    # Calculate angular difference
    diff = torch.abs(preds_deg - targets_deg)
    # The difference can be clockwise or counter-clockwise, take the minimum
    return torch.mean(torch.minimum(diff, 360 - diff))

# --- Main Training and Validation Logic ---

print("Loading data...")
# MOVE THIS LINE UP: Initialize the processor BEFORE defining transforms
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)


# Data Augmentation Transforms for Training
train_transforms = transforms.Compose([
    # NOW processor is defined and processor.size["height"] is available
    transforms.RandomResizedCrop(processor.size["height"], scale=(0.8, 1.0)), # Example resizing
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Example color jitter
    transforms.RandomHorizontalFlip(p=0.5), # Horizontal flip (angle adjusted in dataset)
    # Add more transforms here, e.g., transforms.RandomRotation (requires careful angle adjustment or no adjustment)
])

# No augmentation for validation
val_transforms = None # Or transforms.Compose([...]), but usually no random transforms


train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_dataset = CampusDataset(train_df, IMAGE_DIR_TRAIN, processor, transform=train_transforms)
val_dataset = CampusDataset(val_df, IMAGE_DIR_VAL, processor, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # Added num_workers
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4) # Added num_workers
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

print("Initializing model...")
model = SwinRegressionModel(dropout_rate=DROPOUT_RATE).cuda()

# Use MSE loss for sin/cos outputs
criterion = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning Rate Scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7) # Example scheduler

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"{SAVE_DIR_BASE}_{date_time}"
os.makedirs(save_dir, exist_ok=True) # Use makedirs for potential parent dirs

print(f"Starting training. Saving checkpoints to {save_dir}")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    maae_train = 0
    tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)

    for inputs, targets_sin_cos in tqdm_loader:
        pixel_values = inputs["pixel_values"].cuda()
        targets_sin_cos = targets_sin_cos.cuda() # Target is now sin/cos pair

        preds_sin_cos = model(pixel_values) # Model outputs sin/cos pair

        # Calculate loss using MSE on sin/cos values
        loss = criterion(preds_sin_cos, targets_sin_cos)

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        total_loss += loss.item()

        # Calculate MAAE for monitoring (convert sin/cos to angle)
        with torch.no_grad():
             batch_maae = mean_absolute_angular_error(preds_sin_cos.detach().cpu(), targets_sin_cos.cpu())
             maae_train += batch_maae.item()

        tqdm_loader.set_postfix(loss=loss.item(), batch_maae=batch_maae.item())

    # Step the scheduler after the epoch
    scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_maae = maae_train / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train MSE Loss: {avg_train_loss:.6f} - Train MAAE: {avg_train_maae:.4f}")

    # Save model checkpoint
    checkpoint_path = f'{save_dir}/checkpoint_epoch_{epoch+1}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_train_loss,
        'maae': avg_train_maae,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


    # Validation
    model.eval()
    val_loss = 0
    maae_val = 0
    tqdm_loader_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for inputs, targets_sin_cos in tqdm_loader_val:
            pixel_values = inputs["pixel_values"].cuda()
            targets_sin_cos = targets_sin_cos.cuda()

            preds_sin_cos = model(pixel_values)

            # Calculate loss using MSE on sin/cos values
            loss = criterion(preds_sin_cos, targets_sin_cos)
            val_loss += loss.item()

            # Calculate MAAE (convert sin/cos to angle)
            batch_maae = mean_absolute_angular_error(preds_sin_cos.cpu(), targets_sin_cos.cpu())
            maae_val += batch_maae.item()

            tqdm_loader_val.set_postfix(loss=loss.item(), batch_maae=batch_maae.item())


    avg_val_loss = val_loss / len(val_loader)
    avg_val_maae = maae_val / len(val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Val MSE Loss: {avg_val_loss:.6f} - Val MAAE: {avg_val_maae:.4f}")

print("Training finished.")