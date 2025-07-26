import os
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Import transforms
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR # Import scheduler

# ---------------- Configuration ----------------
CONFIG = {
    "model_name": "facebook/dinov2-base",
    "train_csv": "/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv",
    "val_csv": "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv",
    "image_dir_train": "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train",
    "image_dir_val": "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val",
    "output_dir": f"./dinov2_angle_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 224, # Ensure consistent image size
    "epochs": 50, # Increase epochs
    "freeze_epochs": 5, # Epochs to train only the head
    "lr_head": 1e-4, # Slightly higher LR for the head
    "lr_backbone": 1e-5, # Lower LR for the backbone fine-tuning
    "weight_decay": 1e-4,
    "dropout_rate": 0.1, # Add dropout
    "patience": 10, # Early stopping patience
    "seed": 42,
}

# For reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Dataset ----------------
class AngleDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, augment=False):
        self.df = dataframe
        self.image_dir = image_dir
        self.processor = processor
        self.augment = augment

        # Define augmentations (applied BEFORE processor's normalization/resizing)
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=45, fill=0), # Crucial for angles! fill=0 for black background
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                # transforms.RandomHorizontalFlip(), # Usually BAD for orientation unless you adjust the angle target!
            ])
        else:
            self.transform = None # No augmentation for validation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy item or handle appropriately
            # For simplicity, let's try loading the first image if current fails
            row = self.df.iloc[0]
            image_path = os.path.join(self.image_dir, row['filename'])
            image = Image.open(image_path).convert("RGB")


        # Apply augmentations to PIL image *before* processor
        if self.transform:
            image = self.transform(image)

        # Use processor for resizing, normalization, and tensor conversion
        # The processor usually returns 'pixel_values'. Check its documentation if needed.
        inputs = self.processor(images=image, return_tensors="pt")
        # Squeeze the batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        angle = float(row['angle']) # Keep in [0, 360)
        target = torch.tensor(angle, dtype=torch.float32)

        return inputs, target

# ---------------- Model ----------------
class DinoV2AngleRegressorSinCos(nn.Module):
    def __init__(self, backbone_name=CONFIG["model_name"], dropout_rate=CONFIG["dropout_rate"]):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        # Output 2 values: sin(angle) and cos(angle)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        # Predict sin and cos components
        sin_cos_pred = self.head(pooled)
        return sin_cos_pred # Shape: (batch_size, 2)

# ---------------- Loss Function (using original MAAE logic) ----------------
def angular_regression_loss(preds_deg, targets_deg):
    """Calculates MAAE loss, handling the circular nature of angles."""
    # Ensure targets are in [0, 360)
    targets_deg = targets_deg % 360
    # Ensure predictions are in [0, 360)
    preds_deg = preds_deg % 360

    diff = torch.abs(preds_deg - targets_deg)
    # Calculate shortest distance on the circle
    loss = torch.mean(torch.minimum(diff, 360.0 - diff))
    return loss

# ---------------- Evaluation Metric (same as loss) ----------------
def mean_absolute_angular_error(preds_deg, targets_deg):
    """Calculates MAAE metric."""
    return angular_regression_loss(preds_deg, targets_deg) # Use the same logic

# ---------------- Helper to convert sin/cos prediction to degrees ----------------
def prediction_to_angle_deg(sin_cos_pred):
    """Converts model output (sin, cos) to angles in degrees [0, 360)."""
    # atan2(y, x) -> atan2(sin, cos)
    angle_rad = torch.atan2(sin_cos_pred[:, 0], sin_cos_pred[:, 1])
    # Convert radians [-pi, pi] to degrees [0, 360)
    angle_deg = torch.rad2deg(angle_rad) % 360.0
    return angle_deg

# ---------------- Load Data ----------------
print("Loading data...")
train_df = pd.read_csv(CONFIG["train_csv"])
val_df = pd.read_csv(CONFIG["val_csv"])

# Initialize processor - ensure image size is consistent
processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"], size=CONFIG["image_size"])

train_dataset = AngleDataset(train_df, CONFIG["image_dir_train"], processor, augment=True)
val_dataset = AngleDataset(val_df, CONFIG["image_dir_val"], processor, augment=False) # No augmentation for validation

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
print("Data loaded.")

# ---------------- Training Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DinoV2AngleRegressorSinCos().to(device)

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"Output directory: {CONFIG['output_dir']}")

# Setup optimizer with differential learning rates
def get_optimizer(model, lr_head, lr_backbone, weight_decay):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        # Head parameters with head_lr
        {'params': [p for n, p in param_optimizer if "backbone" not in n and p.requires_grad],
         'lr': lr_head, 'weight_decay': weight_decay},
        # Backbone parameters with backbone_lr, applying weight decay unless specified
        {'params': [p for n, p in param_optimizer if "backbone" in n and p.requires_grad and not any(nd in n for nd in no_decay)],
         'lr': lr_backbone, 'weight_decay': weight_decay},
        # Backbone parameters with backbone_lr, without weight decay
        {'params': [p for n, p in param_optimizer if "backbone" in n and p.requires_grad and any(nd in n for nd in no_decay)],
         'lr': lr_backbone, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)

# Initially freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = get_optimizer(model, CONFIG["lr_head"], CONFIG["lr_backbone"], CONFIG["weight_decay"])
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] - CONFIG["freeze_epochs"], eta_min=1e-7) # Scheduler for the fine-tuning phase

# ---------------- Training & Validation Loop ----------------
best_val_maae = float('inf')
epochs_no_improve = 0
best_model_state = None

print("Starting training...")
for epoch in range(CONFIG["epochs"]):
    print("-" * 30)
    print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")

    # --- Freeze/Unfreeze Logic ---
    if epoch == CONFIG["freeze_epochs"]:
        print("Unfreezing backbone and adjusting optimizer...")
        for param in model.backbone.parameters():
            param.requires_grad = True
        # Re-initialize optimizer with all parameters trainable and differential LRs
        optimizer = get_optimizer(model, CONFIG["lr_head"], CONFIG["lr_backbone"], CONFIG["weight_decay"])
        # Re-initialize scheduler for the fine-tuning phase
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"] - CONFIG["freeze_epochs"], eta_min=CONFIG["lr_backbone"] * 0.01) # Use backbone LR as reference
        print("Backbone unfrozen. Training full model.")
    elif epoch < CONFIG["freeze_epochs"]:
        print("Training head only.")
        # Ensure only head LR is active (AdamW handles this via parameter groups)
        pass # Optimizer already configured correctly
    else:
         # Adjust LR only during the fine-tuning phase
         scheduler.step()
         current_lr_head = optimizer.param_groups[0]['lr'] # Head group
         current_lr_backbone = optimizer.param_groups[1]['lr'] # Backbone group (with decay)
         print(f"Fine-tuning. Head LR: {current_lr_head:.2e}, Backbone LR: {current_lr_backbone:.2e}")


    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    train_preds_all = []
    train_targets_all = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    for batch in pbar:
        inputs, targets = batch
        pixel_values = inputs["pixel_values"].to(device)
        targets_deg = targets.to(device) # Targets are in degrees [0, 360)

        optimizer.zero_grad()

        # Forward pass
        sin_cos_preds = model(pixel_values) # Model outputs (sin, cos)

        # Convert predictions to angles [0, 360) for loss calculation
        preds_deg = prediction_to_angle_deg(sin_cos_preds)

        # Calculate loss
        loss = angular_regression_loss(preds_deg, targets_deg)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        # Store for epoch-level MAAE calculation
        train_preds_all.append(preds_deg.detach())
        train_targets_all.append(targets_deg.detach())

        pbar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader) # This is MAAE on train batches
    train_preds_all = torch.cat(train_preds_all)
    train_targets_all = torch.cat(train_targets_all)
    epoch_train_maae = mean_absolute_angular_error(train_preds_all, train_targets_all).item()
    print(f"Epoch {epoch+1} - Train Avg Batch Loss (MAAE): {avg_train_loss:.2f}°, Train Epoch MAAE: {epoch_train_maae:.2f}°")


    # --- Validation Phase ---
    model.eval()
    val_preds_all = []
    val_targets_all = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            inputs, targets = batch
            pixel_values = inputs["pixel_values"].to(device)
            targets_deg = targets.to(device)

            sin_cos_preds = model(pixel_values)
            preds_deg = prediction_to_angle_deg(sin_cos_preds)

            val_preds_all.append(preds_deg)
            val_targets_all.append(targets_deg)

    val_preds_all = torch.cat(val_preds_all)
    val_targets_all = torch.cat(val_targets_all)
    epoch_val_maae = mean_absolute_angular_error(val_preds_all, val_targets_all).item()
    print(f"Epoch {epoch+1} - Val MAAE: {epoch_val_maae:.2f}°")

    # --- Checkpointing and Early Stopping ---
    if epoch_val_maae < best_val_maae:
        best_val_maae = epoch_val_maae
        epochs_no_improve = 0
        # Save the best model state
        best_model_state = copy.deepcopy(model.state_dict())
        model_save_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
        torch.save(best_model_state, model_save_path)
        print(f"Validation MAAE improved to {best_val_maae:.2f}°. Saving best model to {model_save_path}")
    else:
        epochs_no_improve += 1
        print(f"Validation MAAE did not improve for {epochs_no_improve} epoch(s). Best MAAE: {best_val_maae:.2f}°")

    if epochs_no_improve >= CONFIG["patience"]:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Save the last model state as well
last_model_save_path = os.path.join(CONFIG["output_dir"], "last_model.pth")
torch.save(model.state_dict(), last_model_save_path)
print(f"Training finished. Last model saved to {last_model_save_path}")
print(f"Best validation MAAE achieved: {best_val_maae:.2f}°")