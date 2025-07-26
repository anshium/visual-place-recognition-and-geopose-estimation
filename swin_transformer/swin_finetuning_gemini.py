import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
from datetime import datetime
import joblib
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

# --- Dataset Class ---
class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform

        # Filter out rows where the image file does not exist
        valid_rows = []
        for _, row in dataframe.iterrows():
            image_path = os.path.join(self.image_dir, row['filename'])
            if os.path.isfile(image_path):
                valid_rows.append(row)
            else:
                # print(f"Warning: Image file not found: {image_path}") # Keep this commented unless debugging
                pass # Silently skip missing files to avoid excessive output

        self.df = pd.DataFrame(valid_rows)
        print(f"Loaded {len(self.df)} valid samples from {len(dataframe)} in original dataframe.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Use the processor to get model inputs
        # The processor expects a list of images, even for a single image
        inputs = self.processor(images=[image], return_tensors="pt")
        # Squeeze the batch dimension added by the processor for a single image
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        target = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)

        return inputs, target

# --- Model Definition ---
class SwinRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained Swin model
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        # Add a linear layer for regression on top of the pooled output
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)  # Predict Latitude & Longitude

    def forward(self, pixel_values):
        # Pass pixel values through the backbone
        outputs = self.backbone(pixel_values=pixel_values)
        # Get the pooled output (global features)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_dim)
        # Pass pooled output through the regressor
        return self.regressor(pooled_output)

# --- Data Loading and Preprocessing ---
train_df = pd.read_csv("/home/skills/ansh/delme/dataset/iiit_dataset/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/dataset/iiit_dataset/labels_val.csv")

# Initialize StandardScaler
scaler = StandardScaler()
# Fit on training data and transform both train and validation data
train_df[['latitude', 'longitude']] = scaler.fit_transform(train_df[['latitude', 'longitude']])
val_df[['latitude', 'longitude']] = scaler.transform(val_df[['latitude', 'longitude']])

# Save the scaler for later use (e.g., inference)
joblib.dump(scaler, "latlon_scaler.pkl")
print("Scaler saved to latlon_scaler.pkl")


# Load the image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True)

# Define data augmentation transforms for training
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_processor.size["height"]), # Random crop and resize
    transforms.RandomHorizontalFlip(), # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Random color jitter
    # Note: Swin processor handles normalization, so we don't add ToTensor or Normalize here
])

# No augmentation for validation
val_transforms = None # Or transforms.Compose([...]) if you want basic resizing/cropping without randomness

# Create datasets
train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", image_processor, transform=train_transforms)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", image_processor, transform=val_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4) # Added num_workers
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4) # Added num_workers

# --- Model, Optimizer, Loss ---
model = SwinRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# Learning Rate Scheduler: Reduce learning rate when validation MSE stops improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# --- Training Setup ---
date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"/home/skills/ansh/delme/swin_transformer/training_{date_time}"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving checkpoints to {save_dir}")

best_val_mse = float('inf') # Initialize best validation MSE for early stopping
epochs_no_improve = 0 # Counter for epochs without improvement
early_stop_patience = 10 # Number of epochs to wait before early stopping

# --- Training Loop ---
print("Starting training...")
for epoch in range(50): # Train for up to 50 epochs
    model.train()
    total_train_loss = 0

    # Training phase
    print(f"Epoch {epoch+1}/{50} - Training...")
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        pixel_values = inputs["pixel_values"].cuda()
        targets = targets.cuda()

        preds = model(pixel_values)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Train Loss (Normalized): {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval()
    total_val_loss_normalized = 0
    all_preds_original = []
    all_targets_original = []

    print(f"Epoch {epoch+1}/{50} - Validating...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            pixel_values = inputs["pixel_values"].cuda()
            targets_normalized = targets.cuda() # Keep normalized targets for normalized loss calculation

            preds_normalized = model(pixel_values)
            loss_normalized = loss_fn(preds_normalized, targets_normalized)
            total_val_loss_normalized += loss_normalized.item()

            # Move predictions and targets to CPU for de-normalization and metric calculation
            preds_cpu = preds_normalized.cpu().numpy()
            targets_cpu = targets_normalized.cpu().numpy()

            # De-normalize predictions and targets
            preds_original = scaler.inverse_transform(preds_cpu)
            targets_original = scaler.inverse_transform(targets_cpu)

            all_preds_original.append(preds_original)
            all_targets_original.append(targets_original)

    avg_val_loss_normalized = total_val_loss_normalized / len(val_loader)
    print(f"Epoch {epoch+1} - Average Val Loss (Normalized): {avg_val_loss_normalized:.4f}")

    # Calculate validation MSE on original scale
    all_preds_original = np.vstack(all_preds_original)
    all_targets_original = np.vstack(all_targets_original)
    val_mse_original = mean_squared_error(all_targets_original, all_preds_original)
    print(f"Epoch {epoch+1} - Validation MSE (Original Scale): {val_mse_original:.4f}")

    # Step the learning rate scheduler based on validation MSE
    scheduler.step(val_mse_original)

    # Print sample predictions vs targets on original scale
    print("Sample Predictions vs. Targets (Original Scale):")
    for i in range(min(5, len(all_preds_original))):
        print(f"  Pred: [{all_preds_original[i, 0]:.4f}, {all_preds_original[i, 1]:.4f}], "
              f"True: [{all_targets_original[i, 0]:.4f}, {all_targets_original[i, 1]:.4f}], "
              f"Error: [{np.abs(all_preds_original[i, 0] - all_targets_original[i, 0]):.4f}, {np.abs(all_preds_original[i, 1] - all_targets_original[i, 1]):.4f}]")

    # --- Check for Early Stopping and Save Best Model ---
    if val_mse_original < best_val_mse:
        best_val_mse = val_mse_original
        epochs_no_improve = 0
        # Save the best model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_mse': best_val_mse,
        }, f'{save_dir}/best_checkpoint.pth')
        print(f"Saved best model checkpoint at epoch {epoch+1} with Val MSE: {best_val_mse:.4f}")
    else:
        epochs_no_improve += 1
        print(f"Validation MSE did not improve for {epochs_no_improve} epochs.")
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
            break # Stop training loop

    # Optional: Save checkpoint every few epochs or at the end of each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mse': val_mse_original,
    }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')

print("Training finished.")

# After training, you can load the best model using:
# checkpoint = torch.load(f'{save_dir}/best_checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# best_val_mse = checkpoint['best_val_mse']
# print(f"Loaded best model from epoch {epoch+1} with Val MSE: {best_val_mse:.4f}")

