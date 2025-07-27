import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoImageProcessor, SwinModel
from tqdm import tqdm
import joblib
from datetime import datetime
from torchvision import transforms
import math

TRAIN_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv"
VAL_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
TRAIN_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train"
VAL_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

DATE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_DIR = f"/home/skills/ansh/delme/swin_transformer/training_gemini_2_{DATE_TIME}"
print(f"Results will be saved in: {SAVE_DIR}")

SCALER_PATH = os.path.join(SAVE_DIR, "latlon_scaler.pkl")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'model_best.pth')
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, 'model_final.pth')

MODEL_NAME = "microsoft/swin-base-patch4-window12-384"
IMAGE_SIZE = 384
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 150
HUBER_DELTA = 1.0
DROPOUT_PROB = 0.3
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.2
EARLY_STOPPING_PATIENCE = 40

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(degrees=15),
])

class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, scaler, is_training=False, target_transforms=None):
        self.image_dir = image_dir
        self.processor = processor
        self.scaler = scaler
        self.is_training = is_training
        self.target_transforms = target_transforms

        valid_rows = []
        targets_to_scale = []
        filenames = []
        print("Verifying image files and preparing targets...")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            image_path = os.path.join(self.image_dir, row['filename'])
            if os.path.isfile(image_path):
                valid_rows.append(row)
                targets_to_scale.append([row['latitude'], row['longitude']])
                filenames.append(row['filename'])
            else:
                print(f"Warning: Image file not found and skipped: {image_path}")

        if not valid_rows:
            raise ValueError("No valid image files found for the provided dataframe and image directory.")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.filenames = [row['filename'] for row in valid_rows]

        print("Scaling targets...")
        if targets_to_scale:
             self.scaled_targets = self.scaler.transform(np.array(targets_to_scale))
        else:
             self.scaled_targets = np.array([]) 

        print("Dataset initialized.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        if self.is_training and self.target_transforms:
            image = self.target_transforms(image)

        inputs = self.processor(images=image, return_tensors="pt", do_resize=True, size=(IMAGE_SIZE, IMAGE_SIZE))
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        target = torch.tensor(self.scaled_targets[idx], dtype=torch.float32)

        return inputs, target

class SwinRegressionModel(nn.Module):
    def __init__(self, model_name, dropout_prob=0.3):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 2)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)


print("Fitting new scaler on training data.")
scaler = StandardScaler()

scaler.fit(train_df[['latitude', 'longitude']])

joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")



image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)



train_dataset = CampusDataset(train_df, TRAIN_IMG_DIR, image_processor, scaler, is_training=True, target_transforms=train_transforms)
val_dataset = CampusDataset(val_df, VAL_IMG_DIR, image_processor, scaler, is_training=False)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


model = SwinRegressionModel(MODEL_NAME, dropout_prob=DROPOUT_PROB).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.HuberLoss(delta=HUBER_DELTA).to(device)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)


start_epoch = 0
best_val_mse = float('inf')
epochs_without_improvement = 0
print("Starting training from scratch.")


for epoch in range(start_epoch, NUM_EPOCHS): 
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    model.train()
    total_train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

    for batch_idx, (inputs, targets) in enumerate(train_loop):
        pixel_values = inputs["pixel_values"].to(device)
        targets = targets.to(device) 

        preds_scaled = model(pixel_values)
        loss = loss_fn(preds_scaled, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        if batch_idx % 50 == 0:
             train_loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Train Loss (Scaled): {avg_train_loss:.4f}")

    
    model.eval()
    all_preds_original = []
    all_targets_original = []
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")

    with torch.no_grad():
        for inputs, targets_scaled in val_loop:
            pixel_values = inputs["pixel_values"].to(device)
            preds_scaled = model(pixel_values)
            preds_np_scaled = preds_scaled.cpu().numpy()
            targets_np_scaled = targets_scaled.cpu().numpy()

            if len(preds_np_scaled) == 0: continue

            preds_original = scaler.inverse_transform(preds_np_scaled)
            targets_original = scaler.inverse_transform(targets_np_scaled)

            all_preds_original.append(preds_original)
            all_targets_original.append(targets_original)

    all_preds_original = np.vstack(all_preds_original)
    all_targets_original = np.vstack(all_targets_original)

    val_mse = mean_squared_error(all_targets_original, all_preds_original)
    val_rmse = math.sqrt(val_mse)
    val_mae = mean_absolute_error(all_targets_original, all_preds_original)
    val_mae_lat = mean_absolute_error(all_targets_original[:, 0], all_preds_original[:, 0])
    val_mae_lon = mean_absolute_error(all_targets_original[:, 1], all_preds_original[:, 1])

    print(f"Epoch {epoch+1} - Validation Metrics (Original Scale):")
    print(f"  MSE: {val_mse:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  MAE Lat: {val_mae_lat:.4f}, MAE Lon: {val_mae_lon:.4f}")
    print("-" * 20)
    print("Sample Predictions (Original Scale):")
    for i in range(min(5, len(all_preds_original))):
         pred = all_preds_original[i]
         true = all_targets_original[i]
         error = np.abs(pred - true)
         print(f"  Pred: [{pred[0]:.4f}, {pred[1]:.4f}], True: [{true[0]:.4f}, {true[1]:.4f}], Abs Error: [{error[0]:.4f}, {error[1]:.4f}]")
    print("-" * 20)

    
    
    checkpoint_path = os.path.join(SAVE_DIR, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_mse': val_mse,
        'best_val_mse': best_val_mse, 
    }, checkpoint_path)
    

    scheduler.step(val_mse)

    if val_mse < best_val_mse:
        print(f"Validation MSE improved from {best_val_mse:.4f} to {val_mse:.4f}. Saving best model...")
        best_val_mse = val_mse
        torch.save(model.state_dict(), BEST_MODEL_PATH) 
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"Validation MSE did not improve. ({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
        break


torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Final model state dict saved to {FINAL_MODEL_PATH}")
print(f"Best model state dict saved to {BEST_MODEL_PATH} with Val MSE: {best_val_mse:.4f}")