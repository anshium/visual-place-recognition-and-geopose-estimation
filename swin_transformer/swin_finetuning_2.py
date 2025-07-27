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
                pass # skip missing files :/

        self.df = pd.DataFrame(valid_rows)
        print(f"Loaded {len(self.df)} valid samples from {len(dataframe)} in original dataframe.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        target = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)

        return inputs, target

class SwinRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_dim)
        return self.regressor(pooled_output)

train_df = pd.read_csv("/home/skills/ansh/delme/dataset/iiit_dataset/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/dataset/iiit_dataset/labels_val.csv")

scaler = StandardScaler()
train_df[['latitude', 'longitude']] = scaler.fit_transform(train_df[['latitude', 'longitude']])
val_df[['latitude', 'longitude']] = scaler.transform(val_df[['latitude', 'longitude']])

joblib.dump(scaler, "latlon_scaler.pkl")
print("Scaler saved to latlon_scaler.pkl")


image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(image_processor.size["height"]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

val_transforms = None

train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", image_processor, transform=train_transforms)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", image_processor, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4) 
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4) 


model = SwinRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()


scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f"/home/skills/ansh/delme/swin_transformer/training_{date_time}"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving checkpoints to {save_dir}")

best_val_mse = float('inf') 
epochs_no_improve = 0 
early_stop_patience = 10 


print("Starting training...")
for epoch in range(50): 
    model.train()
    total_train_loss = 0

    
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

    
    model.eval()
    total_val_loss_normalized = 0
    all_preds_original = []
    all_targets_original = []

    print(f"Epoch {epoch+1}/{50} - Validating...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            pixel_values = inputs["pixel_values"].cuda()
            targets_normalized = targets.cuda() 

            preds_normalized = model(pixel_values)
            loss_normalized = loss_fn(preds_normalized, targets_normalized)
            total_val_loss_normalized += loss_normalized.item()

            
            preds_cpu = preds_normalized.cpu().numpy()
            targets_cpu = targets_normalized.cpu().numpy()

            
            preds_original = scaler.inverse_transform(preds_cpu)
            targets_original = scaler.inverse_transform(targets_cpu)

            all_preds_original.append(preds_original)
            all_targets_original.append(targets_original)

    avg_val_loss_normalized = total_val_loss_normalized / len(val_loader)
    print(f"Epoch {epoch+1} - Average Val Loss (Normalized): {avg_val_loss_normalized:.4f}")

    
    all_preds_original = np.vstack(all_preds_original)
    all_targets_original = np.vstack(all_targets_original)
    val_mse_original = mean_squared_error(all_targets_original, all_preds_original)
    print(f"Epoch {epoch+1} - Validation MSE (Original Scale): {val_mse_original:.4f}")

    
    scheduler.step(val_mse_original)

    
    print("Sample Predictions vs. Targets (Original Scale):")
    for i in range(min(5, len(all_preds_original))):
        print(f"  Pred: [{all_preds_original[i, 0]:.4f}, {all_preds_original[i, 1]:.4f}], "
              f"True: [{all_targets_original[i, 0]:.4f}, {all_targets_original[i, 1]:.4f}], "
              f"Error: [{np.abs(all_preds_original[i, 0] - all_targets_original[i, 0]):.4f}, {np.abs(all_preds_original[i, 1] - all_targets_original[i, 1]):.4f}]")

    
    if val_mse_original < best_val_mse:
        best_val_mse = val_mse_original
        epochs_no_improve = 0
        
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
            break 
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mse': val_mse_original,
    }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')

print("Training finished.")
