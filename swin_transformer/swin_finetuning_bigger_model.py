import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, SwinModel
from tqdm import tqdm
import joblib
from datetime import datetime

# Dataset class
class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor

        # Filter out rows where the image file does not exist
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

        inputs = self.processor(images=image, return_tensors="pt", do_resize=True, size=384)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        target = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)

        return inputs, target

# Regression model using Swin Transformer
class SwinRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window12-384")
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384")
# Load data
train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

# Normalize targets
scaler = StandardScaler()
train_df[['latitude', 'longitude']] = scaler.fit_transform(train_df[['latitude', 'longitude']])
val_df[['latitude', 'longitude']] = scaler.transform(val_df[['latitude', 'longitude']])

# Save scaler
# date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# save_dir = f"/home/skills/ansh/delme/swin_transformer/training_{date_time}"
# os.mkdir(save_dir)
# joblib.dump(scaler, f"{save_dir}/latlon_scaler_{date_time}.pkl")

save_dir = "/home/skills/ansh/delme/swin_transformer/training_20250504_152048"

# Processor and datasets

train_dataset = CampusDataset(train_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train", image_processor)
val_dataset = CampusDataset(val_df, "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val", image_processor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model setup
model = SwinRegressionModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

checkpoint_path = f"{save_dir}/checkpoint_49.pth"

# Load model and optimizer states
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  # continue from next epoch


# Training loop
for epoch in range(start_epoch, start_epoch + 20):  # train for 20 more epochs
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader):
        pixel_values = inputs["pixel_values"].cuda()
        targets = targets.cuda()

        preds = model(pixel_values)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'{save_dir}/checkpoint_{epoch}.pth')

    # Validation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            pixel_values = inputs["pixel_values"].cuda()
            preds = model(pixel_values).cpu().numpy()
            targets = targets.cpu().numpy()

            preds_original = scaler.inverse_transform(preds)
            targets_original = scaler.inverse_transform(targets)

            all_preds.append(preds_original)
            all_targets.append(targets_original)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    for pred, true in zip(all_preds[:5], all_targets[:5]):
        print(f"Pred: {pred}, True: {true}, Error: {np.abs(pred - true)}")

    val_loss = F.mse_loss(torch.tensor(all_preds), torch.tensor(all_targets)).item()
    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")


# Final model save
torch.save(model, f"{save_dir}/model_final.pth")
