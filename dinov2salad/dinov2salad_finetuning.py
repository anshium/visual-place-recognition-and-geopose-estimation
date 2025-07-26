import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import os

from sklearn.preprocessing import StandardScaler
import numpy as np


model_base = torch.hub.load("serizba/salad", "dinov2_salad").cuda().eval()

class DINOv2RegressionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.feature_extractor = base_model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(8448, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)  # shape: (B, 8448)
        return self.regressor(features)


from torchvision import transforms

class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

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

        image_tensor = self.transform(image)
        target = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)

        return image_tensor, target


train_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_train.csv")
val_df = pd.read_csv("/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv")

scaler = StandardScaler()
train_df[['latitude', 'longitude']] = scaler.fit_transform(train_df[['latitude', 'longitude']])
val_df[['latitude', 'longitude']] = scaler.transform(val_df[['latitude', 'longitude']])

train_image_dir = "/home/skills/ansh/delme/dataset/iiit_dataset/images_train/images_train"
val_image_dir = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

train_dataset = CampusDataset(train_df, train_image_dir)
val_dataset = CampusDataset(val_df, val_image_dir)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

dinov2_base = torch.hub.load("serizba/salad", "dinov2_salad").cuda().eval()
model = DINOv2RegressionModel(dinov2_base).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

from sklearn.preprocessing import StandardScaler


# dirs and stuff
from datetime import datetime

date_time = datetime.now().strftime('%Y%m%d_%H%M%S')

save_dir = f"/home/skills/ansh/delme/dinov2salad/training_{date_time}"

os.mkdir(save_dir)

import joblib
joblib.dump(scaler, f"latlon_scaler_{date_time}.pkl")

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        preds = model(inputs)
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
    }, f'{save_dir}/checkpoint_{epoch}_.pth')


    # Validation
    model.eval()
    val_loss = 0

    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            preds = model(inputs).cpu().numpy()
            targets = targets.cpu().numpy()
            
            # De-normalize
            preds_original = scaler.inverse_transform(preds)
            targets_original = scaler.inverse_transform(targets)
            
            all_preds.append(preds_original)
            all_targets.append(targets_original)


    # Combine for final reporting
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    for pred, true in zip(all_preds[:5], all_targets[:5]):
        print(f"Pred: {pred}, True: {true}, Error: {np.abs(pred - true)}")

    print(f"Epoch {epoch+1} - Val Loss: {val_loss / len(val_loader):.4f}")