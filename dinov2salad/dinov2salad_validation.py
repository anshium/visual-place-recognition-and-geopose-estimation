import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import joblib
from tqdm import tqdm
from icecream import ic


class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
        return image, target


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
            features = self.feature_extractor(x)
        return self.regressor(features)


def calculate_validation_scores(checkpoint_path, val_csv_path, image_dir):
    val_df = pd.read_csv(val_csv_path)

    filtered_val_df = val_df[val_df['filename'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]
    if len(val_df) != len(filtered_val_df):
        print(f"Warning: {len(val_df) - len(filtered_val_df)} images listed in the validation CSV were not found in the image directory.")

    val_dataset = CampusDataset(filtered_val_df, image_dir)
    val_loader = DataLoader(val_dataset, batch_size=16)

    base_model = torch.hub.load("serizba/salad", "dinov2_salad").cuda().eval()
    model = DINOv2RegressionModel(base_model).cuda()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # scaler = joblib.load("latlon_scaler.pkl")
    scaler = joblib.load("latlon_scaler_20250504_022555.pkl")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.cuda()
            preds = model(inputs).cpu().numpy()
            targets = targets.cpu().numpy()

            preds_original = scaler.inverse_transform(preds)
            targets_original = targets

            all_preds.append(preds_original)
            all_targets.append(targets_original)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    print(all_preds.shape, all_targets.shape)

    lats_preds = all_preds[:, 0]
    longs_preds = all_preds[:, 1]

    lats_target = all_targets[:, 0]
    longs_target = all_targets[:, 1]

    final_loss = 0.5 * (np.sum((lats_preds - lats_target) ** 2) + np.sum((longs_preds - longs_target) ** 2)) / len(all_preds)

    ic(final_loss)
    ic(lats_preds[:5] - lats_target[:5])
    ic(lats_target[:5])
    ic(lats_preds[:5])

    print("\nSample Predictions (Original Scale):")
    for i in range(min(5, len(all_preds))):
        pred_lat, pred_lon = all_preds[i]
        true_lat, true_lon = all_targets[i]
        lat_error = np.abs(pred_lat - true_lat)
        lon_error = np.abs(pred_lon - true_lon)
        print(f"Prediction: (Lat: {pred_lat:.6f}, Lon: {pred_lon:.6f}), "
              f"True: (Lat: {true_lat:.6f}, Lon: {true_lon:.6f}), "
              f"Error: (Lat: {lat_error:.6f}, Lon: {lon_error:.6f})")


if __name__ == "__main__":
    # checkpoint_file = "/home/skills/ansh/delme/dinov2salad/training_20250504_015745/checkpoint_49_.pth"
    checkpoint_file = "/home/skills/ansh/delme/dinov2salad/training_20250504_022555/checkpoint_98_.pth"
    validation_csv = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
    validation_image_dir = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

    calculate_validation_scores(checkpoint_file, validation_csv, validation_image_dir)

## getting 228000 at checkpoint 49 in training_20250504_015745