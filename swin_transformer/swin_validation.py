import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from tqdm import tqdm

from icecream import ic

class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
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
        pooled_output = outputs.pooler_output  
        return self.regressor(pooled_output)

def calculate_validation_scores(checkpoint_path, val_csv_path, image_dir):
    val_df = pd.read_csv(val_csv_path)

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True)

    
    valid_filenames = val_df['filename'].tolist()

    
    filtered_val_df = val_df[val_df['filename'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]

    if len(val_df) != len(filtered_val_df):
        print(f"Warning: {len(val_df) - len(filtered_val_df)} images listed in the validation CSV were not found in the image directory.")

    val_dataset = CampusDataset(filtered_val_df, image_dir, image_processor)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = SwinRegressionModel().cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = joblib.load("latlon_scaler_2.pkl")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            pixel_values = inputs["pixel_values"].cuda()
            preds = model(pixel_values).cpu().numpy()
            targets = targets.cpu().numpy()

            
            preds_original = scaler.inverse_transform(preds)
            targets_original = targets

            all_preds.append(preds_original)
            all_targets.append(targets_original)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    print(all_preds.shape)
    print(all_targets.shape)

    lats_preds = all_preds[:, 0]
    longs_preds = all_preds[:, 1]

    lats_target = all_targets[:, 0]
    longs_target = all_targets[:, 1]

    final_loss = 0.5 * (np.sum((lats_preds - lats_target) ** 2) + np.sum((longs_preds - longs_target) ** 2)) / len(all_preds)

    ic(final_loss)
    ic(lats_preds[:5] - lats_target[:5])
    ic(lats_target[:5])
    ic(lats_preds[:5])

    ic(np.where(lats_preds - lats_target >= 1e4))

    ic(lats_target[158], lats_target[159], lats_target[160])

    print("\nSample Predictions (Original Scale):")
    for i in range(min(5, len(all_preds))):
        pred_lat, pred_lon = all_preds[i]
        true_lat, true_lon = all_targets[i]
        lat_error = np.abs(pred_lat - true_lat)
        lon_error = np.abs(pred_lon - true_lon)
        print(f"Prediction: (Lat: {pred_lat:.6f}, Lon: {pred_lon:.6f}), "
              f"True: (Lat: {true_lat:.6f}, Lon: {true_lon:.6f}), "
              f"Error: (Lat: {lat_error:.6f}, Lon: {lon_error:.6f})")

    def extract_id(filename):
        return int(os.path.splitext(filename)[0].split('_')[-1])

    filtered_val_df["ID"] = filtered_val_df["filename"].apply(extract_id)
    filtered_val_df = filtered_val_df.reset_index(drop=True)

    prediction_df = pd.DataFrame(all_preds, columns=["latitude", "longitude"])
    prediction_df["ID"] = filtered_val_df["ID"]

    prediction_df = prediction_df[["ID", "latitude", "longitude"]]

    prediction_df = prediction_df.sort_values(by="ID")

    prediction_df.to_csv("/home/skills/ansh/delme/swin_transformer/results_csv/preds.csv", index=False)
    print("Saved predictions to preds.csv")


if __name__ == "__main__":
    checkpoint_file = "/home/skills/ansh/delme/swin_transformer/training_20250504_001441/checkpoint_49_.pth"
    validation_csv = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
    validation_image_dir = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

    calculate_validation_scores(checkpoint_file, validation_csv, validation_image_dir)


#

#

#