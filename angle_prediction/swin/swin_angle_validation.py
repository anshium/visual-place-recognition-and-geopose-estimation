import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from icecream import ic


class CampusAngleDataset(Dataset):
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

        target = torch.tensor(row['angle'], dtype=torch.float32)

        return inputs, target


class SwinAngleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output).squeeze(-1)


def mean_absolute_angular_error(preds, targets):
    diff = torch.abs(preds - targets)
    return torch.mean(torch.minimum(diff, 360 - diff))


def validate_model(checkpoint_path, val_csv_path, image_dir):
    val_df = pd.read_csv(val_csv_path)

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True)

    # Filter out rows where image doesn't exist
    filtered_val_df = val_df[val_df['filename'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]

    if len(val_df) != len(filtered_val_df):
        print(f"Warning: {len(val_df) - len(filtered_val_df)} images listed in the CSV were not found.")

    val_dataset = CampusAngleDataset(filtered_val_df, image_dir, image_processor)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = SwinAngleModel().cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            pixel_values = inputs["pixel_values"].cuda()
            targets = targets.cuda()

            preds = model(pixel_values)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Wrap predictions to [0, 360)
    all_preds = all_preds % 360

    # Compute angular error
    diff = np.abs(all_preds - all_targets)
    angular_errors = np.minimum(diff, 360 - diff)
    mae = angular_errors.mean()

    print(f"\nFinal Mean Absolute Angular Error (MAAE): {mae:.4f}°")

    print("\nSample Predictions:")
    for i in range(min(5, len(all_preds))):
        pred = all_preds[i]
        true = all_targets[i]
        err = angular_errors[i]
        print(f"Prediction: {pred:.2f}°, Target: {true:.2f}°, Error: {err:.2f}°")


if __name__ == "__main__":
    checkpoint_file = "/home/skills/ansh/delme/angle_prediction/swin/training_20250504_025037/checkpoint_20_.pth"
    validation_csv = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
    validation_image_dir = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

    validate_model(checkpoint_file, validation_csv, validation_image_dir)
