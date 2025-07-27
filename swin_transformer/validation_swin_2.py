import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoImageProcessor, SwinModel
from tqdm import tqdm
import joblib
import math # For sqrt

SAVE_DIR_TO_EVALUATE = "/home/skills/ansh/delme/swin_transformer/training_gemini_2_20250505_004059"

if not os.path.isdir(SAVE_DIR_TO_EVALUATE):
    raise FileNotFoundError(f"The specified save directory does not exist: {SAVE_DIR_TO_EVALUATE}\nPlease provide the correct path to a completed training run.")

VAL_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
VAL_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

SCALER_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, "latlon_scaler.pkl")
BEST_MODEL_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, 'model_best.pth')
RESULTS_CSV_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, 'validation_predictions.csv')

MODEL_NAME = "microsoft/swin-base-patch4-window12-384"
IMAGE_SIZE = 384
BATCH_SIZE = 32
DROPOUT_PROB = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Evaluating model from: {BEST_MODEL_PATH}")
print(f"Using scaler from: {SCALER_PATH}")
print(f"Saving results to: {RESULTS_CSV_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
         print("Warning: Loaded object might not be a fitted StandardScaler.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

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
        print("Verifying image files and preparing targets for validation...")
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Checking validation files"):
            image_path = os.path.join(self.image_dir, row['filename'])
            if os.path.isfile(image_path):
                valid_rows.append(row)
                targets_to_scale.append([row['latitude'], row['longitude']])
                filenames.append(row['filename'])
            else:
                print(f"Warning: Image file not found and skipped: {image_path}")

        if not valid_rows:
            raise ValueError("No valid image files found for the provided validation dataframe and image directory.")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.filenames = [row['filename'] for row in valid_rows]
        self.original_targets = np.array([[row['latitude'], row['longitude']] for row in valid_rows])

        print("Scaling targets using loaded scaler...")
        if targets_to_scale:
             self.scaled_targets = self.scaler.transform(np.array(targets_to_scale))
        else:
             self.scaled_targets = np.array([])

        print("Validation dataset initialized.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.filenames[idx] # Get filename
        image_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        inputs = self.processor(images=image, return_tensors="pt", do_resize=True, size=(IMAGE_SIZE, IMAGE_SIZE))
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        target_scaled = torch.tensor(self.scaled_targets[idx], dtype=torch.float32)
        target_original = torch.tensor(self.original_targets[idx], dtype=torch.float32)

        return inputs, target_scaled, target_original, filename


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

def collate_fn(batch):
    pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
    inputs = {'pixel_values': pixel_values}
    targets_scaled = torch.stack([item[1] for item in batch])
    targets_original = torch.stack([item[2] for item in batch])
    filenames = [item[3] for item in batch]
    return inputs, targets_scaled, targets_original, filenames


val_df = pd.read_csv(VAL_CSV_PATH)

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

val_dataset = CampusDataset(val_df, VAL_IMG_DIR, image_processor, scaler, is_training=False)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

model = SwinRegressionModel(MODEL_NAME, dropout_prob=DROPOUT_PROB).to(device)

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Best model file not found at: {BEST_MODEL_PATH}")
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(f"Successfully loaded model weights from {BEST_MODEL_PATH}")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.eval()

all_preds_original_list = []
all_targets_original_list = []
all_filenames_list = []

print("Starting prediction on validation set...")
val_loop = tqdm(val_loader, desc="Predicting")

with torch.no_grad():
    for batch_idx, (inputs, targets_scaled, targets_original_batch, filenames_batch) in enumerate(val_loop):
        pixel_values = inputs["pixel_values"].to(device)

        preds_scaled = model(pixel_values)

        preds_np_scaled = preds_scaled.cpu().numpy()
        targets_np_original = targets_original_batch.cpu().numpy()

        if len(preds_np_scaled) > 0:
             preds_original = scaler.inverse_transform(preds_np_scaled)

             all_preds_original_list.extend(preds_original.tolist())
             all_targets_original_list.extend(targets_np_original.tolist())
             all_filenames_list.extend(filenames_batch)


if not all_preds_original_list:
    print("No predictions were made. Check the validation data and paths.")
else:
    all_preds_original_np = np.array(all_preds_original_list)
    all_targets_original_np = np.array(all_targets_original_list)

    val_mse = mean_squared_error(all_targets_original_np, all_preds_original_np)
    val_rmse = math.sqrt(val_mse)
    val_mae = mean_absolute_error(all_targets_original_np, all_preds_original_np)
    val_mae_lat = mean_absolute_error(all_targets_original_np[:, 0], all_preds_original_np[:, 0])
    val_mae_lon = mean_absolute_error(all_targets_original_np[:, 1], all_preds_original_np[:, 1])

    print("\n--- Evaluation Results on Validation Set (Original Scale) ---")
    print(f"  Mean Squared Error (MSE): {val_mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {val_rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {val_mae:.6f}")
    print(f"  MAE Latitude: {val_mae_lat:.6f}")
    print(f"  MAE Longitude: {val_mae_lon:.6f}")
    print("-" * 30)

    results_df = pd.DataFrame({
        'filename': all_filenames_list,
        'true_latitude': all_targets_original_np[:, 0],
        'true_longitude': all_targets_original_np[:, 1],
        'predicted_latitude': all_preds_original_np[:, 0],
        'predicted_longitude': all_preds_original_np[:, 1]
    })

    results_df['error_latitude'] = np.abs(results_df['true_latitude'] - results_df['predicted_latitude'])
    results_df['error_longitude'] = np.abs(results_df['true_longitude'] - results_df['predicted_longitude'])

    try:
        results_df.to_csv(RESULTS_CSV_PATH, index=False, float_format='%.6f')
        print(f"Successfully saved prediction results to: {RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    print("\nSample Predictions vs True Values (Original Scale):")
    print(results_df[['filename', 'predicted_latitude', 'predicted_longitude', 'true_latitude', 'true_longitude']].head(10).to_string())


print("\nPrediction script finished.")