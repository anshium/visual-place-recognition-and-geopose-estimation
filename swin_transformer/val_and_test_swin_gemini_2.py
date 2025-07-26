import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoImageProcessor, SwinModel
from tqdm import tqdm
import joblib
import math
import glob # To find image files in the test directory

# --- !! IMPORTANT !! ---
# Set this to the specific directory created during the training run you want to evaluate
# Example: SAVE_DIR_TO_EVALUATE = "/home/skills/ansh/delme/swin_transformer/training_gemini_2_20231027_103000"
SAVE_DIR_TO_EVALUATE = "/home/skills/ansh/delme/swin_transformer/training_gemini_2_20250505_004059"

# --- !! ADD TEST DATA PATH !! ---
# Set this to the directory containing your test images (without labels)
# Example: TEST_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_test/images_test"
TEST_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_test/images_test"
# ---

# --- Configuration (Derived from Training Script & SAVE_DIR) ---
if not os.path.isdir(SAVE_DIR_TO_EVALUATE):
    raise FileNotFoundError(f"The specified save directory does not exist: {SAVE_DIR_TO_EVALUATE}\nPlease provide the correct path to a completed training run.")
if not os.path.isdir(TEST_IMG_DIR):
     print(f"Warning: Test image directory not found: {TEST_IMG_DIR}. Skipping test set prediction.")
     PERFORM_TEST_PREDICTION = False
else:
     PERFORM_TEST_PREDICTION = True


# Paths (Validation data remains the same)
VAL_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
VAL_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

# Paths relative to the specific SAVE_DIR
SCALER_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, "latlon_scaler.pkl")
BEST_MODEL_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, 'model_best.pth')
VAL_RESULTS_CSV_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, 'validation_predictions.csv')
TEST_RESULTS_CSV_PATH = os.path.join(SAVE_DIR_TO_EVALUATE, 'test_predictions_sorted.csv') # Added sorted to filename

# Model & Processing Hyperparameters (Should match the training run)
MODEL_NAME = "microsoft/swin-base-patch4-window12-384"
IMAGE_SIZE = 384
BATCH_SIZE = 32
DROPOUT_PROB = 0.3

# Image extensions to look for in the test directory
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', "*.JPEG"]

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Evaluating model from: {BEST_MODEL_PATH}")
print(f"Using scaler from: {SCALER_PATH}")
print(f"Saving validation results to: {VAL_RESULTS_CSV_PATH}")
if PERFORM_TEST_PREDICTION:
    print(f"Processing test images from: {TEST_IMG_DIR}")
    print(f"Saving sorted test results to: {TEST_RESULTS_CSV_PATH}") # Updated message

# --- Load Scaler ---
# (Scaler loading code remains the same)
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

# --- Dataset Class for Validation (includes labels) ---
# (CampusDataset class remains the same)
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
                try:
                    img = Image.open(image_path)
                    img.verify()
                    img.close()
                    valid_rows.append(row)
                    targets_to_scale.append([row['latitude'], row['longitude']])
                    filenames.append(row['filename'])
                except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                     print(f"Warning: Skipping invalid/corrupt image file: {image_path} ({e})")
                except Exception as e:
                    print(f"Warning: Skipping file due to unexpected error: {image_path} ({e})")
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
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path} during getitem: {e}")
            raise
        inputs = self.processor(images=image, return_tensors="pt", do_resize=True, size=(IMAGE_SIZE, IMAGE_SIZE))
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        target_scaled = torch.tensor(self.scaled_targets[idx], dtype=torch.float32)
        target_original = torch.tensor(self.original_targets[idx], dtype=torch.float32)
        return inputs, target_scaled, target_original, filename

# --- Dataset Class for Test (No labels) ---
# (TestImageDataset class remains the same)
class TestImageDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_paths = []
        print(f"Scanning for images in {self.image_dir}...")
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))

        # --- Sort image paths alphabetically by filename ---
        # This ensures the dataset iterates in a predictable order if needed,
        # although the final sorting happens on the DataFrame anyway.
        self.image_paths.sort(key=lambda p: os.path.basename(p))
        # ---

        if not self.image_paths:
             raise FileNotFoundError(f"No image files ({', '.join(IMAGE_EXTENSIONS)}) found in {self.image_dir}")

        print(f"Found {len(self.image_paths)} images for testing (sorted by path).")
        self.filenames = [os.path.basename(p) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename = self.filenames[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            print(f"Error loading test image {image_path}: {e}. Skipping.")
            return None
        except Exception as e:
            print(f"Unexpected error loading test image {image_path}: {e}. Skipping.")
            return None
        try:
            inputs = self.processor(images=image, return_tensors="pt", do_resize=True, size=(IMAGE_SIZE, IMAGE_SIZE))
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            print(f"Error processing test image {image_path}: {e}. Skipping.")
            return None
        return inputs, filename

# --- Regression Model Class ---
# (SwinRegressionModel class remains the same)
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

# --- Collate Function for Validation ---
# (val_collate_fn remains the same)
def val_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
    inputs = {'pixel_values': pixel_values}
    targets_scaled = torch.stack([item[1] for item in batch])
    targets_original = torch.stack([item[2] for item in batch])
    filenames = [item[3] for item in batch]
    return inputs, targets_scaled, targets_original, filenames

# --- Collate Function for Test ---
# (test_collate_fn remains the same)
def test_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
    inputs = {'pixel_values': pixel_values}
    filenames = [item[1] for item in batch]
    return inputs, filenames

# --- Main Prediction Script ---

# Load validation dataframe
val_df = pd.read_csv(VAL_CSV_PATH)

# Initialize Image Processor
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# --- Create Validation Dataset & DataLoader ---
val_dataset = CampusDataset(val_df, VAL_IMG_DIR, image_processor, scaler, is_training=False)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=val_collate_fn
)


# --- Create Test Dataset & DataLoader (if directory exists) ---
test_loader = None # Initialize to None
if PERFORM_TEST_PREDICTION:
    try:
        test_dataset = TestImageDataset(TEST_IMG_DIR, image_processor)
        # Set shuffle=False for test loader to maintain order if needed,
        # although we sort the final DataFrame anyway.
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=test_collate_fn
        )
    except FileNotFoundError as e:
        print(e)
        PERFORM_TEST_PREDICTION = False
    except Exception as e:
        print(f"An unexpected error occurred creating the test dataset/loader: {e}")
        PERFORM_TEST_PREDICTION = False


# --- Load Model Architecture ---
model = SwinRegressionModel(MODEL_NAME, dropout_prob=DROPOUT_PROB).to(device)

# --- Load Best Model Weights ---
# (Model loading code remains the same)
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Best model file not found at: {BEST_MODEL_PATH}")
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(f"Successfully loaded model weights from {BEST_MODEL_PATH}")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# --- Set model to evaluation mode ---
model.eval()

# ============================================
# --- 1. Validation Set Prediction & Eval ---
# ============================================
# (Validation prediction loop and saving code remains the same)
print("\n--- Starting Prediction on Validation Set ---")
all_val_preds_original_list = []
all_val_targets_original_list = []
all_val_filenames_list = []

val_loop = tqdm(val_loader, desc="Predicting (Validation)")
with torch.no_grad():
    for batch in val_loop:
        if batch is None:
             print("Skipping an empty/failed batch in validation.")
             continue
        inputs, targets_scaled, targets_original_batch, filenames_batch = batch
        pixel_values = inputs["pixel_values"].to(device)
        preds_scaled = model(pixel_values)
        preds_np_scaled = preds_scaled.cpu().numpy()
        targets_np_original = targets_original_batch.cpu().numpy()
        if len(preds_np_scaled) > 0:
             preds_original = scaler.inverse_transform(preds_np_scaled)
             all_val_preds_original_list.extend(preds_original.tolist())
             all_val_targets_original_list.extend(targets_np_original.tolist())
             all_val_filenames_list.extend(filenames_batch)

if not all_val_preds_original_list:
    print("No validation predictions were made. Check validation data and paths.")
else:
    all_val_preds_original_np = np.array(all_val_preds_original_list)
    all_val_targets_original_np = np.array(all_val_targets_original_list)
    val_mse = mean_squared_error(all_val_targets_original_np, all_val_preds_original_np)
    val_rmse = math.sqrt(val_mse)
    val_mae = mean_absolute_error(all_val_targets_original_np, all_val_preds_original_np)
    val_mae_lat = mean_absolute_error(all_val_targets_original_np[:, 0], all_val_preds_original_np[:, 0])
    val_mae_lon = mean_absolute_error(all_val_targets_original_np[:, 1], all_val_preds_original_np[:, 1])
    print("\n--- Evaluation Results on Validation Set (Original Scale) ---")
    print(f"  Mean Squared Error (MSE): {val_mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {val_rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {val_mae:.6f}")
    print(f"  MAE Latitude: {val_mae_lat:.6f}")
    print(f"  MAE Longitude: {val_mae_lon:.6f}")
    print("-" * 30)
    val_results_df = pd.DataFrame({
        'filename': all_val_filenames_list,
        'true_latitude': all_val_targets_original_np[:, 0],
        'true_longitude': all_val_targets_original_np[:, 1],
        'predicted_latitude': all_val_preds_original_np[:, 0],
        'predicted_longitude': all_val_preds_original_np[:, 1]
    })
    val_results_df['error_latitude'] = np.abs(val_results_df['true_latitude'] - val_results_df['predicted_latitude'])
    val_results_df['error_longitude'] = np.abs(val_results_df['true_longitude'] - val_results_df['predicted_longitude'])
    try:
        val_results_df.to_csv(VAL_RESULTS_CSV_PATH, index=False, float_format='%.6f')
        print(f"Successfully saved validation prediction results to: {VAL_RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"Error saving validation results to CSV: {e}")

# ========================================
# --- 2. Test Set Prediction ---
# ========================================
if PERFORM_TEST_PREDICTION and test_loader: # Check if test_loader was created
    print("\n--- Starting Prediction on Test Set ---")
    all_test_preds_list = []
    all_test_filenames_list = []

    test_loop = tqdm(test_loader, desc="Predicting (Test)")
    with torch.no_grad():
        for batch in test_loop:
            if batch is None:
                print("Skipping an empty/failed batch in test.")
                continue
            inputs, filenames_batch = batch

            pixel_values = inputs["pixel_values"].to(device)
            preds_scaled = model(pixel_values)
            preds_np_scaled = preds_scaled.cpu().numpy()

            if len(preds_np_scaled) > 0:
                 preds_original = scaler.inverse_transform(preds_np_scaled)
                 all_test_preds_list.extend(preds_original.tolist())
                 all_test_filenames_list.extend(filenames_batch)

    # --- Saving Test Results ---
    if not all_test_preds_list:
        print("No test predictions were made. Check test data directory and image files.")
    else:
        all_test_preds_np = np.array(all_test_preds_list)

        test_results_df = pd.DataFrame({
            'filename': all_test_filenames_list,
            'predicted_latitude': all_test_preds_np[:, 0],
            'predicted_longitude': all_test_preds_np[:, 1]
        })

        # <<< --- SORTING STEP --- >>>
        print("Sorting test results by filename...")
        test_results_df.sort_values(by='filename', inplace=True)
        # <<< --- END SORTING STEP --- >>>

        try:
            # Save the SORTED DataFrame
            test_results_df.to_csv(TEST_RESULTS_CSV_PATH, index=False, float_format='%.6f')
            print(f"Successfully saved sorted test prediction results to: {TEST_RESULTS_CSV_PATH}")
        except Exception as e:
            print(f"Error saving sorted test results to CSV: {e}")
elif PERFORM_TEST_PREDICTION and not test_loader:
     print("\nSkipping test set prediction as test loader could not be created.")
else:
    print("\nSkipping test set prediction as PERFORM_TEST_PREDICTION is False.")


print("\nPrediction script finished.")