import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F # Kept for potential model variations
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from glob import glob # To find image files in the test directory

# ===============================
# Configuration & Paths
# ===============================
print("Setting up configuration...")

# --- !! IMPORTANT: Update these paths !! ---
# Path to the best model saved during training
BEST_MODEL_PATH = "/home/skills/ansh/delme/angle_prediction/efficientnet/sincos/training_20250505_191535/best_model.pth"
# Replace <YOUR_TRAINING_RUN_FOLDER> with the actual folder name (e.g., training_20231027_103000)

# Validation data paths (for MAAE calculation)
VAL_CSV_PATH = "/home/skills/ansh/delme/cleaned_dataset_files/labels_val.csv"
VAL_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

# Test data path (for prediction output) - NO LABELS EXPECTED
TEST_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_test/images_test" # <<< CHANGE THIS to your test image folder
TEST_PREDICTIONS_CSV = "/home/skills/ansh/delme/angle_prediction/efficientnet/sincos/training_20250505_191535/test_pred.csv" # <<< CHANGE THIS to desired output CSV path

# Other settings
BATCH_SIZE = 32 # Can be larger for evaluation if memory allows
NUM_WORKERS = 4
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif'] # Common image extensions

# --- Input Validation ---
if not os.path.exists(BEST_MODEL_PATH) or BEST_MODEL_PATH.endswith("<YOUR_TRAINING_RUN_FOLDER>/best_model.pth"):
    raise FileNotFoundError(f"Best model file path seems incorrect or not updated: {BEST_MODEL_PATH}\nPlease update the BEST_MODEL_PATH variable.")
if not os.path.isdir(VAL_IMG_DIR):
     print(f"Warning: Validation image directory not found: {VAL_IMG_DIR}. MAAE calculation will fail if VAL_CSV_PATH is used.")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"Test image directory not found: {TEST_IMG_DIR}\nPlease update the TEST_IMG_DIR variable.")
# Ensure output directory exists for test predictions
os.makedirs(os.path.dirname(TEST_PREDICTIONS_CSV), exist_ok=True)


# ===============================
# Dataset Definition (Outputs cos/sin) - For Validation Set (with labels)
# ===============================
class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        # Assuming files were verified during training or are reliable
        self.df = dataframe
        print(f"Initialized CampusDataset (Validation) with {len(self.df)} samples.")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        image_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"ERROR loading validation image {image_path}: {e}. Skipping/Returning dummy.")
            # Return dummy data consistent with expected types if needed by collate_fn
            dummy_img = torch.zeros((3, input_size, input_size)) # Match transform output size
            dummy_target = torch.zeros(2, dtype=torch.float32)
            return {"pixel_values": dummy_img, "filename": "error_file", "true_angle_deg": 0.0}, dummy_target

        image = self.transform(image)

        # Target angle in degrees
        angle_deg = row['angle']
        angle_rad = np.deg2rad(angle_deg)
        target = torch.tensor([np.cos(angle_rad), np.sin(angle_rad)], dtype=torch.float32)

        return {"pixel_values": image, "filename": filename, "true_angle_deg": angle_deg}, target

# ===============================
# Dataset Definition - For Test Set (NO labels)
# ===============================
class TestImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform, img_extensions):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        print(f"Scanning for test images in: {self.image_dir}")
        for ext in img_extensions:
            self.image_files.extend(glob(os.path.join(self.image_dir, ext)))

        if not self.image_files:
             raise FileNotFoundError(f"No image files found in {self.image_dir} with extensions {img_extensions}")
        print(f"Initialized TestImageFolderDataset with {len(self.image_files)} samples.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        filename = os.path.basename(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"ERROR loading test image {image_path}: {e}. Skipping/Returning dummy.")
            # Return dummy data consistent with expected types if needed by collate_fn
            dummy_img = torch.zeros((3, input_size, input_size)) # Match transform output size
            return {"pixel_values": dummy_img, "filename": "error_file"}


        image = self.transform(image)
        # Return pixel values and filename
        return {"pixel_values": image, "filename": filename}


# ===============================
# EfficientNet Model for Sin/Cos Regression - Copied from training script
# ===============================
class EfficientNetSinCosModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None) # Load architecture only
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Match dropout used in training
            nn.Linear(num_features, 2)
        )

    def forward(self, pixel_values):
        preds_cos_sin = self.backbone(pixel_values)
        # Optional Normalization (should match if used in training's forward pass)
        # preds_cos_sin = F.normalize(preds_cos_sin, p=2, dim=1)
        return preds_cos_sin

# ===============================
# Angular Loss Function (for validation reporting ONLY) - Copied from training script
# ===============================
def mean_absolute_angular_error(preds_deg, targets_deg):
    """Calculates the MAAE, handling the circular nature of angles (0-360)."""
    diff = torch.abs(preds_deg - targets_deg)
    angular_errors = torch.minimum(diff, 360.0 - diff)
    return torch.mean(angular_errors) # Return mean over the batch

# ===============================
# Data Transforms - Copied from training script (using val_transform)
# ===============================
weights = EfficientNet_B0_Weights.IMAGENET1K_V1 # Needed for stats
imagenet_stats = weights.transforms().mean, weights.transforms().std
input_size = 224

# Use the SAME transform for validation and test inference
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1]),
])

# ===============================
# Main Prediction Script
# ===============================
if __name__ == "__main__":
    print("Starting prediction script...")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from: {BEST_MODEL_PATH}")
    model = EfficientNetSinCosModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- 1. Process Validation Set (Calculate MAAE) ---
    print("\n--- Processing Validation Set ---")
    try:
        val_df = pd.read_csv(VAL_CSV_PATH)
        val_dataset = CampusDataset(val_df, VAL_IMG_DIR, inference_transform)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )

            total_angular_error_sum = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation MAAE"):
                    inputs_dict = batch[0] # Dataset returns (dict, tensor)
                    targets_cos_sin = batch[1].to(device)
                    pixel_values = inputs_dict["pixel_values"].to(device)
                    # Skip potential error samples
                    if pixel_values.nelement() == 0 or inputs_dict["filename"][0] == "error_file": continue

                    batch_true_angles_deg = inputs_dict["true_angle_deg"]

                    preds_cos_sin = model(pixel_values)
                    pred_angle_rad = torch.atan2(preds_cos_sin[:, 1], preds_cos_sin[:, 0])
                    pred_angle_deg = (torch.rad2deg(pred_angle_rad) + 360.0) % 360.0

                    batch_true_angles_deg_tensor = batch_true_angles_deg.clone().detach().to(device=pred_angle_deg.device, dtype=pred_angle_deg.dtype)
                    diff = torch.abs(pred_angle_deg - batch_true_angles_deg_tensor)
                    batch_angular_errors = torch.minimum(diff, 360.0 - diff)

                    total_angular_error_sum += torch.sum(batch_angular_errors).item()
                    total_val_samples += pixel_values.size(0)

            if total_val_samples > 0:
                final_maae = total_angular_error_sum / total_val_samples
                print("\nValidation Set Results:")
                print(f"Total validation samples processed: {total_val_samples}")
                print(f"Mean Absolute Angular Error (MAAE): {final_maae:.4f} degrees")
            else:
                print("\nNo valid validation samples processed. MAAE could not be calculated.")
        else:
            print("\nValidation dataset is empty. Skipping MAAE calculation.")

    except FileNotFoundError:
        print(f"\nValidation CSV not found at {VAL_CSV_PATH}. Skipping MAAE calculation.")
    except Exception as e:
        print(f"\nAn error occurred during validation processing: {e}. Skipping MAAE calculation.")


    # --- 2. Process Test Set (Predict and Save) ---
    print("\n--- Processing Test Set ---")
    try:
        test_dataset = TestImageFolderDataset(TEST_IMG_DIR, inference_transform, IMAGE_EXTENSIONS)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False, # No need to shuffle for prediction
                num_workers=NUM_WORKERS,
                pin_memory=True # May speed up data transfer to GPU
            )

            test_results = [] # List to store results as dicts

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Test Prediction"):
                    inputs_dict = batch # Dataset returns dict
                    pixel_values = inputs_dict["pixel_values"].to(device)
                    filenames = inputs_dict["filename"] # List of filenames in the batch
                     # Skip potential error samples
                    if pixel_values.nelement() == 0 or filenames[0] == "error_file": continue

                    # Get model predictions (cos, sin)
                    preds_cos_sin = model(pixel_values)

                    # Convert predictions back to angles (degrees)
                    pred_angle_rad = torch.atan2(preds_cos_sin[:, 1], preds_cos_sin[:, 0])
                    pred_angle_deg = (torch.rad2deg(pred_angle_rad) + 360.0) % 360.0

                    # Store results
                    pred_angles_list = pred_angle_deg.cpu().tolist()
                    for fname, angle in zip(filenames, pred_angles_list):
                         test_results.append({"filename": fname, "predicted_angle_degrees": angle})


            if test_results:
                # Create DataFrame and save to CSV
                test_pred_df = pd.DataFrame(test_results)
                test_pred_df.sort_values(by="filename", inplace=True) # Optional: sort by filename
                test_pred_df.to_csv(TEST_PREDICTIONS_CSV, index=False)
                print(f"\nTest set predictions saved successfully to: {TEST_PREDICTIONS_CSV}")
                print(f"Total test samples processed: {len(test_results)}")
            else:
                 print("\nNo valid test samples were processed. No predictions saved.")
        else:
             print("\nTest dataset is empty. Skipping test prediction.")

    except FileNotFoundError as e:
        print(f"\nError setting up test dataset: {e}. Skipping test prediction.")
    except Exception as e:
        print(f"\nAn error occurred during test prediction: {e}. Skipping test prediction.")

    print("\nScript finished.")