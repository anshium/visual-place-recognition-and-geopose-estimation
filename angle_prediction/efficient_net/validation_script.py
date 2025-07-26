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

# ===============================
# Configuration & Paths
# ===============================
print("Setting up configuration...")

# --- !! IMPORTANT: Update this path !! ---
# You need to specify the exact path to the 'best_model.pth' file
# from the specific training run you want to evaluate.
# Example: If your training run was saved in '/home/skills/ansh/delme/angle_prediction/efficientnet/sincos/training_20231027_103000'
# then the path would be:
BEST_MODEL_PATH = "/home/skills/ansh/delme/angle_prediction/efficientnet/sincos/training_20250505_191535/best_model.pth"
# Replace <YOUR_TRAINING_RUN_FOLDER> with the actual folder name (e.g., training_20231027_103000)

# Validation data paths (should match your training script)
VAL_CSV_PATH = "/home/skills/ansh/delme/dataset/iiit_dataset/labels_val.csv"
VAL_IMG_DIR = "/home/skills/ansh/delme/dataset/iiit_dataset/images_val/images_val"

# Other settings
BATCH_SIZE = 32 # Can be larger for evaluation if memory allows
NUM_WORKERS = 4

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Best model file not found at: {BEST_MODEL_PATH}\nPlease update the BEST_MODEL_PATH variable.")


# ===============================
# Dataset Definition (Outputs cos/sin) - Copied from training script
# ===============================
class CampusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform

        valid_rows = []
        print("Verifying image files for validation set...")
        # Simplified check for prediction script - assumes files were checked during training
        self.df = dataframe
        # Optional: Add back the file existence check if needed for prediction robustness
        # for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Checking files"):
        #     image_path = os.path.join(self.image_dir, row['filename'])
        #     if os.path.isfile(image_path):
        #         valid_rows.append(row)
        # self.df = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Handle error: return a dummy item or skip (here returning None)
            # Note: DataLoader's default collate_fn might need adjustment if returning None
            # For simplicity, we'll rely on the check during training or let it error here.
            if idx == 0: raise e
            return self.__getitem__(0) # Re-try with first item (not ideal)

        image = self.transform(image)

        # Target angle in degrees
        angle_deg = row['angle']
        # Convert target angle to radians
        angle_rad = np.deg2rad(angle_deg)
        # Target is now a tensor [cos(angle), sin(angle)]
        target = torch.tensor([np.cos(angle_rad), np.sin(angle_rad)], dtype=torch.float32)
        # Return filename and original angle as well for potential analysis
        return {"pixel_values": image, "filename": row['filename'], "true_angle_deg": angle_deg}, target

# ===============================
# EfficientNet Model for Sin/Cos Regression - Copied from training script
# ===============================
class EfficientNetSinCosModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load architecture definition WITHOUT pretrained weights here,
        # we will load our trained weights later.
        self.backbone = efficientnet_b0(weights=None) # Load architecture only

        num_features = self.backbone.classifier[1].in_features
        # Modify the classifier head exactly as in training
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), # Match dropout used in training
            nn.Linear(num_features, 2) # Output dimension is 2
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
    # minimum(abs(pred-target), 360 - abs(pred-target))
    angular_errors = torch.minimum(diff, 360.0 - diff)
    return torch.mean(angular_errors) # Return mean over the batch

# ===============================
# Data Transforms - Copied from training script (using val_transform)
# ===============================
weights = EfficientNet_B0_Weights.IMAGENET1K_V1 # Needed for stats
imagenet_stats = weights.transforms().mean, weights.transforms().std
input_size = 224

val_transform = transforms.Compose([
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

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from: {BEST_MODEL_PATH}")
    model = EfficientNetSinCosModel().to(device)
    # Load the saved state dictionary
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode (important!)
    print("Model loaded successfully.")

    # Load Validation Data
    print("Loading validation data...")
    val_df = pd.read_csv(VAL_CSV_PATH)
    val_dataset = CampusDataset(val_df, VAL_IMG_DIR, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Loaded {len(val_dataset)} validation samples.")

    # Prediction and MAAE Calculation
    print("Running predictions on validation set...")
    all_pred_angles_deg = []
    all_target_angles_deg = []
    total_angular_error_sum = 0.0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch in tqdm(val_loader, desc="Predicting"):
            inputs_dict = batch[0] # Dataset returns (dict, tensor)
            targets_cos_sin = batch[1].to(device) # Targets are [cos, sin]
            pixel_values = inputs_dict["pixel_values"].to(device)
            batch_true_angles_deg = inputs_dict["true_angle_deg"] # Get true angles directly

            # Get model predictions (cos, sin)
            preds_cos_sin = model(pixel_values) # Shape: [batch, 2]

            # --- Convert predictions back to angles (degrees) ---
            pred_angle_rad = torch.atan2(preds_cos_sin[:, 1], preds_cos_sin[:, 0]) # Output is [-pi, pi]
            pred_angle_deg = torch.rad2deg(pred_angle_rad) # Output is [-180, 180]
            pred_angle_deg = (pred_angle_deg + 360.0) % 360.0 # Map to [0, 360]

            # --- Calculate angular difference for the batch ---
            # Ensure batch_true_angles_deg is a tensor on the same device for calculation
            batch_true_angles_deg_tensor = batch_true_angles_deg.clone().detach().to(device=pred_angle_deg.device, dtype=pred_angle_deg.dtype)

            diff = torch.abs(pred_angle_deg - batch_true_angles_deg_tensor)
            batch_angular_errors = torch.minimum(diff, 360.0 - diff)

            # --- Accumulate results ---
            total_angular_error_sum += torch.sum(batch_angular_errors).item()
            total_samples += pixel_values.size(0) # Add number of samples in the batch

            # Optional: Store individual predictions/targets if needed later
            all_pred_angles_deg.extend(pred_angle_deg.cpu().tolist())
            all_target_angles_deg.extend(batch_true_angles_deg.tolist())


    # Calculate final MAAE
    if total_samples > 0:
        final_maae = total_angular_error_sum / total_samples
        print("\n===================================")
        print(f"Prediction complete.")
        print(f"Total validation samples processed: {total_samples}")
        print(f"Mean Absolute Angular Error (MAAE): {final_maae:.4f} degrees")
        print("===================================")
    else:
        print("No samples were processed. Check data loading.")

    # Optional: Save predictions to a file
    if all_pred_angles_deg and all_target_angles_deg:
        results_df = pd.DataFrame({
            'filename': val_df['filename'][:len(all_pred_angles_deg)], # Ensure alignment if dataset had errors
            'true_angle': all_target_angles_deg,
            'predicted_angle': all_pred_angles_deg
        })
        results_df['angular_error'] = results_df.apply(lambda row: min(abs(row['predicted_angle'] - row['true_angle']), 360 - abs(row['predicted_angle'] - row['true_angle'])), axis=1)
        results_save_path = os.path.join(os.path.dirname(BEST_MODEL_PATH), "validation_predictions.csv")
        results_df.to_csv(results_save_path, index=False)
        print(f"Prediction results saved to: {results_save_path}")