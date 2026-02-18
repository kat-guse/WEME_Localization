import os
import pandas as pd
import numpy as np
import json
import shelve
from pathlib import Path
import torch
import bioacoustics_model_zoo as bmz


# Hardware Setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')
print(f"Using device: {device}")

# ==========================================
# CONFIG
# ==========================================
csv_path        = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/OKLG-8_Western_Meadowlark_CLEANED.csv"
clip_dir        = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/minspec_clips"
out_dir         = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/minspec_output"
confirmed_shelf = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/weml_confirmed.out"
hawkears_species = "Western Meadowlark"  # or check what bmz calls it

Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(confirmed_shelf).parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# RELOAD POSITIONS
# ==========================================
class CsvPositionEstimate:
    def __init__(self, row):
        self.class_name      = row["class_name"]
        self.start_timestamp = row["start_timestamp"]
        self.duration        = float(row["duration"])
        self.mean_residual   = float(row["mean_residual"])
        self.residual_rms    = float(row["residual_rms"])
        self.mean_cc_max     = float(row["mean_cc_max"])
        self.location_estimate = np.array([row["pred_x"], row["pred_y"], row["pred_z"]])
        self.receiver_files              = json.loads(row["receiver_files"])
        self.receiver_locations          = np.array(json.loads(row["receiver_locations"]))
        self.receiver_start_time_offsets = np.array(json.loads(row["receiver_start_time_offsets"]))
        self.tdoas                       = np.array(json.loads(row["tdoas"]))
        self.cc_maxs                     = np.array(json.loads(row["cc_maxs"]))
        self.distance_residuals          = np.array(json.loads(row["distance_residuals"]))

df_raw = pd.read_csv(csv_path)
position_estimates = [CsvPositionEstimate(row) for _, row in df_raw.iterrows()]
positions = [p for p in position_estimates if p.residual_rms < 20]
print(f"Loaded {len(positions)} positions")

# ==========================================
# RUN HAWKEARS VIA BMZ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading HawkEars on {device}...")

model = bmz.HawkEars()
model.to(device)

# Get all minspec clips
clip_files = sorted([str(f) for f in Path(clip_dir).glob("*.wav")])
print(f"Found {len(clip_files)} minspec clips to analyze")

if len(clip_files) > 0:
    # Run predictions
    scores = model.predict(
        clip_files,
        activation_layer='sigmoid',
        batch_size=8,
        num_workers=4
    )
    
    # Save raw scores
    scores.to_csv(Path(out_dir) / "hawkears_scores.csv")
    print(f"Saved raw scores to {out_dir}/hawkears_scores.csv")
    
    # Find the Western Meadowlark column
    meadowlark_cols = [c for c in scores.columns if 'Meadowlark' in c and 'Western' in c]
    
    if not meadowlark_cols:
        print("\nAvailable species columns:")
        print(scores.columns.tolist())
        raise ValueError("Could not find Western Meadowlark column in HawkEars output")
    
    target_col = meadowlark_cols[0]
    print(f"Using column: {target_col}")
    
    # Filter detections above threshold
    threshold = 0.4  # adjust based on your needs
    verified_df = scores[scores[target_col] >= threshold].copy()
    print(f"\nDetections above {threshold} threshold: {len(verified_df)}")
    
    # Extract clip indices from file paths
    verified_indices = set()
    for filepath in verified_df.index.get_level_values(0):
        clip_idx = int(Path(filepath).stem)
        verified_indices.add(clip_idx)
    
    # Map back to positions
    confirmed_positions = [p for i, p in enumerate(positions) if i in verified_indices]
    print(f"Final confirmed positions: {len(confirmed_positions)}")
    
    # Save to shelve
    with shelve.open(confirmed_shelf, "n") as db:
        db["position_estimates"] = confirmed_positions
    print(f"\n✓ Saved {len(confirmed_positions)} confirmed positions to {confirmed_shelf}")
    
else:
    print("No clips found to analyze.")
