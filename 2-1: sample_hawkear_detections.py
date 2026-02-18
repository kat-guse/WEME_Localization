#importing packages currently installed in the venv
import bioacoustics_model_zoo as bmz
import opensoundscape as ocs
from opensoundscape.audio import Audio
import torch
import pandas as pd
import os, glob, inspect

#TO USE BEASTMASTER COMPUTER CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')
print(f"Using device: {device}")

"""
Phase 1: Run Point Counts through Hawkears and format to ensure I can use the
validation app created by Megan Edgar.
"""

#directory
audio_directory = "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/"
output_directory = "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/PC_Validation_1234567/"
os.makedirs(output_directory, exist_ok=True)

timestamps = [
    "20250516_152000", "20250516_182000", "20250517_160000",
    "20250522_154000", "20250522_174000", "20250604_163000",
    "20250531_153000", "20250531_160000", "20250602_163000",
    "20250609_170000", "20250608_153000", "20250609_150000",
    "20250610_153000"
]

audio_files = []
for ts in timestamps:
    pattern = os.path.join(audio_directory, f"OKLG-*-Sync/**/*{ts}_synced.*")
    found = glob.glob(pattern, recursive=True)
    audio_files += found

audio_files = sorted(set(audio_files))
print(f"Unique files total: {len(audio_files)}")
confidence_threshold = 0.05

if not audio_files:
    print("No audio files found in the specified directory.")
    exit()

print(f"Found {len(audio_files)} audio files.")

def run_prediction_and_save(model_name, audio_files, output_dir):
    print(f"\n--- Running predictions with {model_name} ---")
    model_class = getattr(bmz, model_name)
    model = model_class()
    model.to(device) 

    output_csv = os.path.join(output_dir, f"{model_name.lower()}_detections.csv")
    print(f"Analyzing {len(audio_files)} files...")
    
    scores = model.predict(
        audio_files, 
        activation_layer='sigmoid', 
        overlap_fraction=0.5
    )

    scores.to_csv(output_csv, index=True)
    print(f"Raw detection scores saved to {output_csv}")
    return scores, output_csv

def format_for_validator(scores_df, output_dir):
    # Ensure the first column (file path) is available to be processed
    df = scores_df.reset_index()
    if 'file' not in df.columns and 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'file'})

    # 1. Strip the directory path for portability
    # This creates the clean "OKLG-6-D5_..." format
    df['filename'] = df['file'].apply(lambda x: os.path.basename(str(x)))

    # 2. Rename columns
    df = df.rename(columns={'clip_start_s': 'start_time', 'clip_end_s': 'end_time'})

    # 3. Wide → long
    long_df = df.melt(
        id_vars=['filename', 'start_time', 'end_time'],
        var_name='class_name',
        value_name='score'
    )

    # --- THE FIX: Force score to numeric to prevent TypeError ---
    long_df['score'] = pd.to_numeric(long_df['score'], errors='coerce')

    # Exclusion list
    ignore_list = [
        'American Bullfrog', 'American Toad', 'Canadian Toad', 'Canine', 
        'Boreal Chorus Frog', 'Gray Treefrog', 'Great Plains Toad', 
        'Green Frog', 'Northern Leopard Frog', 'Mashup', 'Noise', 'Other', 
        'Pickerel Frog', 'Plains Spadefoot Toad', 'Rooster', 'Spring Peeper', 
        'Squirrel', 'Western Toad', 'Wood Frog'
    ]
    
    # Filter: Keep only scores above threshold AND NOT in the non_bird list
    # We drop any rows that became NaN during the numeric conversion
    long_df = long_df.dropna(subset=['score'])
    long_df = long_df[
        (~long_df['class_name'].isin(ignore_list)) & 
        (long_df['score'] >= confidence_threshold)
    ].copy()

    # 5. Final Formatting
    long_df['class_code'] = long_df['class_name']
    long_df = long_df[['filename', 'start_time', 'end_time', 'class_name', 'class_code', 'score']]

    output_csv = os.path.join(output_dir, "HawkEars_labels1.csv")
    long_df.to_csv(output_csv, index=False)

    print(f"Validator-ready CSV written to: {output_csv}")
    print(f"Total Bird Detections: {len(long_df)}")
    
    return long_df

# --- Main workflow execution ---
scores_hawkears, _ = run_prediction_and_save(
    'HawkEars',
    audio_files,
    output_directory
)

validator_df = format_for_validator(
    scores_hawkears,
    output_directory
)

print("\n--- Final Verification ---")
print("First 5 filenames in output:")
print(validator_df['filename'].head().to_list())
print("\nAll model pipelines completed!")
