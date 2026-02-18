import bioacoustics_model_zoo as bmz
import opensoundscape as ocs
from opensoundscape.audio import Audio
import torch
import pandas as pd
import os, glob, inspect
import sqlite3
from datetime import datetime, timedelta

# Hardware Setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')
print(f"Using device: {device}")

#---CONFIGURATION-----
audio_directory = "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/OKLG-8-Sync/"
output_directory = "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/"
os.makedirs(output_directory, exist_ok=True)
target_suffix = "20250616_120000_synced.WAV"

#---SPECIES OF INTEREST AND THRESHOLDS-------
TARGET_SPECIES = [
    'Western Meadowlark'
]
SPECIES_THRESHOLDS = [
    0.15
]

if len(TARGET_SPECIES) != len(SPECIES_THRESHOLDS):
    raise ValueError("Species/threshold length mismatch")
SPECIES_THRESHOLD_MAP = dict(zip(TARGET_SPECIES, SPECIES_THRESHOLDS))
print(f"\n--- Species Configuration ---")
print(f"Target Species: {TARGET_SPECIES}")
print(f"Thresholds: {SPECIES_THRESHOLD_MAP}")

#---FILE DISCOVERY----------
print(f"Searching for files ending in {target_suffix} in {audio_directory}")
all_wavs = glob.glob(os.path.join(audio_directory, "**/*.wav"), recursive=True)
all_wavs += glob.glob(os.path.join(audio_directory, "**/*.WAV"), recursive=True)
audio_files = [f for f in all_wavs if f.endswith(target_suffix)]
if not audio_files:
    print("No matching files found.")
    exit()
print(f"Found {len(audio_files)} files.")

#---FUNCTIONS--------------
def run_prediction_and_save(model_name, audio_files, output_dir, target_species):
    print(f"\n--- Running predictions with {model_name} ---")
    model_class = getattr(bmz, model_name)
    model = model_class()
    
    # ---- CACHE FILES ----
    raw_cache_csv = os.path.join(output_dir, f"{model_name.lower()}_cache_raw_predictions.csv")
    filtered_cache_csv = os.path.join(output_dir, f"{model_name.lower()}_cache_detections_targeted.csv")
    
    # ---- LOAD CACHE IF EXISTS ----
    if os.path.exists(raw_cache_csv):
        print(f"Loading cached predictions from {raw_cache_csv}")
        scores = pd.read_csv(raw_cache_csv, index_col=[0, 1, 2])
    else:
        print("No cache found — running model prediction (this is the slow part)")
        scores = model.predict(
            audio_files,
            activation_layer='sigmoid',
            overlap_fraction=0.5
        )
        scores.to_csv(raw_cache_csv)
        print(f"Saved raw predictions to {raw_cache_csv}")
    
    return scores, raw_cache_csv

def extract_and_format(scores_df, target_species, species_threshold_map, output_dir):
    """Filter predictions for target species with exact timestamps"""
    print(f"\n--- Filtering for target species with timestamps ---")
    
    # --RESET INDEX-----
    scores_reset = scores_df.reset_index()
    
    # ---FILTER FOR TARGET SPECIES ABOVE THRESHOLD----
    detections = []
    for species in target_species:
        if species in scores_reset.columns:
            threshold = species_threshold_map[species]
            species_detections = scores_reset[scores_reset[species] >= threshold].copy()
            species_detections['detected_species'] = species
            species_detections['confidence'] = species_detections[species]
            species_detections['threshold'] = threshold
            detections.append(species_detections)
    
    if not detections:
        print("No detections found for target species")
        return pd.DataFrame()
    
    # Combine all detections
    all_detections = pd.concat(detections, ignore_index=True)
    
    # Select relevant columns
    result_columns = ['file', 'start_time', 'end_time', 'detected_species', 'confidence', 'threshold']
    result_df = all_detections[result_columns]

    # --- ADD UTC TIMESTAMPS ---
    def parse_file_start(file_path):
        # Example filename: OKLG-8-A1_20250616_120000_synced.WAV
        filename = os.path.basename(file_path)
        try:
            parts = filename.split('_')
            date_str, time_str = parts[-3], parts[-2]  # '20250616', '120000'
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
            return dt
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None

    # Add file_start_utc column
    result_df['file_start_utc'] = result_df['file'].apply(parse_file_start)

    # Add clip_start_utc and clip_end_utc columns
    result_df['clip_start_utc'] = result_df.apply(
        lambda row: row['file_start_utc'] + timedelta(seconds=row['start_time'])
        if row['file_start_utc'] is not None else None,
        axis=1
    )
    result_df['clip_end_utc'] = result_df.apply(
        lambda row: row['file_start_utc'] + timedelta(seconds=row['end_time'])
        if row['file_start_utc'] is not None else None,
        axis=1
    )
    
    # ---SORT BY FILE AND START_TIME----
    result_df = result_df.sort_values(['file', 'start_time'])
    
    # ----SAVE TO CSV----
    output_csv = os.path.join(output_dir, "detections_with_timestamps.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {len(result_df)} detections to {output_csv}")
    
    # ---PRINT SUMMARY----
    print(f"\n--- Detection Summary ---")
    for species in target_species:
        count = len(result_df[result_df['detected_species'] == species])
        print(f"{species}: {count} detections")
    
    return result_df

#----MAIN EXECUTION-----
model_name = "HawkEars"
scores, cache_file = run_prediction_and_save(model_name, audio_files, output_directory, TARGET_SPECIES)

# Filter and extract timestamps
detections_df = extract_and_format(
    scores, 
    TARGET_SPECIES, 
    SPECIES_THRESHOLD_MAP, 
    output_directory
)

# Display sample results
if not detections_df.empty:
    print("\n--- Sample Detections ---")
    print(detections_df.head(10))
    print(f"\nTotal detections: {len(detections_df)}")        


    

    
