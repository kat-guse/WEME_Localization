import pandas as pd
import numpy as np
import os
import json
from opensoundscape.localization.spatial_event import localize_events_parallel
from spatial_utils import SynchronizedRecorderArray, calc_speed_of_sound
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. SPECIES CONFIGURATION
# ==========================================
SPECIES_TO_PROCESS = "Western Meadowlark"

CONFIG = {
    "Western Meadowlark": {
        "bandpass": [2000, 8000],
        "min_channels": 4,
        "dist_filter": 60,
        "temp_window": 2.0
    }
}

# ==========================================
# 2. PATHS
# ==========================================
coords_path  = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/RTK_Coordinates - BC Alberts (EPSG-2955).csv'
det_path     = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/Western_Meadowlark_detections_with_BI_filtering.csv'
output_base  = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/'

species_slug    = SPECIES_TO_PROCESS.replace(' ', '_')
unfiltered_path = os.path.join(output_base, f"OKLG-8_{species_slug}_UNFILTERED.csv")
diag_dir        = os.path.join(output_base, f"diagnostics_{species_slug}")
os.makedirs(diag_dir, exist_ok=True)

# ==========================================
# 3. LOAD COORDINATES
# ==========================================
all_coords = pd.read_csv(coords_path)
grid_8_coords = all_coords[all_coords['localization_grid'] == 'OKLG-8'].copy()
grid_8_coords['device_id'] = grid_8_coords['device_id'].str.split('-').str[-1]
grid_8_coords = grid_8_coords.set_index('device_id')
grid_8_coords = grid_8_coords.rename(columns={
    'ground_truth_easting': 'x',
    'ground_truth_northing': 'y',
    'orthometric_height': 'z'
})[['x', 'y', 'z']]

# ==========================================
# 4. LOAD DETECTIONS AND CREATE EVENTS
# ==========================================
current_speed = calc_speed_of_sound(temperature=25)
array = SynchronizedRecorderArray(file_coords=grid_8_coords, speed_of_sound=current_speed)

df = pd.read_csv(det_path)
df = df[df['original_start_s'] > 2.0]

MAX_FILE_DURATION = 1197
initial_count = len(df)
df = df[df['original_start_s'] < MAX_FILE_DURATION].copy()
print(f"Removed {initial_count - len(df)} detections exceeding the 20-minute window.")

spec_cfg = CONFIG[SPECIES_TO_PROCESS]
events = array.create_candidate_events(
    df,
    species_name=SPECIES_TO_PROCESS,
    min_n_receivers=spec_cfg["min_channels"],
    max_receiver_dist=spec_cfg["dist_filter"],
    temporal_window=spec_cfg["temp_window"],
    bandpass_range=spec_cfg["bandpass"]
)

if len(events) == 0:
    raise SystemExit("✗ No candidate events found. Check detections or temporal_window.")

# ==========================================
# 5. LOCALIZE
# ==========================================
def arr_to_json(x):
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    return json.dumps(list(x))

def position_to_row(p):
    """
    Convert a PositionEstimate to a dict row.
    Returns None if the localization failed (non-finite coordinates).
    """
    loc       = np.asarray(p.location_estimate)
    residuals = np.asarray(p.distance_residuals)
    
    # Discard failed localizations where solver returned NaN/Inf
    if not np.all(np.isfinite(loc)):
        return None
    if not np.all(np.isfinite(residuals)):
        return None
    
    # Also discard obviously wrong z-coordinates (more than 2km from ARU median)
    # This catches cases where the solver converged to nonsense
    z_median = 690  # your site elevation (could compute from grid_8_coords)
    if abs(loc[2] - z_median) > 2000:
        return None

    return {
        "class_name":                  p.class_name,
        "start_timestamp":             str(p.start_timestamp),
        "duration":                    float(p.duration),
        "pred_x":                      float(loc[0]),
        "pred_y":                      float(loc[1]),
        "pred_z":                      float(loc[2]),
        "mean_residual":               float(np.mean(np.abs(residuals))),
        "residual_rms":                float(np.sqrt(np.mean(residuals ** 2))),
        "mean_cc_max":                 float(np.mean(p.cc_maxs)),
        "receiver_files":              arr_to_json(p.receiver_files),
        "receiver_locations":          arr_to_json(p.receiver_locations),
        "receiver_start_time_offsets": arr_to_json(p.receiver_start_time_offsets),
        "tdoas":                       arr_to_json(p.tdoas),
        "cc_maxs":                     arr_to_json(p.cc_maxs),
        "distance_residuals":          arr_to_json(residuals),
        "location_estimate":           arr_to_json(loc),
    }

print(f"Localizing {len(events)} candidate events...")
localized_positions = localize_events_parallel(
    events,
    num_workers=4,
    localization_algorithm="least_squares"
)

if not localized_positions:
    raise SystemExit("⚠ Localization returned no results.")

# Convert to rows, filtering out failed localizations
rows = [position_to_row(p) for p in localized_positions]
n_failed = sum(1 for r in rows if r is None)
rows = [r for r in rows if r is not None]
results_df = pd.DataFrame(rows)

print(f"  Successfully localized: {len(localized_positions)}")
print(f"  Valid (finite coords):  {len(results_df)}")
print(f"  Failed/discarded:       {n_failed}")

if results_df.empty:
    raise SystemExit("⚠ All localizations failed validity check.")

# Save unfiltered valid results
results_df.to_csv(unfiltered_path, index=False)
print(f"  Saved to: {unfiltered_path}\n")

print("=" * 50)
print("Run the interactive filter script to review plots")
print("and choose quality thresholds:")
print("  python filter_localizations.py")
print("=" * 50)
