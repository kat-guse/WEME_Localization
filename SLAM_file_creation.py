import pandas as pd
from pathlib import Path
import re
import json
import shelve
import numpy as np

#Reformat localized_events.csv
# Load your current data
df = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/weml_confirmed_locations.csv')

class CsvPositionEstimate:
    # ... (same definition as before)
    pass

shelf_path = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/pythonoutput/weml_confirmed.out'
with shelve.open(shelf_path, 'r') as db:
    positions = db['position_estimates']

# Create SLAM-formatted table
slam_events = []
for i, p in enumerate(positions):
    # Generate event_id
    event_id = f"OKLG8_WEML_{i:04d}"
    
    # Extract file_ids from receiver_files paths
    file_ids = [f.split('/')[-1].replace('.WAV', '').replace('_synced', '') for f in p.receiver_files]
    
    slam_events.append({
        'event_id': event_id,
        'label': 'Western Meadowlark',
        'start_timestamp': p.start_timestamp,
        'duration': p.duration,
        'utm_x': p.location_estimate[0],
        'utm_y': p.location_estimate[1],
        'elevation': p.location_estimate[2],
        'utm_zone': '11U',  # Alberta is in UTM zone 11U for NAD83
        'file_ids': json.dumps(file_ids),
        'file_start_time_offsets': json.dumps(p.receiver_start_time_offsets.tolist()),
        'tdoas': json.dumps(p.tdoas.tolist()),
        'distance_residuals': json.dumps(p.distance_residuals.tolist()),
        'residual_rms': p.residual_rms,
        'mean_cc_max': p.mean_cc_max,
        'hawkears_confirmed': True
    })

slam_df = pd.DataFrame(slam_events)
slam_df.to_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/localized_events.csv', index=False)
print(f"Created {len(slam_df)} SLAM-formatted events")

# Create audio file table
audio_base = Path('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/OKLG-8-Sync')

audio_files = []
for recorder_dir in audio_base.glob('OKLG-8-*'):
    point_id = recorder_dir.name  # e.g., 'OKLG-8-A1'
    
    for wav_file in recorder_dir.glob('*_synced.WAV'):
        # Extract timestamp from filename: OKLG-8-A1_20250616_120000_synced.WAV
        match = re.search(r'(\d{8})_(\d{6})', wav_file.name)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            timestamp = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}+00:00"
            
            audio_files.append({
                'file_id': wav_file.stem.replace('_synced', ''),
                'relative_path': str(wav_file.relative_to(audio_base.parent)),
                'point_id': point_id,
                'start_timestamp': timestamp
            })

audio_df = pd.DataFrame(audio_files)
audio_df.to_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/audio_file_table.csv', index=False)
print(f"Created audio file table with {len(audio_df)} files")

# ----Reformat point_table.csv----
coords = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/RTK_Coordinates - BC Alberts (EPSG-2955).csv')
grid_8 = coords[coords['localization_grid'] == 'OKLG-8'].copy()

point_table = grid_8[['device_id', 'ground_truth_easting', 'ground_truth_northing', 'orthometric_height']].rename(columns={
    'device_id': 'point_id',
    'ground_truth_easting': 'utm_easting',
    'ground_truth_northing': 'utm_northing',
    'orthometric_height': 'elevation'
})
point_table['utm_zone'] = '11U'
point_table['array'] = 'OKLG-8'

point_table.to_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/point_table.csv', index=False)

# ----Create classes.csv-----
classes = pd.DataFrame([{
    'class': 'Western Meadowlark',
    'scientific_name': 'Sturnella neglecta',
    'description': 'Western Meadowlark vocalizations detected and confirmed by HawkEars classifier',
    'vocalization_type': 'song and calls'
}])
classes.to_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/classes.csv', index=False)


