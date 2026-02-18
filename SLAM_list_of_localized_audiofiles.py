import pandas as pd
import json
from pathlib import Path

#-----list of audio_files of only localized events -----
events_df = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/localized_events.csv')

# Extract all unique file_ids that participated in localizations
used_file_ids = set()
for file_ids_json in events_df['file_ids']:
    file_ids = json.loads(file_ids_json)
    used_file_ids.update(file_ids)

print(f"Number of unique audio files used in localizations: {len(used_file_ids)}")

# Load full audio file table
audio_df = pd.read_csvaudio_df = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/audio_file_table.csv')


# Filter to only files used in localizations
audio_df_filtered = audio_df[audio_df['file_id'].isin(used_file_ids)].copy()

print(f"Reduced from {len(audio_df)} to {len(audio_df_filtered)} files")

output_path = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/audio_file_table_filtered.csv'
audio_df_filtered.to_csv(output_path, index=False)
print(f"✓ Saved filtered table to: {output_path}")

##verifications
import pandas as pd
import json
from collections import Counter

events_df = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/SLAM_dataset/localized_events.csv')

# Count how many times each file appears
file_usage = Counter()
for file_ids_json in events_df['file_ids']:
    file_ids = json.loads(file_ids_json)
    for fid in file_ids:
        file_usage[fid] += 1

print(f"Total unique files: {len(file_usage)}")
print(f"Total events: {len(events_df)}")
print(f"\nMost-used files:")
for file_id, count in file_usage.most_common(10):
    print(f"  {file_id}: used in {count} events")

# Check unique timestamps
unique_timestamps = events_df['start_timestamp'].nunique()
print(f"\nUnique event timestamps: {unique_timestamps}")
print(f"Total events: {len(events_df)}")
print(f"Average events per timestamp: {len(events_df) / unique_timestamps:.1f}")
