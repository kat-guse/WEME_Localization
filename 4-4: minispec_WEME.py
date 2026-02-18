import pandas as pd
import numpy as np
import json
import shelve
from pathlib import Path
import librosa
from joblib import Parallel, delayed
from opensoundscape import Audio, Spectrogram

# ==========================================
# CONFIG
# ==========================================
csv_path        = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/OKLG-8_Western_Meadowlark_CLEANED.csv"
clip_dir        = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/minspec_clips"
out_dir         = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/minspec_output"
confirmed_shelf = r"/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/weml_confirmed.out"
hawkears_label_prefix = "WEML"

Path(clip_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)
Path(confirmed_shelf).parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# POSITION WRAPPER CLASS
# ==========================================
class CsvPositionEstimate:
    def __init__(self, row):
        # Plain scalars
        self.class_name      = row["class_name"]
        self.start_timestamp = row["start_timestamp"]
        self.duration        = float(row["duration"])
        self.mean_residual   = float(row["mean_residual"])
        self.residual_rms    = float(row["residual_rms"])
        self.mean_cc_max     = float(row["mean_cc_max"])

        # Location — already flat columns
        self.location_estimate = np.array([row["pred_x"], row["pred_y"], row["pred_z"]])

        # Arrays — clean json.loads
        self.receiver_files              = json.loads(row["receiver_files"])
        self.receiver_locations          = np.array(json.loads(row["receiver_locations"]))
        self.receiver_start_time_offsets = np.array(json.loads(row["receiver_start_time_offsets"]))
        self.tdoas                       = np.array(json.loads(row["tdoas"]))
        self.cc_maxs                     = np.array(json.loads(row["cc_maxs"]))
        self.distance_residuals          = np.array(json.loads(row["distance_residuals"]))

    def load_aligned_audio_segments(self):
        clips = []
        for filepath, offset_sec, tdoa in zip(
            self.receiver_files,
            self.receiver_start_time_offsets,
            self.tdoas,
        ):
            start = float(offset_sec) + float(tdoa)
            clips.append(Audio.from_file(filepath, offset=start, duration=self.duration))
        return clips

# ==========================================
# LOAD CSV
# ==========================================
df_raw = pd.read_csv(csv_path)
position_estimates = [CsvPositionEstimate(row) for _, row in df_raw.iterrows()]
print(f"Loaded {len(position_estimates)} positions from CSV")

# ==========================================
# FILTER
# ==========================================
positions = [p for p in position_estimates if p.residual_rms < 20]
print(f"Processing {len(positions)} positions after residual_rms < 20 filter")

# ==========================================
# MINSPEC GENERATION
# ==========================================
def spec_to_audio(spec, sr):
    y_inv = librosa.griffinlim(spec.spectrogram, hop_length=256, win_length=512)
    return Audio(y_inv, sr)

def distances_to_receivers(p, dims=2):
    return [
        np.linalg.norm(p.location_estimate[:dims] - r[:dims])
        for r in p.receiver_locations
    ]

def min_spec_to_audio(position, discard_over_distance=50):
    clips     = position.load_aligned_audio_segments()
    distances = distances_to_receivers(position)
    clips     = [c for i, c in enumerate(clips) if distances[i] < discard_over_distance]
    
    if len(clips) == 0:
        return None
    
    specs     = [Spectrogram.from_audio(c, dB_scale=False) for c in clips]
    minspec   = specs[0]._spawn(
        spectrogram=np.min(np.array([s.spectrogram for s in specs]), axis=0)
    )
    max_val = np.max([c.samples.max() for c in clips])
    return (
        spec_to_audio(minspec, clips[0].sample_rate)
        .normalize(max_val)
        .extend_to(clips[0].duration)
    )

def process(p, i):
    try:
        audio = min_spec_to_audio(p, discard_over_distance=35)
        if audio is None:
            return 1
        audio.save(f"{clip_dir}/{i}.wav")
        return 0
    except Exception as e:
        print(f"  clip {i} failed: {e}")
        return 1

results = Parallel(n_jobs=4)(delayed(process)(p, i) for i, p in enumerate(positions))
print(f"Failures: {sum(results)} of {len(results)}")

# ==========================================
# RUN HAWKEARS (manual step)
# ==========================================
print("\nTo run HawkEars detection on the clips:")
print(f"python /path/to/HawkEars/analyze.py -i \"{clip_dir}\" -o \"{out_dir}\"")
print("\nAfter HawkEars finishes, run parse_hawkears.py\n")
