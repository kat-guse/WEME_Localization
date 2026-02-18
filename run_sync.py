
import os
import pandas as pd
from opensoundscape.audio import Audio

# import YOUR functions
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
import audiomothsync as amsync

# ---------------- USER SETTINGS ----------------
input_folder = "/media/UofA/BU_Work/BayneLabWorkSpace/2025DataIntake/Sorted/OKLG/Grids/OKLG-8"
output_folder = "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/OKLG-8-Sync"

desired_sr = 48000
expected_sr = 48000
# -----------------------------------------------

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    wav_files = [f for f in files if f.upper().endswith(".WAV")]

    if not wav_files:
        continue

    # Preserve subfolder structure
    rel_path = os.path.relpath(root, input_folder)
    out_dir = os.path.join(output_folder, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for wav_name in sorted(wav_files):
        try:
            wav_path = os.path.join(root, wav_name)
            csv_path = wav_path.replace(".WAV", ".CSV")

            if not os.path.exists(csv_path):
                print(f"⚠ Missing CSV for {wav_path}")
                continue

            print(f"\nProcessing {wav_path}")

            audio = Audio.from_file(wav_path)
            pps_table = pd.read_csv(csv_path, index_col=0)

            sample_timestamp_df = amsync.associate_pps_samples_timestamps(
                pps_table,
                expected_sr=expected_sr
            )

            passed, qc = amsync.check_pps_quality(
                sample_timestamp_df,
                expected_sr=expected_sr
            )

            if not passed:
                print(f"Skipping {wav_path}: {qc.get('fail', 'QC failed')}")
                continue

            synced_audio = amsync.correct_sample_rate(
                audio,
                sample_timestamp_df,
                desired_sr=desired_sr,
                expected_sr=expected_sr,
                interpolation_method="nearest"
            )

            out_wav = os.path.join(
                out_dir,
                wav_name.replace(".WAV", "_synced.WAV")
            )

            synced_audio.save(out_wav)

            pd.DataFrame([qc]).to_csv(
                out_wav.replace(".WAV", "_qc.csv"),
                index=False
            )

            sample_timestamp_df.to_csv(
                out_wav.replace(".WAV", "_timestamps.csv")
            )

            print("  Start time:", synced_audio.metadata["recording_start_time"])
            print("  Sample rate:", synced_audio.sample_rate)

        except Exception as e:
            print(f"💥 Error processing {wav_path}: {e}")
            continue

print("\nAll files processed.")
