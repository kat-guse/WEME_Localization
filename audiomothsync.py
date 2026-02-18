#Using opensoundscape - audiomothsync.py example 
import warnings
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
import pytz

def parse_RMC_time_to_seconds(timestamp: str) -> float:
    stamp_format = "%Y-%m-%dT%H:%M:%S.%f"
    return datetime.strptime(timestamp, stamp_format).timestamp()

def parse_RMC_time_to_datetime(timestamp: str) -> datetime:
    """
    Takes in the GPS-audiomoth formatted timestamp and returns datetime.datetime
    localized with UTC timezone. Assumes the input time string is given in UTC.

    Args:
        timestamp: string in the format used by our GPS-audiomoths
                '%Y-%m-%dT%H:%M:%S.%f' e.g. "2022-05-14T10:30:00.725"
    Returns: datetime object localized to UTC timezone
    """
    stamp_format = "%Y-%m-%dT%H:%M:%S.%f"
    tz = pytz.timezone("UTC")
    return tz.localize(datetime.strptime(timestamp, stamp_format))

def associate_pps_samples_timestamps(
    pps_table,
    expected_sr=48000,
    sr_tol=150,
    cpu_clock_counter_col="TIMER_COUNT"
) -> pd.DataFrame:
    
    samples_written = []
    gps_times_str = []
    gps_times_sec = []
    pps_numbers_out = []
    counter_positions = []
    pps_pings_with_no_RMC = []
    #Buffer overflow check
    final_row = pps_table.iloc[-1]
    assert (
        final_row["BUFFERS_FILLED"] - final_row["BUFFERS_WRITTEN"] < 8
    ), "Buffer overflow detected, WAV file has missing content"

    pps_numbers = pps_table.index

    # PPS Duplicate Check: check for duplicates in PPS number
    if pps_numbers.duplicated().any():
        raise ValueError("PPS numbers are duplicated in this PPS table")

    #Sequential PPS Check: check that PPS numbers are sequential from first to last
    assert np.array_equal(
        pps_numbers.values,
        list(range(pps_numbers[0], pps_numbers[-1] + 1))
    ), "PPS Numbers were not sequential integers"

    for i, pps_number in enumerate(pps_numbers):
        if i == 0:  #: # if first pps ping, ignore
            # TOTAL_SAMPLES starts at zero check
            assert pps_table.loc[pps_numbers[i], "TOTAL_SAMPLES"] == 0
            continue
        
        if pd.isnull(pps_table.loc[pps_numbers[i], "LAST_RMC_AUDIOMOTH_TIME"]):
            # no RMC data in this row
            continue
        #DEFINE ROWS:
        row1 = pps_table.loc[pps_numbers[i]]
        row0 = pps_table.loc[pps_numbers[i-1]]

        #GPS lock check
        if row1["STATUS"] != "A":
            continue

        rmc_gps_time_str = row1["LAST_RMC_GPS_TIME"]
        """Accurate timestamp associated with the PPS, from RMC data packet"""
        rmc_gps_time = parse_RMC_time_to_seconds(rmc_gps_time_str)
        """Accurate timestamp associated with the PPS, from RMC data packet"""

        #GPS time must be integer seconds
        if not np.isclose(rmc_gps_time % 1, 0, atol=1e-3):
            continue  # non-integer RMC GPS TIME! Not a valid behavior

        # parse the timestamp of the LAST_RMC_AUDIOMOTH_TIME from this row, row1
        rmc_arrival_amtime = parse_RMC_time_to_seconds(
            row1["LAST_RMC_AUDIOMOTH_TIME"]
        )

        # parse the timestamp of AUDIOMOTH_TIME (time PPS was recieved, internal clock) for previous row, row0
        pps_arrival_amtime = parse_RMC_time_to_seconds(
            row0["AUDIOMOTH_TIME"]
        )

        sample_count_at_pps_arrival = row0["TOTAL_SAMPLES"]

        cpu_clock_counter = row0[cpu_clock_counter_col]
        
        """Position of tye CPU clock % 1000 at time of PPS ping"""
        if not (pps_arrival_amtime < rmc_arrival_amtime < pps_arrival_amtime + 1.0):
            pps_pings_with_no_RMC.append(pps_number)
            continue

        #Observed sample rate check
        if samples_written:
            observed_sr = (
                sample_count_at_pps_arrival - samples_written[-1]
            ) / (rmc_gps_time - gps_times_sec[-1])
            
            if abs(observed_sr - expected_sr) > sr_tol:
                print(
                    f"Observed sr {observed_sr} did not match expected {expected_sr} at PPS {pps_number}. Discarding RMC."
                )
                continue

        samples_written.append(sample_count_at_pps_arrival)
        gps_times_sec.append(rmc_gps_time)
        gps_times_str.append(rmc_gps_time_str)
        pps_numbers_out.append(pps_numbers[i - 1])
        counter_positions.append(cpu_clock_counter)

    if pps_pings_with_no_RMC:
        print(f"{len(pps_pings_with_no_RMC)} PPS pings did not have corresponding RMC data")

    return pd.DataFrame(
        {
            "PPS": pps_numbers_out,
            "SAMPLES_WRITTEN": samples_written,
            "GPS_TIME_SEC": gps_times_sec,
            "GPS_TIME_STR": gps_times_str,
            "CPU_CLOCK_COUNTER": counter_positions,
        }
    )


def correct_sample_rate(
    audio,
    sample_timestamp_df,
    desired_sr,
    expected_sr=48000,
    interpolation_method="nearest",
    sr_warning_tolerance=100,
):

    dtype = audio.samples.dtype
    new_samples = []  # make container for the resampled samples

    for idx in range(1, len(sample_timestamp_df)):
        this_row = sample_timestamp_df.iloc[idx]
        previous_row = sample_timestamp_df.iloc[idx - 1]

        start_sample = previous_row["SAMPLES_WRITTEN"]
        end_sample_exclude = this_row["SAMPLES_WRITTEN"]

        time_elapsed = this_row["GPS_TIME_SEC"] - previous_row["GPS_TIME_SEC"]
        observed_sr = (end_sample_exclude - start_sample) / time_elapsed

        if abs(observed_sr - expected_sr) > sr_warning_tolerance:
            warnings.warn(
                f"Observed SR {observed_sr:.2f} differs from expected {expected_sr}"
            )

        raw_samples = audio.samples[start_sample:end_sample_exclude]
        x_values = np.arange(len(raw_samples)) / observed_sr

        interpolator = interp1d(
            x_values, raw_samples, kind=interpolation_method, fill_value="extrapolate"
        )
        
        desired_x = np.arange(int(time_elapsed * desired_sr)) / desired_sr
        resampled = interpolator(desired_x).astype(dtype)

        new_samples.append(resampled)

    samples_out = np.concatenate(new_samples)
    # create new audio object with same metadata as the original
    new_audio = audio._spawn(samples=samples_out, sample_rate=desired_sr)
    # update the 'recording_start_time' in audio.metadata to the accurate timestamp
    start_time = sample_timestamp_df.loc[0, "GPS_TIME_STR"]

    if new_audio.metadata is None:
        new_audio.metadata = {}
    new_audio.metadata["recording_start_time"] = parse_RMC_time_to_datetime(start_time)

    return new_audio

def check_pps_quality(sample_timestamp_df, expected_sr=48000):
    reasons = {}

    reasons["n_pps"] = len(sample_timestamp_df)

    if len(sample_timestamp_df) < 5:
        reasons["fail"] = "too_few_pps"
        return False, reasons

    gps_diffs = np.diff(sample_timestamp_df["GPS_TIME_SEC"])
    reasons["max_pps_gap"] = float(gps_diffs.max())

    if gps_diffs.max() > 2.0:
        reasons["fail"] = "pps_gap_too_large"
        return False, reasons

    # Enforce strictly contiguous PPS (1 Hz, no gaps)
    if not np.all(gps_diffs == 1):
        reasons["fail"] = "non_contiguous_pps"
        return False, reasons

    obs_sr = (
        np.diff(sample_timestamp_df["SAMPLES_WRITTEN"]) /
        np.diff(sample_timestamp_df["GPS_TIME_SEC"])
    )

    reasons["max_sr_error_hz"] = float(np.max(np.abs(obs_sr - expected_sr)))

    if np.any(np.abs(obs_sr - expected_sr) > 5):
        reasons["fail"] = "unstable_sample_rate"
        return False, reasons
    pps_residuals = (
    np.diff(sample_timestamp_df["SAMPLES_WRITTEN"]) -
    expected_sr * np.diff(sample_timestamp_df["GPS_TIME_SEC"])
    )

    reasons["max_pps_residual_samples"] = float(np.max(np.abs(pps_residuals)))
    reasons["max_pps_residual_us"] = float(
        np.max(np.abs(pps_residuals)) / expected_sr * 1e6
    )

    reasons["pass"] = True
    return True, reasons




