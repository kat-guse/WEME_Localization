import pandas as pd
import numpy as np
import os
from datetime import timedelta
from opensoundscape.localization.spatial_event import SpatialEvent
from opensoundscape.localization.localization_algorithms import SPEED_OF_SOUND

def calc_speed_of_sound(temperature=22):
    #Calculate speed of sound in air based on temperature in Celsius
    return 331.3 * np.sqrt(1 + float(temperature) / 273.15)

class SynchronizedRecorderArray:
    def __init__(self, file_coords, speed_of_sound):
        self.file_coords = file_coords
        self.speed_of_sound = speed_of_sound
    
    def create_candidate_events(self, detections, species_name, min_n_receivers=4, 
                            max_receiver_dist=60, bandpass_range=None, temporal_window=2.0):
        """
        Create candidate events with proper temporal clustering
        
        Args:
            detections: DataFrame with columns including clip_start_utc, original_mid_s, sensor_id, wav_path
            min_n_receivers: minimum number of recorders (3-4 recommended)
            max_receiver_dist: max distance between recorders in meters (40-50m recommended)
            bandpass_ranges: dict of species:frequency range
            temporal_window: seconds within which detections are considered same event (2-3s recommended)
        """
        nearby_sensors_dict = self.make_nearby_sensors_dict(max_receiver_dist)
        max_delay = (max_receiver_dist / self.speed_of_sound) + 0.2  # Added 0.2s buffer for clock drift
        
        detections = detections.copy()
        # Filter detections for the specific species being processed
        detections = detections[detections['detected_species'] == species_name]
        
        if len(detections) == 0:
            print(f"No detections found for {species_name}")
            return []

        detections['clip_start_utc'] = pd.to_datetime(detections['clip_start_utc'])
        detections['detection_time_utc'] = detections.apply(
            lambda row: row['clip_start_utc'] + timedelta(seconds=row['original_mid_s']), 
            axis=1
        )
        
        # Strip grid prefix to match coordinate index
        detections['device_match'] = detections['sensor_id'].str.split('-').str[-1]
        detections = detections.sort_values('detection_time_utc').reset_index(drop=True)
        
        clusters = self._cluster_detections_temporally(detections, temporal_window)
        candidate_events = []
        
        for cluster_df in clusters:
            if len(cluster_df) < min_n_receivers:
                continue

            best_idx = cluster_df['confidence'].idxmax()
            ref_sensor = cluster_df.loc[best_idx, 'device_match']
            ref_time = cluster_df.loc[best_idx, 'detection_time_utc']

            if ref_sensor not in self.file_coords.index:
                continue

            close_receivers = nearby_sensors_dict.get(ref_sensor, [])
            sensors_in_cluster = cluster_df['device_match'].unique()
            close_det_sensors = [s for s in sensors_in_cluster if s in close_receivers and s != ref_sensor]
            
            if (len(close_det_sensors) + 1) < min_n_receivers:
                continue
            
            event_sensor_ids = [ref_sensor] + close_det_sensors
            receiver_files, receiver_locations, offsets = [], [], []
            
            for sensor in event_sensor_ids:
                row = cluster_df[cluster_df['device_match'] == sensor].iloc[0]
                receiver_files.append(row['wav_path'])
                offsets.append(row['original_start_s'])
                receiver_locations.append(self.file_coords.loc[sensor, ['x', 'y', 'z']].values)

            
            # Create the event
            event = SpatialEvent(
                receiver_files=receiver_files,
                receiver_locations=receiver_locations,
                receiver_start_time_offsets=offsets,
                speed_of_sound=self.speed_of_sound,
                max_delay=max_delay,
                duration=3.0, 
                class_name=species_name,
                start_timestamp=ref_time,
                bandpass_range=bandpass_range
            )
            event.cc_filter = 'phat' 
            candidate_events.append(event)
        
        print(f"Created {len(candidate_events)} candidate events after filtering")
        return candidate_events
    
    def _cluster_detections_temporally(self, detections, temporal_window):
        """
        Cluster detections that occur within temporal_window seconds of each other
        
        Returns: list of DataFrames, each representing one temporal cluster
        """
        if len(detections) == 0: return []
        clusters = []
        current_cluster = [detections.iloc[0]]
        for i in range(1, len(detections)):
            time_diff = (detections.iloc[i]['detection_time_utc'] - current_cluster[-1]['detection_time_utc']).total_seconds()
            if time_diff <= temporal_window:
                current_cluster.append(detections.iloc[i])
            else:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [detections.iloc[i]]
        if current_cluster: clusters.append(pd.DataFrame(current_cluster))
        return clusters
    
    def make_nearby_sensors_dict(self, r_max):
        """Standard Euclidean distance check between your UTM coordinates"""
        coords = self.file_coords[['x', 'y']].astype(float)
        nearby_dict = dict()
        for s in coords.index:
            ref_loc = coords.loc[s].values 
            others = coords.drop([s])
            dist = np.sqrt(np.sum((others.values - ref_loc)**2, axis=1))
            nearby_dict[s] = list(others[dist <= r_max].index)
        return nearby_dict
