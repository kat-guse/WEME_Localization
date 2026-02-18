# Creates final data products and visualizations
# First, it checks the HawkEars score distribution, then loads confirmed positions, 
# prints coordinate statistics and quality metrics, creates a csv, and an interactive html map.
import pandas as pd
import numpy as np
import shelve
import json
import folium
from pyproj import Transformer
from opensoundscape import Audio

# CLASS DEFINITION
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

shelf_path = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/weml_confirmed.out'
scores_path = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/minspec_output/hawkears_scores.csv'
output_csv = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/weml_confirmed_locations.csv'
output_html = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/weml_map.html'
df_arus_all = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/RTK_Coordinates - BC Alberts (EPSG-2955).csv')

# ==========================================
# 1. CHECK HAWKEARS SCORE DISTRIBUTION
# ==========================================
print("=" * 50)
print("HawkEars Score Distribution")
print("=" * 50)
scores = pd.read_csv(scores_path)
weml_col = [c for c in scores.columns if 'Western Meadowlark' in c][0]
print(scores[weml_col].describe())
print()

# ==========================================
# 2. LOAD CONFIRMED POSITIONS
# ==========================================
with shelve.open(shelf_path, 'r') as db:
    positions = db['position_estimates']

print("=" * 50)
print(f"Confirmed Positions: {len(positions)}")
print("=" * 50)

# ==========================================
# 3. COORDINATE STATISTICS (UTM)
# ==========================================
coords = [(p.location_estimate[0], p.location_estimate[1], p.location_estimate[2]) for p in positions]
df_utm = pd.DataFrame(coords, columns=['x', 'y', 'z'])
print("\nUTM Coordinates:")
print(df_utm.describe())

# ==========================================
# 4. QUALITY METRICS
# ==========================================
residuals = [p.residual_rms for p in positions]
cc_maxs = [p.mean_cc_max for p in positions]
print("\nQuality Metrics:")
print(f"  RMS residual: min={min(residuals):.2f}, median={np.median(residuals):.2f}, max={max(residuals):.2f}")
print(f"  CC max:       min={min(cc_maxs):.3f}, median={np.median(cc_maxs):.3f}, max={max(cc_maxs):.3f}")
print()

# ==========================================
# 5. EXPORT TO CSV
# ==========================================
export_data = []
for p in positions:
    export_data.append({
        'timestamp': p.start_timestamp,
        'x': p.location_estimate[0],
        'y': p.location_estimate[1],
        'z': p.location_estimate[2],
        'residual_rms': p.residual_rms,
        'mean_cc_max': p.mean_cc_max
    })

pd.DataFrame(export_data).to_csv(output_csv, index=False)
print(f"✓ CSV exported to: {output_csv}")

# ==========================================
# 6. CREATE INTERACTIVE MAP
# ==========================================
print("\nCreating interactive HTML map...")

# Load ARU coordinates
df_arus_all = pd.read_csv('/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/RTK_Coordinates - BC Alberts (EPSG-2955).csv')
df_arus = df_arus_all[df_arus_all['localization_grid'] == 'OKLG-8'].copy()

# Convert UTM to lat/lon for bird positions
transformer = Transformer.from_crs("EPSG:2955", "EPSG:4326", always_xy=True)

map_coords = []
for p in positions:
    x, y, z = p.location_estimate[0], p.location_estimate[1], p.location_estimate[2]
    lon, lat = transformer.transform(x, y)
    map_coords.append({
        'lat': lat,
        'lon': lon,
        'z': z,
        'timestamp': str(p.start_timestamp),
        'residual_rms': p.residual_rms,
        'mean_cc_max': p.mean_cc_max
    })

df_map = pd.DataFrame(map_coords)

# Transform ARU locations
df_arus['longitude'], df_arus['latitude'] = transformer.transform(
    df_arus['ground_truth_easting'].values,
    df_arus['ground_truth_northing'].values
)

# Create map centered on bird detections
center_lat = df_map['lat'].median()
center_lon = df_map['lon'].median()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=18,
    tiles='OpenStreetMap'
)

# Add satellite imagery layer
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite',
    overlay=False
).add_to(m)

# Add ARU locations (black microphone icons)
aru_group = folium.FeatureGroup(name="ARU Recorders")
for _, row in df_arus.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<b>ARU Recorder</b><br>Grid: {row['localization_grid']}<br>Device: {row['device_id']}<br>Elevation: {row['orthometric_height']:.1f} m",
        icon=folium.Icon(color='black', icon='microphone', prefix='fa')
    ).add_to(aru_group)
aru_group.add_to(m)

# Add bird detections (orange circles)
bird_group = folium.FeatureGroup(name="Western Meadowlark Detections")
for _, row in df_map.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6,
        popup=f"""
        <b>Western Meadowlark</b><br>
        Time: {row['timestamp']}<br>
        Elevation: {row['z']:.1f} m<br>
        RMS Error: {row['residual_rms']:.2f} m<br>
        CC Score: {row['mean_cc_max']:.3f}
        """,
        color='red',
        fill=True,
        fillColor='orange',
        fillOpacity=0.7,
        weight=2
    ).add_to(bird_group)
bird_group.add_to(m)

# Add layer control to toggle ARUs and birds on/off
folium.LayerControl().add_to(m)

m.save(output_html)
print(f"✓ Map saved to: {output_html}")
print(f"\nAll outputs complete!")
