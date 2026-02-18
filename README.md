# WEME_Localization_South_Okanagan_BC

Creators: Katrine Guse 

Corresponding Author: Erin Bayne, Elly Knight, Dan Yip

Contact: kguse@ualberta.ca

Affiliations: University of Alberta, Environment and Climate Change Canada

## General characteristics
- Audio_format: 3-second clips centred on each localized event
- Dimensions localized: 3D (x,y,z)
- Number of localization arrays: 1
- Array geometry: 7x7 grid with 35m spacing
- Number of audio files: TBD
- Size: TBD (GB)
- Species localized: Western Meadowlark (song)

## Study Description

### Study Purpose
This data was collected as part of a Master's thesis conducted at the University of Alberta. The goal of this data is to use localization as a gold-standard dataset to understand mechanisms contributing to the error in survey techniques. The arrays were deployed in the South Okanagan of British Columbia. Specifically, along White Lake Rd. The location was selected for its geographical features (slope, hills, grassland, shrub step) to monitor grassland species in open habitats with no tree cover. 

### Personnel
Katrine Guse, Elly Knight, Dan Yip, Erin Bayne and the help of Sean Vanderluit in the field

### Data Types Collected
The data collected included audio recordings from 49 AudioMoths and 1 SM-2 Mini (additional centroid), paired point counts, spot maps, sound playback, GPS points using real-time kinematic positioning, and weather element data. 

## Files
This section describes the contents of each file. 

- localized_events.csv — main table with event_id, label, start_timestamp, duration, position (utm x/y/z), file_ids, file_start_time_offsets, tdoas (optional), distance_residuals (optional)
- classes.csv — describes each species/class

/localization_metadata/ folder:
  - point_table.csv — ARU locations: point_id, utm_easting, utm_northing, elevation, utm_zone
  - audio_file_table.csv — audio file catalog: file_id, relative_path, point_id, start_timestamp

/scripts/ folder:
- All scripts used (.py, .ipynb, .r files) to create the localizations
- environment.yml — frozen Python environment with package versions

/audio/ folder:
- All audio files organized however you want (e.g., by recorder, flat structure)

## Sites
The Nature Trust of BC Property, 	White Lake Basin Biodiversity Ranch Boundary, South Okanagan, British Columbia, Canada 

