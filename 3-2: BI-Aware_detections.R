# BI-AWARE THRESHOLD FILTERING FOR HAWKEARS DETECTIONS
# This script reads HawkEars detections, calculates BI for each clip,
# assigns the appropriate threshold based on BI condition, and filters results

# --- CONFIGURATION ---
# Input: HawkEars detection summary from Python script (PHASE 1 - LOCALIZATION.py)
detections_path <- "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/detections_with_timestamps.csv"

# BI thresholds CSV (output from your calibration script)
bi_thresholds_path <- "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/scripts/WEME_BI_thresholds.csv"
# Species name (must match what's in your detection file)
species_name <- "Western Meadowlark"

# --- TO ADD MORE SPECIES EXAMPLE---
#bi_thresholds_path <- "~/Desktop/AMRO_BI_thresholds.csv"
#species_name <- "American Robin"

# Audio files location (to calculate BI)
audio_root <- "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/OKLG-8-Sync"

# Output directory
out_dir <- "/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# LIBRARIES
library(dplyr)
library(readr)
library(tuneR)
library(soundecology)
library(purrr)

# --- LOAD BI THRESHOLDS ---
cat("Loading BI thresholds...\n")
bi_thresholds <- read_csv(bi_thresholds_path, show_col_types = FALSE) %>%
  rename(condition = label) %>%
  mutate(species = species_name)

# Define BI z-score boundaries based on your calibration
# These correspond to the z_score values in your CSV
bi_thresholds <- bi_thresholds %>%
  mutate(
    z_lower = case_when(
      condition == "Very Low BI" ~ -Inf,
      condition == "Low BI" ~ -1.0,
      condition == "High BI" ~ 0.0,
      condition == "Very High BI" ~ 1.0,
      TRUE ~ NA_real_
    ),
    z_upper = case_when(
      condition == "Very Low BI" ~ -1.0,
      condition == "Low BI" ~ 0.0,
      condition == "High BI" ~ 1.0,
      condition == "Very High BI" ~ Inf,
      TRUE ~ NA_real_
    )
  )

cat("BI Thresholds loaded:\n")
print(bi_thresholds %>% select(species, condition, z_lower, z_upper, threshold, precision))

# --- HELPER FUNCTIONS ---

# Z-score standardization
z_score <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

# Safe audio reading with time window
safe_read_window <- function(path, mid, win = 6) {
  tryCatch({
    # Read 6-second window centered on detection
    w <- readWave(path,
                  from = max(0, mid - win/2),
                  to   = mid + win/2,
                  units = "seconds")
    if (w@stereo) w <- mono(w, "left")
    if (length(w@left) < 64) stop("Audio too short")
    w
  }, error = function(e) NULL)
}

# Calculate Bioacoustic Index
calc_bi <- function(w) {
  tryCatch(
    bioacoustic_index(w,
                      min_freq = 2000,
                      max_freq = 8000,
                      fft_w = 512)$left_area,
    error = function(e) NA_real_
  )
}

# Assign BI condition based on z-score
assign_bi_condition <- function(bi_z, current_species, threshold_table) {
  # Subset the table once to avoid repeated filtering
  thresh_sub <- threshold_table[threshold_table$species == current_species, ]
  
  res <- rep(NA_character_, length(bi_z))
  
  for(i in seq_along(bi_z)) {
    z <- bi_z[i]
    if (is.na(z)) next
    
    # Use standard R indexing instead of dplyr::filter
    match_idx <- which(z >= thresh_sub$z_lower & z < thresh_sub$z_upper)
    if (length(match_idx) > 0) {
      res[i] <- thresh_sub$condition[match_idx[1]]
    }
  }
  return(res)
}

# Get threshold for a given BI condition and species
get_threshold_for_condition <- function(condition, species_name, threshold_table) {
  threshold_table %>%
    filter(species == species_name, condition == !!condition) %>%
    pull(threshold) %>%
    first()
}

# --- MAIN PROCESSING ---

cat("\nReading HawkEars detections...\n")
detections <- read_csv(detections_path, show_col_types = FALSE)

# Map Python output columns to R script's expected columns
detections <- detections %>%
  mutate(
    filename = basename(file),
    # Extract sensor_id from the file path since it isn't a column in your CSV
    sensor_id = basename(dirname(file)), 
    species = detected_species,         # Match your 'detected_species' header
    confidence_score = confidence,      # Match your 'confidence' header
    
    original_start_s = start_time,
    original_end_s = end_time,
    original_mid_s = (start_time + end_time) / 2,
    
    clip_start_s = start_time,          # Match your 'start_time' header
    clip_end_s = end_time               # Match your 'end_time' header
  ) %>%
  # This removes the baseline 0.15 threshold to make room for BI thresholds
  select(-any_of("threshold")) 

cat(sprintf("Loaded %d detections\n", nrow(detections)))

# Create audio file index
cat("Indexing audio files...\n")
wav_index <- tibble(
  wav_path = list.files(audio_root,
                        pattern = "\\.wav$|\\.WAV$",
                        recursive = TRUE,
                        full.names = TRUE)
) %>%
  mutate(
    match_name = basename(wav_path),
    sensor_id = basename(dirname(wav_path))
  ) %>%
  distinct(match_name, .keep_all = TRUE)

cat(sprintf("Found %d audio files\n", nrow(wav_index)))

# Calculate clip midpoint and match to audio files
cat("Calculating BI for each detection...\n")
detections_with_bi <- detections %>%
  mutate(
    clip_mid = original_mid_s,
    match_name = filename
  ) %>%
  left_join(wav_index, by = c("sensor_id", "match_name")) %>%
  filter(!is.na(wav_path))

cat(sprintf("Matched %d detections to audio files\n", nrow(detections_with_bi)))

# Calculate BI (this may take a while for many detections)
detections_with_bi <- detections_with_bi %>%
  mutate(
    wave = map2(wav_path, clip_mid, safe_read_window,
                .progress = "Calculating BI"),
    BI_raw = map_dbl(wave, calc_bi)
  ) %>%
  select(-wave)  # Remove wave objects to save memory

# Calculate z-scores
detections_with_bi <- detections_with_bi %>%
  group_by(species) %>%
  mutate(BI_z = z_score(BI_raw)) %>%
  ungroup()

# Assign BI conditions using our new robust function
cat("Assigning BI conditions...\n")
detections_with_bi$BI_condition <- assign_bi_condition(
  detections_with_bi$BI_z, 
  species_name, 
  bi_thresholds
)

# Map thresholds using a simple join instead of a rowwise function
cat("Applying thresholds...\n")
detections_with_bi <- detections_with_bi %>%
  left_join(bi_thresholds %>% select(condition, threshold_used_new = threshold), 
            by = c("BI_condition" = "condition")) %>%
  rename(threshold_used = threshold_used_new)

# Filter: keep only detections that meet their BI-specific threshold
detections_filtered <- detections_with_bi %>%
  filter(!is.na(BI_condition),
         !is.na(threshold_used),
         confidence_score >= threshold_used)

#------ENSURE ORIGINAL DETECTION TIMES ARE IN OUTPUT------
detections_filtered <- detections_filtered %>%
  select(
    file, sensor_id, wav_path,
    original_start_s, original_end_s, original_mid_s,  # TRUE detection times
    clip_start_utc, file_start_utc,
    species, confidence_score,
    BI_condition, BI_raw, BI_z, threshold_used,
    everything()
  )

# --- SAVE RESULTS ---

# Save full results with BI info
full_output <- file.path(out_dir, paste0(gsub(" ", "_", species_name), "_detections_with_BI_filtering.csv"))
write_csv(detections_filtered, full_output)

cat("\n--- FILTERING SUMMARY ---\n")
cat(sprintf("Original detections: %d\n", nrow(detections)))
cat(sprintf("After BI filtering: %d\n", nrow(detections_filtered)))
cat(sprintf("Filtered out: %d (%.1f%%)\n", 
            nrow(detections) - nrow(detections_filtered),
            100 * (1 - nrow(detections_filtered) / nrow(detections))))

# Summary by BI condition
cat("\n--- DETECTIONS BY BI CONDITION ---\n")
condition_summary <- detections_filtered %>%
  count(BI_condition, species) %>%
  arrange(BI_condition)

print(condition_summary)

# Summary by sensor
cat("\n--- DETECTIONS BY SENSOR (BI-FILTERED) ---\n")
sensor_summary <- detections_filtered %>%
  count(sensor_id, species) %>%
  arrange(desc(n))

print(sensor_summary)

# Save summary stats
summary_output <- file.path(out_dir, paste0(gsub(" ", "_", species_name), "_BI_filtering_summary.csv"))
summary_stats <- detections_filtered %>%
  group_by(species, BI_condition) %>%
  summarise(
    n_detections = n(),
    mean_confidence = mean(confidence_score),
    median_confidence = median(confidence_score),
    mean_BI_raw = mean(BI_raw, na.rm = TRUE),
    mean_BI_z = mean(BI_z, na.rm = TRUE),
    threshold_used = first(threshold_used),
    .groups = "drop"
  )

write_csv(summary_stats, summary_output)

cat(sprintf("\nResults saved to:\n  %s\n  %s\n", full_output, summary_output))

# --- OPTIONAL: VISUALIZATION ---
library(ggplot2)

# Plot: BI distribution of kept vs. removed detections
comparison_data <- detections_with_bi %>%
  filter(!is.na(BI_z)) %>%
  mutate(
    kept = confidence_score >= threshold_used & !is.na(threshold_used)
  )

p1 <- ggplot(comparison_data, aes(x = BI_z, fill = kept)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  labs(
    title = "BI Distribution: Kept vs. Filtered Detections",
    subtitle = paste(species_name, "detections after BI-aware thresholding"),
    x = "BI (z-score)",
    y = "Count",
    fill = "Kept"
  ) +
  theme_minimal()

ggsave(file.path(out_dir, paste0(gsub(" ", "_", species_name), "_BI_distribution_filtered.png")), 
       p1, width = 10, height = 6, dpi = 300)

# Plot: Confidence scores by BI condition
p2 <- ggplot(detections_filtered, aes(x = BI_condition, y = confidence_score)) +
  geom_boxplot(fill = "skyblue", alpha = 0.7) +
  geom_hline(data = bi_thresholds, 
             aes(yintercept = threshold), 
             color = "red", linetype = "dashed") +
  labs(
    title = "Detection Confidence by BI Condition",
    subtitle = "Red lines show BI-specific thresholds",
    x = "BI Condition",
    y = "Confidence Score"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(out_dir, paste0(gsub(" ", "_", species_name), "_confidence_by_BI_condition.png")), 
       p2, width = 8, height = 6, dpi = 300)

cat("\nPlots saved to output directory\n")
cat("\nDone!\n")
