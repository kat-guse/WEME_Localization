#This code is influenced by Meg Edgars Validation - acoustic indices.R. 
#It, calibrates score thresholds for a species’ detector by modeling how 
#detection score (and background noise, via the Bioacoustic Index) relates to 
#true positives, then choosing the lowest score threshold that achieves a target 
#precision. It then applies those thresholds across noise conditions, saves the  
#results, and visualizes the chosen thresholds as a heatmap.

# CONFIG: paths, species, thresholds
validation_path <- "/Users/katrine/Desktop/HawkEars_validation_results.csv"
labels_path     <- "/Users/katrine/Desktop/HAWKEARS_Labels6(A1)-Stripped.csv"
audio_root      <- "/Volumes/BUworkspace/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/OKLG-6-Sync"
out_dir <- "~/Desktop"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

species_code <- "Western Meadowlark"
species_tag  <- "WEME"   # used for filenames

min_conf <- 0.10
max_conf <- 1.00
step     <- 0.05
target_precision   <- 0.90
fallback_threshold <- 0.95
threshold_grid <- seq(min_conf, fallback_threshold, by = step)

# LIBRARIES
library(dplyr)
library(purrr)
library(tidyr)
library(readr)
library(ggplot2)

# READ + PREP VALIDATION
validation <- read_csv(validation_path, show_col_types = FALSE) %>%
  mutate(
    class_code = as.character(class_code),
    score = as.numeric(score),
    label = as.character(label)
  )

validation_sp <- validation %>%
  filter(class_code == species_code) %>%
  mutate(
    tp = case_when(
      tolower(label) %in% c("yes","y","true","1","present") ~ 1L,
      tolower(label) %in% c("no","n","false","0","absent")  ~ 0L,
      TRUE ~ NA_integer_
    )
  ) %>%
  filter(!is.na(tp), !is.na(score),
         score >= min_conf, score <= max_conf)

# LABEL POPULATION (bins)

if (!is.null(labels_path) && file.exists(labels_path)) {
  labels_all <- read_csv(labels_path, show_col_types = FALSE) %>%
    mutate(class_code = as.character(class_code),
           score = as.numeric(score)) %>%
    filter(class_code == species_code,
           !is.na(score),
           score >= min_conf, score <= max_conf)
} else {
  labels_all <- validation_sp %>% select(-tp)
}

lab_bins <- labels_all %>%
  mutate(
    bin_lower = min_conf + floor((score - min_conf) / step) * step,
    bin_lower = pmin(fallback_threshold, pmax(min_conf, bin_lower)),
    bin_mid = bin_lower + step / 2
  ) %>%
  count(bin_lower, bin_mid, name = "N_i")

if (nrow(lab_bins) == 0)
  stop("No label bins found — check species or score range.")


# MODELS

fit_glm <- glm(tp ~ score, family = binomial(), data = validation_sp)

compute_tseng_threshold <- function(df_bins, model, BI_z_fixed = NULL) {
  
  if (is.null(BI_z_fixed)) {
    df_bins <- df_bins %>%
      mutate(TPR_i = predict(model,
                             newdata = data.frame(score = bin_mid),
                             type = "response"))
  } else {
    df_bins <- df_bins %>%
      mutate(TPR_i = predict(model,
                             newdata = data.frame(score = bin_mid,
                                                  BI_z = BI_z_fixed),
                             type = "response"))
  }
  
  df_bins <- df_bins %>%
    mutate(
      NTP_i = TPR_i * N_i,
      NFP_i = (1 - TPR_i) * N_i
    )
  
  curve <- map_dfr(threshold_grid, function(T) {
    kept <- df_bins %>% filter(bin_lower >= T)
    denom <- sum(kept$NTP_i + kept$NFP_i)
    tibble(
      threshold = T,
      precision = ifelse(denom == 0, NA_real_,
                         sum(kept$NTP_i) / denom),
      prop_retained = sum(kept$N_i) / sum(df_bins$N_i),
      n_retained = sum(kept$N_i)
    )
  })
  
  hit <- curve %>%
    filter(!is.na(precision), precision >= target_precision) %>%
    arrange(threshold) %>%
    slice(1)
  
  if (nrow(hit) == 0)
    hit <- curve %>% filter(threshold == fallback_threshold)
  
  hit
}

# BIOACOUSTIC INDEX 
if (!is.null(audio_root) && dir.exists(audio_root)) {
  
  library(tuneR)
  library(soundecology)
  
  z <- function(x) {
    s <- sd(x, na.rm = TRUE)
    if (!is.finite(s) || s == 0) return(rep(0, length(x)))
    (x - mean(x, na.rm = TRUE)) / s
  }
  
  safe_read_window <- function(path, mid, win = 6) {
    tryCatch({
      w <- readWave(path,
                    from = max(0, mid - win/2),
                    to   = mid + win/2,
                    units = "seconds")
      if (w@stereo) w <- mono(w, "left")
      if (length(w@left) < 64) stop()
      w
    }, error = function(e) NULL)
  }
  
  calc_bi <- function(w) {
    tryCatch(
      bioacoustic_index(w,
                        min_freq = 2000,
                        max_freq = 8000,
                        fft_w = 512)$left_area,
      error = function(e) NA_real_
    )
  }
  
  wav_index <- tibble(
    wav_path = list.files(audio_root,
                          pattern = "\\.wav$|\\.WAV$",
                          recursive = TRUE,
                          full.names = TRUE)
  ) %>%
    mutate(match_name = basename(wav_path)) %>%
    distinct(match_name, .keep_all = TRUE)
  
  val_bi <- validation_sp %>%
    mutate(
      clip_mid = (as.numeric(start_time) +
                    as.numeric(end_time)) / 2,
      match_name = basename(filename)
    ) %>%
    left_join(wav_index, by = "match_name") %>%
    filter(!is.na(wav_path)) %>%
    mutate(
      wave = map2(wav_path, clip_mid, safe_read_window),
      BI_raw = map_dbl(wave, calc_bi)
    ) %>%
    filter(!is.na(BI_raw)) %>%
    mutate(BI_z = z(BI_raw))
  
  fit_bi <- glm(tp ~ score * BI_z,
                family = binomial(),
                data = val_bi)
  
  target_levels <- tibble(
    label   = c("Very Low BI","Low BI","High BI","Very High BI"),
    z_score = c(-1.5, -0.5, 0.5, 1.5)
  )
  
  bi_results <- target_levels %>%
    mutate(
      summary = map(z_score,
                    ~ compute_tseng_threshold(lab_bins, fit_bi, .x))
    ) %>%
    unnest(summary)
  
} else {
  stop("audio_root not found — BI thresholds skipped.")
}

# SAVE RESULTS
outfile_csv <- file.path(out_dir, paste0(species_tag, "_BI_thresholds.csv"))
write_csv(bi_results, outfile_csv)

#  PLOTTING FUNCTION

plot_bi_thresholds <- function(df, species_name, outfile) {
  
  df <- df %>%
    mutate(
      label = factor(
        label,
        levels = c("Very Low BI","Low BI","High BI","Very High BI")
      )
    )
  
  p <- ggplot(df, aes(x = label, y = 1, fill = threshold)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(
      aes(label = sprintf(
        "%.2f\np=%.3f\nret=%.1f%%",
        threshold, precision, prop_retained * 100
      )),
      color = "white",
      fontface = "bold",
      size = 4
    ) +
    scale_fill_gradient2(
      low = "#2166ac",
      mid = "#f7f7f7",
      high = "#b2182b",
      midpoint = 0.5,
      limits = c(0, 1),
      name = "Threshold"
    ) +
    labs(
      title = paste("Chosen Thresholds by BI condition"),
      subtitle = species_name,
      x = "BI condition",
      y = ""
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      legend.position = "top"
    )
  
  ggsave(outfile, p, width = 8, height = 3.5, dpi = 300)
  p
}

# PLOT
plot_file  <- file.path(out_dir, paste0(species_tag, "_BI_threshold_heatmap.png"))
plot_bi_thresholds(bi_results, species_code, plot_file)
