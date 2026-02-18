
# Run this interactively AFTER 4-2: WEME_localizations.py has finished:
#creates plots to choose thresholds and lets you input them directly into the output 
# example: python /media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/scripts/4-3: filter_localizations.py

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# CONFIG — match your localization script
# ==========================================
SPECIES_TO_PROCESS = "Western Meadowlark"
output_base = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/LocalizationOutput-8/'
coords_path = '/media/UofA/BU_Work/BayneLabWorkSpace/Katrine_workspace/AudioMothSync/RTK_Coordinates - BC Alberts (EPSG-2955).csv'

species_slug    = SPECIES_TO_PROCESS.replace(' ', '_')
unfiltered_path = os.path.join(output_base, f"OKLG-8_{species_slug}_UNFILTERED.csv")
filtered_path   = os.path.join(output_base, f"OKLG-8_{species_slug}_CLEANED.csv")
diag_dir        = os.path.join(output_base, f"diagnostics_{species_slug}")
os.makedirs(diag_dir, exist_ok=True)

# ==========================================
# LOAD UNFILTERED RESULTS
# ==========================================
print(f"\nLoading: {unfiltered_path}")
df = pd.read_csv(unfiltered_path)
print(f"Loaded {len(df)} localizations (already filtered for validity).\n")

# ==========================================
# LOAD COORDS FOR DIAGNOSTICS
# ==========================================
all_coords = pd.read_csv(coords_path)
grid_8_coords = all_coords[all_coords['localization_grid'] == 'OKLG-8'].copy()
grid_8_coords['device_id'] = grid_8_coords['device_id'].str.split('-').str[-1]
grid_8_coords = grid_8_coords.set_index('device_id')
grid_8_coords = grid_8_coords.rename(columns={
    'ground_truth_easting': 'x',
    'ground_truth_northing': 'y',
    'orthometric_height': 'z'
})[['x', 'y', 'z']]

# ── Coordinate system check ────────────────────────────────────────────────────
print("=" * 55)
print("  ARU coordinate summary")
print("=" * 55)
print(grid_8_coords[['x', 'y', 'z']].describe().round(2))
print()

# ── Auto-suggest z-range based on ARU elevations ────────────────────────────
Z_MEDIAN = grid_8_coords['z'].median()
Z_MIN_AUTO = max(0, Z_MEDIAN - 300)
Z_MAX_AUTO = Z_MEDIAN + 300

# ==========================================
# PRINT SUMMARY STATS
# ==========================================
print("=" * 55)
print(f"  {SPECIES_TO_PROCESS} — Localization quality summary")
print("=" * 55)

print("\n  Residual RMS (m):")
print(f"    Min    : {df['residual_rms'].min():.2f}")
print(f"    Q1     : {df['residual_rms'].quantile(0.25):.2f}")
print(f"    Median : {df['residual_rms'].median():.2f}")
print(f"    Q3     : {df['residual_rms'].quantile(0.75):.2f}")
print(f"    Max    : {df['residual_rms'].max():.2f}")

print("\n  Mean Residual (m):")
print(f"    Min    : {df['mean_residual'].min():.2f}")
print(f"    Median : {df['mean_residual'].median():.2f}")
print(f"    Max    : {df['mean_residual'].max():.2f}")

print("\n  Mean CC Max:")
print(f"    Min    : {df['mean_cc_max'].min():.4f}")
print(f"    Median : {df['mean_cc_max'].median():.4f}")
print(f"    Max    : {df['mean_cc_max'].max():.4f}")

print("\n  Predicted Z / elevation (m):")
print(f"    Min    : {df['pred_z'].min():.1f}")
print(f"    Median : {df['pred_z'].median():.1f}")
print(f"    Max    : {df['pred_z'].max():.1f}")
print(f"    ARU median elevation: {Z_MEDIAN:.1f} m")

print(f"\n  Detections surviving each RMS cutoff:")
for t in [2, 5, 10, 15, 20, 30, 50]:
    n = len(df[df['residual_rms'] < t])
    bar = '█' * int(40 * n / max(len(df), 1))
    print(f"    < {t:>2} m : {n:>5}  {bar}")
print()

# ==========================================
# GENERATE DIAGNOSTIC PLOTS (BEFORE asking for input)
# ==========================================
print("=" * 55)
print("  Generating diagnostic plots...")
print("=" * 55)

# Plot 1 — RMS histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['residual_rms'], bins=50, edgecolor='black', color='steelblue')
for cutoff, style in [(5, '--'), (10, '-.'), (20, ':')]:
    ax.axvline(cutoff, color='grey', linestyle=style, alpha=0.6, label=f'{cutoff} m')
ax.set_xlabel('Residual RMS (m)')
ax.set_ylabel('Count')
ax.set_title(f'{SPECIES_TO_PROCESS} — Residual RMS distribution')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(diag_dir, '1_residual_rms_histogram.png'), dpi=150)
plt.close(fig)
print(f"  ✓ Saved: 1_residual_rms_histogram.png")

# Plot 2 — Mean residual histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['mean_residual'], bins=50, edgecolor='black', color='teal')
for cutoff, style in [(2, '--'), (5, '-.'), (15, ':')]:
    ax.axvline(cutoff, color='grey', linestyle=style, alpha=0.6, label=f'{cutoff} m')
ax.set_xlabel('Mean Residual (m)')
ax.set_ylabel('Count')
ax.set_title(f'{SPECIES_TO_PROCESS} — Mean residual distribution')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(diag_dir, '2_mean_residual_histogram.png'), dpi=150)
plt.close(fig)
print(f"  ✓ Saved: 2_mean_residual_histogram.png")

# Plot 3 — CC Max histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['mean_cc_max'], bins=50, edgecolor='black', color='darkorange')
for cutoff, style in [(0.02, '--'), (0.05, '-.'), (0.1, ':')]:
    ax.axvline(cutoff, color='grey', linestyle=style, alpha=0.6, label=f'{cutoff}')
ax.set_xlabel('Mean CC Max')
ax.set_ylabel('Count')
ax.set_title(f'{SPECIES_TO_PROCESS} — Cross-correlation score distribution')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(diag_dir, '3_mean_cc_max_histogram.png'), dpi=150)
plt.close(fig)
print(f"  ✓ Saved: 3_mean_cc_max_histogram.png")

# Plot 4 — Elevation histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['pred_z'], bins=50, edgecolor='black', color='slategrey')
ax.axvline(Z_MIN_AUTO, color='red', linestyle='--', linewidth=2, label=f'suggested min = {Z_MIN_AUTO:.0f}')
ax.axvline(Z_MAX_AUTO, color='red', linestyle=':',  linewidth=2, label=f'suggested max = {Z_MAX_AUTO:.0f}')
ax.set_xlabel('Predicted elevation (m)')
ax.set_ylabel('Count')
ax.set_title(f'{SPECIES_TO_PROCESS} — Predicted elevation distribution')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(diag_dir, '4_pred_z_histogram.png'), dpi=150)
plt.close(fig)
print(f"  ✓ Saved: 4_pred_z_histogram.png")

# Plot 5 — Spatial scatter (unfiltered)
padding = 300
X_MIN = grid_8_coords['x'].min() - padding
X_MAX = grid_8_coords['x'].max() + padding
Y_MIN = grid_8_coords['y'].min() - padding
Y_MAX = grid_8_coords['y'].max() + padding

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(
    df['pred_x'], df['pred_y'],
    c=df['residual_rms'], cmap='jet', alpha=0.6,
    edgecolors='none', s=30,
    vmin=0, vmax=df['residual_rms'].quantile(0.95)
)
ax.plot(grid_8_coords['x'], grid_8_coords['y'],
        '^', color='black', markersize=8, label='ARU')
plt.colorbar(sc, ax=ax).set_label('Residual RMS (m)')
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
ax.set_title(f'{SPECIES_TO_PROCESS} — All localizations (unfiltered)')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(diag_dir, '5_all_localizations_unfiltered.png'), dpi=150)
plt.close(fig)
print(f"  ✓ Saved: 5_all_localizations_unfiltered.png")

# Plot 6 — RMS threshold tradeoff
thresholds = [2, 5, 10, 15, 20, 30, 50]
counts = [len(df[df['residual_rms'] < t]) for t in thresholds]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, counts, 'o-', color='steelblue')
for t, c in zip(thresholds, counts):
    ax.annotate(str(c), (t, c), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=8)
ax.set_xlabel('Residual RMS threshold (m)')
ax.set_ylabel('Localizations retained')
ax.set_title(f'{SPECIES_TO_PROCESS} — Detections surviving each RMS cutoff')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(diag_dir, '6_rms_threshold_tradeoff.png'), dpi=150)
plt.close(fig)
print(f"  ✓ Saved: 6_rms_threshold_tradeoff.png")

print(f"\n✓ All diagnostic plots saved to: {diag_dir}")
print("\nReview the plots (download them if running remotely),")
print("then return here to enter filter thresholds.\n")

# ==========================================
# HELPER — prompt with validation
# ==========================================
def ask_threshold(prompt, min_val=None, max_val=None, allow_none=True):
    while True:
        raw = input(prompt).strip()
        if raw == '':
            if allow_none:
                return None
            else:
                print("  A value is required here.")
                continue
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  Must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  Please enter a number (or press Enter to skip).")

# ==========================================
# INTERACTIVE THRESHOLD ENTRY
# ==========================================
print("=" * 55)
print("  Enter quality filter thresholds")
print("=" * 55)
print("Press Enter to skip any filter.\n")

rms_max      = ask_threshold("  Max Residual RMS (m)   [e.g. 20, Enter to skip]: ", min_val=0)
mean_res_max = ask_threshold("  Max Mean Residual (m)  [e.g. 5,  Enter to skip]: ", min_val=0)
cc_min       = ask_threshold("  Min Mean CC Max        [e.g. 0.02, Enter to skip]: ", min_val=0, max_val=1)

print(f"\n  Suggested z range based on ARU elevations: {Z_MIN_AUTO:.0f} – {Z_MAX_AUTO:.0f} m")
z_min = ask_threshold(f"  Min pred_z elevation   [default {Z_MIN_AUTO:.0f}]: ", min_val=0) or Z_MIN_AUTO
z_max = ask_threshold(f"  Max pred_z elevation   [default {Z_MAX_AUTO:.0f}]: ", min_val=0) or Z_MAX_AUTO

# ==========================================
# APPLY FILTERS
# ==========================================
mask = (df['pred_z'] > z_min) & (df['pred_z'] < z_max)
if rms_max:      mask &= (df['residual_rms']  < rms_max)
if mean_res_max: mask &= (df['mean_residual'] < mean_res_max)
if cc_min:       mask &= (df['mean_cc_max']   > cc_min)

filtered = df[mask].copy()

print(f"\n  Filters applied:")
print(f"    residual_rms  < {rms_max}")
print(f"    mean_residual < {mean_res_max}")
print(f"    mean_cc_max   > {cc_min}")
print(f"    pred_z          {z_min:.0f} – {z_max:.0f} m")
print(f"\n  Localizations retained: {len(filtered)} / {len(df)}")

# ==========================================
# REGENERATE KEY PLOTS WITH USER'S THRESHOLDS MARKED
# ==========================================
def regenerate_plots_with_thresholds(rms_max, mean_res_max, cc_min, z_min, z_max):
    print("\nRegenerating plots with your chosen thresholds marked...")
    
    # Plot 1 — RMS histogram with user's threshold
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['residual_rms'], bins=50, edgecolor='black', color='steelblue')
    if rms_max:
        ax.axvline(rms_max, color='red', linewidth=2, label=f'your cutoff = {rms_max} m')
    for cutoff, style in [(5, '--'), (10, '-.'), (20, ':')]:
        ax.axvline(cutoff, color='grey', linestyle=style, alpha=0.6, label=f'{cutoff} m')
    ax.set_xlabel('Residual RMS (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'{SPECIES_TO_PROCESS} — Residual RMS (with your threshold)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(diag_dir, '1_residual_rms_histogram_FILTERED.png'), dpi=150)
    plt.close(fig)

    # Plot 6 — RMS tradeoff with user's threshold
    thresholds = [2, 5, 10, 15, 20, 30, 50]
    counts = [len(df[df['residual_rms'] < t]) for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, counts, 'o-', color='steelblue')
    if rms_max:
        ax.axvline(rms_max, color='red', linewidth=2, label=f'your cutoff = {rms_max}')
    for t, c in zip(thresholds, counts):
        ax.annotate(str(c), (t, c), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=8)
    ax.set_xlabel('Residual RMS threshold (m)')
    ax.set_ylabel('Localizations retained')
    ax.set_title(f'{SPECIES_TO_PROCESS} — RMS threshold tradeoff (with your choice)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(diag_dir, '6_rms_threshold_tradeoff_FILTERED.png'), dpi=150)
    plt.close(fig)

    # Plot 5 — Spatial scatter of FILTERED points only
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        filtered['pred_x'], filtered['pred_y'],
        c=filtered['residual_rms'], cmap='jet', alpha=0.6,
        edgecolors='black', s=40, linewidths=0.5,
        vmin=0, vmax=df['residual_rms'].quantile(0.95)
    )
    ax.plot(grid_8_coords['x'], grid_8_coords['y'],
            '^', color='black', markersize=8, label='ARU')
    plt.colorbar(sc, ax=ax).set_label('Residual RMS (m)')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_title(f'{SPECIES_TO_PROCESS} — Filtered localizations (n={len(filtered)})')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(diag_dir, '5_filtered_localizations_FINAL.png'), dpi=150)
    plt.close(fig)
    
    print(f"  ✓ Updated plots saved to: {diag_dir}")

# ==========================================
# CONFIRM AND SAVE
# ==========================================
while True:
    confirm = input("\n  Save filtered CSV and regenerate plots with your thresholds? [y/n]: ").strip().lower()
    if confirm == 'y':
        filtered.to_csv(filtered_path, index=False)
        regenerate_plots_with_thresholds(rms_max, mean_res_max, cc_min, z_min, z_max)
        print(f"\n  ✓ Filtered CSV saved to: {filtered_path}")
        print(f"  ✓ {len(filtered)} localizations retained from {len(df)} total.")
        break
    elif confirm == 'n':
        print("  Nothing saved. Rerun the script to try different thresholds.")
        break
    else:
        print("  Please enter y or n.")
