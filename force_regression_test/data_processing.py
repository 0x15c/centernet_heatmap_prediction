import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

gt_filename = "Session_20260302_155834.jsonl"
disp_filename = "Session_20260302_155834_full_data_logging.jsonl"

# 1. Load Data
gt_df = pd.read_json(gt_filename, lines=True)
disp_df = pd.read_json(disp_filename, lines=True)

# 2. Filter for "Shearing (Wait)" status
# This creates a mask to select only the relevant rows
mask = gt_df['status'] == "Holding_Shear"
gt_filtered = gt_df[mask].copy()
disp_filtered = disp_df[mask].copy() # Assumes 1-to-1 row mapping

# 3. Process Forces (XY Plane)
# Convert the list column 'force' into a numpy array
forces = np.stack(gt_filtered['force'].values)
forces_z = forces[:, 2:3]
forces_total = np.linalg.norm(forces, axis=1)
forces_total = forces_total[:,np.newaxis]
# forces_mag_xy = np.linalg.norm(forces_xy, axis=1)

# 4. Process Displacement (XY Plane)
# Using filtered displacement dataframe
# disp_xy = disp_filtered[['disp_x', 'disp_y']].to_numpy()
# disp_mag = np.linalg.norm(disp_xy, axis=1)

div = disp_filtered[['Phi_max_diff']].to_numpy()

# 5. Regression Logic
# Filter out zero displacements to avoid noise at the origin
valid_idx = div > 10.0
x = div[valid_idx]
y = forces_z[valid_idx]

res = linregress(x, y)
line = res.slope * x + res.intercept

# 6. Visualization
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(x, y, label='Data Points', alpha=0.6, s=10)
ax.plot(x, line, color='red', linewidth=2, 
        label=f'Fitted Line (R²={res.rvalue**2:.3f})')

stats_text = (f"Slope: {res.slope:.4f}\n"
              f"Intercept: {res.intercept:.4f}\n"
              f"R²: {res.rvalue**2:.4f}\n"
              f"P-value: {res.pvalue:.4e}")

ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel("Displacement Magnitude (mm/units)")
ax.set_ylabel("Shearing Force Magnitude (N)")
ax.set_title("Shearing Force vs. Displacement (Wait Status Only)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.show()