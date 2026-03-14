import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import linregress

# 1. Load and Sync Data
gt_df = pd.read_json("Session_20260311_223951.jsonl", lines=True)
disp_df = pd.read_json("Session_20260311_223951_MLP.jsonl", lines=True)
df = pd.merge(gt_df, disp_df, on='frame')

# Multi-status selection
active_statuses = ["Shearing (Wait)"] # "Shearing (Wait)", "Holding_Shear"
shearing_df = df[df['status'].isin(active_statuses)].copy()

# 2. Extract Raw Components
forces_raw = np.stack(shearing_df['force'].values)[:, :2] 
dxy = shearing_df[['disp_x_sample_based_c0', 'disp_y_sample_based_c0']].values
dx= -dxy[:, 0]
dy= dxy[:, 1]

# Pre-calculate global limits for fixed scope
f_limit = np.max(np.abs(forces_raw)) * 1.2
d_limit_x = [dx.min() - 0.05, dx.max() + 0.05]
d_limit_y = [dy.min() - 0.05, dy.max() + 0.05]

# 3. Setup Interactive Figure
fig, ax = plt.subplots(2, 2, figsize=(14, 12))
plt.subplots_adjust(bottom=0.15)

# Initialize plot objects
scat_x = ax[0, 0].scatter(dx, forces_raw[:, 0], marker='.', c="blue", alpha=0.3)
line_x_plot, = ax[0, 0].plot([], [], color='red', linewidth=2)

scat_y = ax[0, 1].scatter(dy, forces_raw[:, 1], marker='.', c="orange", alpha=0.3)
line_y_plot, = ax[0, 1].plot([], [], color='red', linewidth=2)

scat_pred_x = ax[1, 0].scatter([], [], marker='.', c="purple", alpha=0.4)
line_diag_x, = ax[1, 0].plot([-f_limit, f_limit], [-f_limit, f_limit], 'k--', alpha=0.5)

scat_pred_y = ax[1, 1].scatter([], [], marker='.', c="brown", alpha=0.4)
line_diag_y, = ax[1, 1].plot([-f_limit, f_limit], [-f_limit, f_limit], 'k--', alpha=0.5)

# Set Fixed Limits
ax[0, 0].set_xlim(d_limit_x); ax[0, 0].set_ylim(-f_limit, f_limit)
ax[0, 1].set_xlim(d_limit_y); ax[0, 1].set_ylim(-f_limit, f_limit)
ax[1, 0].set_xlim(-f_limit, f_limit); ax[1, 0].set_ylim(-f_limit, f_limit)
ax[1, 1].set_xlim(-f_limit, f_limit); ax[1, 1].set_ylim(-f_limit, f_limit)

# Dynamic Text Labels
text_x = ax[0, 0].text(0.05, 0.95, "", transform=ax[0, 0].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
text_y = ax[0, 1].text(0.05, 0.95, "", transform=ax[0, 1].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
mae_text_x = ax[1, 0].text(0.05, 0.95, "", transform=ax[1, 0].transAxes, verticalalignment='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
mae_text_y = ax[1, 1].text(0.05, 0.95, "", transform=ax[1, 1].transAxes, verticalalignment='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

# 4. Rotation Slider
ax_theta = plt.axes([0.25, 0.05, 0.5, 0.03])
theta_slider = Slider(ax_theta, 'Rotation Angle (deg)', -120.0, 120.0, valinit=0.0)

def update(val):
    angle_rad = np.radians(theta_slider.val)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array(((c, -s), (s, c)))
    
    # Rotate forces based on current slider angle
    forces_rot = forces_raw @ R.T
    fx_rot, fy_rot = forces_rot[:, 0], forces_rot[:, 1]
    
    # Linear Regressions
    res_x = linregress(dx, fx_rot)
    res_y = linregress(dy, fy_rot)
    
    # Predictions
    fx_hat = res_x.slope * dx + res_x.intercept
    fy_hat = res_y.slope * dy + res_y.intercept
    
    # Update Row 0 (Stiffness Plots)
    scat_x.set_offsets(np.c_[dx, fx_rot])
    line_x_plot.set_data(dx, fx_hat)
    text_x.set_text(f"X-Slope: {res_x.slope:.4f}\nX-Intercept: {res_x.intercept:.4f}\nR²: {res_x.rvalue**2:.4f}")
    
    scat_y.set_offsets(np.c_[dy, fy_rot])
    line_y_plot.set_data(dy, fy_hat)
    text_y.set_text(f"Y-Slope: {res_y.slope:.4f}\nY-Intercept: {res_y.intercept:.4f}\nR²: {res_y.rvalue**2:.4f}")
    
    # Update Row 1 (Prediction vs Truth)
    scat_pred_x.set_offsets(np.c_[fx_rot, fx_hat])
    mae_text_x.set_text(f"MAE X: {np.mean(np.abs(fx_rot - fx_hat)):.4f} N")
    
    scat_pred_y.set_offsets(np.c_[fy_rot, fy_hat])
    mae_text_y.set_text(f"MAE Y: {np.mean(np.abs(fy_rot - fy_hat)):.4f} N")
    
    fig.canvas.draw_idle()

theta_slider.on_changed(update)

# Initial labels and aesthetic touches
ax[0, 0].set_title("X-Axis Stiffness"); ax[0, 0].set_ylabel("Rotated Force [N]")
ax[0, 1].set_title("Y-Axis Stiffness")
ax[1, 0].set_title("Model Accuracy (X)"); ax[1, 0].set_xlabel("Ground Truth [N]"); ax[1, 0].set_ylabel("Predicted [N]")
ax[1, 1].set_title("Model Accuracy (Y)"); ax[1, 1].set_xlabel("Ground Truth [N]")

for a in ax.flatten():
    a.grid(True, linestyle='--', alpha=0.5)

update(0)
plt.show()