import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Load and Sync Data
gt_df = pd.read_json("Session_20260302_155834.jsonl", lines=True)
disp_df = pd.read_json("Session_20260302_155834_full_data_logging.jsonl", lines=True)
df = pd.merge(gt_df, disp_df, on='frame')
shearing_df = df[df['status'] == "Holding_Shear"].copy()

# 2. Extract Components
forces = np.stack(shearing_df['force'].values)
fz_all = forces[:, 2]
div_potential = np.stack(shearing_df['Phi_max_diff'].values)
fxy_2d_force = forces[:, :2]
y_small_idx = np.abs(fxy_2d_force[:,1])<0.5
x_small_idx = np.abs(fxy_2d_force[:,0])<0.25
fxy_y_small = fxy_2d_force[y_small_idx]
fxy_x_small = fxy_2d_force[x_small_idx]
dxy = shearing_df[['disp_x_sample_based_c1', 'disp_y_sample_based_c1']].values
dxy_y_small = dxy[y_small_idx]
dxy_x_small = dxy[x_small_idx]
# fxy_all = np.linalg.norm(fxy_y_small, axis=1)
# dxy_all = np.linalg.norm(dxy_y_small, axis=1)
fxy_all = fxy_x_small[:,1]
dxy_all = dxy_x_small[:,1]
div_potential_y_small = div_potential[x_small_idx]

# 3. Setup the Interactive Plot
fig = plt.figure(figsize=(14, 7))
plt.subplots_adjust(bottom=0.25)

# Left Axis: 3D Reference
ax3d = fig.add_subplot(121, projection='3d')
# Static background points
ax3d.scatter(dxy_all, div_potential_y_small, fxy_all, c='green', alpha=0.05, s=2)

# Dynamic Highlight: This scatter object will show the points in the current slice
slice_scatter_3d = ax3d.scatter([], [], [], color='red', s=10, alpha=0.8, label='Active Slice')

ax3d.set_title("3D Force-Disp Map (Red = Active Slice)")
ax3d.set_xlabel("Displacement")
ax3d.set_ylabel("Divergence")
ax3d.set_zlabel("Shear Force Magnitude")

# Right Axis: 2D Sliced Projection
ax2d = fig.add_subplot(122)
scatter_2d = ax2d.scatter([], [], color='blue', alpha=0.6, s=20)
ax2d.set_xlim(dxy_all.min(), dxy_all.max())
ax2d.set_ylim(fxy_all.min(), fxy_all.max())
ax2d.set_title("Sliced Projection ($D_{xy}$ vs $F_{xy}$)")
ax2d.set_xlabel("Displacement Magnitude")
ax2d.set_ylabel("Shear Force Magnitude")
ax2d.grid(True)

# 4. Add Sliders
ax_fz = plt.axes([0.25, 0.1, 0.5, 0.03])
ax_eps = plt.axes([0.25, 0.05, 0.5, 0.03])

fz_slider = Slider(ax_fz, 'Target divergence diff', div_potential_y_small.min(), div_potential_y_small.max(), valinit=div_potential_y_small.mean())
eps_slider = Slider(ax_eps, 'Tolerance', 0.1, 10, valinit=0.1)

# 5. Update Function
def update(val):
    k = fz_slider.val
    eps = eps_slider.val
    
    # Filter data based on slice
    mask = np.abs(div_potential_y_small - k) < eps
    
    # Data for the slice
    d_slice = dxy_all[mask]
    fz_slice = div_potential_y_small[mask]
    fxy_slice = fxy_all[mask]
    
    # Update 2D Scatter (Projection)
    scatter_2d.set_offsets(np.c_[d_slice, fxy_slice])
    
    # Update 3D Highlighted Points
    # For 3D scatter updates, we must use _offsets3d
    slice_scatter_3d._offsets3d = (d_slice, fz_slice, fxy_slice)
    
    fig.canvas.draw_idle()

fz_slider.on_changed(update)
eps_slider.on_changed(update)

# Initialize
update(None)
plt.show()