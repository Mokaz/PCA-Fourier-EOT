import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.exp5_real_data import base_config, load_zpos_sequence, load_ground_truth_sequence
from src.utils.geometry_utils import compute_exact_vessel_shape_global

# Matplotlib configuration for thesis
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

def plot_real_data_measurements():
    MAT_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.mat"
    
    H5_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.h5"
    
    if not os.path.exists(MAT_FILE_PATH):
        print(f"Error: Could not find {MAT_FILE_PATH}")
        return

    dt = 0.1
    # This might take a moment to load
    measurements_ts = load_zpos_sequence(MAT_FILE_PATH, dt=dt)
    if os.path.exists(H5_FILE_PATH):
        ground_truth_ts, _, _, _ = load_ground_truth_sequence(H5_FILE_PATH, dt=dt, N_pca=4)
    else:
        ground_truth_ts = None
        
    config = base_config

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot all measurements (less subsampled for detail)
    all_meas_x = []
    all_meas_y = []
    for i, (ts, scan) in enumerate(measurements_ts.items()):
        if scan.x.size > 0 and i % 5 == 0:  # plot every 5th scan
            all_meas_x.extend(scan.x)
            all_meas_y.extend(scan.y)

    # Plot (East, North) -> (y, x) with darker, clearer points
    ax.scatter(all_meas_y, all_meas_x, s=5, color="red", alpha=0.5, label="Lidar Scans", edgecolors='none')

    # Plot Lidar Position
    lidar_north, lidar_east = config.lidar.lidar_position
    ax.scatter([lidar_east], [lidar_north], s=120, color="orange", marker="^", label="Lidar", edgecolors='black', zorder=5)

    # Plot initial vessel extent
    initial_state = config.tracker.initial_state
    
    try:
        # Get shape coords in body frame from config
        shape_body_coords = config.extent.shape_coords_body
        
        shape_x, shape_y = compute_exact_vessel_shape_global(initial_state, shape_body_coords)
        # Close the polygon for plotting
        shape_x = np.append(shape_x, shape_x[0])
        shape_y = np.append(shape_y, shape_y[0])
        ax.plot(shape_y, shape_x, color="purple", linewidth=2.5, label="Initial Shape", zorder=4)

    except Exception as e:
        print(f"Could not compute exact vessel shape: {e}")

    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    
    # You can comment out the title for a paper, usually captions are used instead
    ax.set_title("Real Data Scans and Initial Vessel Prior")
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend
    leg = ax.legend(loc="upper left", framealpha=0.9, edgecolor='black')
    for handle in leg.legend_handles:
        if isinstance(handle, plt.matplotlib.collections.PathCollection):
            handle.set_alpha(1.0)
            if handle.get_facecolor()[0][2] > 0.5: # if it's the blue lidar scans
                handle.set_sizes([30.0]) # Make legend marker bigger

    # Save to PDF
    output_dir = os.path.join(PROJECT_ROOT, "figures", "real_datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "real_data_measurements.pdf")
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_real_data_measurements()
