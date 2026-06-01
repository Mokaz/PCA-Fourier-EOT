import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# Increase base font sizes for thesis readability
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Setup paths
SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.tools import generate_fourier_function
from src.extent_model.boat_pca_utils import get_gt_pca_coeffs_for_boat

PCA_FILE = PROJECT_ROOT / "data" / "input_parameters" / "ShipDatasetPCAParameters.npz"
JSON_FILE = PROJECT_ROOT / "data" / "processed_ships.json"

def check_pca_spillage(N_pca=4, plot_top_violators=3):
    """
    Checks all boats in the dataset to see if their PCA-truncated reconstructions
    spill outside the[-0.5, 0.5] normalized bounding box.
    """
    pca_data = np.load(PCA_FILE)
    pca_mean = pca_data['mean']
    pca_eigenvectors = pca_data['eigenvectors'][:, :N_pca].real

    with open(JSON_FILE, 'r') as f:
        boats_data = json.load(f)

    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    g_func = generate_fourier_function(N_f=64)
    G = g_func(angles).T  # Shape: (360, 64)

    violators =[]

    for boat in boats_data:
        if not boat.get('is_boat') or boat.get('is_kayak'):
            continue
            
        boat_id = boat['id']
        try:
            coeffs = get_gt_pca_coeffs_for_boat(boat_id, N_pca=N_pca, pca_path=PCA_FILE)
        except Exception:
            continue
            
        # Reconstruct the shape with truncated PCA
        fourier_coeffs = pca_mean + pca_eigenvectors @ coeffs.reshape(-1, 1)
        r_vals = (G @ fourier_coeffs).flatten()
        
        # Calculate Cartesian coordinates
        x_vals = r_vals * np.cos(angles)
        y_vals = r_vals * np.sin(angles)
        
        # Check max bounds
        max_x, min_x = np.max(x_vals), np.min(x_vals)
        max_y, min_y = np.max(y_vals), np.min(y_vals)
        min_r = np.min(r_vals)
        
        spillage_x = max(0, max_x - 0.5, -0.5 - min_x)
        spillage_y = max(0, max_y - 0.5, -0.5 - min_y)
        total_spillage = spillage_x + spillage_y
        
        if total_spillage > 1e-3 or min_r < 0:
            violators.append({
                'id': boat_id, 'x_vals': x_vals, 'y_vals': y_vals, 
                'spillage': total_spillage, 'min_r': min_r,
                'true_r': boat['radii']
            })

    violators = sorted(violators, key=lambda x: x['spillage'], reverse=True)
    print(f"Out of {len(boats_data)} boats, {len(violators)} spill outside the 1x1 bounds or have negative radii when truncated to N={N_pca}.")

    # --- Plot the Top Violators ---
    if len(violators) > 0:
        fig, axs = plt.subplots(1, min(plot_top_violators, len(violators)), figsize=(15, 5))
        if plot_top_violators == 1: axs = [axs]
        
        for i, ax in enumerate(axs):
            v = violators[i]
            # True Shape
            true_r = np.array(v['true_r'])
            true_ang = np.linspace(-np.pi, np.pi, len(true_r), endpoint=False)
            ax.plot(true_r * np.cos(true_ang), true_r * np.sin(true_ang), 'k:', label="True Shape")
            
            # Reconstructed Shape
            x_rec, y_rec = v['x_vals'], v['y_vals']
            x_rec = np.append(x_rec, x_rec[0])
            y_rec = np.append(y_rec, y_rec[0])
            ax.plot(x_rec, y_rec, 'r-', linewidth=2, label=f"PCA N={N_pca}")
            
            # 1x1 Bounding Box
            box_x =[-0.5, 0.5, 0.5, -0.5, -0.5]
            box_y =[-0.5, -0.5, 0.5, 0.5, -0.5]
            ax.plot(box_x, box_y, 'b--', label="1x1 Bounds")
            
            ax.set_title(f"Boat {v['id']} | Spillage: {v['spillage']:.3f}m")
            ax.set_aspect('equal')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle=':', alpha=0.6)
            
        plt.suptitle(f"Top PCA Truncation Overshoots (N={N_pca})")
        plt.tight_layout()
        plt.show()

def visualize_feasible_pca_space(N_pca=4):
    """
    Visualizes 2D slices of the N-dimensional PCA space, highlighting the 
    strictly convex feasible region where the shape stays inside the 1x1 box 
    and radius > 0.
    """
    pca_data = np.load(PCA_FILE)
    pca_mean = pca_data['mean']
    pca_eigenvectors = pca_data['eigenvectors'][:, :N_pca].real
    if 'eigenvalues' in pca_data:
        pca_eigenvalues = pca_data['eigenvalues'].real[:N_pca]
    else:
        pca_eigenvalues = np.ones(N_pca)

    angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    g_func = generate_fourier_function(N_f=64)
    G = g_func(angles).T  # Shape: (180, 64)

    # Get the coefficients for actual boats to overlay them
    with open(JSON_FILE, 'r') as f:
        boats_data = json.load(f)
        
    actual_coeffs =[]
    actual_cats = []
    
    for boat in boats_data:
        if not boat.get('is_boat') or boat.get('is_kayak'):
            continue
        try:
            coeffs = get_gt_pca_coeffs_for_boat(boat['id'], N_pca=N_pca, pca_path=PCA_FILE)
            
            # calculate category
            r_4d = (G @ (pca_mean + pca_eigenvectors @ coeffs.reshape(-1, 1))).flatten()
            x_4d = r_4d * np.cos(angles)
            y_4d = r_4d * np.sin(angles)
            
            if np.any(r_4d < 0):
                cat = 2 # Red
            elif np.max(np.abs(x_4d)) > 0.505 or np.max(np.abs(y_4d)) > 0.505:
                cat = 1 # Orange
            else:
                cat = 0 # Green
                
            actual_coeffs.append(coeffs)
            actual_cats.append(cat)
        except Exception:
            continue
            
    actual_coeffs = np.array(actual_coeffs)  # Shape: (N_boats, N_pca)
    actual_cats = np.array(actual_cats)

    # Print total counts
    total_boats = len(actual_cats)
    green_count = np.sum(actual_cats == 0)
    orange_count = np.sum(actual_cats == 1)
    red_count = np.sum(actual_cats == 2)

    chi2_val = float(chi2.ppf(0.99, df=N_pca))
    mahalanobis_sq = np.sum((actual_coeffs ** 2) / pca_eigenvalues.reshape(1, -1), axis=1)
    within_mahala = np.sum(mahalanobis_sq <= chi2_val)

    print(f"\nBoat Categories (Out of {total_boats}):")
    print(f"  Green (Feasible): {green_count}")
    print(f"  Orange (Spills > 0.5): {orange_count}")
    print(f"  Red (Neg R): {red_count}")
    print(f"  Within 99% Mahalanobis Bound: {within_mahala}")

    # Get all combinations of dimensions (e.g., PC0 vs PC1, PC0 vs PC2...)
    pairs = list(itertools.combinations(range(N_pca), 2))
    cols = 3
    rows = int(np.ceil(len(pairs) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    cmap = ListedColormap(['#e74c3c', '#f39c12', '#2ecc71']) # Red (Neg R), Orange (Spill), Green (Feasible)

    res = 200
    grid_vals = np.linspace(-10, 10, res)
    PCA_grid, PCB_grid = np.meshgrid(grid_vals, grid_vals)
    
    # Pre-calculate trig arrays for fast bounding box projection
    cos_ang = np.cos(angles)[:, None]
    sin_ang = np.sin(angles)[:, None]

    for idx, (d1, d2) in enumerate(pairs):
        ax = axes[idx]
        
        # --- Vectorized Evaluation over the Grid ---
        # e matrix holds state vectors [PC0, PC1, PC2, PC3]^T for all 40,000 points
        E = np.zeros((N_pca, res * res))
        E[d1, :] = PCA_grid.ravel()
        E[d2, :] = PCB_grid.ravel()
        
        # Calculate Radii for all 40,000 points simultaneously
        # G: (180, 64) | PCA_Mean: (64, 1) | Eigenvectors: (64, 4) | E: (4, 40000)
        # R_vals shape: (180, 40000)
        R_vals = G @ (pca_mean + pca_eigenvectors @ E)
        
        # Calculate X and Y coordinates: shape (180, 40000)
        X_vals = R_vals * cos_ang
        Y_vals = R_vals * sin_ang
        
        # Condition 1: Negative radius (Looping geometry)
        neg_mask = np.any(R_vals < 0, axis=0)
        
        # Condition 2: Spills outside 1x1 box
        spill_mask = np.any(np.abs(X_vals) > 0.505, axis=0) | np.any(np.abs(Y_vals) > 0.505, axis=0)
        
        # Assign Scores: 1 (Feasible), -1 (Spill), -2 (Neg R)
        feasibility = np.ones(res * res)
        feasibility[spill_mask] = -1
        feasibility[neg_mask] = -2
        feasibility_grid = feasibility.reshape(res, res)
        
        # --- Plotting ---
        ax.pcolormesh(PCA_grid, PCB_grid, feasibility_grid, cmap=cmap, alpha=0.6, shading='auto')

        # Plot Mahalanobis 99% Bound
        chi2_val = float(chi2.ppf(0.99, df=N_pca))
        width = 2 * np.sqrt(chi2_val * pca_eigenvalues[d1])
        height = 2 * np.sqrt(chi2_val * pca_eigenvalues[d2])
        ellipse = Ellipse((0, 0), width=width, height=height, 
                          edgecolor='mediumorchid', fill=False, linewidth=2, linestyle='--',
                          label='99% Mahalanobis')
        ax.add_patch(ellipse)

        # Plot actual boats
        if len(actual_coeffs) > 0:
            colors = np.array(['#2ecc71', '#f39c12', '#e74c3c'])
            ax.scatter(actual_coeffs[:, d1], actual_coeffs[:, d2], c=colors[actual_cats], s=12, alpha=0.8, edgecolors='black', linewidths=0.5)

        # Mark Mean
        ax.scatter(0, 0, marker='X', color='blue', s=80, label='Mean', zorder=5)

        ax.set_title(f"PC{d1} vs PC{d2}")
        ax.set_xlabel(f"PCA Coefficient {d1}")
        ax.set_ylabel(f"PCA Coefficient {d2}")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    # Turn off unused axes if any
    for idx in range(len(pairs), len(axes)):
        axes[idx].axis('off')

    # Custom global legend
    green_patch = mpatches.Patch(color='#2ecc71', alpha=0.6, label='Feasible (Inside 1x1)')
    orange_patch = mpatches.Patch(color='#f39c12', alpha=0.6, label='Infeasible (Spills > 0.5)')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.6, label='Infeasible (r < 0)')
    mahala_line = plt.Line2D([0], [0], color='mediumorchid', linestyle='--', linewidth=2, label='99% Mahalanobis')
    mean_marker = plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', markersize=10, label='Mean')
    fig.legend(handles=[green_patch, orange_patch, red_patch, mahala_line, mean_marker], 
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=14)

    plt.suptitle("Feasible PCA Space (All 2D Slices for N=4)\nUnplotted dimensions held at 0 (Mean)", fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def verify_mahalanobis_bound_monte_carlo(N_pca=4, num_samples=100000):
    """
    Verifies via Monte Carlo sampling if any points within the N-dimensional 99% Mahalanobis 
    ellipsoid result in a negative radius (Red category).
    """
    pca_data = np.load(PCA_FILE)
    pca_mean = pca_data['mean']
    pca_eigenvectors = pca_data['eigenvectors'][:, :N_pca].real
    if 'eigenvalues' in pca_data:
        pca_eigenvalues = pca_data['eigenvalues'].real[:N_pca]
    else:
        pca_eigenvalues = np.ones(N_pca)

    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    g_func = generate_fourier_function(N_f=64)
    G = g_func(angles).T  # Shape: (360, 64)

    # 99% Mahalanobis threshold
    chi2_val = float(chi2.ppf(0.99, df=N_pca))
    
    # 1. Generate points uniformly in an N-dimensional unit hyperball
    norm_vals = np.random.randn(num_samples, N_pca)
    radii = np.random.rand(num_samples) ** (1 / N_pca)
    points_in_ball = norm_vals / np.linalg.norm(norm_vals, axis=1, keepdims=True) * radii[:, None]
    
    # 2. Scale points strictly so the bounds of the unit ball map exactly to the Mahalanobis ellipsoid limits
    axis_scales = np.sqrt(chi2_val * pca_eigenvalues)
    E_samples = points_in_ball * axis_scales  # Shape: (num_samples, N_pca)
    
    # 3. Evaluate the radii bounds simultaneously via vectorization
    fourier_samples = pca_mean + pca_eigenvectors @ E_samples.T
    R_vals = G @ fourier_samples  # Shape: (360, num_samples)
    
    # Check conditions
    neg_mask = np.any(R_vals < 0, axis=0) # True if ANY angle has radius < 0
    
    cos_ang = np.cos(angles)[:, None]
    sin_ang = np.sin(angles)[:, None]
    X_vals = R_vals * cos_ang
    Y_vals = R_vals * sin_ang
    spill_mask = np.any(np.abs(X_vals) > 0.505, axis=0) | np.any(np.abs(Y_vals) > 0.505, axis=0)
    
    red_count = np.sum(neg_mask)
    orange_count = np.sum(spill_mask & ~neg_mask)
    green_count = np.sum(~spill_mask & ~neg_mask)
    
    print("\n--- Mahalanobis Bound Verification (Monte Carlo) ---")
    print(f"Sampled {num_samples:,} random configurations inside the N={N_pca} 99% Mahalanobis ellipsoid.")
    print(f"  Green (Feasible): {green_count:,} ({green_count/num_samples*100:.2f}%)")
    print(f"  Orange (Spills) : {orange_count:,} ({orange_count/num_samples*100:.2f}%)")
    print(f"  Red (Neg R)     : {red_count:,} ({red_count/num_samples*100:.2f}%)")
    
    if red_count == 0:
        print("=> VERIFIED: NO red points found. The 99% Mahalanobis hyperspace strictly excludes overlapping/looping geometry.")
    else:
        print(f"=> FAILED: Found {red_count} points with negative radii inside the boundaries.")

if __name__ == "__main__":
    N_pca = 4
    print("1. Checking PCA Truncation Spillage...")
    check_pca_spillage(N_pca=N_pca, plot_top_violators=3)

    print("\n2. Running Monte Carlo Verification...")
    verify_mahalanobis_bound_monte_carlo(N_pca=N_pca, num_samples=100_000)
    
    print("\n3. Generating All 2D Convex Space Slices...")
    visualize_feasible_pca_space(N_pca=N_pca)