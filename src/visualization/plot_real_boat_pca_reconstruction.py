import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.extent_model.boat_pca_utils import get_pca_coeffs_from_radii
from src.utils.tools import fourier_basis_matrix

def inverse_fourier_transform(coeffs, num_points=360, symmetry=True):
    target_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
    # The basis matrix evaluates r(theta) = \sum coeffs * basis
    # basis matrix shape is (num_coeffs, num_points)
    # coeffs shape is (num_coeffs,)
    basis = fourier_basis_matrix(target_angles, len(coeffs)) # shape: (len(coeffs), num_points)
    return (basis.T @ coeffs).flatten()

def plot_reconstruction():
    h5_filepath = "data/real_datasets/Nicholasdata_filtered.h5"
    pca_path = "data/input_parameters/ShipDatasetPCAParameters.npz"
    N_pca = 4
    
    if not os.path.exists(h5_filepath):
        print(f"File not found: {h5_filepath}")
        return
        
    if not os.path.exists(pca_path):
        print(f"File not found: {pca_path}")
        return

    # Load PCA Model to easily do the reverse transform
    pca_data = np.load(pca_path)
    mean_fourier = pca_data['mean'].flatten()
    eigenvectors = pca_data['eigenvectors'] 
    
    # 1. Load actual radii and angles from HDF5
    with h5py.File(h5_filepath, 'r') as f:
        if 'ship_trajectory_0' not in f or 'extendRadii' not in f['ship_trajectory_0']:
            print("Required GT shape info missing from HDF5.")
            return
            
        extend_radii = f['ship_trajectory_0']['extendRadii'][:].flatten()
        extend_angles = f['ship_trajectory_0']['extendAngles'][:].flatten()
        
    xs_raw = extend_radii * np.cos(extend_angles)
    ys_raw = extend_radii * np.sin(extend_angles)
    L_gt = np.max(xs_raw) - np.min(xs_raw)
    W_gt = np.max(ys_raw) - np.min(ys_raw)

    print(f"Original Length: {L_gt:.2f}, Width: {W_gt:.2f}")

    # Center the shape
    cx = (np.max(xs_raw) + np.min(xs_raw)) / 2.0
    cy = (np.max(ys_raw) + np.min(ys_raw)) / 2.0
    xs_centered = xs_raw - cx
    ys_centered = ys_raw - cy

    # SQUASH to unit square bounding box to match PCA training data
    # (PCA training data processed_ships.json normalized BOTH X and Y to [-0.5, 0.5])
    xs_squashed = xs_centered / L_gt
    ys_squashed = ys_centered / W_gt

    extend_radii_squashed = np.sqrt(xs_squashed**2 + ys_squashed**2)
    extend_angles_squashed = np.arctan2(ys_squashed, xs_squashed)

    # Sort sequentially for interpolation
    sort_idx = np.argsort(extend_angles_squashed)
    extend_angles = extend_angles_squashed[sort_idx]
    extend_radii = extend_radii_squashed[sort_idx]

    # 2. Get PCA coeffs using length=1.0 since it's fully squashed
    pca_coeffs = get_pca_coeffs_from_radii(extend_radii, extend_angles, 1.0, N_pca, pca_path=pca_path)
    
    print(f"Calculated PCA Coefficients (N={N_pca}):")
    print(pca_coeffs)
    
    # 3. Reconstruct shape
    fourier_coeffs = mean_fourier + (eigenvectors[:, :N_pca] @ pca_coeffs)
    
    dense_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    
    # These are radii of the squashed 1x1 unit-square shape
    normalized_radii_recon = inverse_fourier_transform(fourier_coeffs, num_points=360, symmetry=True)
    
    # Convert squashed shape back to Cartesian unit square coordinates
    xs_unit = normalized_radii_recon * np.cos(dense_angles)
    ys_unit = normalized_radii_recon * np.sin(dense_angles)
    
    # STRETCH back to real-world dimensions
    xs_recon = xs_unit * L_gt
    ys_recon = ys_unit * W_gt
    
    # Re-calculate real radii/angles for MSE evaluation
    radii_recon = np.sqrt(xs_recon**2 + ys_recon**2)
    angles_recon = np.arctan2(ys_recon, xs_recon)
    
    # Evaluation
    # Compare against original centered physical radii
    extend_radii_centered = np.sqrt(xs_centered**2 + ys_centered**2)
    extend_angles_centered = np.arctan2(ys_centered, xs_centered)
    sort_idx_eval = np.argsort(extend_angles_centered)
    
    radii_interp = np.interp(angles_recon, extend_angles_centered[sort_idx_eval], extend_radii_centered[sort_idx_eval])
    mse = np.mean((radii_interp - radii_recon)**2)
    print(f"\nReconstruction Accuracy (N={N_pca}):")
    print(f"Mean Squared Error (MSE): {mse:.4f} m^2")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f} m")

    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Raw extent (Centered for comparison)
    plt.plot(xs_centered, ys_centered, 'b--', label='Centered HDF5 Shape', linewidth=2)
    # Reconstructed extent
    plt.plot(xs_recon, ys_recon, 'r-', label=f'PCA Reconstruction (N_pca={N_pca})', linewidth=2)
    
    # Origin / Coordinate Center
    plt.plot(0, 0, 'kx', markersize=12, markeredgewidth=2, label='Origin (0,0)')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title("Ground Truth Shape vs PCA Reconstruction")
    plt.xlabel("Local X (m)")
    plt.ylabel("Local Y (m)")
    plt.legend()
    
    out_file = "figures/real_boat_pca_reconstruction.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
    plt.show()

if __name__ == '__main__':
    plot_reconstruction()
