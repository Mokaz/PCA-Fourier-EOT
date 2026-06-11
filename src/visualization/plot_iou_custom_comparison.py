import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the IoU calculation function from your existing script
from plot_iou_over_time import _calc_iou_for_pkl

def plot_custom_comparison():
    # Define the exact paths provided
    path_scaled = Path(r"E:\GitHub2\GP-PCA-EOT\data\results\single_runs\test_exp2_scaled_R")
    path_non_scaled = Path(r"E:\GitHub2\GP-PCA-EOT\data\results\mc_experiments\exp2_complex_maneuvers_noise015\implicit_iekf\run_000_results.pkl")
    
    # Resolve the 'scaled R' path (in case it's a folder or missing the .pkl extension)
    if path_scaled.is_dir():
        pkl_scaled = list(path_scaled.rglob("*.pkl"))[0]
    elif path_scaled.suffix != '.pkl':
        pkl_scaled = path_scaled.with_name(path_scaled.name + '.pkl')
    else:
        pkl_scaled = path_scaled
        
    print(f"Loading 'Scaled R' data from: {pkl_scaled}")
    ious_scaled = _calc_iou_for_pkl(pkl_scaled)
    
    print(f"Loading 'Non scaled' data from: {path_non_scaled}")
    ious_non_scaled = _calc_iou_for_pkl(path_non_scaled)
    
    # Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Plot both IoU arrays on the same figure
    plt.plot(np.arange(len(ious_scaled)), ious_scaled, label='Scaled R', linewidth=2, color='C0')
    plt.plot(np.arange(len(ious_non_scaled)), ious_non_scaled, label='Non scaled', linewidth=2, color='C1')
    
    # Formatting the plot
    plt.title("Tracking IoU over Time: Scaled R vs Non-scaled", fontsize=14)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
    plt.tight_layout()
    
    # Define output directory and save
    output_dir = Path(r"E:\GitHub2\GP-PCA-EOT\figures\iou")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "iou_scaled_vs_non_scaled.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    plot_custom_comparison()