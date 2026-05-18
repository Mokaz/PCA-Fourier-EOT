import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from global_project_paths import SIMDATA_PATH

from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global, calculate_iou

def extract_iou_data(sim_name):
    """Loads the PKL file and calculates IoU for each frame."""
    base_path = Path(SIMDATA_PATH)
    pkl_matches = list(base_path.rglob(f"{sim_name}.pkl"))
    
    if not pkl_matches:
        logging.warning(f"Could not find {sim_name}.pkl")
        return None
        
    pkl_path = pkl_matches[0]
    with open(pkl_path, "rb") as f:
        sim_result = pickle.load(f)
        
    config = sim_result.config
    pca_params = np.load(Path(PROJECT_ROOT) / config.tracker.PCA_parameters_path)
    
    ground_truth_ts = sim_result.ground_truth_ts.values
    tracker_results_ts = sim_result.tracker_results_ts.values
    extent_cfg = config.extent
    
    frames = []
    ious = []
    
    # Calculate IoU per frame
    for i, res in enumerate(tracker_results_ts):
        if i < len(ground_truth_ts):
            gt_state = ground_truth_ts[i]
            est_state = res.state_posterior.mean
            
            try:
                gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, extent_cfg.shape_coords_body)
                est_x, est_y = compute_estimated_shape_global(est_state, config, pca_params)
                iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                frames.append(i)
                ious.append(iou)
            except Exception:
                pass
                
    return frames, ious


def plot_iou_over_time():
    """Iterates through runs and plots grouped IoU line charts using Matplotlib."""
    simdata_root = Path(SIMDATA_PATH)
    json_files = glob.glob(os.path.join(simdata_root, "**", "*.json"), recursive=True)
    json_files = [f for f in json_files if not f.endswith('_config.json')]
    
    noiseless_data = []
    noisy_data = []
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            
        sim_name = data.get('name')
        if not sim_name:
            continue
            
        method = data.get('method', 'unknown').upper()
        # Differentiate groups by checking for "zeronoise" in the filename
        is_noisy = "zeronoise" not in sim_name.lower()
        
        logging.info(f"Extracting IoU for {sim_name}...")
        res = extract_iou_data(sim_name)
        if res:
            frames, ious = res
            if is_noisy:
                noisy_data.append((method, frames, ious))
            else:
                noiseless_data.append((method, frames, ious))
                
    # Sort data alphabetically by method name for consistent legends
    noiseless_data.sort(key=lambda x: x[0])
    noisy_data.sort(key=lambda x: x[0])

    # 1. Plot Noiseless Scenario
    if noiseless_data:
        plt.figure(figsize=(10, 6))
        for method, frames, ious in noiseless_data:
            plt.plot(frames, ious, label=method, linewidth=2)
            
        plt.title('Tracking IoU over Time (Zero-Noise)\nSimulation 1: Constant Speed Trajectory', fontsize=14)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('IoU', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        # Position legend below the chart
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(noiseless_data)), fancybox=True, shadow=True)
        plt.tight_layout()
        
        plt.savefig('iou_noiseless_mpl.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved iou_noiseless_mpl.png")
            
    # 2. Plot Noisy Scenario
    if noisy_data:
        plt.figure(figsize=(10, 6))
        for method, frames, ious in noisy_data:
            plt.plot(frames, ious, label=method, linewidth=2)
            
        plt.title('Tracking IoU over Time (Noisy)\nSimulation 1: Constant Speed Trajectory', fontsize=14)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('IoU', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        # Position legend below the chart
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(noisy_data)), fancybox=True, shadow=True)
        plt.tight_layout()
        
        plt.savefig('iou_noisy_mpl.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Saved iou_noisy_mpl.png")

if __name__ == '__main__':
    plot_iou_over_time()