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

SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global, calculate_iou

def _calc_iou_for_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        sim_result = pickle.load(f)
        
    config = sim_result.config
    pca_params = np.load(Path(PROJECT_ROOT) / config.tracker.PCA_parameters_path)
    
    ground_truth_ts = sim_result.ground_truth_ts.values
    tracker_results_ts = sim_result.tracker_results_ts.values
    extent_cfg = config.extent
    
    ious = []
    
    for i, res in enumerate(tracker_results_ts):
        if i < len(ground_truth_ts):
            gt_state = ground_truth_ts[i]
            est_state = res.state_posterior.mean
            
            try:
                gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, extent_cfg.shape_coords_body)
                est_x, est_y = compute_estimated_shape_global(est_state, config, pca_params)
                iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                ious.append(iou)
            except Exception:
                ious.append(np.nan)
                
    return np.array(ious)

def get_single_run_iou(sim_name):
    base_path = Path(SIMDATA_PATH)
    pkl_matches = list(base_path.rglob(f"{sim_name}.pkl"))
    if not pkl_matches: return None
    ious = _calc_iou_for_pkl(pkl_matches[0])
    return np.arange(len(ious)), ious

FILTER_COLORS = {
    'Explicit EKF': 'C0',
    'Explicit IEKF': 'C1',
    'Implicit EKF': 'C2',
    'Implicit IEKF': 'C3'
}

def plot_iou_over_time(test_prefix, make_square=False, make_short=False, mode='single'):
    if make_square:
        figsize = (8, 8)
    elif make_short:
        figsize = (10, 4)
    else:
        figsize = (10, 6)
    
    output_dir = PROJECT_ROOT / 'figures' / 'iou'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == 'single':
        simdata_root = Path(SIMDATA_PATH)
        json_files = glob.glob(os.path.join(simdata_root, "**", "*.json"), recursive=True)
        json_files = [f for f in json_files if not f.endswith('_config.json')]
        
        noiseless_data, noisy_data, real_data_list = [], [], []
        
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                if not isinstance(data, dict): continue
            except Exception: continue
                
            sim_name = data.get('name')
            if not sim_name or not sim_name.startswith(test_prefix): continue
                
            method_raw = data.get('method', 'unknown').upper()
            method_map = {'EKF': 'Explicit EKF', 'IEKF': 'Explicit IEKF', 'IMPLICIT_EKF': 'Implicit EKF', 'IMPLICIT_IEKF': 'Implicit IEKF'}
            method = method_map.get(method_raw, method_raw)
                
            is_real = "real_data" in sim_name.lower()
            is_noisy = not ("zeronoise" in sim_name.lower() or "noiseless" in sim_name.lower())
            
            logging.info(f"Extracting single-run IoU for {sim_name}...")
            res = get_single_run_iou(sim_name)
            if res:
                if is_real:
                    real_data_list.append((method, res[0], res[1]))
                elif is_noisy:
                    noisy_data.append((method, res[0], res[1]))
                else:
                    noiseless_data.append((method, res[0], res[1]))
                    
        noiseless_data.sort(key=lambda x: x[0])
        noisy_data.sort(key=lambda x: x[0])
        real_data_list.sort(key=lambda x: x[0])
        
        # Plot Real Data
        if real_data_list:
            plt.figure(figsize=figsize)
            for method, frames, ious in real_data_list:
                plt.plot(frames, ious, label=method, linewidth=2, color=FILTER_COLORS.get(method))
            plt.title(f"Tracking IoU over Time (Real Data)\nDataset: {test_prefix}", fontsize=14)
            plt.xlabel('Frame', fontsize=12); plt.ylabel('IoU', fontsize=12)
            plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(real_data_list)), fancybox=True, shadow=True)
            plt.tight_layout()
            save_path = output_dir / f"iou_{test_prefix}_real_data_mpl.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(); logging.info(f"Saved {save_path}")
                
        # Plot Noiseless
        if noiseless_data:
            plt.figure(figsize=figsize)
            for method, frames, ious in noiseless_data:
                plt.plot(frames, ious, label=method, linewidth=2, color=FILTER_COLORS.get(method))
            plt.title(f"Tracking IoU over Time (Zero-Noise)\nSimulation: {test_prefix}", fontsize=14)
            plt.xlabel('Frame', fontsize=12); plt.ylabel('IoU', fontsize=12)
            plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(noiseless_data)), fancybox=True, shadow=True)
            plt.tight_layout()
            save_path = output_dir / f"iou_{test_prefix}_noiseless_mpl.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(); logging.info(f"Saved {save_path}")
                
        # Plot Noisy
        if noisy_data:
            plt.figure(figsize=figsize)
            for method, frames, ious in noisy_data:
                plt.plot(frames, ious, label=method, linewidth=2, color=FILTER_COLORS.get(method))
            plt.title(f"Tracking IoU over Time (Noisy)\nSimulation: {test_prefix}", fontsize=14)
            plt.xlabel('Frame', fontsize=12); plt.ylabel('IoU', fontsize=12)
            plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(noisy_data)), fancybox=True, shadow=True)
            plt.tight_layout()
            save_path = output_dir / f"iou_{test_prefix}_noisy_mpl.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(); logging.info(f"Saved {save_path}")

    else: # mode == 'mc'
        experiment_dir = PROJECT_ROOT / "data" / "results" / "mc_experiments" / test_prefix
        if not experiment_dir.exists():
            logging.error(f"MC Experiment directory {experiment_dir} not found.")
            return
            
        filter_keys = ['EKF', 'IEKF', 'IMPLICIT_EKF', 'IMPLICIT_IEKF']
        mc_data = []

        for key in filter_keys:
            method_dir = experiment_dir / key.lower()
            if not method_dir.exists(): continue
            
            result_files = list(method_dir.glob("*_results.pkl"))
            if not result_files: continue
                 
            all_ious = []
            method_map = {'EKF': 'Explicit EKF', 'IEKF': 'Explicit IEKF', 'IMPLICIT_EKF': 'Implicit EKF', 'IMPLICIT_IEKF': 'Implicit IEKF'}
            method = method_map.get(key, key)
            
            logging.info(f"Extracting ensemble IoU for {key} over {len(result_files)} runs...")
            for rf in result_files:
                run_ious = _calc_iou_for_pkl(rf)
                all_ious.append(run_ious)
                
            try:
                # Shape (M, K)
                iou_matrix = np.array(all_ious)
                avg_iou_ts = np.nanmean(iou_matrix, axis=0)
                mc_data.append((method, np.arange(len(avg_iou_ts)), avg_iou_ts))
            except ValueError:
                logging.error(f"Run lengths vary in {key}. Cannot stack.")
                continue
                
        if mc_data:
            plt.figure(figsize=figsize)
            for method, frames, ious in mc_data:
                plt.plot(frames, ious, label=method, linewidth=2, color=FILTER_COLORS.get(method))
                
            plt.title(f"Average Tracking IoU (Noisy - MC)\nExperiment: {test_prefix}", fontsize=14)
            plt.xlabel('Frame', fontsize=12); plt.ylabel('Average IoU', fontsize=12)
            plt.ylim(0, 1.05); plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower left', ncol=2, fancybox=True, shadow=True)
            plt.tight_layout()
            
            save_path = output_dir / f"iou_{test_prefix}_noisy_mc_mpl.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(); logging.info(f"Saved {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot IoU from PKL files")
    parser.add_argument('--test', type=str, default='exp1_linear_noise015', help='Prefix or MC folder name')
    parser.add_argument('--square', action='store_true', help='Make the output plot have a 1:1 aspect ratio')
    parser.add_argument('--short', action='store_true', help='Make the output plot shorter vertically')
    parser.add_argument('--mode', type=str, choices=['single', 'mc'], default='mc', help='Plotting mode')
    args = parser.parse_args()
    
    plot_iou_over_time(args.test, make_square=args.square, make_short=args.short, mode=args.mode)
