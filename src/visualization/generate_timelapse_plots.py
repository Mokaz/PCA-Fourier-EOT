import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import matplotlib.patches as patches

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH, FIGURES_PATH
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global
from src.states.states import State_PCA, State_GP

def generate_timelapse_for_file(filename: Path, mode: str = 'both', num_snapshots: int = 4):
    print(f"Processing {filename.name}...")
    
    try:
        with open(filename, "rb") as f:
            sim_result = pickle.load(f)
    except Exception as e:
        print(f"Failed to load {filename.name}: {e}")
        return

    config = sim_result.config
    
    pca_params = None
    if hasattr(config.tracker, 'PCA_parameters_path'):
        pca_path = Path(config.tracker.PCA_parameters_path)
        if pca_path.exists():
            pca_params = np.load(pca_path)
        elif (PROJECT_ROOT / config.tracker.PCA_parameters_path).exists():
             pca_params = np.load(PROJECT_ROOT / config.tracker.PCA_parameters_path)

    tracker_results = list(sim_result.tracker_results_ts.values)
    ground_truth_states = list(sim_result.ground_truth_ts.values)
    
    num_frames = len(tracker_results)
    if num_frames < num_snapshots:
        print(f"Skipping {filename.name}: Not enough frames ({num_frames})")
        return

    indices = np.linspace(0, num_frames - 1, num_snapshots, dtype=int)
    
    all_x = []
    all_y = []
    
    lidar_pos = config.lidar.lidar_position
    all_x.append(lidar_pos[0])
    all_y.append(lidar_pos[1])

    for idx in indices:
        if idx < len(ground_truth_states):
            s = ground_truth_states[idx]
            all_x.append(s.x)
            all_y.append(s.y)
        else:
            s_est = tracker_results[idx].state_posterior.mean
            all_x.append(s_est.x)
            all_y.append(s_est.y)
    
    padding = 20
    min_x, max_x = min(all_x) - padding, max(all_x) + padding
    min_y, max_y = min(all_y) - padding, max(all_y) + padding
    
    data_height = max_x - min_x
    data_width = max_y - min_y
    
    modes_to_run = ['single', 'four'] if mode == 'both' else [mode]

    for current_mode in modes_to_run:
        if current_mode == 'single':
            fig_width = 12
            fig_height = fig_width * (data_height / data_width)
            fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            axes_list = [axs] * len(indices)
            
            full_history_x = [s.x for s in ground_truth_states]
            full_history_y = [s.y for s in ground_truth_states]
            axs.plot(full_history_y, full_history_x, color='royalblue', linewidth=1, label='Path', zorder=1)
            
            axs.set_xlim(min_y, max_y)
            axs.set_ylim(min_x, max_x)
        else:
            fig_width = 5 * num_snapshots
            fig_height = (fig_width / num_snapshots) * (data_height / data_width) + 2
            fig, axs = plt.subplots(1, num_snapshots, figsize=(fig_width, fig_height), sharey=True, sharex=True)
            axes_list = [axs] if num_snapshots == 1 else axs
        
        for i, idx in enumerate(indices):
            ax = axes_list[i]
            
            tracker_res = tracker_results[idx]
            est_state = tracker_res.state_posterior.mean
            
            if tracker_res.measurements is not None:
                z_lidar = tracker_res.measurements.reshape((-1, 2))
            else:
                z_lidar = np.empty((0, 2))
            
            do_label = (i == 0)
            
            if current_mode == 'four':
                history_end = min(idx+1, len(ground_truth_states))
                history_x = [s.x for s in ground_truth_states[:history_end]]
                history_y = [s.y for s in ground_truth_states[:history_end]]
                if history_x:
                    ax.plot(history_y, history_x, color='royalblue', linewidth=1, label='Path' if do_label else "", zorder=1)
            
            if idx < len(ground_truth_states):
                gt_state = ground_truth_states[idx]
                gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
                ax.plot(gt_shape_y, gt_shape_x, color='black', linewidth=1.5, label='GT Shape' if do_label else "", zorder=2)
            
            if tracker_res.state_prior is not None:
                prior_state = tracker_res.state_prior.mean
                prior_shape_x, prior_shape_y = compute_estimated_shape_global(prior_state, config, pca_params)
                ax.plot(prior_shape_y, prior_shape_x, color='purple', linestyle=':', linewidth=1.5, label='Prior Shape' if do_label else "", zorder=2.5)
                ax.scatter(prior_state.y, prior_state.x, color='purple', marker='D', s=20, label='Prior Centroid' if do_label else "", zorder=2.6)

            est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
            ax.plot(est_shape_y, est_shape_x, color='green', linewidth=1.5, linestyle='-', label='Est Shape' if do_label else "", zorder=3)

            for z in z_lidar:
                dist = np.linalg.norm(z - np.array(lidar_pos))
                if dist < config.lidar.max_distance:
                    ax.plot([lidar_pos[1], z[1]], [lidar_pos[0], z[0]], color='red', alpha=0.3, linewidth=0.5, zorder=4)
                    ax.scatter(z[1], z[0], s=5, color='red', marker='.', zorder=5) 
            
            if do_label:
                ax.plot([], [], color='red', label='LiDAR', zorder=4)
            
            if hasattr(tracker_res, 'predicted_measurement') and tracker_res.predicted_measurement is not None:
                z_pred_cart = tracker_res.predicted_measurement.mean.reshape((-1, 2))
                ax.scatter(z_pred_cart[:, 1], z_pred_cart[:, 0], color='orange', marker='x', s=10, label='Pred Meas' if do_label else "", zorder=5.5)

            arrow_len = 5.0
            ax.arrow(est_state.y, est_state.x, 
                     arrow_len * np.sin(est_state.yaw), arrow_len * np.cos(est_state.yaw),
                     head_width=1.5, head_length=1.5, fc='purple', ec='purple', label='Heading' if do_label else "", zorder=6)

            if current_mode == 'single':
                ax.text(est_state.y, est_state.x + 8, f"t={idx}", fontsize=11, zorder=10, clip_on=True, ha='center')
            else: 
                ax.set_title(f"Frame {idx}")
                ax.set_aspect('equal', 'box')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlim(min_y, max_y)
                ax.set_ylim(min_x, max_x)
                if i == 0:
                    ax.set_ylabel("North [m]")
                ax.set_xlabel("East [m]")
        
        if current_mode == 'single':
            axs.set_title(f"Timelapse: {filename.stem}")
            axs.set_aspect('equal', 'box')
            axs.grid(True, linestyle=':', alpha=0.6)
            axs.set_ylabel("North [m]")
            axs.set_xlabel("East [m]")
            
            handles, labels = axs.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs.legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.9)
            plt.tight_layout()
            suffix = "_timelapse_single"
        else:
            handles, labels = axes_list[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=8, bbox_to_anchor=(0.5, 0.92), frameon=False)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            suffix = "_timelapse_four"

        output_dir = FIGURES_PATH / "timelapses"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{filename.stem}{suffix}.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved {current_mode} mode to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate timelapse plots.')
    parser.add_argument('--mode', type=str, default='both', choices=['single', 'four', 'both'], help='Plotting mode')
    parser.add_argument('--num_snapshots', type=int, default=4, help='Number of snapshot timesteps to plot (default: 4)')
    args = parser.parse_args()

    # Search recursively to handle runs inside their own directories
    all_files = list(SIMDATA_PATH.rglob("*.pkl"))
    # Filter out historical/old runs to process only the active new results
    files = sorted([f for f in all_files if not f.parent.name.startswith("old") and "old" not in f.parts])
    
    if not files:
        print("No .pkl files found in results folder.")
        return

    for f in files:
        generate_timelapse_for_file(f, mode=args.mode, num_snapshots=args.num_snapshots)

if __name__ == "__main__":
    main()
