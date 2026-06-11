import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH, FIGURES_PATH
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global

FILTER_COLORS = {
    'EKF': 'C0',
    'IEKF': 'C1',
    'IMPLICIT_EKF': 'C2',
    'IMPLICIT_IEKF': 'C3',
    'progressive_1_barebones': 'C0',
    'progressive_2_safeguards': 'C1',
    'progressive_3_neginfo': 'C2',
    'progressive_4_full': 'C3'
}

FILTER_LABELS = {
    'EKF': 'Explicit EKF',
    'IEKF': 'Explicit IEKF',
    'IMPLICIT_EKF': 'Implicit EKF',
    'IMPLICIT_IEKF': 'Implicit IEKF',
    'progressive_1_barebones': 'Barebones',
    'progressive_2_safeguards': 'Safeguards',
    'progressive_3_neginfo': 'Neg Info',
    'progressive_4_full': 'Full'
}

def load_pca_params(config):
    pca_params = None
    if hasattr(config.tracker, 'PCA_parameters_path'):
        pca_path = Path(config.tracker.PCA_parameters_path)
        if pca_path.exists():
            pca_params = np.load(pca_path)
        elif (PROJECT_ROOT / config.tracker.PCA_parameters_path).exists():
             pca_params = np.load(PROJECT_ROOT / config.tracker.PCA_parameters_path)
    return pca_params

def get_filter_runs(prefix, folder=None, run_id=None):
    """Finds the .pkl files matching the prefix and groups them by filter method."""
    all_files = list(SIMDATA_PATH.rglob("*.pkl"))
    files = [f for f in all_files if not f.parent.name.startswith("old") and "old" not in f.parts]
    
    if folder:
        files = [f for f in files if folder.replace('\\', '/') in str(f).replace('\\', '/')]
        
    # Match if prefix is start of filename, or if it's part of the directory path
    matched_files = [f for f in files if f.stem.startswith(prefix) or any(prefix in p for p in f.parts)]
    
    if run_id:
        matched_files = [f for f in matched_files if run_id in f.stem]
        
    runs = {}
    for f in matched_files:
        path_str = str(f).lower()
        if 'progressive_1_barebones' in path_str:
            if 'progressive_1_barebones' not in runs: runs['progressive_1_barebones'] = f
        elif 'progressive_2_safeguards' in path_str:
            if 'progressive_2_safeguards' not in runs: runs['progressive_2_safeguards'] = f
        elif 'progressive_3_neginfo' in path_str:
            if 'progressive_3_neginfo' not in runs: runs['progressive_3_neginfo'] = f
        elif 'progressive_4_full' in path_str:
            if 'progressive_4_full' not in runs: runs['progressive_4_full'] = f
        elif 'implicit_iekf' in path_str or 'exp4_wrong_init' in path_str:
            runs['IMPLICIT_IEKF'] = f
        elif 'implicit_ekf' in path_str:
            runs['IMPLICIT_EKF'] = f
        elif 'iekf' in path_str:
            runs['IEKF'] = f
        elif 'ekf' in path_str:
            runs['EKF'] = f
            
    return runs

def plot_overlay(runs, prefix):
    """Plots the final frame of all filters overlaid onto a single plot."""
    print(f"Generating Final Frame Overlay for {prefix}...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    gt_plotted = False
    
    for method_key in ['EKF', 'IEKF', 'IMPLICIT_EKF', 'IMPLICIT_IEKF', 'progressive_1_barebones', 'progressive_2_safeguards', 'progressive_3_neginfo', 'progressive_4_full']:
        if method_key not in runs:
            continue
            
        with open(runs[method_key], "rb") as f:
            sim_result = pickle.load(f)
            
        config = sim_result.config
        pca_params = load_pca_params(config)
        
        gt_states = list(sim_result.ground_truth_ts.values)
        tracker_results = list(sim_result.tracker_results_ts.values)
        
        final_idx = min(len(gt_states) - 1, len(tracker_results) - 1)
        
        # Plot GT and LiDAR only once
        if not gt_plotted:
            gt_state = gt_states[final_idx]
            gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
            ax.plot(gt_shape_y, gt_shape_x, color='black', linewidth=3, label='Ground Truth', zorder=2)
            
            # Ground truth path
            history_x = [s.x for s in gt_states[:final_idx+1]]
            history_y = [s.y for s in gt_states[:final_idx+1]]
            ax.plot(history_y, history_x, color='gray', linestyle='--', linewidth=1, label='Path', zorder=1)
            
            # LiDAR at final frame
            res = tracker_results[final_idx]
            if res.measurements is not None:
                lidar_pos = config.lidar.lidar_position
                z_lidar = res.measurements.reshape((-1, 2))
                for z in z_lidar:
                    dist = np.linalg.norm(z - np.array(lidar_pos))
                    if dist < config.lidar.max_distance:
                        ax.plot([lidar_pos[1], z[1]], [lidar_pos[0], z[0]], color='gray', alpha=0.1, linewidth=0.5, zorder=0)
                        ax.scatter(z[1], z[0], s=5, color='gray', marker='.', zorder=0.5) 
            gt_plotted = True
            
        # Plot the filter's estimate
        res = tracker_results[final_idx]
        est_state = res.state_posterior.mean
        est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
        
        color = FILTER_COLORS.get(method_key, 'green')
        label = FILTER_LABELS.get(method_key, method_key)
        
        ax.plot(est_shape_y, est_shape_x, color=color, linewidth=2, label=label, zorder=3)
        
        # Heading arrow
        arrow_len = 5.0
        ax.arrow(est_state.y, est_state.x, 
                 arrow_len * np.sin(est_state.yaw), arrow_len * np.cos(est_state.yaw),
                 head_width=1.0, head_length=1.0, fc=color, ec=color, zorder=4)

    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylabel("North [m]", fontsize=18)
    ax.set_xlabel("East [m]", fontsize=18)
    ax.set_title(f"Final Frame Overlay - {prefix}", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(loc='upper left', fontsize=16)
    
    output_dir = FIGURES_PATH / "compare_timelapse_filters"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}_overlay.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved overlay plot to {out_path}")

def plot_2x2(runs, prefix, num_snapshots=4, legend_y_pos=0.08):
    """Plots a 2x2 grid showing the full timelapse for each filter."""
    print(f"Generating 2x2 Timelapse for {prefix}...")
    
    # Before creating the figure, load one run to find the limits
    with open(runs[list(runs.keys())[0]], "rb") as f:
        sim_result = pickle.load(f)
        gt_states = list(sim_result.ground_truth_ts.values)

    max_x = max([s.x for s in gt_states]) # North
    min_x = min([s.x for s in gt_states])
    max_y = max([s.y for s in gt_states]) # East
    min_y = min([s.y for s in gt_states])

    x_span = max_x - min_x
    y_span = max_y - min_y
    if y_span == 0: y_span = 1.0
    if x_span == 0: x_span = 1.0
    
    # If the track is extremely wide, artificially expand the vertical limits
    # so the axes natively take up more vertical space without making the boats tiny
    if (x_span / y_span) < 0.2:
        missing_x = (0.2 * y_span) - x_span
        min_x -= missing_x / 2
        max_x += missing_x / 2
        x_span = max_x - min_x

    # Calculate data aspect ratio (Height / Width)
    data_ratio = x_span / y_span

    # Set figure width scaling
    fig_width = 14
    fig_height = fig_width * data_ratio
    
    # Add a little padding for the legend and layout
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height + 2.0), sharex=True, sharey=True, layout='compressed')
    axs = axs.flatten()
    
    if any(k.startswith('progressive') for k in runs):
        method_order = ['progressive_1_barebones', 'progressive_2_safeguards', 'progressive_3_neginfo', 'progressive_4_full']
    else:
        method_order = ['EKF', 'IEKF', 'IMPLICIT_EKF', 'IMPLICIT_IEKF']
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='GT Path'),
        Line2D([0], [0], color='black', linewidth=1.5, label='GT Shape')
    ]
    
    legend_method_order = method_order.copy()
    if 'progressive_2_safeguards' in legend_method_order and 'progressive_3_neginfo' in legend_method_order:
        idx_safe = legend_method_order.index('progressive_2_safeguards')
        idx_neg = legend_method_order.index('progressive_3_neginfo')
        legend_method_order[idx_safe], legend_method_order[idx_neg] = legend_method_order[idx_neg], legend_method_order[idx_safe]
        
    for method_key in legend_method_order:
        if method_key in runs:
            legend_elements.append(Line2D([0], [0], color=FILTER_COLORS.get(method_key, 'green'), linewidth=2, label=FILTER_LABELS.get(method_key, method_key)))
    
    for i, method_key in enumerate(method_order):
        ax = axs[i]
        color = FILTER_COLORS.get(method_key, 'green')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if method_key not in runs:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)
            continue
            
        with open(runs[method_key], "rb") as f:
            sim_result = pickle.load(f)
            
        config = sim_result.config
        pca_params = load_pca_params(config)
        
        gt_states = list(sim_result.ground_truth_ts.values)
        tracker_results = list(sim_result.tracker_results_ts.values)
        
        num_frames = min(len(gt_states), len(tracker_results))
        indices = np.linspace(0, num_frames - 1, num_snapshots, dtype=int)
        
        # Ground truth path
        history_x = [s.x for s in gt_states]
        history_y = [s.y for s in gt_states]
        ax.plot(history_y, history_x, color='gray', linestyle='--', linewidth=1, label='GT Path', zorder=1)
        
        for snapshot_idx, idx in enumerate(indices):
            gt_state = gt_states[idx]
            res = tracker_results[idx]
            est_state = res.state_posterior.mean
            
            # Plot GT Shape
            gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
            ax.plot(gt_shape_y, gt_shape_x, color='black', linewidth=1.5, zorder=2)
            
            # Plot Estimated Shape
            est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
            ax.plot(est_shape_y, est_shape_x, color=color, linewidth=2, zorder=3)
            
            # Heading arrow
            arrow_len = 5.0
            ax.arrow(est_state.y, est_state.x, 
                     arrow_len * np.sin(est_state.yaw), arrow_len * np.cos(est_state.yaw),
                     head_width=1.0, head_length=1.0, fc=color, ec=color, zorder=4)
            
            # Annotate timestep
            text_x = est_state.x + 4.0
            text_y = est_state.y
            if text_x > max_x + x_span * 0.05:
                text_x = est_state.x - 6.0
                text_y = est_state.y + 2.0

            # MANUAL ADJUSTMENTS FOR LEGIBILITY ON FINAL SNAPSHOT (if needed)
            # 9th snapshot (index 8)
            # if snapshot_idx == 8:
            #     if method_key == 'IMPLICIT_IEKF' or method_key == 'IMPLICIT_EKF':
            #         text_x = est_state.x - 3.0
            #         text_y = est_state.y + 7.0
            #     else:
            #         text_x = est_state.x - 5.0
            #         text_y = est_state.y + 8.0

            # if snapshot_idx == 9:
            #     if method_key == 'EKF' or method_key == 'IEKF':
            #         # text_x = est_state.x - 3.0
            #         text_y = est_state.y - 2.0

            ax.text(text_y, text_x, f"t={idx}", fontsize=16, zorder=10, clip_on=True, ha='center', color=color)

        ax.set_aspect('equal', 'box')
        # Explicitly set limits so we apply the visual vertical aspect override natively
        ax.set_xlim(min_y - y_span * 0.1, max_y + y_span * 0.1)
        ax.set_ylim(min_x - x_span * 0.1, max_x + x_span * 0.1)
        ax.tick_params(axis='both', which='major', labelsize=14)

    for i in [0, 2]: axs[i].set_ylabel("North [m]", fontsize=18)
    for i in [2, 3]: axs[i].set_xlabel("East [m]", fontsize=18)
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, legend_y_pos), fontsize=16)
    
    output_dir = FIGURES_PATH / "compare_timelapse_filters"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}_2x2_timelapse.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved 2x2 plot to {out_path}")


def plot_single_timelapse(runs, prefix, frames_str=None):
    """Plots a single run's timelapse with specified frames overlayed on one figure."""
    print(f"Generating Single Timelapse for {prefix}...")
    
    target_method = 'IMPLICIT_IEKF'
    if target_method not in runs:
        if not runs:
            print("No runs to plot.")
            return
        target_method = list(runs.keys())[0]
        
    with open(runs[target_method], "rb") as f:
        sim_result = pickle.load(f)
        
    config = sim_result.config
    pca_params = load_pca_params(config)
    
    gt_states = list(sim_result.ground_truth_ts.values)
    tracker_results = list(sim_result.tracker_results_ts.values)
    
    # Extract limits
    max_x = max([s.x for s in gt_states]) # North
    min_x = min([s.x for s in gt_states])
    x_span = max_x - min_x
    if x_span == 0: x_span = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    
    num_frames = min(len(gt_states), len(tracker_results))
    
    if frames_str:
        frames = [int(f.strip()) for f in frames_str.split(',')]
        frames = [f for f in frames if f < num_frames]
    else:
        custom_frames = [0, 5, 10, 20, 50] 
        if num_frames > 150:
            custom_frames.extend(np.linspace(150, num_frames - 1, 5, dtype=int).tolist())
        frames = sorted(list(set([int(f) for f in custom_frames if f < num_frames])))
    
    history_x = [s.x for s in gt_states]
    history_y = [s.y for s in gt_states]
    ax.plot(history_y, history_x, color='gray', linestyle='--', linewidth=1, label='GT Path', zorder=1)
    
    color = FILTER_COLORS.get(target_method, 'green')
    label = FILTER_LABELS.get(target_method, target_method)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='GT Path'),
        Line2D([0], [0], color='black', linewidth=1.5, label='GT Shape'),
        Line2D([0], [0], color=color, linewidth=2, label=label)
    ]
    
    for idx in frames:
        gt_state = gt_states[idx]
        res = tracker_results[idx]
        est_state = res.state_posterior.mean
        
        gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
        ax.plot(gt_shape_y, gt_shape_x, color='black', linewidth=1.5, zorder=2)
        
        est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
        ax.plot(est_shape_y, est_shape_x, color=color, linewidth=2, zorder=3)
        
        arrow_len = x_span * 0.02
        if arrow_len < 1.0: arrow_len = 5.0
        ax.arrow(est_state.y, est_state.x, 
                 arrow_len * np.sin(est_state.yaw), arrow_len * np.cos(est_state.yaw),
                 head_width=arrow_len*0.2, head_length=arrow_len*0.2, fc=color, ec=color, zorder=4)
        
        text_offset = x_span * 0.03
        if text_offset < 1.0: text_offset = 4.0
        ax.text(est_state.y, est_state.x + text_offset, f"t={idx}", fontsize=14, zorder=10, clip_on=True, ha='center', color=color)

    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylabel("North [m]", fontsize=18)
    ax.set_xlabel("East [m]", fontsize=18)
    ax.set_title(f"Single Filter Timelapse - {prefix}", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=16)
    
    output_dir = FIGURES_PATH / "compare_timelapse_filters"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}_single_timelapse.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved single timelapse plot to {out_path}")

def plot_single_timelapse_subplots(runs, prefix, frames_str=None):
    """Plots a single run's timelapse with specified frames in individual subplots."""
    print(f"Generating Single Timelapse Subplots for {prefix}...")
    
    target_method = 'IMPLICIT_IEKF'
    if target_method not in runs:
        if not runs:
            print("No runs to plot.")
            return
        target_method = list(runs.keys())[0]
        
    with open(runs[target_method], "rb") as f:
        sim_result = pickle.load(f)
        
    config = sim_result.config
    pca_params = load_pca_params(config)
    
    gt_states = list(sim_result.ground_truth_ts.values)
    tracker_results = list(sim_result.tracker_results_ts.values)
    
    max_x = max([s.x for s in gt_states]) # North
    min_x = min([s.x for s in gt_states])
    max_y = max([s.y for s in gt_states]) # East
    min_y = min([s.y for s in gt_states])
    x_span = max_x - min_x
    if x_span == 0: x_span = 1.0
    y_span = max_y - min_y
    if y_span == 0: y_span = 1.0

    num_frames = min(len(gt_states), len(tracker_results))
    
    if frames_str:
        frames = [int(f.strip()) for f in frames_str.split(',')]
        frames = [f for f in frames if f < num_frames]
    else:
        custom_frames = [0, 5, 10, 20, 50] 
        if num_frames > 150:
            custom_frames.extend(np.linspace(150, num_frames - 1, 5, dtype=int).tolist())
        frames = sorted(list(set([int(f) for f in custom_frames if f < num_frames])))
        
    num_subplots = len(frames)
    
    # Try dynamic layout. For 12 subplots, force 3 columns to get a 4-row structure
    cols = min(3, num_subplots)
    rows = int(np.ceil(num_subplots / cols))
    
    # Balance size. 5x5 per plot was nice, maybe slightly smaller so it doesn't get ridiculously large
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), sharex=True, sharey=True, layout='compressed')
    if num_subplots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
        
    history_x = [s.x for s in gt_states]
    history_y = [s.y for s in gt_states]
    
    color = FILTER_COLORS.get(target_method, 'green')
    label = FILTER_LABELS.get(target_method, target_method)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='GT Path'),
        Line2D([0], [0], color='black', linewidth=1.5, label='GT Shape'),
        Line2D([0], [0], color=color, linewidth=2, label=label)
    ]
    
    for i, idx in enumerate(frames):
        ax = axs[i]
        
        ax.plot(history_y, history_x, color='gray', linestyle='--', linewidth=1, label='GT Path', zorder=1)
        
        gt_state = gt_states[idx]
        res = tracker_results[idx]
        est_state = res.state_posterior.mean
        
        gt_shape_x, gt_shape_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
        ax.plot(gt_shape_y, gt_shape_x, color='black', linewidth=1.5, zorder=2)
        
        est_shape_x, est_shape_y = compute_estimated_shape_global(est_state, config, pca_params)
        ax.plot(est_shape_y, est_shape_x, color=color, linewidth=2, zorder=3)
        
        arrow_len = x_span * 0.02
        if arrow_len < 1.0: arrow_len = 5.0
        ax.arrow(est_state.y, est_state.x, 
                 arrow_len * np.sin(est_state.yaw), arrow_len * np.cos(est_state.yaw),
                 head_width=arrow_len*0.2, head_length=arrow_len*0.2, fc=color, ec=color, zorder=4)
                 
        if res.measurements is not None:
            lidar_pos = config.lidar.lidar_position
            z_lidar = res.measurements.reshape((-1, 2))
            for z in z_lidar:
                dist = np.linalg.norm(z - np.array(lidar_pos))
                if dist < config.lidar.max_distance:
                    ax.plot([lidar_pos[1], z[1]], [lidar_pos[0], z[0]], color='gray', alpha=0.1, linewidth=0.5, zorder=0)
                    ax.scatter(z[1], z[0], s=3, color='gray', marker='.', zorder=0.5) 
        
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(f"Frame {idx}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)

    for j in range(len(frames), len(axs)):
        axs[j].set_visible(False)

    # Adjust span slightly more to ensure large initial estimate doesn't clip
    axs[0].set_xlim(min_y - y_span * 0.15, max_y + y_span * 0.15)
    axs[0].set_ylim(min_x - x_span * 0.15, max_x + x_span * 0.15)
    
    fig.text(0.5, -0.02, 'East [m]', ha='center', fontsize=18)
    fig.text(-0.02, 0.5, 'North [m]', va='center', rotation='vertical', fontsize=18)

    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=16)
    
    output_dir = FIGURES_PATH / "compare_timelapse_filters"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}_single_timelapse_subplots.pdf"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved subplots timelapse plot to {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare Filter Results.')
    parser.add_argument('prefix', type=str, help='Prefix to match the simulation names (e.g., exp1_linear_noiseless)')
    parser.add_argument('--folder', type=str, default='single_runs', help='Folder to restrict the search (default: single_runs)')
    parser.add_argument('--mode', type=str, default='both', choices=['overlay', '2x2', 'single_timelapse', 'single_timelapse_subplots', 'both'], help='Plotting mode')
    parser.add_argument('--num_snapshots', type=int, default=4, help='Number of snapshots for 2x2 mode')
    parser.add_argument('--legend_y_pos', type=float, default=0.08, help='Y position for the 2x2 plot legend (bbox_to_anchor value, lower moves it downwards, default: 0.08)')
    parser.add_argument('--run_id', type=str, default=None, help='Specific run ID to plot (e.g., run_011)')
    parser.add_argument('--custom_frames', type=str, default=None, help='Comma separated list of frames for single_timelapse')
    args = parser.parse_args()

    runs = get_filter_runs(args.prefix, args.folder, args.run_id)
    
    if not runs:
        print(f"No runs found for prefix '{args.prefix}' in folder '{args.folder}'" + (f" with run_id '{args.run_id}'." if args.run_id else "."))
        return
        
    print(f"Found runs: {list(runs.keys())}")
    
    out_prefix = f"{Path(args.folder).name}_{args.prefix}" if args.folder and args.folder != 'single_runs' else args.prefix
    if args.run_id:
        out_prefix += f"_{args.run_id}"
    
    if args.mode in ['overlay', 'both']:
        plot_overlay(runs, out_prefix)
        
    if args.mode in ['2x2', 'both']:
        plot_2x2(runs, out_prefix, args.num_snapshots, args.legend_y_pos)
        
    if args.mode in ['single_timelapse', 'both']:
        plot_single_timelapse(runs, out_prefix, args.custom_frames)
        
    if args.mode in ['single_timelapse_subplots', 'both']:
        plot_single_timelapse_subplots(runs, out_prefix, args.custom_frames)

if __name__ == "__main__":
    main()