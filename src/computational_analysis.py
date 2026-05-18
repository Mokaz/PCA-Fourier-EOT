import time
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.main import get_common_configs, get_pca_tracker_config
from src.utils.config_classes import Config
from src.experiment_runner import run_single_simulation

def run_timing_analysis():
    # --- BOAT SELECTION & TRAJECTORY ---
    selected_boat_id = "Havfruen" 
    selected_trajectory = "complex_maneuvers"
    N_pca = 4
    
    # Load Base Configs
    sim_base, lidar_base, extent_base = get_common_configs(traj_type=selected_trajectory, N_pca=N_pca, selected_boat_id=selected_boat_id)
    
    print(f"Simulating boat from database (ID: {extent_base.shape_params_true.get('id', 'Unknown')})")
    print(f"L={sim_base.initial_state_gt.length:.2f}, W={sim_base.initial_state_gt.width:.2f}")

    # Methods to compare
    methods = ["implicit_ekf", "implicit_iekf", "iplf"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*40}")
        print(f"--- Running Timing Analysis for: {method} ---")
        print(f"{'='*40}")
        
        tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, sim_base.initial_state_gt, N_pca)
        tracker_cfg.process_model = 'inflation'
        tracker_cfg.method = method

        if method == "implicit_ekf":
            tracker_cfg.max_iterations = 1

        config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)

        # Apply specific test2_complex_maneuvers_noise015 settings
        tracker_cfg.debug_prints = False

        boat_id = extent_base.shape_params_true.get('id', 'custom')
        config.sim.use_cache = True # Enable cache so no time is spent on measurement generation
        config.sim.num_frames = 800
        config.lidar.lidar_gt_std_dev = 0.15
        
        # Tracker settings
        config.tracker.use_initialize_centroid = False
        config.tracker.use_D_imp_for_R = False
        config.tracker.use_state_clamping = True
        config.tracker.use_mahalanobis_projection = True
        config.tracker.use_negative_info_angular = True
        config.tracker.use_negative_info_front = True
        config.tracker.use_negative_info_centroid = True
        config.tracker.use_absolute_L_W_prior = False
        config.tracker.use_L_W_aspect_ratio_prior = False
        config.tracker.use_scaled_R = False
        config.tracker.force_kinematic_unobservability = False
        config.tracker.smoother_window_size = 10

        # Unique Name
        config.sim.name = f"timing_test2_complex_maneuvers_noise015_{method}"

        # Run and time it 
        start_time = time.time()
        sim_result = run_single_simulation(config=config)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_per_frame = elapsed_time / config.sim.num_frames
        
        results[method] = {
            "total_time": elapsed_time,
            "time_per_frame": time_per_frame
        }
        
    print(f"\n{'='*50}")
    print("=== Computational Analysis Summary ===")
    print(f"{'='*50}")
    print(f"{'Method':<16} | {'Total Time (s)':<14} | {'Time per Frame (s)':<18}")
    print("-" * 50)
    for method in methods:
        stats = results[method]
        print(f"{method:<16} | {stats['total_time']:<14.4f} | {stats['time_per_frame']:<18.6f}")

if __name__ == "__main__":
    run_timing_analysis()
