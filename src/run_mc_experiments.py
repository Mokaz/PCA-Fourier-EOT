import os
import pickle
import json
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import copy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from src.experiment_runner import _setup_tracker_and_data, run_filtering_loop
from src.utils.geometry_utils import compute_exact_vessel_shape_global, compute_estimated_shape_global, calculate_iou
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.utils.SimulationResult import SimulationResult

def run_mc_experiment(experiment_name, methods):
    """
    Loads pre-generated MC datasets and runs specific trackers over them.
    """
    from src.utils.config_classes import Config, SimulationConfig, LidarConfig, ExtentConfig, TrackerConfig

    experiment_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / experiment_name
    if not experiment_dir.exists():
        logging.error(f"Experiment directory {experiment_dir} not found.")
        return

    # Load base config
    config_file = experiment_dir / "base_config.json"
    with open(config_file, "r") as f:
        base_config_data = json.load(f)
        
    # Reconstruct the config parts manually or using a helper, for simplicity passing kwargs mostly works
    # We will reconstruct using the classes
    # Because of nested structures, reconstructing from dict might be a bit tricky, relying on config classes.
    # Clean up fields that should not be passed to __init__
    extent_dict = base_config_data['extent'].copy()
    extent_dict.pop('angles', None)
    extent_dict.pop('shape_coords_body', None)
    
    sim_base = SimulationConfig(**base_config_data['sim'])
    lidar_base = LidarConfig(**base_config_data['lidar'])
    extent_base = ExtentConfig(**extent_dict)
    tracker_base = TrackerConfig(**base_config_data['tracker'])
    
    # Needs handling of trajectory dict 
    from src.utils.config_classes import TrajectoryConfig
    sim_base.trajectory = TrajectoryConfig(**base_config_data['sim']['trajectory'])
    
    # Needs handling of States 
    from src.states.states import State_PCA
    
    def parse_state_pca(data):
        if not data:
            return None
        d = dict(data)
        
        # Collect pca coefficients dynamically
        pca_coeffs = []
        i = 0
        while f'pca_coeff_{i}' in d:
            pca_coeffs.append(d[f'pca_coeff_{i}'])
            i += 1
            
        # Fallback if pca_coeffs already exists in the dict as a list
        if not pca_coeffs and 'pca_coeffs' in d:
            pca_coeffs = d['pca_coeffs']
            
        return State_PCA(
            x=d.get('x', 0.0), y=d.get('y', 0.0), yaw=d.get('yaw', 0.0),
            vel_x=d.get('vel_x', 0.0), vel_y=d.get('vel_y', 0.0), yaw_rate=d.get('yaw_rate', 0.0),
            length=d.get('length', 0.0), width=d.get('width', 0.0),
            pca_coeffs=pca_coeffs
        )

    sim_base.initial_state_gt = parse_state_pca(base_config_data['sim']['initial_state_gt'])
    tracker_base.initial_state = parse_state_pca(base_config_data['tracker']['initial_state'])
    tracker_base.initial_std_devs = parse_state_pca(base_config_data['tracker']['initial_std_devs'])
    
    import numpy as np
    tracker_base.pca_eigenvalues = np.array(base_config_data['tracker']['pca_eigenvalues'])
    tracker_base.lidar_position = np.array(base_config_data['tracker']['lidar_position'])
    
    base_config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_base, extent=extent_base)

    run_files = sorted(list(experiment_dir.glob("run_*.pkl")))
    if not run_files:
        logging.warning("No run files found in the experiment directory.")
        return

    for method in methods:
        method_dir = experiment_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"--- Running method {method} ---")
        
        for run_file in tqdm(run_files, desc=f"Evaluating {method}"):
            run_id = run_file.stem  # e.g. "run_000"
            result_file = method_dir / f"{run_id}_results.pkl"
            
            if result_file.exists():
                logging.info(f"Skipping {run_id} for {method}, results already exist.")
                continue

            with open(run_file, "rb") as f:
                run_data = pickle.load(f)
                
            gt_ts = run_data["gt_ts"]
            meas_lidar_ts = run_data["meas_lidar_ts"]
            meas_global_ts = run_data["meas_global_ts"]
            static_covariances = run_data["static_covariances"]
            
            # --- Apply Exact Requested Tracker Settings for Experiment 1 ---
            tracker_cfg_run = copy.deepcopy(base_config.tracker)
            tracker_cfg_run.method = method
            
            tracker_cfg_run.use_gt_state_for_bodyangles_calc = False
            tracker_cfg_run.use_initialize_centroid = False
            tracker_cfg_run.N_pca = 4
            tracker_cfg_run.PCA_parameters_path = 'data/input_parameters/ShipDatasetPCAParameters.npz'
            tracker_cfg_run.process_model = 'inflation'
            tracker_cfg_run.temporal_eta = 0.1
            tracker_cfg_run.temporal_pca_var = 1.0
            tracker_cfg_run.inflation_lambda = 1.0
            tracker_cfg_run.N_gp_points = 20
            tracker_cfg_run.gp_length_scale = 0.5
            tracker_cfg_run.gp_signal_var = 1.0
            tracker_cfg_run.gp_forgetting_factor = 0.05
            tracker_cfg_run.gp_use_negative_info = True
            tracker_cfg_run.pos_north_std_dev = 0.3
            tracker_cfg_run.pos_east_std_dev = 0.3
            tracker_cfg_run.heading_std_dev = 0.2
            tracker_cfg_run.length_std_dev = 0.01
            tracker_cfg_run.width_std_dev = 0.01
            tracker_cfg_run.pca_std_dev_scale = 0.05
            tracker_cfg_run.use_proportional_pca_random_walk = True
            tracker_cfg_run.lidar_std_dev = 0.15
            tracker_cfg_run.debug_prints = False
            tracker_cfg_run.initial_state = State_PCA(
                x=0.0, y=-40.0, yaw=1.5707963267948966, vel_x=0.0, vel_y=3.0, yaw_rate=0.0,
                length=7.511096477508545, width=2.802800178527832,
                pca_coeffs=np.array([0.0, 0.0, 0.0, 0.0])
            )
            tracker_cfg_run.initial_std_devs = State_PCA(
                x=2.0, y=2.0, yaw=0.2, vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
                length=2.0, width=2.0,
                pca_coeffs=np.array([1.0, 0.42200755611910706, 0.13513648823791097, 0.10970908525104822])
            )
            tracker_cfg_run.lidar_position = np.array([30.0, 0.0])
            tracker_cfg_run.pca_eigenvalues = np.array([1.0, 0.1780903774216213, 0.01826187045327505, 0.012036083386621767])
            
            if method == "implicit_ekf":
                tracker_cfg_run.max_iterations = 1
            else:
                tracker_cfg_run.max_iterations = 20
                
            tracker_cfg_run.convergence_threshold = 1e-06
            tracker_cfg_run.use_state_clamping = True
            tracker_cfg_run.use_mahalanobis_projection = True
            tracker_cfg_run.mahalanobis_projection_prob = 0.99
            tracker_cfg_run.use_negative_info_angular = True
            tracker_cfg_run.use_negative_info_front = True
            tracker_cfg_run.use_negative_info_centroid = True
            tracker_cfg_run.radial_margin = 0.1
            tracker_cfg_run.use_exact_extreme_angle = False
            tracker_cfg_run.use_D_imp_for_R = False
            tracker_cfg_run.use_scaled_R = False
            tracker_cfg_run.force_kinematic_unobservability = False
            tracker_cfg_run.use_arc_length_residual = True
            tracker_cfg_run.R_neg_info_std_angle = 0.05
            tracker_cfg_run.R_neg_info_std_front = 0.01
            tracker_cfg_run.R_neg_info_std_centroid = 0.01
            tracker_cfg_run.use_absolute_L_W_prior = False
            tracker_cfg_run.prior_target_L = 20.0
            tracker_cfg_run.prior_target_W = 6.0
            tracker_cfg_run.prior_size_std = 5.0
            tracker_cfg_run.use_L_W_aspect_ratio_prior = True
            tracker_cfg_run.prior_aspect_ratio = 3.8
            tracker_cfg_run.prior_ratio_std = 5.0
            tracker_cfg_run.smoother_window_size = 10
            
            config_run = Config(sim=base_config.sim, lidar=base_config.lidar, tracker=tracker_cfg_run, extent=base_config.extent)
            
            # Needs dt for Q
            tracker, filter_dyn_model, lidar_model, _, pca_params = _setup_tracker_and_data(config_run)
            static_covariances["Q"] = filter_dyn_model.Q_d(dt=config_run.sim.dt)
            
            # --- Fire the tracker sequence
            # Disable tqdm on filtering to keep logs clean during batch
            # Temporarily redirect or manage tqdm if needed, but we can just use the regular loop without desc
            # Since run_filtering_loop has tqdm inside, let's just let it run. It will print quite a bit for 100 runs.
            # You might want to remove the nested tqdm in run_filtering_loop or set disable=True
            
            results_ts = run_filtering_loop(tracker, meas_lidar_ts, gt_ts)
            
            if method == "full_batch_smoother":
                 results_ts = tracker.smooth_trajectory(results_ts)
                 
            data_to_save = SimulationResult(
                config=config_run, ground_truth_ts=gt_ts,
                measurements_global_ts=meas_global_ts, tracker_results_ts=results_ts,
                static_covariances=static_covariances
            )
            
            with open(result_file, "wb") as f:
                pickle.dump(data_to_save, f)
                
            # Compute partial metrics per run to JSON sidecar if desired, but we can also do that in analysis
            

if __name__ == "__main__":
    EXPERIMENT_NAME = "exp2_complex_maneuvers_noise015"
    METHODS = ["ekf", "iekf", "implicit_ekf", "implicit_iekf"]
    
    run_mc_experiment(EXPERIMENT_NAME, METHODS)