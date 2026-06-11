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

def run_mc_experiment(source_experiment_name, target_experiment_name):
    """
    Loads pre-generated MC datasets from source_experiment_name and runs specific progressive trackers over them,
    saving to target_experiment_name.
    """
    from src.utils.config_classes import Config, SimulationConfig, LidarConfig, ExtentConfig, TrackerConfig

    source_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / source_experiment_name
    target_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / target_experiment_name
    
    if not source_dir.exists():
        logging.error(f"Source experiment directory {source_dir} not found.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    config_file = source_dir / "base_config.json"
    with open(config_file, "r") as f:
        base_config_data = json.load(f)
    
    # Save a copy to target_dir as well
    with open(target_dir / "base_config.json", "w") as f:
        json.dump(base_config_data, f, indent=4)
        
    # Clean up fields that should not be passed to __init__
    extent_dict = base_config_data['extent'].copy()
    extent_dict.pop('angles', None)
    extent_dict.pop('shape_coords_body', None)
    
    sim_base = SimulationConfig(**base_config_data['sim'])
    lidar_base = LidarConfig(**base_config_data['lidar'])
    extent_base = ExtentConfig(**extent_dict)
    tracker_base = TrackerConfig(**base_config_data['tracker'])
    
    from src.utils.config_classes import TrajectoryConfig
    sim_base.trajectory = TrajectoryConfig(**base_config_data['sim']['trajectory'])
    
    from src.states.states import State_PCA
    
    def parse_state_pca(data):
        if not data:
            return None
        d = dict(data)
        pca_coeffs = []
        i = 0
        while f'pca_coeff_{i}' in d:
            pca_coeffs.append(d[f'pca_coeff_{i}'])
            i += 1
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

    run_files = sorted(list(source_dir.glob("run_*.pkl")))
    if not run_files:
        logging.warning("No run files found in the source experiment directory.")
        return

    progressive_configs = [
        # 1. Barebones
        {
            "config_name": "progressive_1_barebones",
            "use_state_clamping": False,
            "use_mahalanobis_projection": False,
            "use_negative_info_angular": False,
            "use_negative_info_front": False,
            "use_negative_info_centroid": False,
            "use_L_W_aspect_ratio_prior": False,
        },
        # 2. + Safeguards
        {
            "config_name": "progressive_2_safeguards",
            "use_state_clamping": True,
            "use_mahalanobis_projection": True,
            "use_negative_info_angular": False,
            "use_negative_info_front": False,
            "use_negative_info_centroid": False,
            "use_L_W_aspect_ratio_prior": False,
        },
        # 3. + Negative Info
        {
            "config_name": "progressive_3_neginfo",
            "use_state_clamping": True,
            "use_mahalanobis_projection": True,
            "use_negative_info_angular": True,
            "use_negative_info_front": True,
            "use_negative_info_centroid": True,
            "use_L_W_aspect_ratio_prior": False,
        }
    ]

    for ab_cfg in progressive_configs:
        config_name = ab_cfg["config_name"]
        method_dir = target_dir / config_name
        method_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"--- Running progressive config {config_name} ---")
        
        for run_file in tqdm(run_files, desc=f"Evaluating {config_name}"):
            run_id = run_file.stem  # e.g. "run_000"
            result_file = method_dir / f"{run_id}_results.pkl"
            
            if result_file.exists():
                logging.info(f"Skipping {run_id} for {config_name}, results already exist.")
                continue

            with open(run_file, "rb") as f:
                run_data = pickle.load(f)
                
            gt_ts = run_data["gt_ts"]
            meas_lidar_ts = run_data["meas_lidar_ts"]
            meas_global_ts = run_data["meas_global_ts"]
            static_covariances = run_data.get("static_covariances", {})
            
            # --- Apply Exact Requested Tracker Settings for Experiment ---
            tracker_cfg_run = copy.deepcopy(base_config.tracker)
            tracker_cfg_run.method = "implicit_iekf"
            
            # progressive overrides
            tracker_cfg_run.use_state_clamping = ab_cfg["use_state_clamping"]
            tracker_cfg_run.use_mahalanobis_projection = ab_cfg["use_mahalanobis_projection"]
            tracker_cfg_run.use_negative_info_angular = ab_cfg["use_negative_info_angular"]
            tracker_cfg_run.use_negative_info_front = ab_cfg["use_negative_info_front"]
            tracker_cfg_run.use_negative_info_centroid = ab_cfg["use_negative_info_centroid"]
            tracker_cfg_run.use_L_W_aspect_ratio_prior = ab_cfg["use_L_W_aspect_ratio_prior"]
            
            # Keep other standard settings
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
            tracker_cfg_run.max_iterations = 20
            tracker_cfg_run.convergence_threshold = 1e-06
            tracker_cfg_run.mahalanobis_projection_prob = 0.99
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
            tracker_cfg_run.prior_aspect_ratio = 3.8
            tracker_cfg_run.prior_ratio_std = 5.0
            tracker_cfg_run.smoother_window_size = 10
            
            config_run = Config(sim=base_config.sim, lidar=base_config.lidar, tracker=tracker_cfg_run, extent=base_config.extent)
            
            tracker, filter_dyn_model, lidar_model, _, pca_params = _setup_tracker_and_data(config_run)
            static_covariances["Q"] = filter_dyn_model.Q_d(dt=config_run.sim.dt)
            
            # --- Fire the tracker sequence
            results_ts = run_filtering_loop(tracker, meas_lidar_ts, gt_ts)
            
            data_to_save = SimulationResult(
                config=config_run, ground_truth_ts=gt_ts,
                measurements_global_ts=meas_global_ts, tracker_results_ts=results_ts,
                static_covariances=static_covariances
            )
            
            with open(result_file, "wb") as f:
                pickle.dump(data_to_save, f)
            

if __name__ == "__main__":
    GT_TRAJ_SOURCE = "exp2_complex_maneuvers_noise015"
    EXPERIMENT_NAME = "exp3_progressive_noise015"
    
    run_mc_experiment(GT_TRAJ_SOURCE, EXPERIMENT_NAME)