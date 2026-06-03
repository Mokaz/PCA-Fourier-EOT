import os
import pickle
import json
import logging
import itertools
from pathlib import Path
import sys
from tqdm import tqdm
import copy
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from src.experiment_runner import _setup_tracker_and_data, run_filtering_loop
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.utils.SimulationResult import SimulationResult

def set_nested_value(obj, path, value):
    """Sets a value deep in a nested object using a dot-notation string."""
    parts = path.split('.')
    last = parts.pop()
    for part in parts:
        obj = getattr(obj, part)
    setattr(obj, last, value)

def parse_state_pca(data):
    from src.states.states import State_PCA
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

def tune_mc_experiment(experiment_name, param_grid):
    """
    Loads pre-generated MC datasets and evaluates them across a grid of tracking parameters.
    """
    from src.utils.config_classes import Config, SimulationConfig, LidarConfig, ExtentConfig, TrackerConfig

    experiment_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / experiment_name
    if not experiment_dir.exists():
        logging.error(f"Experiment directory {experiment_dir} not found.")
        return

    # Load base config
    config_file = experiment_dir / "base_config.json"
    if not config_file.exists():
        logging.error("base_config.json not found.")
        return
        
    with open(config_file, "r") as f:
        base_config_data = json.load(f)
        
    extent_dict = base_config_data['extent'].copy()
    extent_dict.pop('angles', None)
    extent_dict.pop('shape_coords_body', None)
    
    sim_base = SimulationConfig(**base_config_data['sim'])
    lidar_base = LidarConfig(**base_config_data['lidar'])
    extent_base = ExtentConfig(**extent_dict)
    tracker_base = TrackerConfig(**base_config_data['tracker'])
    
    from src.utils.config_classes import TrajectoryConfig
    sim_base.trajectory = TrajectoryConfig(**base_config_data['sim']['trajectory'])
    
    sim_base.initial_state_gt = parse_state_pca(base_config_data['sim'].get('initial_state_gt'))
    tracker_base.initial_state = parse_state_pca(base_config_data['tracker'].get('initial_state'))
    tracker_base.initial_std_devs = parse_state_pca(base_config_data['tracker'].get('initial_std_devs'))
    
    tracker_base.pca_eigenvalues = np.array(base_config_data['tracker']['pca_eigenvalues'])
    tracker_base.lidar_position = np.array(base_config_data['tracker']['lidar_position'])
    
    base_config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_base, extent=extent_base)

    run_files = sorted(list(experiment_dir.glob("run_000.pkl")))
    if not run_files:
        logging.warning("run_000.pkl not found in the experiment directory.")
        return

    # Generate parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"--- Queued {len(combinations)} tuning configurations ---")

    for i, combination in enumerate(combinations):
        current_params = dict(zip(keys, combination))
        
        # Build descriptor name for the configuration
        desc_parts = []
        for path, val in current_params.items():
            short_key = path.split('.')[-1].replace("use_", "").replace("_std_dev", "")
            val_str = str(val).replace('.', 'p') if isinstance(val, float) else str(val)
            desc_parts.append(f"{short_key}_{val_str}")
            
        config_name = "tuning_" + "_".join(desc_parts)
        config_dir = experiment_dir / "tuning" / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"[{i+1}/{len(combinations)}] Evaluating config: {config_name}")
        
        for run_file in tqdm(run_files, desc=f"Runs for {config_name}", leave=False):
            run_id = run_file.stem
            result_file = config_dir / f"{run_id}_results.pkl"
            
            if result_file.exists():
                continue

            with open(run_file, "rb") as f:
                run_data = pickle.load(f)
                
            gt_ts = run_data["gt_ts"]
            meas_lidar_ts = run_data["meas_lidar_ts"]
            meas_global_ts = run_data["meas_global_ts"]
            static_covariances = run_data["static_covariances"]
            
            tracker_cfg_run = copy.deepcopy(base_config.tracker)
            tracker_cfg_run.method = "implicit_iekf"
            tracker_cfg_run.max_iterations = 20
            
            # Default tuning base setup
            tracker_cfg_run.process_model = "inflation"
            tracker_cfg_run.use_initialize_centroid = False
            tracker_cfg_run.use_D_imp_for_R = False
            tracker_cfg_run.use_state_clamping = True
            tracker_cfg_run.use_mahalanobis_projection = True
            tracker_cfg_run.use_negative_info_angular = True
            tracker_cfg_run.use_negative_info_front = True
            tracker_cfg_run.use_negative_info_centroid = True
    
            config_run = Config(sim=base_config.sim, lidar=base_config.lidar, tracker=tracker_cfg_run, extent=base_config.extent)
            
            # Apply grid parameters explicitly
            for path, val in current_params.items():
                set_nested_value(config_run, path, val)

            # Tie pos_east_std_dev to pos_north_std_dev for isotropic positioning noise
            config_run.tracker.pos_east_std_dev = config_run.tracker.pos_north_std_dev
            
            tracker, filter_dyn_model, lidar_model, _, pca_params = _setup_tracker_and_data(config_run)
            static_covariances["Q"] = filter_dyn_model.Q_d(dt=config_run.sim.dt)
            
            results_ts = run_filtering_loop(tracker, meas_lidar_ts, gt_ts)
                 
            data_to_save = SimulationResult(
                config=config_run, ground_truth_ts=gt_ts,
                measurements_global_ts=meas_global_ts, tracker_results_ts=results_ts,
                static_covariances=static_covariances
            )
            
            with open(result_file, "wb") as f:
                pickle.dump(data_to_save, f)

if __name__ == "__main__":
    EXPERIMENT_NAME = "exp1_linear_noise015"
    
    # Define hyperparameter grid search specifically for Implicit IEKF
    # WARNING: Too many parameters will cause an explosion of combinations.
    # Comment some out to perform targeted tuning.
    param_grid = {
        "tracker.inflation_lambda": [0.95, 0.99, 1.0],
        "tracker.pca_std_dev_scale": [0.05, 0.1, 0.3],
        "tracker.pos_north_std_dev": [0.1, 0.3],
        "tracker.heading_std_dev": [0.05, 0.1, 0.2],
        "tracker.R_neg_info_std_angle": [0.005, 0.01, 0.05],
        # "tracker.R_neg_info_std_front": [0.01, 0.05],
        # "tracker.R_neg_info_std_centroid": [0.01, 0.05],
        # "tracker.mahalanobis_projection_prob": [0.95, 0.99],
        "tracker.use_arc_length_residual": [False],
    }
    
    tune_mc_experiment(EXPERIMENT_NAME, param_grid)
