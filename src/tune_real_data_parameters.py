import os
import sys
import numpy as np
import pickle
import json
import logging
import itertools
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from global_project_paths import SIMDATA_PATH

from src.run_real_data import setup_real_data_config, load_zpos_sequence, load_ground_truth_sequence
from src.dynamics.process_models import Model_PCA_Inflation
from src.sensors.LidarModel import LidarMeasurementModel
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.senfuslib.timesequence import TimeSequence
from src.utils.SimulationResult import SimulationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

def set_nested_value(obj, path, value):
    """Sets a value deep in a nested object using a dot-notation string."""
    parts = path.split('.')
    last = parts.pop()
    for part in parts:
        obj = getattr(obj, part)
    setattr(obj, last, value)

def tune_real_data_experiment(param_grid):
    # 1. Load real data once
    MAT_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.mat"
    H5_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.h5"
    dt = 0.1
    
    if not os.path.exists(MAT_FILE_PATH):
        print(f"Error: Could not find {MAT_FILE_PATH}")
        return
        
    measurements_ts = load_zpos_sequence(MAT_FILE_PATH, dt=dt)
    if os.path.exists(H5_FILE_PATH):
        ground_truth_ts, L_gt, W_gt, pca_coeffs_gt = load_ground_truth_sequence(H5_FILE_PATH, dt=dt, N_pca=4)
        gt_first = ground_truth_ts.get(0.0) if ground_truth_ts else None
        if gt_first is not None:
            init_x = gt_first.x
            init_y = gt_first.y
            init_yaw = gt_first.yaw
        else:
            init_x, init_y, init_yaw = 30.0, 30.0, np.pi/4
    else:
        ground_truth_ts = TimeSequence()
        L_gt, W_gt, pca_coeffs_gt = 20.0, 6.0, np.zeros(4)
        init_x, init_y, init_yaw = 30.0, 30.0, np.pi/4

    # 2. Prepare Tuning output directory
    tuning_dir = Path(SIMDATA_PATH) / "single_runs" / "real_data_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n{'='*50}")
    print(f"Queued {len(combinations)} tuning configurations for Real Data")
    print(f"{'='*50}\n")
    
    for i, combination in enumerate(combinations):
        current_params = dict(zip(keys, combination))
        
        # Build descriptor name for the configuration
        desc_parts = []
        for path, val in current_params.items():
            short_key = path.split('.')[-1].replace("use_", "").replace("_std_dev", "")
            val_str = str(val).replace('.', 'p') if isinstance(val, float) else str(val)
            desc_parts.append(f"{short_key}_{val_str}")
            
        config_name = "tuning_real_" + "_".join(desc_parts)
        
        sim_dir = tuning_dir / config_name
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = sim_dir / f"{config_name}.pkl"
        json_filename = sim_dir / f"{config_name}.json"
        
        if result_file.exists():
            continue

        logging.info(f"[{i+1}/{len(combinations)}] Evaluating config: {config_name}")
        
        # Generate Base Config
        config = setup_real_data_config(init_x=init_x, init_y=init_y, init_yaw=init_yaw, 
                                        L_gt=L_gt, W_gt=W_gt, init_pca_coeffs=None, dt=dt, 
                                        method="implicit_iekf")
        config.sim.num_frames = len(measurements_ts)
        config.sim.name = config_name
        
        # Set Baseline Tuning Constraints for the grid search
        config.tracker.process_model = "inflation"
        config.tracker.max_iterations = 20
        config.tracker.use_initialize_centroid = False
        config.tracker.use_D_imp_for_R = False
        config.tracker.use_state_clamping = True
        config.tracker.use_mahalanobis_projection = True
        config.tracker.use_negative_info_angular = True
        config.tracker.use_negative_info_front = True
        config.tracker.use_negative_info_centroid = True
        
        # Apply grid parameters explicitly onto config
        for path, val in current_params.items():
            set_nested_value(config, path, val)
            
        # Tie pos_east_std_dev to pos_north_std_dev for isotropic positioning noise
        config.tracker.pos_east_std_dev = config.tracker.pos_north_std_dev

        # Initialize models using config
        filter_dyn_model = Model_PCA_Inflation(
            x_pos_std_dev=config.tracker.pos_north_std_dev, 
            y_pos_std_dev=config.tracker.pos_east_std_dev,
            yaw_std_dev=config.tracker.heading_std_dev, 
            N_pca=config.tracker.N_pca,
            length_std_dev=config.tracker.length_std_dev, 
            width_std_dev=config.tracker.width_std_dev,
            lambda_f=getattr(config.tracker, 'inflation_lambda', 0.1),
            pca_std_dev_scale=config.tracker.pca_std_dev_scale,
            pca_eigenvalues=config.tracker.pca_eigenvalues
        )

        pca_params = np.load(config.tracker.PCA_parameters_path)
        lidar_model = LidarMeasurementModel(
            lidar_position=np.array(config.lidar.lidar_position),
            lidar_std_dev=config.tracker.lidar_std_dev,
            pca_mean=pca_params['mean'],
            pca_eigenvectors=pca_params['eigenvectors'][:, :config.tracker.N_pca].real,
            extent_cfg=config.extent
        )
        
        tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        
        # Tracking Loop
        results_ts = TimeSequence()
        results_ts.insert(-dt, tracker.get_initial_update_result())

        for ts, measurement in tqdm(measurements_ts.items(), desc=f"Filtering {config_name}", leave=False):
            gt_state = ground_truth_ts.get(ts) if ground_truth_ts else None
            if measurement.x.size == 0:
                tracker.predict()
                continue
            tracker.predict()
            update_result = tracker.update(measurement, ground_truth=gt_state) 
            results_ts.insert(ts, update_result)

        # Save Pickled Results
        data_to_save = SimulationResult(
            config=config, 
            ground_truth_ts=ground_truth_ts,
            measurements_global_ts=measurements_ts,
            tracker_results_ts=results_ts,
            static_covariances={"Q": filter_dyn_model.Q_d(dt=dt), "R_point": lidar_model.R_single_point()}
        )
        
        with open(result_file, "wb") as f:
            pickle.dump(data_to_save, f)
            
        # Extract Metrics & Save JSON sidecar
        from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
        from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global, calculate_iou
        
        try:
            consistency_analyzer = create_consistency_analysis_from_sim_result(data_to_save)
            
            nees_data = consistency_analyzer.get_nees(indices='all')
            avg_nees = nees_data.a if nees_data else None
            nees_in_interval = nees_data.in_interval * 100 if nees_data else None
            
            nis_data = consistency_analyzer.get_nis(indices='all')
            avg_nis = nis_data.a if nis_data else None
            nis_in_interval = nis_data.in_interval * 100 if nis_data else None

            if hasattr(consistency_analyzer, 'x_err_gauss') and consistency_analyzer.x_err_gauss is not None:
                err_arrays = [e.mean for e in consistency_analyzer.x_err_gauss.values]
                full_state_rmse = np.sqrt(np.mean(np.square(err_arrays)))
                
                pos_errs = np.array([e.mean[:2] for e in consistency_analyzer.x_err_gauss.values])
                rmse_pos = np.sqrt(np.mean(np.sum(pos_errs**2, axis=1)))
            else:
                full_state_rmse = None
                rmse_pos = None

            ious = []
            for j, res in enumerate(results_ts.values):
                # The real dataset has results prepended at index -dt
                gt_idx = j - 1
                if gt_idx >= 0 and gt_idx < len(ground_truth_ts.values):
                    gt_state = ground_truth_ts.values[gt_idx]
                    est_state = res.state_posterior.mean
                    try:
                        gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
                        est_x, est_y = compute_estimated_shape_global(est_state, config, pca_params)
                        iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                        ious.append(iou)
                    except Exception:
                        pass
                        
            avg_iou = np.mean(ious) if ious else None
            final_iou = ious[-1] if ious else None

        except Exception as e:
            print(f"Error calculating metrics for JSON sidecar: {e}")
            avg_nees, avg_nis, nees_in_interval, nis_in_interval = None, None, None, None
            full_state_rmse, rmse_pos, avg_iou, final_iou = None, None, None, None

        summary_data = {
            "name": config_name,
            "method": config.tracker.method,
            "trajectory_type": "real_data",
            "avg_nees": float(avg_nees) if avg_nees is not None else None,
            "avg_nis": float(avg_nis) if avg_nis is not None else None,
            "full_state_rmse": float(full_state_rmse) if full_state_rmse is not None else None,
            "rmse_pos": float(rmse_pos) if rmse_pos is not None else None,
            "avg_iou": float(avg_iou) if avg_iou is not None else None,
            "final_iou": float(final_iou) if final_iou is not None else None,
            "nees_in_interval_95": float(nees_in_interval) if nees_in_interval is not None else None,
            "nis_in_interval_95": float(nis_in_interval) if nis_in_interval is not None else None,
        }
        # Dump config values cleanly to sidecar
        for path, val in current_params.items():
            short_key = path.split('.')[-1]
            summary_data[short_key] = val
        
        with open(json_filename, "w") as f:
            json.dump(summary_data, f, indent=4)

if __name__ == "__main__":
    # Define hyperparameter grid search specifically for Real Data Implicit IEKF
    # Target some typical noise ranges and parameters specific to real tracking
    param_grid = {
        # --- SWEEPING (The Flexibility Parameters) ---
        "tracker.pos_north_std_dev": [0.75, 1.0, 1.25, 1.5, 2.0],  # Pushing position flexibility
        "tracker.pca_std_dev_scale": [0.3, 0.4, 0.5, 0.6],         # Pushing shape flexibility
        
        # --- LOCKED (Based on Data) ---
        "tracker.inflation_lambda": [1.0],
        "tracker.heading_std_dev": [0.1],
        "tracker.lidar_std_dev": [0.3],
        "tracker.R_neg_info_std_angle": [0.05],
        "tracker.R_neg_info_std_front": [0.01],
        "tracker.use_arc_length_residual": [False], 
    }
    
    tune_real_data_experiment(param_grid)
