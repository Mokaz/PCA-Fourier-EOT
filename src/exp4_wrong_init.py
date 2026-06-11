import os
import pickle
import json
import logging
from pathlib import Path
import sys
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
from src.utils.geometry_utils import compute_exact_vessel_shape_global, compute_estimated_shape_global, calculate_iou
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.utils.SimulationResult import SimulationResult

def run_exp4():
    """
    Runs exp4: uses the wrong_pos_wrong_shape from run_batch on run_000 of exp2
    with Implicit IEKF with the Full system of helping constraints active.
    """
    from src.utils.config_classes import Config, SimulationConfig, LidarConfig, ExtentConfig, TrackerConfig

    experiment_name = "exp2_complex_maneuvers_noise015"
    method = "implicit_iekf"
    run_id = "run_000"

    experiment_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / experiment_name
    if not experiment_dir.exists():
        logging.error(f"Experiment directory {experiment_dir} not found.")
        return

    # Load base config
    config_file = experiment_dir / "base_config.json"
    with open(config_file, "r") as f:
        base_config_data = json.load(f)
        
    # Reconstruct configs
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
            pca_coeffs=np.array(pca_coeffs) if pca_coeffs else np.zeros(4)
        )

    sim_base.initial_state_gt = parse_state_pca(base_config_data['sim']['initial_state_gt'])
    tracker_base.initial_state = parse_state_pca(base_config_data['tracker']['initial_state'])
    tracker_base.initial_std_devs = parse_state_pca(base_config_data['tracker']['initial_std_devs'])
    
    tracker_base.pca_eigenvalues = np.array(base_config_data['tracker']['pca_eigenvalues'])
    tracker_base.lidar_position = np.array(base_config_data['tracker']['lidar_position'])
    
    base_config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_base, extent=extent_base)

    run_file = experiment_dir / f"{run_id}.pkl"
    if not run_file.exists():
        logging.error(f"Run file {run_file} not found.")
        return

    # Output directory for exp4
    exp4_dir = Path(PROJECT_ROOT) / "data" / "results" / "single_runs" / "exp4_wrong_init"
    exp4_dir.mkdir(parents=True, exist_ok=True)
    run_name = "exp4_wrong_init"
    result_file = exp4_dir / f"{run_name}.pkl"

    logging.info(f"--- Running {method} for exp4 ({run_id}) ---")

    with open(run_file, "rb") as f:
        run_data = pickle.load(f)
        
    gt_ts = run_data["gt_ts"]
    meas_lidar_ts = run_data["meas_lidar_ts"]
    meas_global_ts = run_data["meas_global_ts"]
    static_covariances = run_data["static_covariances"]
    
    tracker_cfg_run = copy.deepcopy(base_config.tracker)
    tracker_cfg_run.method = method
    
    tracker_cfg_run.N_pca = 4
    tracker_cfg_run.PCA_parameters_path = 'data/input_parameters/ShipDatasetPCAParameters.npz'
    tracker_cfg_run.process_model = 'inflation'

    # Retrieve gt baseline
    gt_initial = base_config.sim.initial_state_gt
    
    # Apply wrong_pos_wrong_shape offsets
    init_pos_offset = (1.0, -1.0)
    init_yaw_offset = np.deg2rad(10.0)
    init_size_scale = (2.5, 0.5)

    tracker_cfg_run.initial_state = State_PCA(
        x=gt_initial.x + init_pos_offset[0],
        y=gt_initial.y + init_pos_offset[1],
        yaw=gt_initial.yaw + init_yaw_offset,
        vel_x=gt_initial.vel_x,
        vel_y=gt_initial.vel_y,
        yaw_rate=gt_initial.yaw_rate,
        length=gt_initial.length * init_size_scale[0],
        width=gt_initial.width * init_size_scale[1],
        pca_coeffs=np.zeros(4)
    )
    
    tracker_cfg_run.initial_std_devs = State_PCA(
        x=2.0, y=2.0, yaw=0.2, vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
        length=2.0, width=2.0,
        pca_coeffs=np.array([1.0, 0.42200755611910706, 0.13513648823791097, 0.10970908525104822])
    )
    
    tracker_cfg_run.max_iterations = 20
    tracker_cfg_run.convergence_threshold = 1e-06

    # Full system of helping constraints active
    tracker_cfg_run.use_state_clamping = True
    tracker_cfg_run.use_mahalanobis_projection = True
    tracker_cfg_run.mahalanobis_projection_prob = 0.99
    
    tracker_cfg_run.use_negative_info_angular = True
    tracker_cfg_run.use_negative_info_front = True
    tracker_cfg_run.use_negative_info_centroid = True
    
    tracker_cfg_run.use_absolute_L_W_prior = False
    tracker_cfg_run.use_L_W_aspect_ratio_prior = True
    
    tracker_cfg_run.prior_target_L = 20.0
    tracker_cfg_run.prior_target_W = 6.0
    tracker_cfg_run.prior_size_std = 5.0
    tracker_cfg_run.prior_aspect_ratio = 3.8
    tracker_cfg_run.prior_ratio_std = 5.0
    tracker_cfg_run.smoother_window_size = 10
    
    tracker_cfg_run.R_neg_info_std_angle = 0.05
    tracker_cfg_run.R_neg_info_std_front = 0.01
    tracker_cfg_run.R_neg_info_std_centroid = 0.01
    tracker_cfg_run.radial_margin = 0.1
    tracker_cfg_run.use_exact_extreme_angle = False
    tracker_cfg_run.use_D_imp_for_R = False
    tracker_cfg_run.use_scaled_R = False
    tracker_cfg_run.force_kinematic_unobservability = False
    tracker_cfg_run.use_arc_length_residual = True

    tracker_cfg_run.use_gt_state_for_bodyangles_calc = False
    tracker_cfg_run.use_initialize_centroid = False
    tracker_cfg_run.temporal_eta = 0.1
    tracker_cfg_run.temporal_pca_var = 1.0
    tracker_cfg_run.inflation_lambda = 1.0
    tracker_cfg_run.pos_north_std_dev = 0.3
    tracker_cfg_run.pos_east_std_dev = 0.3
    tracker_cfg_run.heading_std_dev = 0.2
    tracker_cfg_run.length_std_dev = 0.01
    tracker_cfg_run.width_std_dev = 0.01
    tracker_cfg_run.pca_std_dev_scale = 0.05
    tracker_cfg_run.use_proportional_pca_random_walk = True
    tracker_cfg_run.lidar_std_dev = 0.15
    tracker_cfg_run.debug_prints = False
    
    config_run = Config(sim=base_config.sim, lidar=base_config.lidar, tracker=tracker_cfg_run, extent=base_config.extent)
    config_run.sim.name = "exp4_wrong_init"
    
    tracker, filter_dyn_model, lidar_model, _, pca_params = _setup_tracker_and_data(config_run)
    static_covariances["Q"] = filter_dyn_model.Q_d(dt=config_run.sim.dt)
    
    logging.info("Starting filtering loop...")
    results_ts = run_filtering_loop(tracker, meas_lidar_ts, gt_ts)
    
    data_to_save = SimulationResult(
        config=config_run, ground_truth_ts=gt_ts,
        measurements_global_ts=meas_global_ts, tracker_results_ts=results_ts,
        static_covariances=static_covariances
    )
    
    with open(result_file, "wb") as f:
        pickle.dump(data_to_save, f)
        
    logging.info(f"Results saved to {result_file}")
    
    # Evaluate metrics and save JSONs
    try:
        consistency_analyzer = create_consistency_analysis_from_sim_result(data_to_save)
        nees_data = consistency_analyzer.get_nees(indices='all')
        avg_nees = nees_data.a if nees_data else None
        
        nis_data = consistency_analyzer.get_nis(indices='all')
        avg_nis = nis_data.a if nis_data else None

        nees_in_interval = nees_data.in_interval * 100 if nees_data else None
        nis_in_interval = nis_data.in_interval * 100 if nis_data else None
        
        if hasattr(consistency_analyzer, 'x_err_gauss') and consistency_analyzer.x_err_gauss is not None:
            err_arrays =[e.mean for e in consistency_analyzer.x_err_gauss.values]
            full_state_rmse = np.sqrt(np.mean(np.square(err_arrays)))
            
            # Calculate 2D Positional RMSE (assuming indices 0 and 1 are North and East)
            pos_errs = np.array([e.mean[:2] for e in consistency_analyzer.x_err_gauss.values])
            rmse_pos = np.sqrt(np.mean(np.sum(pos_errs**2, axis=1)))
        else:
            full_state_rmse = None
            rmse_pos = None

        ious = []
        for i, res in enumerate(results_ts.values):
            if i < len(gt_ts.values):
                gt_state = gt_ts.values[i]
                est_state = res.state_posterior.mean
                gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, base_config.extent.shape_coords_body)
                try:
                    est_x, est_y = compute_estimated_shape_global(est_state, config_run, pca_params)
                    iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                    ious.append(iou)
                except Exception:
                    pass
        avg_iou = np.mean(ious) if ious else None
        final_iou = ious[-1] if ious else None

        # Summary JSON
        summary_data = {
            "name": run_name,
            "method": tracker_cfg_run.method,
            "trajectory_type": config_run.sim.trajectory.type,
            "scenario": getattr(config_run.sim, "scenario", None),
            "num_rays": getattr(config_run.lidar, "num_rays", None),
            "use_D_imp_for_R": getattr(tracker_cfg_run, 'use_D_imp_for_R', False),
            "use_scaled_R": getattr(tracker_cfg_run, 'use_scaled_R', False),
            "use_negative_info_angular": getattr(tracker_cfg_run, 'use_negative_info_angular', False),
            "use_negative_info_front": getattr(tracker_cfg_run, 'use_negative_info_front', False),
            "use_negative_info_centroid": getattr(tracker_cfg_run, 'use_negative_info_centroid', False),
            "use_initialize_centroid": getattr(tracker_cfg_run, 'use_initialize_centroid', False),
            "avg_nees": float(avg_nees) if avg_nees is not None else None,
            "avg_nis": float(avg_nis) if avg_nis is not None else None,
            "full_state_rmse": float(full_state_rmse) if full_state_rmse is not None else None,
            "rmse_pos": float(rmse_pos) if rmse_pos is not None else None,
            "avg_iou": float(avg_iou) if avg_iou is not None else None,
            "final_iou": float(final_iou) if final_iou is not None else None,
            "nees_in_interval_95": float(nees_in_interval) if nees_in_interval is not None else None,
            "nis_in_interval_95": float(nis_in_interval) if nis_in_interval is not None else None,
        }
        
        json_filename = exp4_dir / f"{run_name}.json"
        with open(json_filename, "w") as f:
            json.dump(summary_data, f, indent=4)
            
        logging.info(f"Summary metrics saved to {json_filename}")

        # Config JSON
        class ConfigEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if hasattr(obj, '__dataclass_fields__'):
                    from dataclasses import asdict
                    return asdict(obj)
                return str(obj)

        try:
            from dataclasses import asdict
            with open(exp4_dir / f"{run_name}_config.json", "w") as f:
                json.dump(asdict(config_run), f, indent=4, cls=ConfigEncoder)
            logging.info(f"Config JSON saved to {exp4_dir / f'{run_name}_config.json'}")
        except Exception:
            pass

    except Exception as e:
        logging.error(f"Failed to generate metrics, error: {e}")

if __name__ == "__main__":
    run_exp4()