import os
import sys
import numpy as np
import pickle
import json
from bokeh.plotting import figure
from pathlib import Path
from tqdm import tqdm

from global_project_paths import PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from global_project_paths import SIMDATA_PATH

from src.dynamics.process_models import GroundTruthModel, Model_GP_CV, Model_PCA_CV, Model_PCA_Temporal, Model_PCA_Inflation
from src.dynamics.trajectories import CircleTrajectory, WaypointTrajectory, ConstantVelocityTrajectory
from src.sensors.LidarModel import LidarSimulator

from tracker.EKF import EKF
from src.tracker.IterativeEKF import IterativeEKF
# from src.tracker.gauss_newton import GaussNewton
# from src.tracker.levenberg_marquardt import LevenbergMarquardt
from src.tracker.BFGS import BFGS
# from src.tracker.SLSQP import SLSQP
# from src.tracker.smoothing_SLSQP import SmoothingSLSQP
# from src.tracker.UnscentedKalmanFilter import UKF
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.tracker.IPLF import IPLF
from src.tracker.UT_IPLF import UT_IPLF

from src.tracker.ImplicitSmoother import ImplicitSmoother
from src.tracker.IPLFSmoother import IPLFSmoother
from src.tracker.FullBatchSmoother import FullBatchSmoother

from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.sensors.LidarModel import LidarMeasurementModel

from src.senfuslib.simulator import Simulator
from src.senfuslib.timesequence import TimeSequence

from src.utils.SimulationResult import SimulationResult
from src.utils.geometry_utils import compute_exact_vessel_shape_global, compute_estimated_shape_global, calculate_iou

from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result

from src.utils.config_classes import Config

from src.utils.GaussianProcess import GaussianProcess
from src.sensors.LidarModelGP import LidarModelGP 
from src.tracker.GP_IEKF import GP_IEKF

def _show_gt_plot(config, ground_truth_ts, extent_cfg, lidar_cfg, traj_cfg):
    """Helper function to show the ground truth trajectory plot"""
    ts_values = list(ground_truth_ts.values)
    # NED Frame: x is North, y is East.
    # We want North on Y-axis (vertical) and East on X-axis (horizontal).
    # So Plot X = state.y (East), Plot Y = state.x (North)
    
    east_vals = [state.y for state in ts_values]
    north_vals = [state.x for state in ts_values]
    
    p = figure(title="Debug: Ground Truth Trajectory with Vessel Extent", 
               x_axis_label='East (y)', y_axis_label='North (x)',
               match_aspect=True, width=800, height=800)

    # Plot Trajectory
    p.line(east_vals, north_vals, legend_label="GT Path", line_width=2, color="blue")
    p.circle(east_vals, north_vals, size=4, color="blue", alpha=0.5)

    # Start/End markers
    p.scatter([east_vals[0]], [north_vals[0]], size=10, color="green", legend_label="Start")
    p.scatter([east_vals[-1]], [north_vals[-1]], size=10, color="red", legend_label="End")

    # Plot Center of Trajectory (if circle)
    if traj_cfg.type == "circle":
        center_north, center_east = traj_cfg.center
        p.scatter([center_east], [center_north], size=10, color="orange", marker="cross", legend_label="Traj Center")

    # Plot Waypoints (if waypoints)
    if traj_cfg.type == "waypoints" and hasattr(traj_cfg, 'waypoints'):
        wp_north = [wp[0] for wp in traj_cfg.waypoints]
        wp_east = [wp[1] for wp in traj_cfg.waypoints]
        p.scatter(wp_east, wp_north, size=10, color="purple", marker="triangle", legend_label="Waypoints")
        # Connect waypoints with a dashed line to show intended path
        p.line(wp_east, wp_north, line_color="purple", line_dash="dashed", line_width=1, alpha=0.5)

    # Plot Lidar Position
    lidar_north, lidar_east = lidar_cfg.lidar_position
    p.scatter([lidar_east], [lidar_north], size=15, color="orange", marker="triangle", legend_label="Lidar")

    # Plot vessel extent every 20 frames
    for i in range(0, len(ts_values), 20):
        state = ts_values[i]
        shape_x, shape_y = compute_exact_vessel_shape_global(state, extent_cfg.shape_coords_body)
        # shape_x is North, shape_y is East. Plot (East, North) -> (shape_y, shape_x)
        p.line(shape_y, shape_x, line_color="black", line_width=1, line_alpha=0.5)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    # Open in browser
    import webbrowser
    from bokeh.io import show
    # If running in a notebook, output_notebook() is needed, but assuming script execution here:
    try:
        show(p) 
    except Exception as e:
        print(f"Could not show Bokeh plot: {e}")

def run_single_simulation(config: Config) -> SimulationResult:
    """
    Runs a single simulation using the senfuslib.Simulator and new architecture.
    """
    sim_cfg = config.sim
    tracker_cfg = config.tracker
    lidar_cfg = config.lidar
    extent_cfg = config.extent
    
    method = tracker_cfg.method
    rng = np.random.default_rng(seed=sim_cfg.seed)

    traj_cfg = sim_cfg.trajectory
    if traj_cfg.type == "linear":
        trajectory_strategy = ConstantVelocityTrajectory()
    elif traj_cfg.type == "circle":
        trajectory_strategy = CircleTrajectory(
            center=traj_cfg.center, target_speed=traj_cfg.speed,
            radius=traj_cfg.radius, clockwise=traj_cfg.clockwise
        )
    elif traj_cfg.type == "waypoints":
        trajectory_strategy = WaypointTrajectory(
            waypoints=traj_cfg.waypoints, target_speed=traj_cfg.speed
        )
    else:
        raise ValueError(f"Unknown trajectory type: {traj_cfg.type}")

    gt_dynamic_model = GroundTruthModel(
        rng=rng, yaw_rate_std_dev=sim_cfg.gt_yaw_rate_std_dev,
        trajectory_strategy=trajectory_strategy
    )

    common_kwargs = dict(
        x_pos_std_dev=tracker_cfg.pos_north_std_dev, y_pos_std_dev=tracker_cfg.pos_east_std_dev,
        yaw_std_dev=tracker_cfg.heading_std_dev, N_pca=tracker_cfg.N_pca,
        length_std_dev=config.tracker.length_std_dev, width_std_dev=config.tracker.width_std_dev 
    )

    if tracker_cfg.process_model == "cv":
        filter_dyn_model = Model_PCA_CV(**common_kwargs)
    elif tracker_cfg.process_model == "temporal":
        filter_dyn_model = Model_PCA_Temporal(**common_kwargs, eta_f=tracker_cfg.temporal_eta, pca_process_var=config.tracker.pca_eigenvalues)
    elif tracker_cfg.process_model == "inflation":
        filter_dyn_model = Model_PCA_Inflation(**common_kwargs, lambda_f=tracker_cfg.inflation_lambda, pca_std_dev_scale=tracker_cfg.pca_std_dev_scale, use_proportional_pca_random_walk=tracker_cfg.use_proportional_pca_random_walk, pca_eigenvalues=config.tracker.pca_eigenvalues)
    else:
        raise ValueError(f"Unknown process model: {tracker_cfg.process_model}")

    pca_params = None
    if tracker_cfg.PCA_parameters_path:
        pca_params = np.load(Path(tracker_cfg.PCA_parameters_path))
        
    lidar_model = LidarMeasurementModel(
        lidar_position=np.array(lidar_cfg.lidar_position),
        lidar_std_dev=tracker_cfg.lidar_std_dev,
        pca_mean=pca_params['mean'],
        pca_eigenvectors=pca_params['eigenvectors'][:, :tracker_cfg.N_pca].real,
        extent_cfg=extent_cfg
    )

    if method == "bfgs":
        tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "ekf":
        tracker = EKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "iekf":
        tracker = IterativeEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "implicit_iekf" or method == "implicit_ekf":
        tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "iplf":
        tracker = IPLF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "ut_iplf":
        tracker = UT_IPLF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "full_batch_smoother":
        tracker = FullBatchSmoother(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    elif method == "iplf_smoother":
        tracker = IPLFSmoother(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
    else:
        raise ValueError(f"Unknown method {method}")

    lidar_simulator = LidarSimulator(
        lidar_position=np.array(lidar_cfg.lidar_position),
        num_rays=lidar_cfg.num_rays, max_distance=lidar_cfg.max_distance,
        lidar_gt_std_dev=lidar_cfg.lidar_gt_std_dev, rng=rng, extent_cfg=extent_cfg
    )

    simulator = Simulator(
        dynamic_model=gt_dynamic_model, sensor_model=lidar_simulator, sensor_setter=None,
        init_state=sim_cfg.initial_state_gt, end_time=sim_cfg.num_frames * sim_cfg.dt,
        dt=sim_cfg.dt, config=config, seed=str(sim_cfg.seed), use_cache=sim_cfg.use_cache
    )

    # --- Generate Simulation Data ---
    print(f"Generating simulation data for {sim_cfg.num_frames} frames...")
    ground_truth_ts = simulator.get_gt()
    
    if config.sim.show_gt_plot:
        _show_gt_plot(config, ground_truth_ts, extent_cfg, lidar_cfg, traj_cfg)

    measurements_lidar_frame_ts = simulator.get_meas()
    lidar_pos_global = np.array(lidar_cfg.lidar_position).reshape(2, 1)
    measurements_global_ts = measurements_lidar_frame_ts.map(lambda scan: scan + lidar_pos_global)

    # --- Run Filtering Loop ---
    results_ts: TimeSequence[TrackerUpdateResult] = TimeSequence() 
    initial_result = tracker.get_initial_update_result()
    results_ts.insert(0.0, initial_result)

    for ts, measurement in tqdm(measurements_lidar_frame_ts.items(), desc="Filtering measurements"):
        tracker.predict()
        update_result = tracker.update(measurement, ground_truth=ground_truth_ts.get_t(ts))
        results_ts.insert(ts, update_result)

    static_covariances = {
        "Q": filter_dyn_model.Q_d(dt=sim_cfg.dt),
        "R_point": lidar_model.R_single_point()
    }

    # --- Helper Function to Save and Evaluate ---
    def save_and_evaluate_results(results_sequence, suffix=""):
        sim_dir = os.path.join(SIMDATA_PATH, sim_cfg.name)
        os.makedirs(sim_dir, exist_ok=True)
        
        run_name = f"{sim_cfg.name}{suffix}"
        filename = os.path.join(sim_dir, f"{run_name}.pkl")

        data_to_save = SimulationResult(
            config=config, ground_truth_ts=ground_truth_ts,
            measurements_global_ts=measurements_global_ts, tracker_results_ts=results_sequence,
            static_covariances=static_covariances
        )
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Simulation run data saved to {filename}")

        # Metrics
        try:
            consistency_analyzer = create_consistency_analysis_from_sim_result(data_to_save)
            nees_data = consistency_analyzer.get_nees(indices='all')
            avg_nees = nees_data.a if nees_data else None
            
            if hasattr(consistency_analyzer, 'x_err_gauss') and consistency_analyzer.x_err_gauss is not None:
                err_arrays =[e.mean for e in consistency_analyzer.x_err_gauss.values]
                rmse = np.sqrt(np.mean(np.square(err_arrays)))
            else:
                rmse = None

            ious =[]
            for i, res in enumerate(results_sequence.values):
                if i < len(ground_truth_ts.values):
                    gt_state = ground_truth_ts.values[i]
                    est_state = res.state_posterior.mean
                    gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, extent_cfg.shape_coords_body)
                    try:
                        est_x, est_y = compute_estimated_shape_global(est_state, config, pca_params)
                        iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                        ious.append(iou)
                    except Exception:
                        pass
            avg_iou = np.mean(ious) if ious else None
            final_iou = ious[-1] if ious else None

        except Exception as e:
            print(f"Error calculating metrics for JSON sidecar: {e}")
            avg_nees, rmse, avg_iou, final_iou = None, None, None, None

        # Summary JSON
        summary_data = {
            "name": run_name,
            "method": tracker_cfg.method,
            "trajectory_type": traj_cfg.type,
            "scenario": getattr(sim_cfg, "scenario", None),
            "num_rays": getattr(lidar_cfg, "num_rays", None),
            "use_D_imp_for_R": getattr(tracker_cfg, 'use_D_imp_for_R', False),
            "use_scaled_R": getattr(tracker_cfg, 'use_scaled_R', False),
            "use_negative_info_angular": getattr(tracker_cfg, 'use_negative_info_angular', False),
            "use_negative_info_front": getattr(tracker_cfg, 'use_negative_info_front', False),
            "use_negative_info_centroid": getattr(tracker_cfg, 'use_negative_info_centroid', False),
            "use_initialize_centroid": getattr(tracker_cfg, 'use_initialize_centroid', False),
            "avg_nees": float(avg_nees) if avg_nees is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
            "avg_iou": float(avg_iou) if avg_iou is not None else None,
            "final_iou": float(final_iou) if final_iou is not None else None
        }
        
        json_filename = os.path.join(sim_dir, f"{run_name}.json")
        with open(json_filename, "w") as f:
            json.dump(summary_data, f, indent=4)

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
            with open(os.path.join(sim_dir, f"{run_name}_config.json"), "w") as f:
                json.dump(asdict(config), f, indent=4, cls=ConfigEncoder)
        except Exception:
            pass

        return data_to_save

    # =========================================================================
    # 1. SAVE THE FORWARD PASS
    # =========================================================================
    forward_suffix = "_forward" if method == "full_batch_smoother" else ""
    data_to_save_forward = save_and_evaluate_results(results_ts, suffix=forward_suffix)

    # =========================================================================
    # 2. TRIGGER THE BATCH SMOOTHER & SAVE SMOOTHED RESULTS
    # =========================================================================
    if method == "full_batch_smoother":
        smoothed_results_ts = tracker.smooth_trajectory(results_ts)
        data_to_save_smoothed = save_and_evaluate_results(smoothed_results_ts, suffix="_smoothed")
        return data_to_save_smoothed # Returning smoothed so run_batch/main get the final product

    return data_to_save_forward