import os
import sys
import numpy as np
import scipy.io as sio
import pickle
from pathlib import Path
from tqdm import tqdm
import h5py
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from global_project_paths import SIMDATA_PATH
from src.utils.config_classes import TrackerConfig, SimulationConfig, Config, ExtentConfig, LidarConfig
from src.states.states import State_PCA, LidarScan
from src.senfuslib.timesequence import TimeSequence
from src.dynamics.process_models import Model_PCA_CV, Model_PCA_Inflation
from src.sensors.LidarModel import LidarMeasurementModel
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.tracker.EKF import EKF
from src.tracker.IterativeEKF import IterativeEKF
from src.tracker.IPLF import IPLF
from src.utils.SimulationResult import SimulationResult
from bokeh.plotting import figure
from bokeh.io import show

DEBUG = False

def show_real_data_plot(config, measurements_ts, init_x, init_y):
    """Helper function to show the real data measurements and initial vessel extent."""
    p = figure(title="Debug: Real Data Scans and Initial Vessel Extent", 
               x_axis_label='East (y)', y_axis_label='North (x)',
               match_aspect=True, width=800, height=800)

    # Plot all measurements (subsampled for performance)
    all_meas_x = []
    all_meas_y = []
    for i, (ts, scan) in enumerate(measurements_ts.items()):
        if scan.x.size > 0 and i % 5 == 0:  # plot every 5th scan
            all_meas_x.extend(scan.x)
            all_meas_y.extend(scan.y)

    p.scatter(all_meas_y, all_meas_x, size=2, color="blue", alpha=0.3, legend_label="Lidar Scans (East, North)")

    # Plot Lidar Position
    lidar_north, lidar_east = config.lidar.lidar_position
    p.scatter([lidar_east], [lidar_north], size=15, color="orange", marker="triangle", legend_label="Lidar")

    # Plot initial vessel extent
    initial_state = config.tracker.initial_state
    
    # We can compute the body coords from extent_cfg if it supports it
    try:
        from src.utils.geometry_utils import compute_exact_vessel_shape_global
        shape_x, shape_y = compute_exact_vessel_shape_global(initial_state, config.extent.shape_coords_body)
        p.line(shape_y, shape_x, line_color="red", line_width=2, legend_label="Initial Shape")
    except Exception as e:
        print(f"Could not compute exact vessel shape: {e}")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    
    try:
        show(p) 
    except Exception as e:
        print(f"Could not show Bokeh plot: {e}")

def load_zpos_sequence(mat_filepath, dt=0.1):
    """Loads zPos from the .mat file and converts it to a TimeSequence of LidarScans."""
    print(f"Loading measurements from {mat_filepath}...")
    
    # squeeze_me=True simplifies the loaded MATLAB cell arrays
    mat_data = sio.loadmat(mat_filepath, squeeze_me=True)
    zPos_cells = mat_data['zPos']
    
    meas_ts = TimeSequence()
    
    for i, cell in enumerate(zPos_cells):
        time = i * dt
        # If the cell is empty (no measurements at this timestep)
        if isinstance(cell, np.ndarray) and cell.size == 0:
            scan = LidarScan(x=np.array([]), y=np.array([]))
        else:
            # cell is typically a 2xN array [North, East]
            cell = np.atleast_2d(cell)
            # Check orientation. If it's Nx2, transpose it to 2xN
            if cell.shape[0] != 2 and cell.shape[1] == 2:
                cell = cell.T
                
            scan = LidarScan(x=cell[0, :], y=cell[1, :])
            
        meas_ts.insert(time, scan)
        
    return meas_ts

def load_ground_truth_sequence(h5_filepath, dt=0.1, N_pca=4):
    """Loads ground truth kinematics from the HDF5 file into a TimeSequence of State_PCA."""
    print(f"Loading ground truth from {h5_filepath}...")
    gt_ts = TimeSequence()


    # Load PCA parameters to compute exact PCA coeffs for the target shape
    from src.extent_model.boat_pca_utils import get_pca_coeffs_from_radii
    pca_path = "data/input_parameters/ShipDatasetPCAParameters.npz"
    
    L_gt = 20.0
    W_gt = 6.0
    pca_coeffs_gt = np.zeros(N_pca)
    
    try:
        with h5py.File(h5_filepath, 'r') as f:
            # Extract extent and compute L, W and PCA coeffs
            if 'extendRadii' in f['ship_trajectory_0']:
                extend_radii = f['ship_trajectory_0']['extendRadii'][:].flatten()
                extend_angles = f['ship_trajectory_0']['extendAngles'][:].flatten()
                
                xs_raw = extend_radii * np.cos(extend_angles)
                ys_raw = extend_radii * np.sin(extend_angles)
                L_gt = np.max(xs_raw) - np.min(xs_raw)
                W_gt = np.max(ys_raw) - np.min(ys_raw)
                
                # Center the true bounding box to (0,0) before taking PCA
                cx = (np.max(xs_raw) + np.min(xs_raw)) / 2.0
                cy = (np.max(ys_raw) + np.min(ys_raw)) / 2.0
                xs_cen = xs_raw - cx
                ys_cen = ys_raw - cy
                
                # The PCA model was trained on dimension-squashed 1x1 bounding boxes
                xs_squashed = xs_cen / L_gt
                ys_squashed = ys_cen / W_gt

                extend_radii = np.sqrt(xs_squashed**2 + ys_squashed**2)
                extend_angles = np.arctan2(ys_squashed, xs_squashed)

                # Must sort specifically after wrapping/centering to maintain monotonic order
                sort_idx = np.argsort(extend_angles)
                extend_angles = extend_angles[sort_idx]
                extend_radii = extend_radii[sort_idx]

                pca_coeffs_gt = get_pca_coeffs_from_radii(extend_radii, extend_angles, 1.0, N_pca, pca_path=pca_path)

            xKin = f['ship_trajectory_0']['xKin'][:]
            num_frames = xKin.shape[1]
            for i in range(num_frames):
                time = i * dt
                # Pos-X, Pos-Y, Heading, Vel-X, Vel-Y, Yaw Rate
                x = xKin[0, i]
                y = xKin[1, i]
                yaw = xKin[2, i]
                vel_x = xKin[3, i]
                vel_y = xKin[4, i]
                yaw_rate = xKin[5, i]
                
                gt_state = State_PCA(
                    x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, yaw_rate=yaw_rate,
                    length=L_gt, width=W_gt, pca_coeffs=pca_coeffs_gt
                )
                gt_ts.insert(time, gt_state)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        
    return gt_ts, L_gt, W_gt, pca_coeffs_gt

def setup_real_data_config(init_x=30.0, init_y=30.0, init_yaw=np.pi/4, L_gt=20.0, W_gt=6.0, init_pca_coeffs=None, dt=0.1, N_pca=4, method='implicit_iekf'):
    """Creates the configurations needed for the tracker."""
    
    # 1. Simulation Config (Used mostly for metadata here since we aren't simulating)
    sim_config = SimulationConfig(
        name=f"NicholasData_{method}",
        num_frames=478, # Length of the dataset
        dt=dt,
    )

    # 2. Lidar Config (Assume sensor is at origin for this dataset)
    lidar_config = LidarConfig(
        lidar_position=(0.0, 0.0), 
        max_distance=150.0
    )

    # 3. Extent Config
    # If the user has a pre-converted HDF5 file with the actual extent, we could load it:
    try:
        with h5py.File("data/real_datasets/Nicholasdata_filtered.h5", 'r') as f:
            extend_radii = f['ship_trajectory_0']['extendRadii'][:]
            extend_angles = f['ship_trajectory_0']['extendAngles'][:]
        shape_params = {"type": "custom_extent", "radii": extend_radii, "angles": extend_angles}
    except:
        print("Could not load real extent, falling back to ellipse.")
        shape_params = {"type": "ellipse", "L": 20.0, "W": 6.0}

    extent_config = ExtentConfig(
        N_fourier=64,
        d_angle=np.deg2rad(1.0),
        shape_params_true=shape_params
    )

    # 4. Tracker Config
    if init_pca_coeffs is None:
        init_pca_coeffs = np.zeros(N_pca)

    initial_state_tracker = State_PCA(
        x=init_x, y=init_y, yaw=init_yaw, vel_x=0.0, vel_y=0.0, yaw_rate=0.0,
        length=L_gt, width=W_gt, pca_coeffs=init_pca_coeffs
    )

    initial_std_devs_tracker = State_PCA(
        x=5.0, y=5.0, yaw=0.5, vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
        length=2.0, width=2.0, pca_coeffs=np.ones(N_pca) * 0.5 
    )
    
    # Load PCA dataset eigenvalues
    pca_path = "data/input_parameters/ShipDatasetPCAParameters.npz"
    pca_data = np.load(pca_path)
    eigenvalues = pca_data['eigenvalues'][:N_pca].real

    tracker_config = TrackerConfig(
        method=method,
        process_model='cv',

        use_gt_state_for_bodyangles_calc = False,
        use_initialize_centroid = True,
        N_pca=N_pca,
        PCA_parameters_path=pca_path,
        pos_north_std_dev=0.3,
        pos_east_std_dev=0.3,
        heading_std_dev=0.1,
        length_std_dev=0.1,
        width_std_dev=0.1,
        pca_std_dev_scale=0.3,
        lidar_std_dev=0.15,

        initial_state=initial_state_tracker,
        initial_std_devs=initial_std_devs_tracker,
        lidar_position=np.array(lidar_config.lidar_position),
        pca_eigenvalues=eigenvalues,

        use_D_imp_for_R=True,

        use_state_clamping=True,
        use_mahalanobis_projection=True,

        use_negative_info_angular=True,
        use_negative_info_front=True,
        use_negative_info_centroid=False,

        use_absolute_L_W_prior=False,
        use_L_W_aspect_ratio_prior=True,

        use_scaled_R=False,
        force_kinematic_unobservability=False,
    )

    return Config(sim=sim_config, lidar=lidar_config, tracker=tracker_config, extent=extent_config)

def run_real_dataset():
    MAT_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.mat"
    
    if not os.path.exists(MAT_FILE_PATH):
        print(f"Error: Could not find {MAT_FILE_PATH}")
        return

    # 1. Setup config and load data
    dt = 0.1 # Dataset is 10Hz
    measurements_ts = load_zpos_sequence(MAT_FILE_PATH, dt=dt)
    
    H5_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.h5"
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

    method_list = ["ekf", "iekf", "implicit_ekf", "implicit_iekf", "iplf"]
    
    for target_method in method_list:
        print(f"\n==========================================")
        print(f"Running filtering with method: {target_method}")
        print(f"==========================================")

        # 1. Setup config and load data
        config = setup_real_data_config(init_x=init_x, init_y=init_y, init_yaw=init_yaw, L_gt=L_gt, W_gt=W_gt, init_pca_coeffs=None, dt=dt, method=target_method)
        config.sim.num_frames = len(measurements_ts)
        config.sim.name = f"test3_real_data_{config.tracker.method}"

        # 2. Setup the dynamic and measurement models
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

        # 3. Initialize Tracker
        if target_method == "ekf":
            tracker = EKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif target_method == "iekf":
            tracker = IterativeEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif target_method == "implicit_ekf":
            tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config, max_iterations=1)
        elif target_method == "implicit_iekf":
            tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif target_method == "iplf":
            tracker = IPLF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        else:
            raise ValueError(f"Unknown method {target_method}")

        # Show initial plot before tracking
        if target_method == method_list[0]:
            print("Showing real data measurements and initial vessel shape...")
            # show_real_data_plot(config, measurements_ts, init_x, init_y)

        # 4. Tracking Loop
        from src.senfuslib.timesequence import TimeSequence
        results_ts = TimeSequence()
        
        # Insert Initial Prior at -dt so it doesn't collide with the first measurement
        results_ts.insert(-dt, tracker.get_initial_update_result())

        for ts, measurement in tqdm(measurements_ts.items(), desc=f"Filtering Real Data ({target_method})"):
            gt_state = ground_truth_ts.get(ts) if ground_truth_ts else None
            
            if measurement.x.size == 0:
                # If there are no points, just predict and skip update
                tracker.predict()
                continue
                
            tracker.predict()
            
            update_result = tracker.update(measurement, ground_truth=gt_state) 
            results_ts.insert(ts, update_result)

        # 5. Save Results
        sim_dir = os.path.join(SIMDATA_PATH, config.sim.name)
        os.makedirs(sim_dir, exist_ok=True)
        filename = os.path.join(sim_dir, f"{config.sim.name}.pkl")

        data_to_save = SimulationResult(
            config=config, 
            ground_truth_ts=ground_truth_ts,
            measurements_global_ts=measurements_ts, # Assuming local == global for this dataset
            tracker_results_ts=results_ts,
            static_covariances={"Q": filter_dyn_model.Q_d(dt=dt), "R_point": lidar_model.R_single_point()}
        )
        
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)
            
        # Metrics and Summary JSON
        import json
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
            for i, res in enumerate(results_ts.values):
                if i < len(ground_truth_ts.values):
                    gt_state = ground_truth_ts.values[i]
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
            "name": config.sim.name,
            "method": config.tracker.method,
            "trajectory_type": "real_data",
            "scenario": getattr(config.sim, "scenario", None),
            "num_rays": getattr(config.lidar, "num_rays", None),
            "use_D_imp_for_R": getattr(config.tracker, 'use_D_imp_for_R', False),
            "use_scaled_R": getattr(config.tracker, 'use_scaled_R', False),
            "use_negative_info_angular": getattr(config.tracker, 'use_negative_info_angular', False),
            "use_negative_info_front": getattr(config.tracker, 'use_negative_info_front', False),
            "use_negative_info_centroid": getattr(config.tracker, 'use_negative_info_centroid', False),
            "use_initialize_centroid": getattr(config.tracker, 'use_initialize_centroid', False),
            "avg_nees": float(avg_nees) if avg_nees is not None else None,
            "avg_nis": float(avg_nis) if avg_nis is not None else None,
            "full_state_rmse": float(full_state_rmse) if full_state_rmse is not None else None,
            "rmse_pos": float(rmse_pos) if rmse_pos is not None else None,
            "avg_iou": float(avg_iou) if avg_iou is not None else None,
            "final_iou": float(final_iou) if final_iou is not None else None,
            "nees_in_interval_95": float(nees_in_interval) if nees_in_interval is not None else None,
            "nis_in_interval_95": float(nis_in_interval) if nis_in_interval is not None else None,
        }
        json_filename = os.path.join(sim_dir, f"{config.sim.name}.json")
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
            with open(os.path.join(sim_dir, f"{config.sim.name}_config.json"), "w") as f:
                json.dump(asdict(config), f, indent=4, cls=ConfigEncoder)
        except Exception:
            pass

        print(f"Tracking complete for {target_method}! Saved to {filename}")

if __name__ == "__main__":
    run_real_dataset()