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
from src.dynamics.process_models import Model_PCA_CV
from src.sensors.LidarModel import LidarMeasurementModel
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.utils.SimulationResult import SimulationResult
from bokeh.plotting import figure
from bokeh.io import show

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
    try:
        with h5py.File(h5_filepath, 'r') as f:
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
                    length=20.0, width=6.0, pca_coeffs=np.zeros(N_pca)
                )
                gt_ts.insert(time, gt_state)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        
    return gt_ts

def setup_real_data_config(init_x=30.0, init_y=30.0, dt=0.1, N_pca=4):
    """Creates the configurations needed for the tracker."""
    
    # 1. Simulation Config (Used mostly for metadata here since we aren't simulating)
    sim_config = SimulationConfig(
        name="NicholasData_ImplicitIEKF",
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
    # WARNING: You must set the initial state close to where the ship first appears in zPos!
    # I have set this to an arbitrary position (x=30, y=30). You may need to adjust this.
    initial_state_tracker = State_PCA(
        x=init_x, y=init_y, yaw=np.pi/4, vel_x=0.0, vel_y=0.0, yaw_rate=0.0,
        length=20.0, width=6.0, pca_coeffs=np.zeros(N_pca)
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
        method='implicit_iekf',
        process_model='cv',
        N_pca=N_pca,
        PCA_parameters_path=pca_path,
        pca_eigenvalues=eigenvalues,
        initial_state=initial_state_tracker,
        initial_std_devs=initial_std_devs_tracker,
        lidar_position=np.array(lidar_config.lidar_position),
        use_initialize_centroid=True, # Helps snap the initial state to the first point cloud!
        use_negative_info_angular=True,
        use_negative_info_front=True,
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
        ground_truth_ts = load_ground_truth_sequence(H5_FILE_PATH, dt=dt, N_pca=4)
    else:
        ground_truth_ts = TimeSequence()

    # Extract first valid measurement to initialize position
    init_x = 30.0
    init_y = 30.0
    for ts, scan in measurements_ts.items():
        if scan.x.size > 0:
            init_x = np.mean(scan.x)
            init_y = np.mean(scan.y)
            break

    # 1. Setup config and load data
    config = setup_real_data_config(init_x=init_x, init_y=init_y, dt=dt)
    config.sim.num_frames = len(measurements_ts)

    # 2. Setup the dynamic and measurement models
    filter_dyn_model = Model_PCA_CV(
        x_pos_std_dev=config.tracker.pos_north_std_dev, 
        y_pos_std_dev=config.tracker.pos_east_std_dev,
        yaw_std_dev=config.tracker.heading_std_dev, 
        N_pca=config.tracker.N_pca,
        length_std_dev=config.tracker.length_std_dev, 
        width_std_dev=config.tracker.width_std_dev 
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
    tracker = ImplicitIEKF(
        dynamic_model=filter_dyn_model, 
        lidar_model=lidar_model, 
        config=config
    )

    # Show initial plot before tracking
    print("Showing real data measurements and initial vessel shape...")
    show_real_data_plot(config, measurements_ts, init_x, init_y)

    # 4. Tracking Loop
    from src.senfuslib.timesequence import TimeSequence
    results_ts = TimeSequence()
    
    # Insert Initial Prior at -dt so it doesn't collide with the first measurement
    results_ts.insert(-dt, tracker.get_initial_update_result())

    for ts, measurement in tqdm(measurements_ts.items(), desc="Filtering Real Data"):
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
        
    # Metrics and Summary JSON (GT is empty here, so metrics are None)
    import json
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
        "avg_nees": None,
        "rmse": None,
        "avg_iou": None,
        "final_iou": None
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

    print(f"Tracking complete! Saved to {filename}")
    print("You can now open this folder in your analysis scripts.")

if __name__ == "__main__":
    run_real_dataset()