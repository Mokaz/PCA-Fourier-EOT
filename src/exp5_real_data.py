import os
import sys
import numpy as np
import scipy.io as sio
import pickle
from pathlib import Path
from tqdm import tqdm
import h5py
import logging
import copy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from global_project_paths import SIMDATA_PATH
from src.utils.config_classes import Config, LidarConfig, ExtentConfig, TrackerConfig, SimulationConfig, TrajectoryConfig
from src.states.states import State_PCA, LidarScan
from src.senfuslib.timesequence import TimeSequence
from src.dynamics.process_models import Model_PCA_CV, Model_PCA_Inflation
from src.sensors.LidarModel import LidarMeasurementModel
from src.tracker.ImplicitIEKF import ImplicitIEKF
from src.tracker.EKF import EKF
from src.tracker.IterativeEKF import IterativeEKF
from src.utils.SimulationResult import SimulationResult

def load_zpos_sequence(mat_filepath, dt=0.1):
    """Loads zPos from the .mat file and converts it to a TimeSequence of LidarScans."""
    print(f"Loading measurements from {mat_filepath}...")
    
    mat_data = sio.loadmat(mat_filepath, squeeze_me=True)
    zPos_cells = mat_data['zPos']
    
    meas_ts = TimeSequence()
    
    for i, cell in enumerate(zPos_cells):
        time = i * dt
        if isinstance(cell, np.ndarray) and cell.size == 0:
            scan = LidarScan(x=np.array([]), y=np.array([]))
        else:
            cell = np.atleast_2d(cell)
            if cell.shape[0] != 2 and cell.shape[1] == 2:
                cell = cell.T
            scan = LidarScan(x=cell[0, :], y=cell[1, :])
            
        meas_ts.insert(time, scan)
        
    return meas_ts

def load_ground_truth_sequence(h5_filepath, dt=0.1, N_pca=4):
    """Loads ground truth kinematics from the HDF5 file into a TimeSequence of State_PCA."""
    print(f"Loading ground truth from {h5_filepath}...")
    gt_ts = TimeSequence()

    from src.extent_model.boat_pca_utils import get_pca_coeffs_from_radii
    pca_path = "data/input_parameters/ShipDatasetPCAParameters.npz"
    
    L_gt = 20.0
    W_gt = 6.0
    pca_coeffs_gt = np.zeros(N_pca)
    
    try:
        with h5py.File(h5_filepath, 'r') as f:
            if 'extendRadii' in f['ship_trajectory_0']:
                extend_radii = f['ship_trajectory_0']['extendRadii'][:].flatten()
                extend_angles = f['ship_trajectory_0']['extendAngles'][:].flatten()
                
                xs_raw = extend_radii * np.cos(extend_angles)
                ys_raw = extend_radii * np.sin(extend_angles)
                L_gt = np.max(xs_raw) - np.min(xs_raw)
                W_gt = np.max(ys_raw) - np.min(ys_raw)
                
                cx = (np.max(xs_raw) + np.min(xs_raw)) / 2.0
                cy = (np.max(ys_raw) + np.min(ys_raw)) / 2.0
                xs_cen = xs_raw - cx
                ys_cen = ys_raw - cy
                
                xs_squashed = xs_cen / L_gt
                ys_squashed = ys_cen / W_gt

                extend_radii = np.sqrt(xs_squashed**2 + ys_squashed**2)
                extend_angles = np.arctan2(ys_squashed, xs_squashed)

                sort_idx = np.argsort(extend_angles)
                extend_angles = extend_angles[sort_idx]
                extend_radii = extend_radii[sort_idx]

                pca_coeffs_gt = get_pca_coeffs_from_radii(extend_radii, extend_angles, 1.0, N_pca, pca_path=pca_path)

            xKin = f['ship_trajectory_0']['xKin'][:]
            num_frames = xKin.shape[1]
            for i in range(num_frames):
                time = i * dt
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

base_config = Config(
    sim=SimulationConfig(
        name='exp5_real_data',
        num_simulations=1,
        num_frames=478,
        dt=0.1,
        seed=42,
        gt_yaw_rate_std_dev=0.1,
        initial_state_gt=None,
        trajectory=TrajectoryConfig(
            type='linear',
            speed=3.0,
            center=(30.0, 0.0),
            radius=40.0,
            clockwise=True,
            waypoints=[(0, -40), (0, 40), (60, 40), (60, -40)],
        ),
        use_cache=False,
        show_gt_plot=False,
    ),
    lidar=LidarConfig(
        lidar_position=(0.0, 0.0),
        num_rays=720,
        max_distance=150.0,
        lidar_gt_mean=0.0,
        lidar_gt_std_dev=0.15,
    ),
    tracker=TrackerConfig(
        use_gt_state_for_bodyangles_calc=False,
        use_initialize_centroid=False,
        method='implicit_iekf', # will be replaced in loop
        N_pca=4,
        PCA_parameters_path='data/input_parameters/ShipDatasetPCAParameters.npz',
        process_model='inflation',
        temporal_eta=0.1,
        temporal_pca_var=1.0,
        inflation_lambda=1.0,
        N_gp_points=20,
        gp_length_scale=0.5,
        gp_signal_var=1.0,
        gp_forgetting_factor=0.05,
        gp_use_negative_info=True,
        pos_north_std_dev=0.75,
        pos_east_std_dev=0.75,
        heading_std_dev=0.1,
        length_std_dev=0.1,
        width_std_dev=0.1,
        pca_std_dev_scale=0.4,
        use_proportional_pca_random_walk=True,
        lidar_std_dev=0.3,
        debug_prints=False,
        initial_state=State_PCA(
            x=-5.177493388797188,
            y=-33.55847902854754,
            yaw=1.0771595877943292,
            vel_x=0.0,
            vel_y=0.0,
            yaw_rate=0.0,
            length=5.6000000000000005,
            width=2.020000000000001,
            pca_coeffs=np.array([0.0, 0.0, 0.0, 0.0]),
        ),
        initial_std_devs=State_PCA(
            x=5.0,
            y=5.0,
            yaw=0.5,
            vel_x=2.0,
            vel_y=2.0,
            yaw_rate=0.1,
            length=2.0,
            width=2.0,
            pca_coeffs=np.array([0.5, 0.5, 0.5, 0.5]),
        ),
        lidar_position=np.array([0.0, 0.0]),
        pca_eigenvalues=np.array([1.0, 0.1780903774216213, 0.01826187045327505, 0.012036083386621767]),
        max_iterations=20,
        convergence_threshold=1e-06,
        use_state_clamping=True,
        use_mahalanobis_projection=True,
        mahalanobis_projection_prob=0.99,
        use_negative_info_angular=True,
        use_negative_info_front=True,
        use_negative_info_centroid=True,
        radial_margin=0.1,
        use_exact_extreme_angle=False,
        use_D_imp_for_R=False,
        use_scaled_R=False,
        force_kinematic_unobservability=False,
        use_arc_length_residual=False,
        R_neg_info_std_angle=0.05,
        R_neg_info_std_front=0.01,
        R_neg_info_std_centroid=0.01,
        use_absolute_L_W_prior=False,
        prior_target_L=20.0,
        prior_target_W=6.0,
        prior_size_std=5.0,
        use_L_W_aspect_ratio_prior=True,
        prior_aspect_ratio=3.8,
        prior_ratio_std=5.0,
        smoother_window_size=10,
    ),
    extent=ExtentConfig(
        N_fourier=64,
        d_angle=0.017453292519943295,
        shape_params_true={'type': 'custom_extent', 'radii': np.array([[2.8       ],
       [2.76416   ],
       [2.7291834 ],
       [2.69501931],
       [2.66161812],
       [2.62893138],
       [2.59691163],
       [2.56551222],
       [2.53468719],
       [2.50439112],
       [2.47457897],
       [2.44520593],
       [2.41622734],
       [2.38759847],
       [2.35927446],
       [2.33121012],
       [2.30335986],
       [2.2756775 ],
       [2.24811619],
       [2.22062825],
       [2.19316507],
       [2.16567699],
       [2.13811319],
       [2.1104216 ],
       [2.0825488 ],
       [2.05443999],
       [2.02603894],
       [1.99728797],
       [1.96812798],
       [1.9384986 ],
       [1.90833825],
       [1.87758447],
       [1.84617417],
       [1.81404418],
       [1.78113179],
       [1.74737561],
       [1.71271656],
       [1.67709918],
       [1.64051194],
       [1.60490589],
       [1.57128107],
       [1.53949562],
       [1.50942132],
       [1.48094198],
       [1.45395211],
       [1.4283557 ],
       [1.40406523],
       [1.38100074],
       [1.35908906],
       [1.33826312],
       [1.31846136],
       [1.29962716],
       [1.2817084 ],
       [1.26465701],
       [1.24842866],
       [1.23298233],
       [1.21828013],
       [1.20428693],
       [1.19097019],
       [1.17829973],
       [1.16624754],
       [1.15478761],
       [1.14389575],
       [1.1335495 ],
       [1.12372796],
       [1.1144117 ],
       [1.10558264],
       [1.09722398],
       [1.08932009],
       [1.08185644],
       [1.07481955],
       [1.06819689],
       [1.06197685],
       [1.05614867],
       [1.05070243],
       [1.04562894],
       [1.04091977],
       [1.03656715],
       [1.032564  ],
       [1.02890386],
       [1.02558088],
       [1.02258978],
       [1.01992585],
       [1.01758492],
       [1.01556336],
       [1.01385804],
       [1.01246632],
       [1.01138607],
       [1.01061564],
       [1.01015385],
       [1.01      ],
       [1.01015385],
       [1.01061564],
       [1.01138607],
       [1.01246632],
       [1.01385804],
       [1.01556336],
       [1.01758492],
       [1.01992585],
       [1.02258978],
       [1.02558088],
       [1.02890386],
       [1.032564  ],
       [1.03656715],
       [1.04091977],
       [1.04562894],
       [1.05070243],
       [1.05614867],
       [1.06197685],
       [1.06819689],
       [1.07481955],
       [1.08185644],
       [1.08932009],
       [1.09722398],
       [1.10558264],
       [1.1144117 ],
       [1.12372796],
       [1.1335495 ],
       [1.14389575],
       [1.15478761],
       [1.16624754],
       [1.17829973],
       [1.19097019],
       [1.20428693],
       [1.21828013],
       [1.23298233],
       [1.24842866],
       [1.26465701],
       [1.2817084 ],
       [1.29962716],
       [1.31846136],
       [1.33826312],
       [1.35908906],
       [1.38100074],
       [1.40406523],
       [1.4283557 ],
       [1.45395211],
       [1.48094198],
       [1.50942132],
       [1.53949562],
       [1.57128107],
       [1.60490589],
       [1.64051194],
       [1.67825654],
       [1.71831463],
       [1.76088126],
       [1.80617457],
       [1.85443924],
       [1.90595071],
       [1.96102007],
       [2.02      ],
       [2.08329199],
       [2.15135501],
       [2.22471616],
       [2.30398375],
       [2.3898636 ],
       [2.48317927],
       [2.58489771],
       [2.69616183],
       [2.81833239],
       [2.95304244],
       [2.96133791],
       [2.94409423],
       [2.92793692],
       [2.91283842],
       [2.89877331],
       [2.88571816],
       [2.8736515 ],
       [2.86255367],
       [2.85240675],
       [2.84319451],
       [2.83490235],
       [2.8275172 ],
       [2.82102751],
       [2.81542318],
       [2.81069555],
       [2.80683731],
       [2.80384257],
       [2.80170672],
       [2.80042652],
       [2.8       ],
       [2.80042652],
       [2.80170672],
       [2.80384257],
       [2.80683731],
       [2.81069555],
       [2.81542318],
       [2.82102751],
       [2.8275172 ],
       [2.83490235],
       [2.84319451],
       [2.85240675],
       [2.86255367],
       [2.8736515 ],
       [2.88571816],
       [2.89877331],
       [2.91283842],
       [2.92793692],
       [2.94409423],
       [2.96133791],
       [2.95304244],
       [2.81833239],
       [2.69616183],
       [2.58489771],
       [2.48317927],
       [2.3898636 ],
       [2.30398375],
       [2.22471616],
       [2.15135501],
       [2.08329199],
       [2.02      ],
       [1.96102007],
       [1.90595071],
       [1.85443924],
       [1.80617457],
       [1.76088126],
       [1.71831463],
       [1.67825654],
       [1.64051194],
       [1.60490589],
       [1.57128107],
       [1.53949562],
       [1.50942132],
       [1.48094198],
       [1.45395211],
       [1.4283557 ],
       [1.40406523],
       [1.38100074],
       [1.35908906],
       [1.33826312],
       [1.31846136],
       [1.29962716],
       [1.2817084 ],
       [1.26465701],
       [1.24842866],
       [1.23298233],
       [1.21828013],
       [1.20428693],
       [1.19097019],
       [1.17829973],
       [1.16624754],
       [1.15478761],
       [1.14389575],
       [1.1335495 ],
       [1.12372796],
       [1.1144117 ],
       [1.10558264],
       [1.09722398],
       [1.08932009],
       [1.08185644],
       [1.07481955],
       [1.06819689],
       [1.06197685],
       [1.05614867],
       [1.05070243],
       [1.04562894],
       [1.04091977],
       [1.03656715],
       [1.032564  ],
       [1.02890386],
       [1.02558088],
       [1.02258978],
       [1.01992585],
       [1.01758492],
       [1.01556336],
       [1.01385804],
       [1.01246632],
       [1.01138607],
       [1.01061564],
       [1.01015385],
       [1.01      ],
       [1.01015385],
       [1.01061564],
       [1.01138607],
       [1.01246632],
       [1.01385804],
       [1.01556336],
       [1.01758492],
       [1.01992585],
       [1.02258978],
       [1.02558088],
       [1.02890386],
       [1.032564  ],
       [1.03656715],
       [1.04091977],
       [1.04562894],
       [1.05070243],
       [1.05614867],
       [1.06197685],
       [1.06819689],
       [1.07481955],
       [1.08185644],
       [1.08932009],
       [1.09722398],
       [1.10558264],
       [1.1144117 ],
       [1.12372796],
       [1.1335495 ],
       [1.14389575],
       [1.15478761],
       [1.16624754],
       [1.17829973],
       [1.19097019],
       [1.20428693],
       [1.21828013],
       [1.23298233],
       [1.24842866],
       [1.26465701],
       [1.2817084 ],
       [1.29962716],
       [1.31846136],
       [1.33826312],
       [1.35908906],
       [1.38100074],
       [1.40406523],
       [1.4283557 ],
       [1.45395211],
       [1.48094198],
       [1.50942132],
       [1.53949562],
       [1.57128107],
       [1.60490589],
       [1.64051194],
       [1.67709918],
       [1.71271656],
       [1.74737561],
       [1.78113179],
       [1.81404418],
       [1.84617417],
       [1.87758447],
       [1.90833825],
       [1.9384986 ],
       [1.96812798],
       [1.99728797],
       [2.02603894],
       [2.05443999],
       [2.0825488 ],
       [2.1104216 ],
       [2.13811319],
       [2.16567699],
       [2.19316507],
       [2.22062825],
       [2.24811619],
       [2.2756775 ],
       [2.30335986],
       [2.33121012],
       [2.35927446],
       [2.38759847],
       [2.41622734],
       [2.44520593],
       [2.47457897],
       [2.50439112],
       [2.53468719],
       [2.56551222],
       [2.59691163],
       [2.62893138],
       [2.66161812],
       [2.69501931],
       [2.7291834 ],
       [2.76416   ]]), 'angles': np.array([[0.        ],
       [0.01745329],
       [0.03490659],
       [0.05235988],
       [0.06981317],
       [0.08726646],
       [0.10471976],
       [0.12217305],
       [0.13962634],
       [0.15707963],
       [0.17453293],
       [0.19198622],
       [0.20943951],
       [0.2268928 ],
       [0.2443461 ],
       [0.26179939],
       [0.27925268],
       [0.29670597],
       [0.31415927],
       [0.33161256],
       [0.34906585],
       [0.36651914],
       [0.38397244],
       [0.40142573],
       [0.41887902],
       [0.43633231],
       [0.45378561],
       [0.4712389 ],
       [0.48869219],
       [0.50614548],
       [0.52359878],
       [0.54105207],
       [0.55850536],
       [0.57595865],
       [0.59341195],
       [0.61086524],
       [0.62831853],
       [0.64577182],
       [0.66322512],
       [0.68067841],
       [0.6981317 ],
       [0.71558499],
       [0.73303829],
       [0.75049158],
       [0.76794487],
       [0.78539816],
       [0.80285146],
       [0.82030475],
       [0.83775804],
       [0.85521133],
       [0.87266463],
       [0.89011792],
       [0.90757121],
       [0.9250245 ],
       [0.9424778 ],
       [0.95993109],
       [0.97738438],
       [0.99483767],
       [1.01229097],
       [1.02974426],
       [1.04719755],
       [1.06465084],
       [1.08210414],
       [1.09955743],
       [1.11701072],
       [1.13446401],
       [1.15191731],
       [1.1693706 ],
       [1.18682389],
       [1.20427718],
       [1.22173048],
       [1.23918377],
       [1.25663706],
       [1.27409035],
       [1.29154365],
       [1.30899694],
       [1.32645023],
       [1.34390352],
       [1.36135682],
       [1.37881011],
       [1.3962634 ],
       [1.41371669],
       [1.43116999],
       [1.44862328],
       [1.46607657],
       [1.48352986],
       [1.50098316],
       [1.51843645],
       [1.53588974],
       [1.55334303],
       [1.57079633],
       [1.58824962],
       [1.60570291],
       [1.6231562 ],
       [1.6406095 ],
       [1.65806279],
       [1.67551608],
       [1.69296937],
       [1.71042267],
       [1.72787596],
       [1.74532925],
       [1.76278254],
       [1.78023584],
       [1.79768913],
       [1.81514242],
       [1.83259571],
       [1.85004901],
       [1.8675023 ],
       [1.88495559],
       [1.90240888],
       [1.91986218],
       [1.93731547],
       [1.95476876],
       [1.97222205],
       [1.98967535],
       [2.00712864],
       [2.02458193],
       [2.04203522],
       [2.05948852],
       [2.07694181],
       [2.0943951 ],
       [2.11184839],
       [2.12930169],
       [2.14675498],
       [2.16420827],
       [2.18166156],
       [2.19911486],
       [2.21656815],
       [2.23402144],
       [2.25147474],
       [2.26892803],
       [2.28638132],
       [2.30383461],
       [2.32128791],
       [2.3387412 ],
       [2.35619449],
       [2.37364778],
       [2.39110108],
       [2.40855437],
       [2.42600766],
       [2.44346095],
       [2.46091425],
       [2.47836754],
       [2.49582083],
       [2.51327412],
       [2.53072742],
       [2.54818071],
       [2.565634  ],
       [2.58308729],
       [2.60054059],
       [2.61799388],
       [2.63544717],
       [2.65290046],
       [2.67035376],
       [2.68780705],
       [2.70526034],
       [2.72271363],
       [2.74016693],
       [2.75762022],
       [2.77507351],
       [2.7925268 ],
       [2.8099801 ],
       [2.82743339],
       [2.84488668],
       [2.86233997],
       [2.87979327],
       [2.89724656],
       [2.91469985],
       [2.93215314],
       [2.94960644],
       [2.96705973],
       [2.98451302],
       [3.00196631],
       [3.01941961],
       [3.0368729 ],
       [3.05432619],
       [3.07177948],
       [3.08923278],
       [3.10668607],
       [3.12413936],
       [3.14159265],
       [3.15904595],
       [3.17649924],
       [3.19395253],
       [3.21140582],
       [3.22885912],
       [3.24631241],
       [3.2637657 ],
       [3.28121899],
       [3.29867229],
       [3.31612558],
       [3.33357887],
       [3.35103216],
       [3.36848546],
       [3.38593875],
       [3.40339204],
       [3.42084533],
       [3.43829863],
       [3.45575192],
       [3.47320521],
       [3.4906585 ],
       [3.5081118 ],
       [3.52556509],
       [3.54301838],
       [3.56047167],
       [3.57792497],
       [3.59537826],
       [3.61283155],
       [3.63028484],
       [3.64773814],
       [3.66519143],
       [3.68264472],
       [3.70009801],
       [3.71755131],
       [3.7350046 ],
       [3.75245789],
       [3.76991118],
       [3.78736448],
       [3.80481777],
       [3.82227106],
       [3.83972435],
       [3.85717765],
       [3.87463094],
       [3.89208423],
       [3.90953752],
       [3.92699082],
       [3.94444411],
       [3.9618974 ],
       [3.97935069],
       [3.99680399],
       [4.01425728],
       [4.03171057],
       [4.04916386],
       [4.06661716],
       [4.08407045],
       [4.10152374],
       [4.11897703],
       [4.13643033],
       [4.15388362],
       [4.17133691],
       [4.1887902 ],
       [4.2062435 ],
       [4.22369679],
       [4.24115008],
       [4.25860337],
       [4.27605667],
       [4.29350996],
       [4.31096325],
       [4.32841654],
       [4.34586984],
       [4.36332313],
       [4.38077642],
       [4.39822972],
       [4.41568301],
       [4.4331363 ],
       [4.45058959],
       [4.46804289],
       [4.48549618],
       [4.50294947],
       [4.52040276],
       [4.53785606],
       [4.55530935],
       [4.57276264],
       [4.59021593],
       [4.60766923],
       [4.62512252],
       [4.64257581],
       [4.6600291 ],
       [4.6774824 ],
       [4.69493569],
       [4.71238898],
       [4.72984227],
       [4.74729557],
       [4.76474886],
       [4.78220215],
       [4.79965544],
       [4.81710874],
       [4.83456203],
       [4.85201532],
       [4.86946861],
       [4.88692191],
       [4.9043752 ],
       [4.92182849],
       [4.93928178],
       [4.95673508],
       [4.97418837],
       [4.99164166],
       [5.00909495],
       [5.02654825],
       [5.04400154],
       [5.06145483],
       [5.07890812],
       [5.09636142],
       [5.11381471],
       [5.131268  ],
       [5.14872129],
       [5.16617459],
       [5.18362788],
       [5.20108117],
       [5.21853446],
       [5.23598776],
       [5.25344105],
       [5.27089434],
       [5.28834763],
       [5.30580093],
       [5.32325422],
       [5.34070751],
       [5.3581608 ],
       [5.3756141 ],
       [5.39306739],
       [5.41052068],
       [5.42797397],
       [5.44542727],
       [5.46288056],
       [5.48033385],
       [5.49778714],
       [5.51524044],
       [5.53269373],
       [5.55014702],
       [5.56760031],
       [5.58505361],
       [5.6025069 ],
       [5.61996019],
       [5.63741348],
       [5.65486678],
       [5.67232007],
       [5.68977336],
       [5.70722665],
       [5.72467995],
       [5.74213324],
       [5.75958653],
       [5.77703982],
       [5.79449312],
       [5.81194641],
       [5.8293997 ],
       [5.84685299],
       [5.86430629],
       [5.88175958],
       [5.89921287],
       [5.91666616],
       [5.93411946],
       [5.95157275],
       [5.96902604],
       [5.98647933],
       [6.00393263],
       [6.02138592],
       [6.03883921],
       [6.0562925 ],
       [6.0737458 ],
       [6.09119909],
       [6.10865238],
       [6.12610567],
       [6.14355897],
       [6.16101226],
       [6.17846555],
       [6.19591884],
       [6.21337214],
       [6.23082543],
       [6.24827872],
       [6.26573201]])},
    ),
)


def run_real_dataset():
    MAT_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.mat"
    
    if not os.path.exists(MAT_FILE_PATH):
        print(f"Error: Could not find {MAT_FILE_PATH}")
        return

    dt = 0.1
    measurements_ts = load_zpos_sequence(MAT_FILE_PATH, dt=dt)
    
    H5_FILE_PATH = "data/real_datasets/Nicholasdata_filtered.h5"
    if os.path.exists(H5_FILE_PATH):
        ground_truth_ts, L_gt, W_gt, pca_coeffs_gt = load_ground_truth_sequence(H5_FILE_PATH, dt=dt, N_pca=4)
    else:
        ground_truth_ts = TimeSequence()
        L_gt, W_gt, pca_coeffs_gt = 20.0, 6.0, np.zeros(4)

    method_list = ["ekf", "iekf", "implicit_ekf", "implicit_iekf"]
    
    for target_method in method_list:
        print(f"\n==========================================")
        print(f"Running filtering with method: {target_method}")
        print(f"==========================================")

        config = copy.deepcopy(base_config)
        config.tracker.method = target_method
        config.sim.num_frames = len(measurements_ts)
        config.sim.name = f"exp5_numrays_720_real_data_{target_method}"
        
        # Override specific tracking differences based on method mapping below
        if target_method == "implicit_ekf":
            config.tracker.max_iterations = 1
        elif target_method in ["iekf", "implicit_iekf"]:
            config.tracker.max_iterations = 20
        elif target_method == "ekf":
            config.tracker.max_iterations = 1

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

        if target_method == "ekf":
            tracker = EKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif target_method == "iekf":
            tracker = IterativeEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        elif target_method == "implicit_ekf":
            tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config, max_iterations=1)
        elif target_method == "implicit_iekf":
            tracker = ImplicitIEKF(dynamic_model=filter_dyn_model, lidar_model=lidar_model, config=config)
        else:
            raise ValueError(f"Unknown method {target_method}")

        results_ts = TimeSequence()
        
        results_ts.insert(-dt, tracker.get_initial_update_result())

        for ts, measurement in tqdm(measurements_ts.items(), desc=f"Filtering Real Data ({target_method})"):
            gt_state = ground_truth_ts.get(ts) if ground_truth_ts else None
            
            if measurement.x.size == 0:
                tracker.predict()
                continue
                
            tracker.predict()
            
            update_result = tracker.update(measurement, ground_truth=gt_state) 
            results_ts.insert(ts, update_result)

        sim_dir = os.path.join(SIMDATA_PATH, "single_runs", config.sim.name)
        os.makedirs(sim_dir, exist_ok=True)
        filename = os.path.join(sim_dir, f"{config.sim.name}.pkl")

        data_to_save = SimulationResult(
            config=config, 
            ground_truth_ts=ground_truth_ts,
            measurements_global_ts=measurements_ts,
            tracker_results_ts=results_ts,
            static_covariances={"Q": filter_dyn_model.Q_d(dt=dt), "R_point": lidar_model.R_single_point()}
        )
        
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f)
            
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
            for j, res in enumerate(results_ts.values):
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
