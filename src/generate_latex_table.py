import os
import sys
import numpy as np
from pathlib import Path
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from zlib import crc32
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
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.utils.tools import fourier_basis_matrix

def get_radius_at_angle_pca_custom(theta, L, W, pca_coeffs, pca_mean, pca_eigenvectors):
    pca_coeffs = np.asarray(pca_coeffs).flatten()
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    normalized_angle = np.arctan2(sin_t / W, cos_t / L)
    
    if pca_eigenvectors.shape[1] > len(pca_coeffs):
        current_eigenvectors = pca_eigenvectors[:, :len(pca_coeffs)]
    else:
        current_eigenvectors = pca_eigenvectors

    fourier_coeffs_full = pca_mean + current_eigenvectors @ pca_coeffs
    N_reconstruct = len(fourier_coeffs_full)
    
    g = fourier_basis_matrix(normalized_angle, N_reconstruct) 
    
    if np.isscalar(theta):
        radius_norm = np.dot(g.flatten(), fourier_coeffs_full)
        v_x = L * np.cos(normalized_angle) * radius_norm
        v_y = W * np.sin(normalized_angle) * radius_norm
        return np.sqrt(v_x**2 + v_y**2)
    else:
        radius_norm = g.T @ fourier_coeffs_full
        v_x = L * np.cos(normalized_angle) * radius_norm
        v_y = W * np.sin(normalized_angle) * radius_norm
        return np.sqrt(v_x**2 + v_y**2)

def calculate_iou_radial_custom(r_true_func, r_est_func, num_samples=360):
    theta_vals = np.linspace(-np.pi, np.pi, num_samples, endpoint=False)
    r_true = r_true_func(theta_vals)
    r_est = r_est_func(theta_vals)
    
    r_min = np.minimum(r_true, r_est)
    r_max = np.maximum(r_true, r_est)
    
    intersection_area = 0.5 * np.sum(r_min**2) * (2 * np.pi / num_samples)
    union_area = 0.5 * np.sum(r_max**2) * (2 * np.pi / num_samples)
    
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def get_radius_function_custom(state, pca_mean, pca_eigenvectors):
    return lambda theta: get_radius_at_angle_pca_custom(
        theta, state.length, state.width, state.pca_coeffs, pca_mean, pca_eigenvectors
    )

def main():
    traj_type = "waypoints_star"
    boat_id = "Havfruen"
    N_pca = 4
    
    # Matching the order in your LaTeX document
    methods = ["ekf", "iekf", "implicit_ekf", "implicit_iekf", "iplf"]
    noises = [0.0, 0.15] # 0 noise vs real noise
    
    # Load Real PCA Means exactly as main.py references it
    pca_data_path = PROJECT_ROOT / "data" / "input_parameters" / "ShipDatasetPCAParameters.npz"
    pca_data = np.load(pca_data_path)
    PCA_MEAN = pca_data['mean'].flatten()
    PCA_EIGENVECTORS = pca_data['eigenvectors']
    
    results_table = {}
    
    for method in methods:
        results_table[method] = {}
        for noise in noises:
            print(f"Running simulation -> Method: {method.upper()}, Noise: {noise}")
            sim_base, lidar_base, extent_base = get_common_configs(traj_type=traj_type, N_pca=N_pca, selected_boat_id=boat_id)
            tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, sim_base.initial_state_gt, N_pca)
            
            tracker_cfg.process_model = 'cv'
            tracker_cfg.method = method
            if method == "implicit_ekf":
                tracker_cfg.max_iterations = 1
                tracker_cfg.method = "implicit_iekf"
            
            config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)
            
            # Apply Noise Condition
            config.lidar.lidar_gt_std_dev = noise
            config.lidar.lidar_std_dev = noise if noise > 0 else 0.05  # Slight offset to prevent infinite Kalman gains
            
            # Match recent main configuration constraints
            config.sim.num_frames = 800
            config.lidar.lidar_gt_std_dev = 0.0
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

            config.sim.use_cache = False
            
            # Run Simulator
            sim_result = run_single_simulation(config=config)
            
            # Evaluate Position RMSE and IoU
            gt_ts = sim_result.ground_truth_ts
            est_ts = sim_result.tracker_results_ts
            
            pos_errors = []
            ious = []
            
            for t, est_result in est_ts.items():
                if t not in gt_ts: continue
                
                gt_state = gt_ts.get_t(t)
                est_state = est_result.state_posterior.mean
                
                # Pos RMSE
                e_pos = np.linalg.norm([est_state.x - gt_state.x, est_state.y - gt_state.y])
                pos_errors.append(e_pos)
                
                # Mean IoU
                r_func_true = get_radius_function_custom(gt_state, PCA_MEAN, PCA_EIGENVECTORS)
                r_func_est = get_radius_function_custom(est_state, PCA_MEAN, PCA_EIGENVECTORS)
                iou = calculate_iou_radial_custom(r_func_true, r_func_est)
                ious.append(iou)
                
            mean_pos_rmse = np.sqrt(np.mean(np.array(pos_errors)**2))
            mean_iou = np.mean(ious) * 100
            
            # Evaluate Consistency (NEES / NIS) with project utilities
            analysis = create_consistency_analysis_from_sim_result(sim_result)
            nees_data = analysis.get_nees()
            nis_data = analysis.get_nis()
            
            mean_nees = np.mean(list(nees_data.data.values()))
            mean_nis = np.mean(list(nis_data.data.values()))
            
            results_table[method][noise] = {
                'rmse': mean_pos_rmse,
                'iou': mean_iou,
                'nees': mean_nees,
                'nis': mean_nis
            }

    # Print LaTeX Format Output
    print("\n\n" + "="*50)
    print("OUTPUT READY FOR LATEX TABLE:")
    print("="*50 + "\n")
    
    for method in methods:
        z = results_table[method][0.0]
        n = results_table[method][0.15]
        
        row_name = method.replace('_', '\\_').upper()
        if method == "implicit_ekf": row_name = "Imp-EKF"
        if method == "implicit_iekf": row_name = "Imp-IEKF"
            
        row = f"{row_name:12} & {z['rmse']:.3f} m & {z['iou']:.1f}\\% & {z['nees']:.2f} & {z['nis']:.2f} & {n['rmse']:.3f} m & {n['iou']:.1f}\\% & {n['nees']:.2f} & {n['nis']:.2f} \\\\"
        print(row)

if __name__ == '__main__':
    main()
