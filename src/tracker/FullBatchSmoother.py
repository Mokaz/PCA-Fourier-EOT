import numpy as np
import scipy.linalg
import copy
from scipy.optimize import least_squares
from scipy.stats import chi2
from scipy.sparse import lil_matrix

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import rot2D, ssa, initialize_centroid, cart2pol
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV, DynamicModel
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config
from src.senfuslib.timesequence import TimeSequence

class FullBatchSmoother(Tracker):
    """
    Full Batch Smoother (Offline SLAM) based on the Implicit Analytical Measurement Model.
    Collects all frames via a real-time IEKF forward pass, then runs a massive Sparse 
    Least Squares optimization (trf) over the entire trajectory to find a single, 
    globally optimal shape and smoothed kinematics.
    """
    sensor_model: LidarMeasurementModel
    dynamic_model: DynamicModel

    def __init__(self, 
                 dynamic_model: Model_PCA_CV, 
                 lidar_model: LidarMeasurementModel,
                 config: Config,
                 max_iterations: int | None = None, 
                 convergence_threshold : float | None = None):
        super().__init__(dynamic_model=dynamic_model, sensor_model=lidar_model, config=config)
        
        # Standard Tracker Settings
        self.max_iterations = max_iterations if max_iterations is not None else config.tracker.max_iterations
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else config.tracker.convergence_threshold
        self.use_initialize_centroid = config.tracker.use_initialize_centroid
        self.use_scaled_R = getattr(config.tracker, 'use_scaled_R', False)
        
        # Constraints
        self.use_state_clamping = config.tracker.use_state_clamping
        self.use_mahalanobis_projection = config.tracker.use_mahalanobis_projection
        if self.use_mahalanobis_projection:
            prob = getattr(config.tracker, 'mahalanobis_projection_prob', 0.99)
            self.chi2_thresh = chi2.ppf(prob, df=config.tracker.N_pca)
            
        self.use_negative_info_angular = getattr(config.tracker, 'use_negative_info_angular', False)
        self.use_negative_info_front = getattr(config.tracker, 'use_negative_info_front', False)
        self.use_negative_info_centroid = getattr(config.tracker, 'use_negative_info_centroid', False)
        self.radial_margin = getattr(config.tracker, 'radial_margin', 0.1)
        self.use_exact_extreme_angle = getattr(config.tracker, 'use_exact_extreme_angle', False)

        # Soft Priors
        self.use_absolute_L_W_prior = getattr(config.tracker, 'use_absolute_L_W_prior', False)
        self.prior_target_L = getattr(config.tracker, 'prior_target_L', 20.0)
        self.prior_target_W = getattr(config.tracker, 'prior_target_W', 6.0)
        self.prior_size_std = getattr(config.tracker, 'prior_size_std', 5.0)
        self.use_L_W_aspect_ratio_prior = getattr(config.tracker, 'use_L_W_aspect_ratio_prior', True)
        self.prior_aspect_ratio = getattr(config.tracker, 'prior_aspect_ratio', 3.8)
        self.prior_ratio_std = getattr(config.tracker, 'prior_ratio_std', 5.0)

        # Full Batch Buffer
        self.history_buffer =[]

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        # =================================================================
        # 1. REAL-TIME IEKF PASS (To get a good initial guess for the Batch)
        # =================================================================
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))
        num_meas = measurements_local.x.shape[0]

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_local + lidar_pos
        z_flat = measurements_global_coords.flatten('F')
        
        state_iter_mean = state_prior_mean.copy()
        
        if self.use_initialize_centroid:
            state_iter_mean.pos = initialize_centroid(
                position=state_prior_mean.pos, lidar_pos=self.sensor_model.lidar_position,
                measurements=polar_measurements, L_est=state_prior_mean.length, W_est=state_prior_mean.width
            )

        prev_state_iter_mean = state_prior_mean.copy()
        iterates = [state_iter_mean.copy()]
        self.angular_margin = (2 * np.pi) / getattr(self.config.lidar, 'num_rays', 360)

        # Forward filter (Analytical IEKF)
        for i in range(self.max_iterations):
            H_imp, D_imp, theta_implicit = self.sensor_model.get_implicit_matrices(state_iter_mean, measurements_global_coords)
            z_pred_iter = self.sensor_model.h_from_theta(state_iter_mean, theta_implicit)
            innovation_iter = z_flat - z_pred_iter

            R_std = self.sensor_model.R_scaled(num_meas) if self.use_scaled_R else self.sensor_model.R(num_meas) 
            R_eff = D_imp @ R_std @ D_imp.T

            H_fused = H_imp
            innovation_fused = innovation_iter
            R_fused = R_eff

            any_neg = self.use_negative_info_angular or self.use_negative_info_front or self.use_negative_info_centroid
            if any_neg and num_meas > 2:
                vcs = self._get_virtual_constraints(measurements_local, state_iter_mean)
                for vc in vcs:
                    c_type = vc['type']
                    if c_type in['min_angle', 'max_angle']:
                        H_virt, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(state_iter_mean, vc['body_angle'], is_radial=False)
                        ang_residual = ssa(vc['measured_val'] - gamma_pred)
                        if getattr(self.config.tracker, 'use_arc_length_residual', True):
                            rho = np.maximum(np.sqrt((vc['predicted_point'][0] - lidar_pos[0])**2 + (vc['predicted_point'][1] - lidar_pos[1])**2), 1.0)
                            H_stack = rho * H_virt
                            residual = rho * ang_residual
                        else:
                            H_stack = H_virt
                            residual = ang_residual
                        neg_info_std = getattr(self.config.tracker, 'R_neg_info_std_angle', 0.01)
                    elif c_type == 'front_wall':
                        H_stack, rho_pred = self.sensor_model.get_virtual_measurement_jacobian(state_iter_mean, vc['body_angle'], is_radial=True)
                        residual = vc['measured_val'] - rho_pred
                        neg_info_std = getattr(self.config.tracker, 'R_neg_info_std_front', 0.1)
                    elif c_type == 'centroid_depth':
                        res_val = vc['measured_val'] - vc['rho_c']
                        H_stack = np.zeros((1, len(state_iter_mean)))
                        H_stack[0, 0] = (state_iter_mean[0] - lidar_pos[0]) / vc['rho_c']
                        H_stack[0, 1] = (state_iter_mean[1] - lidar_pos[1]) / vc['rho_c']
                        residual = res_val
                        neg_info_std = getattr(self.config.tracker, 'R_neg_info_std_centroid', 0.1)
                    
                    H_fused = np.vstack((H_fused, H_stack))
                    innovation_fused = np.append(innovation_fused, residual)
                    import scipy.linalg as spla
                    R_fused = spla.block_diag(R_fused, np.array([[neg_info_std**2]]))

            S = H_fused @ P_pred @ H_fused.T + R_fused
            K = np.linalg.solve(S, H_fused @ P_pred.T).T
            
            diff_state = state_prior_mean - state_iter_mean
            diff_state[2] = ssa(diff_state[2])
            state_next = state_prior_mean + K @ (innovation_fused - H_fused @ diff_state)
            state_next[2] = ssa(state_next[2])

            state_iter_mean = state_next
            iterates.append(state_iter_mean.copy())

            step_diff = state_iter_mean - prev_state_iter_mean
            step_diff[2] = ssa(step_diff[2])
            if np.linalg.norm(step_diff) < self.convergence_threshold: break
            prev_state_iter_mean = state_iter_mean.copy()

        # Update Covariance
        I = np.eye(len(state_iter_mean))
        state_post_cov = (I - K @ H_fused) @ P_pred @ (I - K @ H_fused).T + K @ R_fused @ K.T
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_post_cov)

        # Create the TrackerUpdateResult
        z_pred_gauss = MultiVarGauss(mean=z_pred_iter, cov=(H_imp @ P_pred @ H_imp.T + R_std))
        innovation_gauss = MultiVarGauss(mean=innovation_iter, cov=(H_imp @ P_pred @ H_imp.T + R_std))
        
        result = TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z_flat,
            predicted_measurement=z_pred_gauss,
            iterates=iterates,
            predicted_measurements_iterates=[z_pred_iter],
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=H_imp,
            R_covariance=R_std,
            K_gain=K
        )

        # Buffer data for the offline smoother pass
        self.history_buffer.append({
            'z_global_flat': z_flat,
            'meas_local': measurements_local,
            'prior_mean': state_prior_mean.copy(),
            'prior_cov': P_pred.copy(),
            'post_mean': self.state_estimate.mean.copy()
        })

        return result

    def smooth_trajectory(self, forward_results_ts: TimeSequence[TrackerUpdateResult]) -> TimeSequence[TrackerUpdateResult]:
        """
        Public method to be called after the real-time simulation is finished.
        Runs the batch optimization and returns a new TimeSequence containing the smoothed results.
        """
        print("\n" + "="*60)
        print(">>> TRIGGERING FULL BATCH SMOOTHER...")
        print(">>> Assembling Sparse Arrowhead Matrix...")
        
        optimized_states, success = self._run_batch_smoother()
        
        smoothed_results_ts = TimeSequence()
        
        if success:
            print(">>> FULL BATCH SMOOTHER SUCCESS! Generating Smoothed Trajectory.")
            # Iterate through the original forward results
            for k, (ts_time, res) in enumerate(forward_results_ts.items()):
                # Deepcopy so we don't accidentally mutate the forward pass results
                new_res = copy.deepcopy(res)
                
                # Overwrite state with the globally smoothed state
                new_res.state_posterior.mean[:] = optimized_states[k]
                
                # Scale down the covariance drastically to reflect smoothing certainty
                # (A true computation of the Fisher Information Matrix inverse is too slow)
                new_res.state_posterior.cov *= 0.1 
                
                smoothed_results_ts.insert(ts_time, new_res)
        else:
            print(">>> BATCH SMOOTHER FAILED. Returning identical forward trajectory.")
            smoothed_results_ts = forward_results_ts
            
        print("="*60 + "\n")
        return smoothed_results_ts

    def _run_batch_smoother(self):
        """
        Runs a massive Sparse Least Squares optimization over all N frames.
        Builds the Sparse Arrowhead Matrix to drop complexity from O(V^3) to O(V).
        """
        N = len(self.history_buffer)
        state_dim = len(self.history_buffer[0]['post_mean'])
        N_pca = self.config.tracker.N_pca
        
        # 1. INITIAL GUESS VECTOR X =[kin_0, ..., kin_N-1, global_shape]
        X_init =[]
        for k in range(N):
            X_init.extend(self.history_buffer[k]['post_mean'][:6])
        X_init.extend(self.history_buffer[-1]['post_mean'][6:]) # Use shape from last frame
        X_init = np.array(X_init, dtype=float)

        # 2. FREEZE VIRTUAL CONSTRAINTS
        for k in range(N):
            kin_guess = self.history_buffer[k]['post_mean'][:6]
            shape_guess = self.history_buffer[-1]['post_mean'][6:]
            state_k_guess = np.concatenate([kin_guess, shape_guess])
            self.history_buffer[k]['vcs_fixed'] = self._get_virtual_constraints(self.history_buffer[k]['meas_local'], state_k_guess)

        # 3. PRECOMPUTE CHOLESKY/COVARIANCE WEIGHTS 
        Q_kin = self.dynamic_model.Q_d(dt=self.T)[:6, :6]
        L_Q = scipy.linalg.cholesky(Q_kin + np.eye(6)*1e-8, lower=True)
        L_Q_inv = scipy.linalg.solve_triangular(L_Q, np.eye(6), lower=True)
        F_kin = self.dynamic_model.F_d(None, dt=self.T)[:6, :6]
        
        P_0 = self.history_buffer[0]['prior_cov']
        L_P0 = scipy.linalg.cholesky(P_0 + np.eye(state_dim)*1e-6, lower=True)
        L_P0_inv = scipy.linalg.solve_triangular(L_P0, np.eye(state_dim), lower=True)
        state_0_prior = self.history_buffer[0]['prior_mean']
        
        sigma_lidar = self.config.tracker.lidar_std_dev
        
        # 4. CALCULATE EXACT RESIDUAL SIZE AND SPARSITY PATTERN
        total_vars = len(X_init)
        total_res = state_dim             
        total_res += 6 * (N - 1)          
        total_res += sum(len(f['z_global_flat']) for f in self.history_buffer) 
        total_res += sum(len(f['vcs_fixed']) for f in self.history_buffer)     
        if self.use_mahalanobis_projection: total_res += 1
        if self.use_absolute_L_W_prior: total_res += 2
        if self.use_L_W_aspect_ratio_prior: total_res += 1

        print(f"   * Total Variables: {total_vars}")
        print(f"   * Total Residuals: {total_res}")
        
        sparse_jac = lil_matrix((total_res, total_vars), dtype=int)
        
        row = 0
        sparse_jac[row:row+state_dim, :6] = 1 
        sparse_jac[row:row+state_dim, 6*N:] = 1 
        row += state_dim
        
        for k in range(N-1):
            sparse_jac[row:row+6, 6*k:6*k+6] = 1       
            sparse_jac[row:row+6, 6*(k+1):6*(k+1)+6] = 1 
            row += 6
            
        for k in range(N):
            num_z = len(self.history_buffer[k]['z_global_flat'])
            sparse_jac[row:row+num_z, 6*k:6*k+6] = 1
            sparse_jac[row:row+num_z, 6*N:] = 1
            row += num_z
            
            num_vc = len(self.history_buffer[k]['vcs_fixed'])
            if num_vc > 0:
                sparse_jac[row:row+num_vc, 6*k:6*k+6] = 1
                sparse_jac[row:row+num_vc, 6*N:] = 1
                row += num_vc

        if self.use_mahalanobis_projection:
            sparse_jac[row, 6*N:] = 1
            row += 1
        if self.use_absolute_L_W_prior:
            sparse_jac[row:row+2, 6*N:] = 1
            row += 2
        if self.use_L_W_aspect_ratio_prior:
            sparse_jac[row, 6*N:] = 1
            row += 1
            
        assert row == total_res, f"Sparsity map mismatch! Row={row}, Expected={total_res}"

        # 5. RESIDUAL FUNCTION
        def compute_residuals(X_opt):
            kinematics = X_opt[:6*N].reshape(N, 6)
            shape = X_opt[6*N:]
            
            res =[]
            
            # --- A. Prior Residual ---
            diff_0 = np.concatenate([kinematics[0], shape]) - state_0_prior
            diff_0[2] = ssa(diff_0[2])
            res.append(L_P0_inv @ diff_0)

            # --- B. Kinematics Residuals ---
            for k in range(N-1):
                kin_next = kinematics[k+1]
                kin_curr = kinematics[k]
                diff_kin = kin_next - (F_kin @ kin_curr)
                diff_kin[2] = ssa(diff_kin[2])
                res.append(L_Q_inv @ diff_kin)

            # --- C. Measurements & D. Negative Info Residuals ---
            for k in range(N):
                state_k = np.concatenate([kinematics[k], shape])
                frame = self.history_buffer[k]
                
                z_flat = frame['z_global_flat']
                theta_imp = self.sensor_model.get_implicit_angles(state_k, z_flat.reshape(2, -1, order='F'))
                z_pred = self.sensor_model.h_from_theta(state_k, theta_imp)
                
                diff_z = z_flat - z_pred
                sigma_k = sigma_lidar
                if self.use_scaled_R:
                    sigma_k *= np.sqrt(len(z_flat) / 2)
                res.append(diff_z / sigma_k)
                
                for vc in frame['vcs_fixed']:
                    c_type = vc['type']
                    if c_type in['min_angle', 'max_angle']:
                        _, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(state_k, vc['body_angle'], is_radial=False)
                        ang_res = ssa(vc['measured_val'] - gamma_pred)
                        if getattr(self.config.tracker, 'use_arc_length_residual', True):
                            u_x = vc['predicted_point'][0] - self.sensor_model.lidar_position[0]
                            u_y = vc['predicted_point'][1] - self.sensor_model.lidar_position[1]
                            rho = np.maximum(np.sqrt(u_x**2 + u_y**2), 1.0)
                            val = rho * ang_res
                        else:
                            val = ang_res
                        neg_info_std = getattr(self.config.tracker, 'R_neg_info_std_angle', 0.01)
                    elif c_type == 'front_wall':
                        _, rho_pred = self.sensor_model.get_virtual_measurement_jacobian(state_k, vc['body_angle'], is_radial=True)
                        val = vc['measured_val'] - rho_pred
                        neg_info_std = getattr(self.config.tracker, 'R_neg_info_std_front', 0.1)
                    elif c_type == 'centroid_depth':
                        val = vc['measured_val'] - vc['rho_c']
                        neg_info_std = getattr(self.config.tracker, 'R_neg_info_std_centroid', 0.1)
                        
                    res.append(np.array([val / neg_info_std]))

            # --- E. Mahalanobis Penalty ---
            if self.use_mahalanobis_projection:
                pca_coeffs = shape[-N_pca:]
                mahal_sq = np.sum((pca_coeffs ** 2) / self.config.tracker.pca_eigenvalues)
                if mahal_sq > self.chi2_thresh:
                    res.append(np.array([(mahal_sq - self.chi2_thresh) * 100.0]))
                else:
                    res.append(np.array([0.0]))
                    
            # --- F. Soft Priors ---
            if self.use_absolute_L_W_prior:
                res.append(np.array([
                    (shape[0] - self.prior_target_L) / self.prior_size_std,
                    (shape[1] - self.prior_target_W) / self.prior_size_std
                ]))
            if self.use_L_W_aspect_ratio_prior:
                ratio_err = shape[0] - self.prior_aspect_ratio * shape[1]
                res.append(np.array([ratio_err / self.prior_ratio_std]))
                
            return np.concatenate(res)

        # 6. BOUNDS
        lower_bounds = np.array([-np.inf] * (6 * N) +[1.0, 0.5] + [-np.inf] * N_pca)
        upper_bounds = np.array([np.inf] * len(X_init))

        # 7. RUN SPARSE LEAST SQUARES
        print("   * Starting trf Optimizer...")
        result = least_squares(
            fun=compute_residuals,
            x0=X_init,
            method='trf',
            jac_sparsity=sparse_jac, # <--- THE MAGIC BULLET 
            bounds=(lower_bounds, upper_bounds),
            max_nfev=50,
            verbose=0
        )

        # 8. UNPACK
        optimized_states =[]
        global_shape = result.x[6*N:]
        for k in range(N):
            kinematics = result.x[k*6 : (k+1)*6]
            optimized_states.append(np.concatenate([kinematics, global_shape]))
            
        return optimized_states, result.success

    def _get_virtual_constraints(self, measurements: LidarScan, state_pred: np.ndarray) -> list[dict]:
        meas_angles, meas_ranges = cart2pol(measurements.x, measurements.y)
        if len(meas_angles) < 2: return[]

        mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
        diff_angles = ssa(meas_angles - mean_angle)
        
        min_idx_meas = np.argmin(diff_angles)
        max_idx_meas = np.argmax(diff_angles)
        min_meas_dist = np.min(meas_ranges)

        parametric_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        z_pred_flat = self.sensor_model.h_from_theta(state_pred, parametric_angles)
        global_points = z_pred_flat.reshape(2, -1, order='F')
        
        sensor_points = global_points - self.sensor_model.lidar_position.reshape(2, 1)
        pred_angles, pred_ranges = cart2pol(sensor_points[0, :], sensor_points[1, :])
        
        pred_diff_angles = ssa(pred_angles - mean_angle)
        
        min_idx = np.argmin(pred_diff_angles)
        max_idx = np.argmax(pred_diff_angles)
        min_rad_idx = np.argmin(pred_ranges)

        virtual_constraints =[]
        
        if self.use_negative_info_angular:
            if ssa(meas_angles[min_idx_meas] - pred_angles[min_idx]) > self.angular_margin:
                theta_min = self._get_exact_extreme_angle(state_pred, parametric_angles[min_idx], mean_angle, is_max=False) if self.use_exact_extreme_angle else parametric_angles[min_idx]
                pt_global_min = self.sensor_model.h_from_theta(state_pred, np.array([theta_min])).flatten('F')
                virtual_constraints.append({
                    'measured_val': ssa(meas_angles[min_idx_meas] - self.angular_margin),
                    'body_angle': theta_min,
                    'predicted_point': pt_global_min,
                    'type': 'min_angle'
                })

            if ssa(pred_angles[max_idx] - meas_angles[max_idx_meas]) > self.angular_margin:
                theta_max = self._get_exact_extreme_angle(state_pred, parametric_angles[max_idx], mean_angle, is_max=True) if self.use_exact_extreme_angle else parametric_angles[max_idx]
                pt_global_max = self.sensor_model.h_from_theta(state_pred, np.array([theta_max])).flatten('F')
                virtual_constraints.append({
                    'measured_val': ssa(meas_angles[max_idx_meas] + self.angular_margin),
                    'body_angle': theta_max,
                    'predicted_point': pt_global_max,
                    'type': 'max_angle'
                })

        if self.use_negative_info_front:
            if pred_ranges[min_rad_idx] < (min_meas_dist - self.radial_margin):
                virtual_constraints.append({
                    'measured_val': min_meas_dist - self.radial_margin,
                    'body_angle': parametric_angles[min_rad_idx],
                    'type': 'front_wall'
                })

        if self.use_negative_info_centroid:
            u_c_x = state_pred[0] - self.sensor_model.lidar_position[0]
            u_c_y = state_pred[1] - self.sensor_model.lidar_position[1]
            rho_c = np.sqrt(u_c_x**2 + u_c_y**2)
            
            if rho_c < (min_meas_dist + state_pred[7] / 2.0):
                virtual_constraints.append({
                    'measured_val': min_meas_dist + state_pred[7] / 2.0,
                    'rho_c': float(rho_c),
                    'type': 'centroid_depth'
                })

        return virtual_constraints

    def _get_exact_extreme_angle(self, state_pred: np.ndarray, guess_theta: float, mean_angle: float, is_max: bool) -> float:
        from scipy.optimize import minimize_scalar
        def objective(theta):
            pt_global = self.sensor_model.h_from_theta(state_pred, np.array([theta])).flatten('F')
            gamma = np.arctan2(pt_global[1] - self.sensor_model.lidar_position[1], pt_global[0] - self.sensor_model.lidar_position[0])
            unwrapped = ssa(gamma - mean_angle)
            return -unwrapped if is_max else unwrapped

        delta = np.deg2rad(2.0)
        return minimize_scalar(objective, bounds=(guess_theta - delta, guess_theta + delta), method='bounded').x