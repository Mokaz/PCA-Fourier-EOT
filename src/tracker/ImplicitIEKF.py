import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize_scalar
from scipy.linalg import block_diag

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import rot2D, ssa, initialize_centroid, cart2pol
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV, DynamicModel
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class ImplicitIEKF(Tracker):
    """
    Implicit Iterated Extended Kalman Filter (I-IEKF).
    Uses the Implicit Measurement Model constraint g(x, z) = 0.
    Fuses explicit Negative Information bounds and a Soft Extent Prior.
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
        
        self.max_iterations = max_iterations if max_iterations is not None else config.tracker.max_iterations
        self.convergence_threshold = convergence_threshold if convergence_threshold is not None else config.tracker.convergence_threshold
        self.use_initialize_centroid = config.tracker.use_initialize_centroid

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
        self.use_D_imp_for_R = getattr(config.tracker, 'use_D_imp_for_R', True)
        self.use_scaled_R = getattr(config.tracker, 'use_scaled_R', False)
        self.use_absolute_L_W_prior = getattr(config.tracker, 'use_absolute_L_W_prior', False)
        self.prior_target_L = getattr(config.tracker, 'prior_target_L', 20.0)
        self.prior_target_W = getattr(config.tracker, 'prior_target_W', 6.0)
        self.prior_size_std = getattr(config.tracker, 'prior_size_std', 5.0)
        self.use_L_W_aspect_ratio_prior = getattr(config.tracker, 'use_L_W_aspect_ratio_prior', True)
        self.prior_aspect_ratio = getattr(config.tracker, 'prior_aspect_ratio', 3.8)
        self.prior_ratio_std = getattr(config.tracker, 'prior_ratio_std', 5.0)

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        measurements_global_coords = measurements_local + lidar_pos
        z_flat = measurements_global_coords.flatten('F')
        
        state_iter_mean = state_prior_mean.copy()
        
        if self.use_initialize_centroid:
            state_iter_mean.pos = initialize_centroid(
                position=state_prior_mean.pos,
                lidar_pos=self.sensor_model.lidar_position,
                measurements=polar_measurements,
                L_est=state_prior_mean.length,
                W_est=state_prior_mean.width
            )

        prev_state_iter_mean = state_prior_mean.copy()
        iterates = [state_iter_mean.copy()]
        predicted_measurements_iterates =[]

        final_clamped_length = None
        final_clamped_width = None
        final_mahalanobis_projection = None
        final_negative_info_count = 0
        virtual_constraints_iterates =[]

        num_rays = getattr(self.config.lidar, 'num_rays', 360)
        self.angular_margin = (2 * np.pi) / num_rays

        for i in range(self.max_iterations):
            # 1. Implicit Measurement Calculations (Positive Info)
            H_imp, D_imp, theta_implicit = self.sensor_model.get_implicit_matrices(state_iter_mean, measurements_global_coords)
                        
            z_pred_iter = self.sensor_model.h_from_theta(state_iter_mean, theta_implicit)
            predicted_measurements_iterates.append(z_pred_iter.copy())
            
            innovation_iter = z_flat - z_pred_iter

            num_meas = measurements_global_coords.shape[1]
            if self.use_scaled_R:
                R_std = self.sensor_model.R_scaled(num_meas)
            else:
                R_std = self.sensor_model.R(num_meas) 

            if self.use_D_imp_for_R:
                R_eff = D_imp @ R_std @ D_imp.T
            else:
                R_eff = R_std

            H_fused = H_imp
            innovation_fused = innovation_iter
            R_fused = R_eff
            current_virtual_constraints =[]

            # 2. Explicit Constraints Fusion (Negative Info)
            any_negative_info = self.use_negative_info_angular or self.use_negative_info_front or self.use_negative_info_centroid
            if any_negative_info and num_meas > 2:
                # No latching parameters! We re-evaluate cleanly every iteration.
                current_virtual_constraints = self._get_virtual_constraints(measurements_local, state_iter_mean)
                final_negative_info_count = len(current_virtual_constraints)

                for vc in current_virtual_constraints:
                    c_type = vc['type']
                    
                    if c_type in ['min_angle', 'max_angle']:
                        H_virt, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(
                            state_iter_mean, vc['body_angle'], is_radial=False
                        )
                        ang_residual = ssa(vc['measured_val'] - gamma_pred)
                        
                        u_x = vc['predicted_point'][0] - self.sensor_model.lidar_position[0]
                        u_y = vc['predicted_point'][1] - self.sensor_model.lidar_position[1]
                        rho = np.maximum(np.sqrt(u_x**2 + u_y**2), 1.0)
                        
                        # Convert to cartesian arc length to balance power against LiDAR variance
                        residual = rho * ang_residual
                        H_stack = rho * H_virt
                        
                        vc['ang_residual'] = float(ang_residual)
                        vc['arc_residual'] = float(residual)
                        vc['rho'] = float(rho)
                        
                    elif c_type == 'front_wall':
                        H_virt, rho_pred = self.sensor_model.get_virtual_measurement_jacobian(
                            state_iter_mean, vc['body_angle'], is_radial=True
                        )
                        residual = vc['measured_val'] - rho_pred
                        H_stack = H_virt
                        
                        vc['residual'] = float(residual)
                        vc['rho'] = float(rho_pred)

                    elif c_type == 'centroid_depth':
                        rho_c = vc['rho_c']
                        residual = vc['measured_val'] - rho_c
                        
                        H_stack = np.zeros((1, len(state_iter_mean)))
                        H_stack[0, 0] = (state_iter_mean[0] - self.sensor_model.lidar_position[0]) / rho_c
                        H_stack[0, 1] = (state_iter_mean[1] - self.sensor_model.lidar_position[1]) / rho_c
                        
                        vc['residual'] = float(residual)
                        vc['rho'] = float(rho_c)
                    
                    H_fused = np.vstack((H_fused, H_stack))
                    innovation_fused = np.append(innovation_fused, residual)
                    
                    neg_info_std = getattr(self.config.tracker, 'R_arc_std', 0.01)
                    R_arc = np.array([[neg_info_std ** 2]])
                    R_fused = block_diag(R_fused, R_arc)
                    
            virtual_constraints_iterates.append(current_virtual_constraints)

            # =================================================================
            # 3. SOFT L and W EXTENT PRIOR
            # =================================================================
            H_prior_list = []
            innov_prior_list =[]
            R_prior_list =[]

            # Absolute Size Prior
            if self.use_absolute_L_W_prior:
                L_target = self.prior_target_L
                W_target = self.prior_target_W
                std_size = self.prior_size_std

                H_size = np.zeros((2, len(state_iter_mean)))
                H_size[0, 6] = 1.0  # d/dL
                H_size[1, 7] = 1.0  # d/dW

                # Innovation against the prior state
                innov_size = np.array([
                    L_target - state_prior_mean[6],
                    W_target - state_prior_mean[7]
                ])
                R_size = np.diag([std_size**2, std_size**2])

                H_prior_list.append(H_size)
                innov_prior_list.append(innov_size)
                R_prior_list.append(R_size)

            # Aspect Ratio Prior
            if self.use_L_W_aspect_ratio_prior:
                target_ratio = self.prior_aspect_ratio
                std_ratio = self.prior_ratio_std 

                # 1*L - target_ratio*W = 0
                H_ratio = np.zeros((1, len(state_iter_mean)))
                H_ratio[0, 6] = 1.0
                H_ratio[0, 7] = -target_ratio

                innov_ratio = np.array([
                    0.0 - (state_prior_mean[6] - target_ratio * state_prior_mean[7])
                ])
                R_ratio = np.array([[std_ratio**2]])

                H_prior_list.append(H_ratio)
                innov_prior_list.append(innov_ratio)
                R_prior_list.append(R_ratio)

            # C. Fuse active priors into the Kalman matrices
            if H_prior_list:
                H_prior_stacked = np.vstack(H_prior_list)
                innov_prior_stacked = np.concatenate(innov_prior_list)
                R_prior_stacked = block_diag(*R_prior_list)

                H_fused = np.vstack((H_fused, H_prior_stacked))
                innovation_fused = np.append(innovation_fused, innov_prior_stacked)
                R_fused = block_diag(R_fused, R_prior_stacked)

            # 4. IEKF State Update Equation
            S = H_fused @ P_pred @ H_fused.T + R_fused
            
            K_transpose = np.linalg.solve(S, H_fused @ P_pred.T)
            K = K_transpose.T
            
            diff_state = state_prior_mean - state_iter_mean
            diff_state[2] = ssa(diff_state[2])
            
            state_next = state_prior_mean + K @ (innovation_fused - H_fused @ diff_state)
            state_next[2] = ssa(state_next[2])

            # =================================================================
            # ENFORCE OPTIONAL STABILITY CONSTRAINTS
            if self.use_state_clamping:
                orig_length = state_next.length
                orig_width = state_next.width
                if orig_length < 1.0:
                    state_next.length = 1.0
                    final_clamped_length = (float(orig_length), 1.0)
                if orig_width < 0.5:
                    state_next.width = 0.5
                    final_clamped_width = (float(orig_width), 0.5)

            if self.use_mahalanobis_projection:
                eigenvalues = self.config.tracker.pca_eigenvalues
                e = state_next.pca_coeffs
                mahalanobis_sq = np.sum((e ** 2) / eigenvalues)

                if mahalanobis_sq > self.chi2_thresh:
                    orig_coeffs = e.copy()
                    scale_factor = np.sqrt(self.chi2_thresh / mahalanobis_sq)
                    state_next.pca_coeffs = e * scale_factor
                    final_mahalanobis_projection = (orig_coeffs, state_next.pca_coeffs.copy(), mahalanobis_sq)
            # =================================================================
            
            state_iter_mean = state_next
            iterates.append(state_iter_mean.copy())

            # Check convergence
            step_diff = state_iter_mean - prev_state_iter_mean
            step_diff[2] = ssa(step_diff[2])
            if np.linalg.norm(step_diff) < self.convergence_threshold:
                break
            
            prev_state_iter_mean = state_iter_mean.copy()

        # Final Update
        I = np.eye(len(state_iter_mean))
        state_post_cov = (I - K @ H_fused) @ P_pred @ (I - K @ H_fused).T + K @ R_fused @ K.T
        
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_post_cov)
        
        z_pred_gauss = MultiVarGauss(mean=z_pred_iter, cov=(H_imp @ P_pred @ H_imp.T + R_eff))
        innovation_gauss = MultiVarGauss(mean=innovation_iter, cov=(H_imp @ P_pred @ H_imp.T + R_eff))

        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z_flat,
            predicted_measurement=z_pred_gauss,
            iterates=iterates,
            predicted_measurements_iterates=predicted_measurements_iterates,
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=H_imp,
            R_covariance=R_eff,
            K_gain=K,
            clamped_length=final_clamped_length,
            clamped_width=final_clamped_width,
            mahalanobis_projection=final_mahalanobis_projection,
            negative_info_used=final_negative_info_count,
            virtual_constraints_info=virtual_constraints_iterates
        )

    def _get_virtual_constraints(self, measurements: LidarScan, state_pred: np.ndarray) -> list[dict]:
        """
        Determines Negative Information bounds (Left/Right angles, Front boundary, Centroid anchor).
        Evaluated purely per-iteration as True Inequality Constraints.
        """
        meas_angles, meas_ranges = cart2pol(measurements.x, measurements.y)
        if len(meas_angles) < 2:
            return[]

        # Find the measured bounds
        mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
        diff_angles = ssa(meas_angles - mean_angle)
        
        min_idx_meas = np.argmin(diff_angles)
        max_idx_meas = np.argmax(diff_angles)
        
        min_meas_angle = meas_angles[min_idx_meas]
        max_meas_angle = meas_angles[max_idx_meas]
        min_meas_dist = np.min(meas_ranges)

        # Sample the predicted shape using PARAMETRIC angles
        parametric_angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        z_pred_flat = self.sensor_model.h_from_theta(state_pred, parametric_angles)
        global_points = z_pred_flat.reshape(2, -1, order='F')
        
        sensor_points = global_points - self.sensor_model.lidar_position.reshape(2, 1)
        pred_angles, pred_ranges = cart2pol(sensor_points[0, :], sensor_points[1, :])
        
        pred_diff_angles = ssa(pred_angles - mean_angle)
        
        min_idx = np.argmin(pred_diff_angles)
        max_idx = np.argmax(pred_diff_angles)
        min_rad_idx = np.argmin(pred_ranges)
        
        min_pred_angle = pred_angles[min_idx]
        max_pred_angle = pred_angles[max_idx]

        virtual_constraints =[]
        
        # 1. Minimum Angle (Left Wall)
        if self.use_negative_info_angular:
            if ssa(min_meas_angle - min_pred_angle) > self.angular_margin:
                theta_min = parametric_angles[min_idx]
                if self.use_exact_extreme_angle:
                    theta_min = self._get_exact_extreme_angle(state_pred, theta_min, mean_angle, is_max=False)
                
                pt_global_min = self.sensor_model.h_from_theta(state_pred, np.array([theta_min])).flatten('F')

                virtual_constraints.append({
                    'measured_val': ssa(min_meas_angle - self.angular_margin),
                    'body_angle': theta_min,
                    'predicted_point': pt_global_min,
                    'type': 'min_angle'
                })

        # 2. Maximum Angle (Right Wall)
        if self.use_negative_info_angular:
            if ssa(max_pred_angle - max_meas_angle) > self.angular_margin:
                theta_max = parametric_angles[max_idx]
                if self.use_exact_extreme_angle:
                    theta_max = self._get_exact_extreme_angle(state_pred, theta_max, mean_angle, is_max=True)
                
                pt_global_max = self.sensor_model.h_from_theta(state_pred, np.array([theta_max])).flatten('F')

                virtual_constraints.append({
                    'measured_val': ssa(max_meas_angle + self.angular_margin),
                    'body_angle': theta_max,
                    'predicted_point': pt_global_max,
                    'type': 'max_angle'
                })

        # 3. Front Wall (Radial Boundary)
        if self.use_negative_info_front:
            if pred_ranges[min_rad_idx] < (min_meas_dist - self.radial_margin):
                virtual_constraints.append({
                    'measured_val': min_meas_dist - self.radial_margin,
                    'body_angle': parametric_angles[min_rad_idx],
                    'type': 'front_wall'
                })

        # 4. Centroid Depth (Anchor)
        if self.use_negative_info_centroid:
            u_c_x = state_pred[0] - self.sensor_model.lidar_position[0]
            u_c_y = state_pred[1] - self.sensor_model.lidar_position[1]
            rho_c = np.sqrt(u_c_x**2 + u_c_y**2)
            
            centroid_buffer = state_pred[7] / 2.0  # Width / 2 buffer
            if rho_c < (min_meas_dist + centroid_buffer):
                virtual_constraints.append({
                    'measured_val': min_meas_dist + centroid_buffer,
                    'rho_c': float(rho_c),
                    'type': 'centroid_depth'
                })

        return virtual_constraints

    def _get_exact_extreme_angle(self, state_pred: np.ndarray, guess_theta: float, mean_angle: float, is_max: bool) -> float:
        """
        Uses a continuous 1D optimizer to find the exact tangent angle theta_body*,
        preventing discretization error and satisfying Danskin's theorem.
        """
        def objective(theta):
            pt_global = self.sensor_model.h_from_theta(state_pred, np.array([theta])).flatten('F')
            u_x = pt_global[0] - self.sensor_model.lidar_position[0]
            u_y = pt_global[1] - self.sensor_model.lidar_position[1]
            
            gamma = np.arctan2(u_y, u_x)
            
            # Unwrap relative to the mean measured angle to prevent optimizer Pi wraparound
            unwrapped_gamma = ssa(gamma - mean_angle)
            
            return -unwrapped_gamma if is_max else unwrapped_gamma

        delta = np.deg2rad(2.0)
        bounds = (guess_theta - delta, guess_theta + delta)
        
        res = minimize_scalar(objective, bounds=bounds, method='bounded')
        return res.x