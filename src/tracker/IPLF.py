import numpy as np
import scipy.linalg
from scipy.stats import chi2
from scipy.linalg import block_diag

from src.senfuslib import MultiVarGauss
from src.tracker.tracker import Tracker
from src.tracker.TrackerUpdateResult import TrackerUpdateResult
from src.utils.tools import ssa, initialize_centroid, cart2pol
from src.states.states import State_PCA, LidarScan
from src.dynamics.process_models import Model_PCA_CV, DynamicModel
from src.sensors.LidarModel import LidarMeasurementModel
from src.utils.config_classes import Config

class IPLF(Tracker):
    """
    Iterated Posterior Linearization Filter (IPLF).
    Uses Statistical Linear Regression (via Cubature Transform Sigma Points) 
    iteratively evaluated on the posterior distribution.
    Fuses explicit Negative Information analytically.
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
        self.force_kinematic_unobservability = getattr(config.tracker, 'force_kinematic_unobservability', False)
        self.radial_margin = getattr(config.tracker, 'radial_margin', 0.1)
        self.use_exact_extreme_angle = getattr(config.tracker, 'use_exact_extreme_angle', False)
        self.use_scaled_R = getattr(config.tracker, 'use_scaled_R', False)
        self.debug_prints = getattr(config.tracker, 'debug_prints', False)
        self.use_absolute_L_W_prior = getattr(config.tracker, 'use_absolute_L_W_prior', False)
        self.prior_target_L = getattr(config.tracker, 'prior_target_L', 20.0)
        self.prior_target_W = getattr(config.tracker, 'prior_target_W', 6.0)
        self.prior_size_std = getattr(config.tracker, 'prior_size_std', 5.0)
        self.use_L_W_aspect_ratio_prior = getattr(config.tracker, 'use_L_W_aspect_ratio_prior', False)
        self.prior_aspect_ratio = getattr(config.tracker, 'prior_aspect_ratio', 3.5)
        self.prior_ratio_std = getattr(config.tracker, 'prior_ratio_std', 1.0)

    def _get_sigma_points(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates Cubature Transform Sigma Points.
        Guarantees strictly positive weights to avoid negative covariance issues.
        """
        n = len(mean)
        
        # --- DIAGNOSTIC PRINT ---
        if self.debug_prints:
            print(f"\n--- CUBATURE DIAGNOSTICS ---")
            print(f"State Dimension (n) : {n}")
        # ------------------------
        
        # Enforce exact symmetry
        cov = (cov + cov.T) / 2
        
        # Safe Cholesky Decomposition
        try:
            L = scipy.linalg.cholesky(cov + np.eye(n)*1e-6, lower=True)
        except scipy.linalg.LinAlgError:
            # Fallback to Eigendecomposition if Cholesky fails completely
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, 0.0)
            L = evecs @ np.diag(np.sqrt(evals))
            
        sigmas = np.zeros((2*n, n))
        weights = np.full(2*n, 1.0 / (2 * n))
        
        for i in range(n):
            sigmas[i] = mean + np.sqrt(n) * L[:, i]
            sigmas[n + i] = mean - np.sqrt(n) * L[:, i]
            
        # --- DIAGNOSTIC PRINT ---
        if self.debug_prints:
            print(f"Weight W_i (All {2*n} points) : {weights[0]:.4f}  <-- STRICTLY POSITIVE")
        # ------------------------
            
        return sigmas, weights

    def predict(self):
        self.state_estimate = self.dynamic_model.pred_from_est(self.state_estimate, self.T)

    def update(self, measurements_local: LidarScan, ground_truth: State_PCA = None) -> TrackerUpdateResult:
        state_prior_mean = self.state_estimate.mean.copy()        
        P_pred = self.state_estimate.cov.copy()
        state_pred = MultiVarGauss(mean=state_prior_mean, cov=P_pred)

        polar_measurements = list(zip(measurements_local.angle, measurements_local.range))
        num_meas = measurements_local.x.shape[0]

        lidar_pos = self.sensor_model.lidar_position.reshape(2, 1)
        z_measurements_global_coords = measurements_local + lidar_pos
        z_flat = z_measurements_global_coords.flatten('F')
        
        # Initialize Iteration Variables
        state_iter_mean = state_prior_mean.copy()
        state_iter_cov = P_pred.copy() 
        
        if self.use_initialize_centroid:
            state_iter_mean.pos = initialize_centroid(
                position=state_prior_mean.pos,
                lidar_pos=self.sensor_model.lidar_position,
                measurements=polar_measurements,
                L_est=state_prior_mean.length,
                W_est=state_prior_mean.width
            )

        prev_state_iter_mean = state_prior_mean.copy()
        iterates =[state_iter_mean.copy()]
        predicted_measurements_iterates =[]

        final_clamped_length = None
        final_clamped_width = None
        final_mahalanobis_projection = None
        final_negative_info_count = 0
        virtual_constraints_iterates =[]

        num_rays = getattr(self.config.lidar, 'num_rays', 360)
        self.angular_margin = (2 * np.pi) / num_rays

        for i in range(self.max_iterations):
            # =================================================================
            # 1. STATISTICAL LINEAR REGRESSION (SLR) via SIGMA POINTS
            # =================================================================
            sigmas, weights = self._get_sigma_points(state_iter_mean, state_iter_cov)
            
            Z_sigmas =[]
            for X in sigmas:
                # 1a. Find implicit angles for this specific sigma point
                theta_imp = self.sensor_model.get_implicit_angles(X, z_measurements_global_coords)
                
                # 1b. Predict measurements
                Z = self.sensor_model.h_from_theta(X, theta_imp)
                Z_sigmas.append(Z)
                
            Z_sigmas = np.array(Z_sigmas)
            
            # 1c. Calculate Empirical Moments
            z_bar = np.sum(weights[:, None] * Z_sigmas, axis=0)
            
            X_diff = sigmas - state_iter_mean
            X_diff[:, 2] = ssa(X_diff[:, 2]) # Ensure yaw wraps cleanly
            
            Z_diff = Z_sigmas - z_bar
            
            # Cross-covariance (State & Measurement) and Auto-covariance
            Psi = (weights[:, None] * X_diff).T @ Z_diff
            Phi = (weights[:, None] * Z_diff).T @ Z_diff
            
            # 1d. Calculate Linearization Parameters (A, b, Omega)
            A_imp = np.linalg.solve(state_iter_cov + np.eye(len(state_iter_mean))*1e-6, Psi).T
            
            # --- STABILITY FIX: KINEMATIC UNOBSERVABILITY ---
            # Single-frame spatial measurements provide exactly zero information about 
            # instantaneous velocity or yaw_rate. Force to zero to prevent Cubature noise.
            if self.force_kinematic_unobservability:
                A_imp[:, 3:6] = 0.0
            
            # Omega is the Linearization Noise (Captures non-linearity uncertainty)
            Omega_imp = Phi - A_imp @ state_iter_cov @ A_imp.T
            Omega_imp = (Omega_imp + Omega_imp.T) / 2 

            # --- DIAGNOSTIC PRINT ---
            if self.debug_prints:
                evals_omega = np.linalg.eigvalsh(Omega_imp)
                min_eig_omega = np.min(evals_omega)
                print(f"Min Eigenval of Omega_imp : {min_eig_omega:.2e}")
                
                if min_eig_omega < -1e-10:
                    print(">>> WARNING: Omega_imp has significant negative eigenvalues (Failure)!")
                elif min_eig_omega < 0:
                    print(">>> NOTE: Omega_imp has negligible negative eigenvalues (Floating point noise).")
                else:
                    print(">>> SUCCESS: Omega_imp is strictly Positive Definite/Semi-Definite!")
            # ------------------------
            
            # Extract the diagonal, clamp to >= 0
            omega_diag = np.maximum(np.diag(Omega_imp), 0.0)
            Omega_imp = np.diag(omega_diag)
            
            # Tracking predicted measurements
            z_pred_iter = z_bar
            predicted_measurements_iterates.append(z_pred_iter.copy())
            
            # =================================================================
            # 2. CONSTRUCT FUSED KALMAN MATRICES
            # =================================================================
            if self.use_scaled_R:
                R_std = self.sensor_model.R_scaled(num_meas)
            else:
                R_std = self.sensor_model.R(num_meas) 

            # IPLF natively merges measurement noise and linearization noise
            R_eff = R_std + Omega_imp

            H_fused = A_imp
            
            # --- STABILITY FIX: ANGLE WRAPPING IN TAYLOR EXPANSION ---
            # Crucial: Calculate properly wrapped difference between prior and iteration point
            state_diff = state_prior_mean - state_iter_mean
            state_diff[2] = ssa(state_diff[2])
            
            # Evaluate linear approximation at the prior state safely
            z_pred_prior = z_bar + A_imp @ state_diff
            innovation_fused = z_flat - z_pred_prior

            R_fused = R_eff
            current_virtual_constraints =[]

            # =================================================================
            # 3. EXPLICIT CONSTRAINTS FUSION (Negative Info - Analytical)
            # =================================================================
            any_negative_info = self.use_negative_info_angular or self.use_negative_info_front or self.use_negative_info_centroid
            if any_negative_info and num_meas > 2:
                current_virtual_constraints = self._get_virtual_constraints(measurements_local, state_iter_mean)
                final_negative_info_count = len(current_virtual_constraints)

                for vc in current_virtual_constraints:
                    c_type = vc['type']
                    
                    if c_type in['min_angle', 'max_angle']:
                        H_virt, gamma_pred = self.sensor_model.get_virtual_measurement_jacobian(
                            state_iter_mean, vc['body_angle'], is_radial=False
                        )
                        ang_residual = ssa(vc['measured_val'] - gamma_pred)
                        
                        u_x = vc['predicted_point'][0] - self.sensor_model.lidar_position[0]
                        u_y = vc['predicted_point'][1] - self.sensor_model.lidar_position[1]
                        rho = np.maximum(np.sqrt(u_x**2 + u_y**2), 1.0)
                        
                        residual = rho * ang_residual
                        H_stack = rho * H_virt
                        
                    elif c_type == 'front_wall':
                        H_virt, rho_pred = self.sensor_model.get_virtual_measurement_jacobian(
                            state_iter_mean, vc['body_angle'], is_radial=True
                        )
                        residual = vc['measured_val'] - rho_pred
                        H_stack = H_virt

                    elif c_type == 'centroid_depth':
                        rho_c = vc['rho_c']
                        residual = vc['measured_val'] - rho_c
                        
                        H_stack = np.zeros((1, len(state_iter_mean)))
                        H_stack[0, 0] = (state_iter_mean[0] - self.sensor_model.lidar_position[0]) / rho_c
                        H_stack[0, 1] = (state_iter_mean[1] - self.sensor_model.lidar_position[1]) / rho_c
                    
                    H_fused = np.vstack((H_fused, H_stack))
                    
                    # Compute virtual constraint innovation using safely wrapped state_diff
                    innov_virt = residual - H_stack @ state_diff
                    innovation_fused = np.append(innovation_fused, innov_virt)
                    
                    neg_info_std = getattr(self.config.tracker, 'R_arc_std', 0.01)
                    R_arc = np.array([[neg_info_std ** 2]])
                    R_fused = block_diag(R_fused, R_arc)
                    
            virtual_constraints_iterates.append(current_virtual_constraints)

            # =================================================================
            # 4. SOFT L and W EXTENT PRIOR
            # =================================================================
            H_prior_list = []
            innov_prior_list =[]
            R_prior_list =[]

            # A. Absolute Size Prior
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

            # B. Aspect Ratio Prior
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

            # =================================================================
            # 5. KALMAN UPDATE (Solving for next Iteration Posterior)
            # =================================================================
            S = H_fused @ P_pred @ H_fused.T + R_fused
            
            # Safe solve for K to avoid full matrix inversion of large S
            K_transpose = np.linalg.solve(S, H_fused @ P_pred.T)
            K = K_transpose.T
            
            # The next state is calculated directly from the prior
            state_next = state_prior_mean + K @ innovation_fused
            state_next[2] = ssa(state_next[2])

            # Update the iteration covariance (Joseph Form for strict PD bounds)
            I = np.eye(len(state_iter_mean))
            state_iter_cov = (I - K @ H_fused) @ P_pred @ (I - K @ H_fused).T + K @ R_fused @ K.T
            state_iter_cov = (state_iter_cov + state_iter_cov.T) / 2 # Enforce strict symmetry

            # --- DIAGNOSTIC PRINT ---
            if self.debug_prints:
                evals_cov = np.linalg.eigvalsh(state_iter_cov)
                min_eig_cov = np.min(evals_cov)
                print(f"Min Eigenval of State Cov : {min_eig_cov:.2e}")
                
                if min_eig_cov < -1e-10:
                    print(">>> FATAL: state_iter_cov is broken. Cholesky will crash on the next loop!\n")
                elif min_eig_cov < 0:
                    print(">>> NOTE: state_iter_cov has negligible negative eigenvalues (Floating point noise).\n")
                else:
                    print(">>> SUCCESS: state_iter_cov remains Positive Definite!\n")
            # ------------------------

            # =================================================================
            # 6. ENFORCE STABILITY CONSTRAINTS
            # =================================================================
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
            
            state_iter_mean = state_next
            iterates.append(state_iter_mean.copy())

            # Convergence Check
            step_diff = state_iter_mean - prev_state_iter_mean
            step_diff[2] = ssa(step_diff[2])
            if np.linalg.norm(step_diff) < self.convergence_threshold:
                break
            
            prev_state_iter_mean = state_iter_mean.copy()

        # =================================================================
        # 7. FINAL RETURN
        # =================================================================
        self.state_estimate = MultiVarGauss(mean=state_iter_mean, cov=state_iter_cov)
        
        # Use R_fused instead of R_eff to match the augmented H_fused shape, then slice out the real measurements
        z_pred_gauss = MultiVarGauss(
            mean=z_pred_iter, 
            cov=(H_fused @ P_pred @ H_fused.T + R_fused)[:len(z_flat), :len(z_flat)]
        )
        innovation_gauss = MultiVarGauss(
            mean=innovation_fused[:len(z_flat)], 
            cov=(H_fused @ P_pred @ H_fused.T + R_fused)[:len(z_flat), :len(z_flat)]
        )
        
        return TrackerUpdateResult(
            state_prior=state_pred,
            state_posterior=self.state_estimate,
            measurements=z_flat,
            predicted_measurement=z_pred_gauss,
            iterates=iterates,
            predicted_measurements_iterates=predicted_measurements_iterates,
            innovation_gauss=innovation_gauss,
            iterations=i + 1,
            H_jacobian=A_imp,
            R_covariance=R_eff,
            K_gain=K,
            clamped_length=final_clamped_length,
            clamped_width=final_clamped_width,
            mahalanobis_projection=final_mahalanobis_projection,
            negative_info_used=final_negative_info_count,
            virtual_constraints_info=virtual_constraints_iterates
        )

    def _get_virtual_constraints(self, measurements: LidarScan, state_pred: np.ndarray) -> list[dict]:
        meas_angles, meas_ranges = cart2pol(measurements.x, measurements.y)
        if len(meas_angles) < 2: return[]

        mean_angle = np.arctan2(np.mean(np.sin(meas_angles)), np.mean(np.cos(meas_angles)))
        diff_angles = ssa(meas_angles - mean_angle)
        
        min_idx_meas = np.argmin(diff_angles)
        max_idx_meas = np.argmax(diff_angles)
        
        min_meas_angle = meas_angles[min_idx_meas]
        max_meas_angle = meas_angles[max_idx_meas]
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
        
        min_pred_angle = pred_angles[min_idx]
        max_pred_angle = pred_angles[max_idx]

        virtual_constraints =[]
        
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
            
            centroid_buffer = state_pred[7] / 2.0 
            if rho_c < (min_meas_dist + centroid_buffer):
                virtual_constraints.append({
                    'measured_val': min_meas_dist + centroid_buffer,
                    'rho_c': float(rho_c),
                    'type': 'centroid_depth'
                })

        return virtual_constraints

    def _get_exact_extreme_angle(self, state_pred: np.ndarray, guess_theta: float, mean_angle: float, is_max: bool) -> float:
        from scipy.optimize import minimize_scalar
        def objective(theta):
            pt_global = self.sensor_model.h_from_theta(state_pred, np.array([theta])).flatten('F')
            u_x = pt_global[0] - self.sensor_model.lidar_position[0]
            u_y = pt_global[1] - self.sensor_model.lidar_position[1]
            gamma = np.arctan2(u_y, u_x)
            unwrapped_gamma = ssa(gamma - mean_angle)
            return -unwrapped_gamma if is_max else unwrapped_gamma

        delta = np.deg2rad(2.0)
        bounds = (guess_theta - delta, guess_theta + delta)
        res = minimize_scalar(objective, bounds=bounds, method='bounded')
        return res.x