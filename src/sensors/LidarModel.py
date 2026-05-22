import numpy as np

from dataclasses import dataclass, field
from typing import Tuple, Sequence, List
from scipy.linalg import block_diag

from src.senfuslib import SensorModel
from src.states.states import State_PCA, LidarScan
from src.utils.geometry_utils import compute_exact_vessel_shape_global
from src.utils.tools import cast_rays, add_noise_to_distances, ur, rot2D, drot2D, fourier_basis_matrix, pol2cart, fourier_basis_derivative_matrix, unit_vector
from src.utils.config_classes import ExtentConfig

@dataclass
class LidarMeasurementModel(SensorModel[Sequence[LidarScan]]):
    """
    A measurement model for the tracker.
    Represents the mathematical model of the sensor (h(x), H(x), R).
    """
    lidar_position: np.ndarray
    lidar_std_dev: float
    pca_mean: np.ndarray
    pca_eigenvectors: np.ndarray
    extent_cfg: ExtentConfig
    
    def h_lidar(self, x, body_angles: list[float]):
        L = x[6]
        W = x[7]
        
        normalized_angles = np.arctan2(np.sin(body_angles) / W, np.cos(body_angles) / L)

        pos = x[:2].reshape(-1, 1)  # shape (2, 1)
        R_heading = rot2D(x[2])    # shape (2, 2)
        LW_scaling = np.diag([L, W])  # shape (2, 2)

        fourier_coeffs = self.pca_mean + self.pca_eigenvectors @ x[8:].reshape(-1, 1)  # shape (N_f, 1)

        # Vectorized direction vectors: shape (2, N) -- one column per angle
        ur_all = ur(normalized_angles)  # shape (2, N), ur returns [cos(θ); sin(θ)]

        # Vectorized g(theta): shape (N_f, N)
        g_all = fourier_basis_matrix(normalized_angles, N_fourier=self.extent_cfg.N_fourier)  # shape (N_f, N)

        # Compute predicted measurement points in global frame
        #   ur_all * (g_all.T @ fourier_coeffs): shape (2, N)
        #   (g_all.T @ fourier_coeffs): shape (N,)
        #   full expression: shape (2, N) → pos + ... → shape (2, N)
        z_pred = np.squeeze(pos + R_heading @ LW_scaling @ ur_all * (g_all.T @ fourier_coeffs).flatten())

        return z_pred.T  # shape (N, 2)

    def lidar_jacobian(self, x, body_angles: list[float]):
        
        """
        Compute the Jacobian of the measurement function h with respect to the state x.     
        
        Parameters:
        x (np.array): Current state (n-dimensional).
        - x[0:2]: Position (North, East)
        - x[2]: Heading
        - x[3:5]: Velocity (North, East)
        - x[5]: Rate of turn
        - x[6]: Length
        - x[7]: Width
        - x[8:]: Fourier coefficients

        body_angles (list[float]): Body angles for the measurements.

        Returns:
        np.array: Jacobian matrix (m x n).
        """

        L = x[6]
        W = x[7]

        if len(body_angles) == 0:
            return np.zeros((0, x.shape[0]))

        # Normalize the angles
        normalized_angles = np.arctan2(np.sin(body_angles) / W, np.cos(body_angles) / L)

        fourier_coeffs = self.pca_mean + self.pca_eigenvectors @ x[8:].reshape(-1, 1)

        jacobians = []

        for angle in normalized_angles:
            fourier_approx = (fourier_basis_matrix(angle, self.extent_cfg.N_fourier).T @ fourier_coeffs).item()

            dp_dpc = np.eye(2)
            dp_dphi = drot2D(x[2]) @ np.diag([L, W]) @ (ur(angle) * fourier_approx).reshape(-1, 1)
            dp_dv = np.zeros([2, 2])
            dp_dr = np.zeros([2, 1])
            dp_dLW = rot2D(x[2]) @ np.diag(ur(angle).flatten()) * fourier_approx
            dp_de = rot2D(x[2]) @ np.diag([L, W]) @ ur(angle).T * (fourier_basis_matrix(angle, self.extent_cfg.N_fourier).T @ self.pca_eigenvectors).flatten()

            H_i = np.hstack([dp_dpc, dp_dphi, dp_dv, dp_dr, dp_dLW, dp_de])
            jacobians.append(H_i)

        return np.vstack(jacobians)

    def R(self, num_measurements: int) -> np.ndarray:
        """
        Measurement noise covariance (R).
        Assumes independent noise for each point's x and y coordinate.
        """

        R_single_point = self.R_single_point()
        return np.kron(np.eye(num_measurements), R_single_point)
    
    def R_scaled(self, num_measurements: int) -> np.ndarray:
        R_single_point = self.R_single_point()
        R_scaled = R_single_point * num_measurements 
        return np.kron(np.eye(num_measurements), R_scaled)
    
    def R_single_point(self) -> np.ndarray:
        """
        Measurement noise covariance for a single point.
        May be used externally
        """
        return np.eye(2) * self.lidar_std_dev**2
    
    def h_from_theta(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calculates predicted measurements given the state and the 
        PARAMETRIC angles (theta). 
        
        This skips the L/W normalization step found in h_lidar, 
        as theta is assumed to be derived implicitly.
        """
        L = x[6]
        W = x[7]
        pca_coeffs = x[8:]
        
        pos = x[:2].reshape(-1, 1)
        R_heading = rot2D(x[2])
        LW_scaling = np.diag([L, W])

        # 1. Calculate Radius r(theta)
        fourier_coeffs = self.pca_mean + self.pca_eigenvectors @ pca_coeffs.reshape(-1, 1)
        
        # g(theta)
        g_mat = fourier_basis_matrix(theta, N_fourier=self.extent_cfg.N_fourier)
        r_vals = (g_mat.T @ fourier_coeffs).flatten() # Shape (N,)

        # 2. Project to Global Frame
        # u(theta) = [cos, sin]
        u_vec = unit_vector(theta) # Shape (2, N)
        
        # h = p + R * S * u * r
        # We multiply column-wise: (2, N) * (N,) -> (2, N)
        body_points = (LW_scaling @ u_vec) * r_vals
        z_pred = pos + R_heading @ body_points

        return z_pred.flatten('F') # Return as flat array[x1, y1, x2, y2...]
    
    def get_implicit_angles(self, x: np.ndarray, z_measurements_global: np.ndarray) -> np.ndarray:
        """
        Lightweight function to calculate ONLY the implicit parametric angles.
        Bypasses the heavy Jacobian/Matrix assembly, making it incredibly fast for Sigma Points.
        """
        L = x[6]
        W = x[7]
        pos = x[:2].reshape(2, 1)
        R_mat = rot2D(x[2])
        
        # Transform measurements to local body frame
        z_loc = R_mat.T @ (z_measurements_global - pos) # Shape (2, N)
        
        # Normalized coordinates
        x_tilde = z_loc[0, :] / L
        y_tilde = z_loc[1, :] / W
        
        # Calculate and return Theta
        angles = np.arctan2(y_tilde, x_tilde)
        return angles
    
    def get_implicit_matrices(self, x: np.ndarray, z_measurements_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the Implicit State Jacobian (H_total), 
        Measurement Jacobian (D), and the Implicit Angles (theta).
        
        Args:
            x: State vector
            z_measurements_global: Shape (2, N) array of global measurement points
            
        Returns:
            H_total: (2N, state_dim)
            D: (2N, 2N) - Block diagonal scaling of measurement noise
            angles: (N,) - The parametric angles theta derived from x and z
        """
        L = x[6]
        W = x[7]
        pca_coeffs = x[8:]
        N_meas = z_measurements_global.shape[1]
        
        # 1. Calculate derived variables (theta, rho, normalized coords)
        pos = x[:2].reshape(2, 1)
        R_mat = rot2D(x[2])
        
        # Transform measurements to local body frame
        # z_loc = R.T @ (z_global - p_c)
        z_loc = R_mat.T @ (z_measurements_global - pos) # Shape (2, N)
        
        # Normalized coordinates
        x_tilde = z_loc[0, :] / L
        y_tilde = z_loc[1, :] / W
        
        # Rho squared and Theta
        rho2 = x_tilde**2 + y_tilde**2
        angles = np.arctan2(y_tilde, x_tilde)
        
        # 2. Calculate Fourier properties at these angles
        # r(theta)
        g_mat = fourier_basis_matrix(angles, self.extent_cfg.N_fourier) # (N_f, N)
        fourier_weights = self.pca_mean + self.pca_eigenvectors @ pca_coeffs.reshape(-1, 1)
        r_vals = (g_mat.T @ fourier_weights).flatten() # (N,)
        
        # r'(theta)
        g_prime_mat = fourier_basis_derivative_matrix(angles, self.extent_cfg.N_fourier)
        r_prime_vals = (g_prime_mat.T @ fourier_weights).flatten() # (N,)
        
        # 3. Calculate the Tangent Vector h_theta
        # h_theta = R @ E @ (u_perp * r + u * r_prime)
        u_vec = unit_vector(angles)       # [cos, sin], Shape (2, N)
        u_perp = np.stack([-np.sin(angles), np.cos(angles)], axis=0) # [-sin, cos]
        
        E_mat = np.diag([L, W])
        
        # Vectorized term inside parenthesis: (2, N)
        term_inner = u_perp * r_vals + u_vec * r_prime_vals
        
        # Rotate and Scale: (2, N)
        h_theta = R_mat @ E_mat @ term_inner
        
        # 4. Calculate Angle Gradients (dTheta/dX)
        inv_rho2 = 1.0 / np.maximum(rho2, 0.05)  # Avoid division by zero with a small threshold

        if np.any(rho2 < 1e-6):
            print("Warning: Small rho^2 encountered in ImplicitIEKF angle gradient computation.")
        
        # dTheta/dPsi = -1/rho^2 * (y~^2 * W/L + x~^2 * L/W)
        dtheta_dpsi = -inv_rho2 * ((y_tilde**2 * (W/L)) + (x_tilde**2 * (L/W)))
        
        # dTheta/dL = (x~ * y~) / (rho^2 * L)
        dtheta_dL = (x_tilde * y_tilde) / (L * rho2)
        
        # dTheta/dW = -(x~ * y~) / (rho^2 * W)
        dtheta_dW = -(x_tilde * y_tilde) / (W * rho2)
        
        # dTheta/dPc (Position)
        term_L = y_tilde / L
        term_W = x_tilde / W
        
        # Derived from: 1/rho^2 * [ (-y~ * -1/L * R_row1) + (x~ * -1/W * R_row2) ]
        dtheta_dpc_x = inv_rho2 * (term_L * R_mat[0,0] - term_W * R_mat[0,1])
        dtheta_dpc_y = inv_rho2 * (term_L * R_mat[1,0] - term_W * R_mat[1,1])
        
        # 5. Assemble H_total (Implicit State Jacobian)
        jacobians_list = []
        D_blocks_list = []

        # Precompute explicit Rotation/Scaling components reused in loop
        J_rot = np.array([[0, -1], [1, 0]])
        
        for i in range(N_meas):
            # --- Assemble H_i (2 x N_state) ---
            r = r_vals[i]
            u_i = u_vec[:, i].reshape(2, 1)

            # A. Explicit Terms
            dh_dpc_exp = np.eye(2)
            dh_dpsi_exp = R_mat @ J_rot @ E_mat @ u_i * r
            dh_dL_exp = R_mat @ np.diag([1, 0]) @ u_i * r
            dh_dW_exp = R_mat @ np.diag([0, 1]) @ u_i * r

            # PCA
            g_vec = g_mat[:, i].reshape(-1, 1)
            dr_de = g_vec.T @ self.pca_eigenvectors
            dh_de_exp = (R_mat @ E_mat @ u_i) @ dr_de
            
            # B. Implicit Corrections (h_theta * dTheta/dX)
            h_t = h_theta[:, i].reshape(2, 1) # (2, 1)
            
            # Pos
            dth_dpc = np.array([[dtheta_dpc_x[i], dtheta_dpc_y[i]]]) # (1, 2)
            dh_dpc_imp = h_t @ dth_dpc 
            
            # Heading, Length, Width
            dh_dpsi_imp = h_t * dtheta_dpsi[i]
            dh_dL_imp = h_t * dtheta_dL[i]
            dh_dW_imp = h_t * dtheta_dW[i]

            # C. Combine
            H_i = np.hstack([
                dh_dpc_exp + dh_dpc_imp,       # Pos
                dh_dpsi_exp + dh_dpsi_imp,     # Heading
                np.zeros((2, 2)),              # Vel
                np.zeros((2, 1)),              # Rate
                dh_dL_exp + dh_dL_imp,         # Length
                dh_dW_exp + dh_dW_imp,         # Width
                dh_de_exp                      # PCA
            ])
            jacobians_list.append(H_i)
            
            # --- Assemble D_i (2 x 2) ---
            # D = I - h_theta * dTheta/dZ
            # dTheta/dZ = -dTheta/dPc
            dth_dz = -dth_dpc 
            D_i = np.eye(2) - (h_t @ dth_dz)
            D_blocks_list.append(D_i)
            
        H_total = np.vstack(jacobians_list)
        D_total = block_diag(*D_blocks_list)
        
        return H_total, D_total, angles
    
    def get_virtual_measurement_jacobian(self, x: np.ndarray, theta_body: float, is_radial: bool = False) -> Tuple[np.ndarray, float]:
        """
        Computes the analytical explicit Jacobian for Negative Information.
        If is_radial=False, returns Angular Jacobian (H_gamma) and predicted angle (gamma_pred).
        If is_radial=True, returns Radial Jacobian (H_rho) and predicted distance (rho_pred).
        """
        pos_x, pos_y = x[0], x[1]
        psi = x[2]
        L, W = x[6], x[7]
        pca_coeffs = x[8:]
        
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        c_theta, s_theta = np.cos(theta_body), np.sin(theta_body)
        
        # 1. Compute Radius
        fourier_coeffs = self.pca_mean + self.pca_eigenvectors @ pca_coeffs.reshape(-1, 1)
        g_theta = fourier_basis_matrix(np.array([theta_body]), self.extent_cfg.N_fourier) # Shape (N_f, 1)
        r_tilde = (g_theta.T @ fourier_coeffs).item()
        
        # 2. Compute the Cartesian coordinates relative to the LiDAR (u_x, u_y)
        x_body = L * c_theta * r_tilde
        y_body = W * s_theta * r_tilde
        
        u_x = pos_x - self.lidar_position[0] + (c_psi * x_body) - (s_psi * y_body)
        u_y = pos_y - self.lidar_position[1] + (s_psi * x_body) + (c_psi * y_body)
        
        # 3. Compute Gradients
        v_e = g_theta.T @ self.pca_eigenvectors # Shape (1, N_PC)
        
        dux_dpos = np.array([1.0, 0.0])
        dux_dpsi = np.array([-s_psi * x_body - c_psi * y_body])
        dux_dvel = np.array([0.0, 0.0, 0.0])
        dux_dscale = np.array([c_psi * c_theta * r_tilde, -s_psi * s_theta * r_tilde])
        dux_de = (c_psi * L * c_theta - s_psi * W * s_theta) * v_e.flatten()
        grad_ux = np.concatenate([dux_dpos, dux_dpsi, dux_dvel, dux_dscale, dux_de])
        
        duy_dpos = np.array([0.0, 1.0])
        duy_dpsi = np.array([c_psi * x_body - s_psi * y_body])
        duy_dvel = np.array([0.0, 0.0, 0.0])
        duy_dscale = np.array([s_psi * c_theta * r_tilde, c_psi * s_theta * r_tilde])
        duy_de = (s_psi * L * c_theta + c_psi * W * s_theta) * v_e.flatten()
        grad_uy = np.concatenate([duy_dpos, duy_dpsi, duy_dvel, duy_dscale, duy_de])
        
        # 4. Return Requested Formulation
        if is_radial:
            rho_pred = np.sqrt(u_x**2 + u_y**2)
            # H_rho = (u_x * grad_ux + u_y * grad_uy) / rho
            H_virt = (u_x * grad_ux + u_y * grad_uy) / rho_pred
            return H_virt.reshape(1, -1), rho_pred
        else:
            gamma_pred = np.arctan2(u_y, u_x)
            # H_gamma = (u_x * grad_uy - u_y * grad_ux) / rho^2
            H_virt = (u_x * grad_uy - u_y * grad_ux) / (u_x**2 + u_y**2)
            return H_virt.reshape(1, -1), gamma_pred

@dataclass
class LidarSimulator(SensorModel[Sequence[LidarScan]]):
    """
    A simulation model for a LiDAR.
    Handles ray-casting and noise generation.
    """
    lidar_position: np.ndarray
    num_rays: int
    max_distance: float
    lidar_gt_std_dev: float
    rng: np.random.Generator = field(repr=False)
    extent_cfg: ExtentConfig

    def sample_from_state(self, x_gt: State_PCA) -> LidarScan:
        """
        Overrides the base method to simulate LiDAR measurements from the ground truth state.
        """
        shape_x_coords, shape_y_coords = compute_exact_vessel_shape_global(
            x_gt, self.extent_cfg.shape_coords_body
        )

        polar_measurements_list = self.simulate_lidar_measurements(shape_x_coords, shape_y_coords)
        
        # Handle empty measurements
        if not polar_measurements_list:
             polar_measurements = np.zeros((0, 2))
        else:
             polar_measurements = np.array(polar_measurements_list)

        assert polar_measurements.shape[1] == 2, "Expected polar_measurements to have shape (N, 2)"

        if polar_measurements.shape[0] == 0:
             return LidarScan(x=np.array([]), y=np.array([]))

        x_coords, y_coords = pol2cart(polar_measurements[:, 0], polar_measurements[:, 1])
        
        return LidarScan(x=x_coords, y=y_coords)

    def simulate_lidar_measurements(self, shape_x, shape_y) -> List[Tuple[float, float]]:
        """
        Simulate LiDAR measurements by casting rays from the LiDAR position to the vessel shape.
        Returns noisy distance measurements in polar coordinates.
        """
        # Noise characteristics
        lidar_position = self.lidar_position
        num_rays = self.num_rays
        max_distance = self.max_distance

        angles, distances = cast_rays(lidar_position, num_rays, max_distance, shape_x, shape_y)
        noisy_measurements = add_noise_to_distances(self.rng, distances, angles, 0.0, self.lidar_gt_std_dev)
        return noisy_measurements