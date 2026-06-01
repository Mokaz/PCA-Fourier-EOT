from dataclasses import dataclass, field
from typing import TypeVar
import numpy as np
from scipy.linalg import block_diag

from src.senfuslib import DynamicModel
from src.states.states import State_PCA, State_GP
from src.utils.tools import ssa

from src.utils.GaussianProcess import GaussianProcess
from src.dynamics.trajectories import TrajectoryStrategy, ConstantVelocityTrajectory

S = TypeVar('S', bound=np.ndarray)  # State type

@dataclass
class Model_PCA_CV(DynamicModel):
    """
    Implements the decoupled Constant Velocity process model for the full GP-PCA state.
    """
    # Noise parameters
    x_pos_std_dev: float
    y_pos_std_dev: float
    yaw_std_dev: float
    length_std_dev: float
    width_std_dev: float

    # Number of PCA components
    N_pca: int

    def F_d(self, x: State_PCA, dt: float) -> np.ndarray:
        """
        Discrete-time state transition jacobian (F).
        """
        F = np.eye(8 + self.N_pca)
        t = np.array([[1, dt], [0, 1]])
        # Create the 6x6 kinematic block
        F_kin = np.kron(t, np.eye(3))
        F[:6, :6] = F_kin
        return F

    def Q_d(self, x: State_PCA = None, dt: float = 0.1) -> np.ndarray:
        """
        Discrete-time process noise covariance (Q).
        """
        Q_c_matrix = np.diag([
            self.x_pos_std_dev**2, 
            self.y_pos_std_dev**2, 
            self.yaw_std_dev**2
        ])
        
        t_int = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
        Qk = np.kron(t_int, Q_c_matrix)
        
        Q_extent_pca = np.zeros((2 + self.N_pca, 2 + self.N_pca))
        
        # INJECT L & W RANDOM WALK NOISE HERE
        Q_extent_pca[0, 0] = (self.length_std_dev ** 2) * dt
        Q_extent_pca[1, 1] = (self.width_std_dev ** 2) * dt
        
        Q = np.block([[Qk,np.zeros((6, 2 + self.N_pca))],[np.zeros((2 + self.N_pca, 6)), Q_extent_pca]
        ])
        return Q

@dataclass
class Model_PCA_Temporal(DynamicModel):
    """
    Implements a Temporal GP (Ornstein-Uhlenbeck) model for the PCA coefficients.
    Independent of Model_PCA_CV.
    Eq (5): F^f = exp(-eta * T), Q^f = (1 - exp(-2 * eta * T)) * K
    """
    x_pos_std_dev: float
    y_pos_std_dev: float
    yaw_std_dev: float
    length_std_dev: float
    width_std_dev: float

    N_pca: int

    # Temporal parameters
    eta_f: float = 0.1
    pca_process_var: np.ndarray = None

    def F_d(self, x: State_PCA, dt: float) -> np.ndarray:
        # Kinematics (CV)
        F = np.eye(8 + self.N_pca)
        t = np.array([[1, dt], [0, 1]])
        F[:6, :6] = np.kron(t, np.eye(3))
        
        # Length/Width (Indices 6, 7): Random Walk / Static
        # Keep F=1.0 for these states (no change)
        F[6:8, 6:8] = np.eye(2) 

        # PCA Coefficients (Indices 8+): Ornstein-Uhlenbeck (Decay)
        decay = np.exp(-self.eta_f * dt)
        pca_dim = self.N_pca
        F[8:, 8:] = np.eye(pca_dim) * decay
        
        return F

    def Q_d(self, x: State_PCA = None, dt: float = 0.1) -> np.ndarray:
        # Kinematics Noise
        Q_c_matrix = np.diag([
            self.x_pos_std_dev**2, 
            self.y_pos_std_dev**2, 
            self.yaw_std_dev**2
        ])
        t_int = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
        Qk = np.kron(t_int, Q_c_matrix)
        
        # UPDATE THIS: Use the class attributes instead of hardcoded 0.001
        Q_lw = np.diag([self.length_std_dev**2, self.width_std_dev**2]) * dt

        scaling = (1 - np.exp(-2 * self.eta_f * dt))
        if self.pca_process_var is not None:
            Q_pca = np.diag(scaling * self.pca_process_var)
        else:
            Q_pca = np.eye(self.N_pca) * scaling
        
        Q = block_diag(Qk, Q_lw, Q_pca)
        return Q

@dataclass
class Model_PCA_Inflation(DynamicModel):
    """
    Implements Random Walk with Covariance Inflation for the Extent.
    Independent of Model_PCA_CV.
    Eq (14): Q^f = (1/lambda - 1) * P_extent
    """
    x_pos_std_dev: float
    y_pos_std_dev: float
    yaw_std_dev: float
    N_pca: int
    length_std_dev: float = 0.01
    width_std_dev: float = 0.01
    pca_std_dev_scale: float = 0.05
    use_proportional_pca_random_walk: bool = True

    # Inflation parameter
    lambda_f: float = 0.99

    pca_eigenvalues: np.ndarray = None

    def F_d(self, x: State_PCA, dt: float) -> np.ndarray:
        # Kinematics (CV) + Fixed Extent
        F = np.eye(8 + self.N_pca)
        t = np.array([[1, dt], [0, 1]])
        F_kin = np.kron(t, np.eye(3))
        F[:6, :6] = F_kin
        return F

    def Q_d(self, x: State_PCA = None, dt: float = 0.1) -> np.ndarray:
        # Kinematics Noise Only (Extent noise added via inflation in pred_from_est)
        Q_c_matrix = np.diag([
            self.x_pos_std_dev**2, 
            self.y_pos_std_dev**2, 
            self.yaw_std_dev**2
        ])
        t_int = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
        Qk = np.kron(t_int, Q_c_matrix)
        Q_extent_pca = np.zeros((2 + self.N_pca, 2 + self.N_pca))
        
        # L & W random walk noise
        Q_extent_pca[0, 0] = (self.length_std_dev ** 2) * dt
        Q_extent_pca[1, 1] = (self.width_std_dev ** 2) * dt
        
        if self.use_proportional_pca_random_walk:
            for i in range(self.N_pca):
                eigen_std = np.sqrt(self.pca_eigenvalues[i]) if self.pca_eigenvalues is not None else 1.0
                
                scaled_std_dev = self.pca_std_dev_scale * eigen_std
                
                Q_extent_pca[2 + i, 2 + i] = (scaled_std_dev ** 2) * dt
        else:
            # Constant variance random walk for PCA components
            for i in range(self.N_pca):
                Q_extent_pca[2 + i, 2 + i] = (self.pca_std_dev_scale ** 2) * dt

        Q = np.block([[Qk, np.zeros((6, 2 + self.N_pca))],
                      [np.zeros((2 + self.N_pca, 6)), Q_extent_pca]
        ])
        return Q

    def pred_from_est(self, x_est, dt: float):
        x_pred = super().pred_from_est(x_est, dt)
        
        # Apply Inflation to Extent Block
        idx_extent = slice(6, None)
        inflation_factor = 1.0 / self.lambda_f
        x_pred.cov[idx_extent, idx_extent] *= inflation_factor
        
        return x_pred

@dataclass
class GroundTruthModel(DynamicModel):
    rng: np.random.Generator = field(repr=False)
    yaw_rate_std_dev: float = 0.01
    
    trajectory_strategy: TrajectoryStrategy = field(default=None, repr=False)

    def step_simulation(self, x: State_PCA, dt: float) -> State_PCA:
        new_state = x.copy()

        target_speed, cmd_yaw_rate = self.trajectory_strategy.compute_velocity_commands(x, dt)

        # Yaw rate noise (only for CV trajectory to prevent drifting off perfect circles/paths)
        if isinstance(self.trajectory_strategy, ConstantVelocityTrajectory):
            noise = self.rng.normal(0.0, self.yaw_rate_std_dev)
        else:
            noise = 0.0
        
        new_state.yaw_rate = cmd_yaw_rate + noise
        new_state.yaw = ssa(x.yaw + new_state.yaw_rate * dt)
        
        # Calculate velocity components based on target speed and new heading
        new_state.vel_x = target_speed * np.cos(new_state.yaw)
        new_state.vel_y = target_speed * np.sin(new_state.yaw)

        new_state.x += new_state.vel_x * dt
        new_state.y += new_state.vel_y * dt
        
        return new_state

@dataclass
class Model_GP_CV(DynamicModel):
    """
    Implements the decoupled Constant Velocity process model for the full GP-EOT state.
    Kinematics evolve via CV, Extent evolves via an Ornstein-Uhlenbeck process.
    """
    # GP Utility class (holds the math for F and Q of the extent)
    gp_utils: GaussianProcess
    
    # Noise parameters
    x_pos_std_dev: float
    y_pos_std_dev: float
    yaw_std_dev: float
    
    # Extent parameters
    forgetting_factor: float

    def F_d(self, x: State_GP, dt: float) -> np.ndarray:
        """
        Discrete-time state transition jacobian (F).
        """
        # 1. Kinematic Transition (Constant Velocity)
        # Order: [x, y, yaw, vx, vy, yaw_rate]
        # Maps to: pos += vel*dt, vel += 0
        t = np.array([[1, dt], [0, 1]])
        F_kin = np.kron(t, np.eye(3)) # 6x6 Matrix
        
        # 2. Extent Transition (GP Forgetting Factor)
        F_gp = self.gp_utils.F(dt, self.forgetting_factor)
        
        # 3. Combine into full block diagonal matrix
        F = block_diag(F_kin, F_gp)
        
        return F

    def Q_d(self, x: State_GP = None, dt: float = 0.1) -> np.ndarray:
        """
        Discrete-time process noise covariance (Q).
        """
        # 1. Kinematic Noise
        # Integrated White Noise assumption
        Q_c_matrix = np.diag([
            self.x_pos_std_dev**2, 
            self.y_pos_std_dev**2, 
            self.yaw_std_dev**2
        ])
        
        t_int = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
        Q_kin = np.kron(t_int, Q_c_matrix) # 6x6 Matrix
        
        # 2. Extent Noise
        # Noise injected to maintain the stationary variance of the GP
        Q_gp = self.gp_utils.Q(dt, self.forgetting_factor)
        
        # 3. Combine
        Q = block_diag(Q_kin, Q_gp)
        
        return Q