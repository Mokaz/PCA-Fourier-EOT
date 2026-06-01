import numpy as np
from typing import Any, Dict, Optional, Tuple, List

from dataclasses import dataclass, field

from src.utils.tools import cart2pol, pol2cart, ssa
from src.utils.ship_database import get_boat_radii
from src.states.states import State_PCA

@dataclass
class TrajectoryConfig:
    type: str = "linear"  # "linear", "circle", "waypoints"
    speed: float = 3.0
    
    # For Circle
    center: Tuple[float, float] = (30.0, 0.0) # E.g., circle around LiDAR
    radius: float = 40.0
    clockwise: bool = True
    
    # For Waypoints
    waypoints: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, -40), (0, 40), (60, 40), (60, -40)
    ])

@dataclass
class SimulationConfig:
    name: str = "default_simulation"

    num_simulations: int = 1
    num_frames: int = 100
    dt: float = 0.1
    seed: int = 42

    gt_yaw_rate_std_dev: float = 0.1
    initial_state_gt: State_PCA = None
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    use_cache: bool = False

    # Visualization
    show_gt_plot: bool = False

@dataclass
class ExtentConfig:
    N_fourier: int = 64
    d_angle: float = np.deg2rad(1.0)
    shape_params_true: Dict[str, Any] = field(default_factory=lambda: {"type": "ellipse", "L": 20.0, "W": 6.0})

    angles: np.ndarray = field(init=False)
    shape_coords_body: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.angles = np.arange(-np.pi, np.pi, self.d_angle)

        self.shape_coords_body = self.compute_exact_vessel_shape_local_from_params(
            self.angles, self.shape_params_true
        )

    def compute_exact_vessel_shape_local_from_params(self, angles: np.ndarray, shape_params_true: Dict[str, Any]) -> np.ndarray:
        """
        Generates the ground truth Cartesian coordinates of the vessel's shape in local body frame.

        Args:
            angles: Angles at which to compute the vessel shape.
            shape_params_true: A dictionary containing the true shape parameters of the vessel.
        Returns:
            A tuple (shape_x, shape_y) of the vessel's global coordinates.
        """
        radii = np.zeros_like(angles)
        nAngles = angles.size

        # Extract parameters from state and config
        L = shape_params_true.get("L")
        W = shape_params_true.get("W")
        S = shape_params_true.get("S")
        P = shape_params_true.get("P")
        shape_type = shape_params_true.get("type")

        if shape_type == "database":
            boat_id = shape_params_true.get("id")
            if boat_id is None:
                raise ValueError("Shape parameters for 'database' type must include 'id'")
                
            db_radii = get_boat_radii(str(boat_id))
            
            # 1. Reconstruct unit shape from DB radii
            db_angles = np.linspace(-np.pi, np.pi, len(db_radii), endpoint=False)
            x_unit = db_radii * np.cos(db_angles)
            y_unit = db_radii * np.sin(db_angles)

            # 2. Scale X by L and Y by W independently
            # Note: Database shapes are normalized to Length=1. 
            # We assume their width is proportional. To force specific W, we need 
            # to know original aspect ratio or just stretch Y unit to W.
            # Assuming x_unit refers to a ~1.0 length object.
            
            x_scaled = x_unit * L 
            
            # Use W scaling relative to the original aspect ratio found in the DB radius?
            # Or force fit to W? Below forces fit to W box.
            max_y_db = np.max(np.abs(y_unit))
            if max_y_db > 0:
                y_scaled = y_unit * ( (W/2.0) / max_y_db )
            else:
                y_scaled = y_unit * W

            # 3. Convert back to radii for the requested 'angles'
            target_angles_pol, target_radii_pol = cart2pol(x_scaled, y_scaled)
            
            # Sort is required for interp if angles wrap or aren't monotonic after conversion
            sort_idx = np.argsort(target_angles_pol)
            radii = np.interp(angles, target_angles_pol[sort_idx], target_radii_pol[sort_idx], period=2*np.pi)
        
        elif shape_type == "box":
            theta_0 = np.arctan2(W, L)
            radii = np.zeros(angles.shape)

            for idx, angle in enumerate(angles):
                if angle <= theta_0:
                    radii[idx] = L / 2 / np.cos(angle)
                elif angle <= np.pi - theta_0:
                    radii[idx] = W / 2 / np.cos(np.pi/2 - angle)
                else:
                    radii[idx] = L / 2 / np.cos(np.pi - angle)

        elif shape_type == "ellipse":
            x_interpol = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_interpol = (W / 2) * np.sqrt(1 - ((2 / L) * x_interpol)**2)

            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)

            radii = np.interp(np.abs(ssa(angles)), angles_interpol, r_interpol, period=2*np.pi)

        elif shape_type == "box_elliptic_sides":
            x_ellipse = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_ellipse = S/2 + ((W - S) / 2) * np.sqrt(1 - ((2 / L) * x_ellipse)**2)

            angles_interpol, r_interpol = cart2pol(x_ellipse, y_ellipse)

            theta_corner = np.arctan2(S, L)
            radii = np.zeros(angles.shape)

            for idx, angle in enumerate(angles):
                if angle <= theta_corner:
                    radii[idx] = (L / 2) / np.cos(angle)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                else:
                    radii[idx] = (L / 2) / np.cos(np.pi - angle)

        elif shape_type == "box_parabolic_bow_and_stern":
            x_interpol = np.linspace((L / 2) - P, L / 2, num=nAngles, endpoint=True)
            y_interpol = (-L**2 * W + 4*L*P*W) / (8 * P**2) + ((L*W - 2*P*W) * x_interpol) / (2*P**2) - (W * x_interpol**2) / (2 * P**2)  
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            theta_corner = np.arctan2(W, L - 2*P)
            for idx, angle in enumerate(angles):
                if angle <= theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = (W / 2) / np.sin(angle)
                else:
                    radii[idx] = np.interp(np.pi - angle, angles_interpol, r_interpol, period=2*np.pi)

        elif shape_type == "elliptic_bow_and_stern":
            y_ellipse = np.linspace(-W / 2, W / 2, num=nAngles, endpoint=True)
            x_ellipse = (L / 2) - P + P * np.sqrt(1 - ((2 / W) * y_ellipse)**2)
            angles_interpol, r_interpol = cart2pol(x_ellipse, y_ellipse)
            theta_corner = np.arctan2(W, L - 2*P)
            for idx, angle in enumerate(angles):
                if angle <= theta_corner:
                    radii[idx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= np.pi - theta_corner:
                    radii[idx] = (W / 2) / np.sin(angle)
                else:
                    radii[idx] = np.interp(np.pi - angle, angles_interpol, r_interpol, period=2*np.pi)

        elif shape_type == "custom_extent":
            true_radii = shape_params_true.get("radii")
            true_angles = shape_params_true.get("angles")
            if true_radii is None or true_angles is None:
                raise ValueError("custom_extent requires 'radii' and 'angles'")
            true_radii = np.array(true_radii).flatten()
            true_angles = np.array(true_angles).flatten()
            # Ensure angles are sorted and unwrap if needed for interpolation
            sort_idx = np.argsort(true_angles)
            radii = np.interp(angles, true_angles[sort_idx], true_radii[sort_idx], period=2*np.pi)

        else:
            raise ValueError(f"Unknown ground truth shape type: {shape_type}")

        # Convert the perfect polar shape to Cartesian coordinates in the body frame
        shape_coords_body = np.stack(pol2cart(angles, radii), axis=0)

        return shape_coords_body


@dataclass
class TrackerConfig:
    use_gt_state_for_bodyangles_calc: bool = False
    use_initialize_centroid: bool = False
    
    method: str = 'implicit_iekf'

    N_pca: int = 4
    PCA_parameters_path : str = 'data/input_parameters/FourierPCAParameters_scaled.npz'

    # Process Model selection: 'cv', 'temporal', 'inflation'
    process_model: str = 'cv'
    
    # Process Model Params
    temporal_eta: float = 0.1
    temporal_pca_var: float = 1.0
    inflation_lambda: float = 0.99

    # GP Specifics
    N_gp_points: int = 20
    gp_length_scale: float = 0.5
    gp_signal_var: float = 1.0
    gp_forgetting_factor: float = 0.05
    gp_use_negative_info: bool = True

    # Q and R noise parameters
    pos_north_std_dev: float = 0.3
    pos_east_std_dev: float = 0.3
    heading_std_dev: float = 0.1

    length_std_dev: float = 0.1
    width_std_dev: float = 0.1
    pca_std_dev_scale: float = 0.3
    use_proportional_pca_random_walk: bool = True

    lidar_std_dev: float = 0.15
    
    debug_prints: bool = False

    # Accepts State_GP or State_PCA
    initial_state: Optional[Any] = None 
    initial_std_devs: Optional[Any] = None
    
    lidar_position: Optional[np.ndarray] = None
    pca_eigenvalues: Optional[np.ndarray] = None

    # Iterative solver params
    max_iterations: int = 10
    convergence_threshold: float = 1e-6

    # Implicit EKF specific
    use_state_clamping: bool = True
    use_mahalanobis_projection: bool = True
    mahalanobis_projection_prob: float = 0.99
    use_negative_info_angular: bool = False
    force_kinematic_unobservability: bool = False
    use_negative_info_front: bool = False
    use_negative_info_centroid: bool = False
    radial_margin: float = 0.1
    use_exact_extreme_angle: bool = False
    use_D_imp_for_R: bool = True
    use_scaled_R: bool = False
    
    # Negative information parameters
    use_arc_length_residual: bool = True
    R_neg_info_std_angle: float = 0.01
    R_neg_info_std_front: float = 0.01
    R_neg_info_std_centroid: float = 0.01

    # --- Soft Extent Priors ---
    use_absolute_L_W_prior: bool = False
    prior_target_L: float = 20.0
    prior_target_W: float = 6.0
    prior_size_std: float = 5.0

    use_L_W_aspect_ratio_prior: bool = True
    prior_aspect_ratio: float = 3.8  # L / W
    prior_ratio_std: float = 5.0     # Tolerance in meters

    # Smoother specific
    smoother_window_size: int = 10

    def __post_init__(self):
        if self.initial_state is None:
            from src.states.states import State_PCA
            self.initial_state = State_PCA(
                x=0.0, y=-40.0, yaw=np.pi / 2, vel_x=0.0, vel_y=0.0, yaw_rate=0.0,
                length=20.0, width=6.0, pca_coeffs=np.zeros(self.N_pca)
            )
        
        if self.initial_std_devs is None:
            from src.states.states import State_PCA
            self.initial_std_devs = State_PCA(
                x=2.0, y=2.0, yaw=0.2, 
                vel_x=2.0, vel_y=2.0, yaw_rate=0.1,
                length=2.0, width=2.0,
                pca_coeffs=np.ones(self.N_pca) * 0.5 
            )

@dataclass
class LidarConfig:
    """
    Configuration parameters for the LiDAR sensor.
    """
    # LiDAR Parameters
    lidar_position: Tuple[float, float] = (30.0, 0.0)
    num_rays: int = 360
    max_distance: float = 140.0
    lidar_gt_mean: float = 0.0
    lidar_gt_std_dev: float = 0.15

@dataclass
class Config:
    sim: SimulationConfig
    lidar: LidarConfig
    tracker: TrackerConfig
    extent: ExtentConfig
