import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

from src.senfuslib.gaussian import MultiVarGauss

@dataclass
class TrackerUpdateResult:
    """
    Holds all relevant data from a single tracker update step.
    Moved here to avoid circular imports between Tracker logic and SimulationResult.
    """
    # Core filter states
    state_prior: MultiVarGauss           # State estimate before the update (x_k|k-1)
    state_posterior: MultiVarGauss       # State estimate after the update (x_k|k)

    # Measurement and Innovation
    measurements: Optional[np.ndarray]      # The flattened measurement vector used (z_k)
    predicted_measurement: Optional[MultiVarGauss]    # The predicted measurement
    innovation_gauss: Optional[MultiVarGauss]         # The innovation

    # --- DEBUGGING VALUES ---
    cost_prior: Optional[float] = None
    cost_likelihood: Optional[float] = None
    cost_penalty: Optional[float] = None
    H_jacobian: Optional[np.ndarray] = None
    R_covariance: Optional[np.ndarray] = None
    K_gain: Optional[np.ndarray] = None
    
    # Optional Debugging / Analysis Info
    iterations: Optional[int] = None
    iterates: Optional[list] = None
    predicted_measurements_iterates: Optional[list] = None
    cost: Optional[float] = None
    raw_optimizer_result: Any = None

    # Constraints Logging
    clamped_length: Optional[tuple[float, float]] = None # (old, new)
    clamped_width: Optional[tuple[float, float]] = None # (old, new)
    mahalanobis_projection: Optional[tuple[np.ndarray, np.ndarray, float]] = None # (old_coeffs, new_coeffs, initial_chi2_dist)
    negative_info_used: Optional[int] = None # Number of negative info constraints applied
    virtual_constraints_info: Optional[list] = None # Information about applied negative info virtual constraints
    H_fused_iterates: Optional[list] = None # History of the fused observation matrix (positive+negative+prior) per iterate
