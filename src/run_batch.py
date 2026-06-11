import itertools
import pickle
from pathlib import Path
import sys
import logging
from zlib import crc32
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from src.global_project_paths import SIMDATA_PATH
from src.utils.config_classes import Config
from src.experiment_runner import run_single_simulation
from main import get_common_configs, get_pca_tracker_config


def set_nested_value(obj, path, value):
    """Sets a value deep in a nested object using a dot-notation string."""
    parts = path.split('.')
    last = parts.pop()
    for part in parts:
        obj = getattr(obj, part)
    setattr(obj, last, value)

# --- SCENARIO REGISTRY ---
SCENARIOS = {
    "baseline": {
        "init_pos_offset": (0.0, 0.0),
        "init_size_scale": (1.0, 1.0),
    },
    "offset_5m": {
        "init_pos_offset": (5.0, 5.0),
        "init_size_scale": (1.0, 1.0),
    },
    "wrong_shape": {
        "init_pos_offset": (0.0, 0.0),
        "init_size_scale": (2.0, 0.5), # Starts way too long and narrow
    },
    "wrong_pos_wrong_shape": {
        "init_pos_offset": (1.0, -1.0),
        "init_yaw_offset": np.deg2rad(10.0),
        "init_size_scale": (2.5, 0.5), # Starts way too long and narrow
    }
}

if __name__ == "__main__":
    
    # --- 1. Define the Parameter Grid ---
    # Keys: dot-notation path to attribute (or "method" for the tracker type)
    # Values: List of options to sweep over
    param_grid = {
        "scenario": ["baseline"],
        # "lidar.num_rays": [1024],
        "method": ["implicit_ekf"],
        "selected_boat_id": ["Havfruen"],
        "selected_trajectory": ["complex_maneuvers"],
        "tracker.max_iterations": [1, 2, 3, 5, 10, 20],
        # "tracker.pca_std_dev_scale": [0.1, 0.3, 0.5],
        # "tracker.lidar_std_dev": [0.05, 0.15, 0.2, 0.3, 0.5],
        # "tracker.use_D_imp_for_R": [True, False],
        # "tracker.use_scaled_R": [True, False],
        # "tracker.use_negative_info_angular": [True, False],
        # "tracker.use_negative_info_front": [True, False],
        # "tracker.use_negative_info_centroid": [True, False],
        # "tracker.use_initialize_centroid": [True, False],
        # "tracker.use_absolute_L_W_prior": [True],
        # "tracker.use_L_W_aspect_ratio_prior": [True]
        # "tracker.use_exact_extreme_angle": [True, False],
        # "tracker.use_arc_length_residual": [True, False],
        # "tracker.R_neg_info_std_angle": [0.005, 0.01, 0.05, 0.1, 0.2],
        "lidar.lidar_gt_std_dev": [0.15],
    }

    # --- 2. Generate Combinations ---
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"--- Queued {len(combinations)} simulations ---")
    
    # Constants
    N_pca = 4

    seen_signatures = set()

    # --- 3. Run Loop ---
    for i, combination in enumerate(combinations):
        current_params = dict(zip(keys, combination))
        scenario_name = current_params.pop("scenario")
        method_name = current_params.pop("method")
        selected_boat_id = current_params.pop("selected_boat_id")
        selected_trajectory = current_params.pop("selected_trajectory")

        scenario_params = SCENARIOS.get(scenario_name, {})

        # Create a dictionary of only the *effective* parameters
        effective_params = {}
        for path, val in current_params.items():
            # Skip parameters that are not relevant to the current process model
            if "inflation_lambda" in path and current_params.get("tracker.process_model") != "inflation":
                continue
            if "temporal_eta" in path and current_params.get("tracker.process_model") != "temporal":
                continue
            effective_params[path] = val

        # Skip redundant parameter combinations
        config_signature = (scenario_name, method_name, selected_boat_id, selected_trajectory, tuple(sorted(effective_params.items())))
        if config_signature in seen_signatures:
            continue
        seen_signatures.add(config_signature)
        
        # --- A. Reset Base Config ---
        sim_base, lidar_base, extent_base = get_common_configs(
            traj_type=selected_trajectory,
            N_pca=N_pca,
            selected_boat_id=selected_boat_id
        )

        if "gp" in method_name:
            raise NotImplementedError("GP batch runs are not supported via main_database configuration helpers.")

        tracker_cfg = get_pca_tracker_config(
            lidar_base.lidar_position,
            sim_base.initial_state_gt,
            N_pca,
            init_pos_offset=scenario_params.get("init_pos_offset", (0.0, 0.0)),
            init_yaw_offset=scenario_params.get("init_yaw_offset", 0.0),
            init_size_scale=scenario_params.get("init_size_scale", (1.0, 1.0))
        )

        # If using Implicit EKF, set max_iterations to 1 for EKF behavior
        if method_name == "implicit_ekf":
            tracker_cfg.max_iterations = 1
        else:
            tracker_cfg.max_iterations = 20

        if selected_trajectory == "waypoints":
            sim_base.num_frames = 500
        elif selected_trajectory == "waypoints2":
            sim_base.num_frames = 800
        else:
            sim_base.num_frames = 800

        config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)

        # --- B. Apply Parameters ---
        param_desc_parts = []
        for path, val in current_params.items():
            set_nested_value(config, path, val)
        
        # Only add effective parameters to the file name to keep names clean
        for path, val in effective_params.items():
            short_key = path.split('.')[-1].replace("use_", "")
            
            # Format numbers to look cleaner in filenames (e.g. 0.99 -> 0p99)
            if isinstance(val, float):
                val_str = str(val).replace('.', 'p')
            else:
                val_str = str(val)
                
            param_desc_parts.append(f"{short_key}_{val_str}")

        # --- C. Naming & Execution ---
        param_suffix = ("_" + "_".join(param_desc_parts)) if param_desc_parts else ""

        # Custom user settings
        config.sim.use_cache = True
        # config.lidar.lidar_gt_std_dev = 0.15
        # config.tracker.use_initialize_centroid = False
        config.tracker.process_model = "inflation"

        config.tracker.use_initialize_centroid = False
        config.tracker.use_D_imp_for_R = False

        config.tracker.use_state_clamping = True
        config.tracker.use_mahalanobis_projection = True

        config.tracker.use_negative_info_angular = True
        config.tracker.use_negative_info_front = True
        config.tracker.use_negative_info_centroid = True

        config.tracker.use_absolute_L_W_prior = True
        config.tracker.use_L_W_aspect_ratio_prior = True

        config.tracker.use_scaled_R = False

        config.tracker.force_kinematic_unobservability = False
        
        # Standardize naming: method + scenario + boat + params + traj + seed
        config.sim.name = f"{method_name}_{scenario_name}_boat{selected_boat_id}{param_suffix}_{config.sim.trajectory.type}_seed_{config.sim.seed}"
        
        print(f"\n[{i+1}/{len(combinations)}] Running: {config.sim.name}")
        print(f"   > Effective Params: {effective_params}")
        
        filename = f"{config.sim.name}.pkl"

        try:
            config.tracker.method = method_name
            run_single_simulation(config=config)
            print(f"   > Success. Saved to {filename}")
        except Exception as e:
            logging.error(f"Simulation {config.sim.name} failed!", exc_info=True)
