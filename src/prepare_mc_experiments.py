import os
import pickle
import json
import logging
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)


def prepare_mc_experiment(experiment_name, num_runs, base_seed, sim_base, lidar_base, extent_base, tracker_cfg):
    """
    Generates Ground Truth and lidar measurements for multiple MC runs and saves them.
    """
    from src.utils.config_classes import Config
    from src.experiment_runner import _setup_tracker_and_data
    from dataclasses import asdict

    # Ensure output directory exists
    output_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Preparing MC Experiment '{experiment_name}' with {num_runs} runs.")
    logging.info(f"Output directory: {output_dir}")
    
    # Save the base config (JSON sidecar)
    dummy_config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)
    
    class ConfigEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.ndarray): return obj.tolist()
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            return str(obj)
            
    with open(output_dir / "base_config.json", "w") as f:
        json.dump(asdict(dummy_config), f, indent=4, cls=ConfigEncoder)


    # Run data generation loop
    for i in tqdm(range(num_runs), desc="Generating MC runs"):
        current_seed = base_seed + i
        
        # Override seed - we must NOT use the cache to guarantee fresh generation
        sim_base.seed = current_seed
        sim_base.use_cache = False 
        
        # We wrap in a config required by the setup method
        config = Config(sim=sim_base, lidar=lidar_base, tracker=tracker_cfg, extent=extent_base)
        
        # Set up everything to get the simulator
        tracker, _, lidar_model, simulator, _ = _setup_tracker_and_data(config)
        
        # Generating simulation data
        gt_ts = simulator.get_gt()
        meas_lidar_ts = simulator.get_meas()
        
        # Build global measurements to save as well
        import numpy as np
        lidar_pos_global = np.array(lidar_base.lidar_position).reshape(2, 1)
        meas_global_ts = meas_lidar_ts.map(lambda scan: scan + lidar_pos_global)
        
        # Save run data
        run_data = {
            "seed": current_seed,
            "gt_ts": gt_ts,
            "meas_lidar_ts": meas_lidar_ts,
            "meas_global_ts": meas_global_ts,
            "static_covariances": {"R_point": lidar_model.R_single_point()} # Q relies on dt which is in config
        }
        
        run_file = output_dir / f"run_{i:03d}.pkl"
        with open(run_file, "wb") as f:
            pickle.dump(run_data, f)
            
    logging.info(f"Successfully generated {num_runs} runs for experiment '{experiment_name}'.")

if __name__ == "__main__":
    from src.main import get_common_configs, get_pca_tracker_config
    
    EXPERIMENT_NAME = "exp1_linear_noise015"
    NUM_RUNS = 100
    BASE_SEED = 1000
    
    N_pca = 4
    selected_boat_id = "Havfruen"
    selected_trajectory = "linear"
    
    sim_base, lidar_base, extent_base = get_common_configs(traj_type=selected_trajectory, N_pca=N_pca, selected_boat_id=selected_boat_id)
    
    sim_base.num_frames = 300
    lidar_base.lidar_gt_std_dev = 0.15
    
    tracker_cfg = get_pca_tracker_config(lidar_base.lidar_position, sim_base.initial_state_gt, N_pca)
    tracker_cfg.method = "implicit_iekf" # doesn't matter for the generation phase
    
    # Needs to match main.py custom user settings to ensure consistency in standard deviation initialization if used etc
    
    prepare_mc_experiment(EXPERIMENT_NAME, NUM_RUNS, BASE_SEED, sim_base, lidar_base, extent_base, tracker_cfg)
