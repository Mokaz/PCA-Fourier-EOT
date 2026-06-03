import os
import pickle
import json
import logging
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

from src.utils.geometry_utils import calculate_iou, compute_exact_vessel_shape_global, compute_estimated_shape_global

def analyze_mc_experiment(experiment_name, methods):
    print(f"\n{'='*50}")
    print(f"Starting analysis for MC Experiment: {experiment_name}")
    print(f"{'='*50}")
    experiment_dir = Path(PROJECT_ROOT) / "data" / "results" / "mc_experiments" / experiment_name
    if not experiment_dir.exists():
        logging.error(f"Experiment directory {experiment_dir} not found.")
        return

    # Load Base Config for metadata
    config_file = experiment_dir / "base_config.json"
    if not config_file.exists():
         logging.error(f"Base config not found.")
         return
         
    with open(config_file, "r") as f:
        base_config_data = json.load(f)

    if not methods:
        # Standard Methods
        methods = [d.name for d in experiment_dir.iterdir() if d.is_dir() and d.name not in ["results", "tuning"]]
        
    tuning_dir = experiment_dir / "tuning"
    tuning_methods = []
    if tuning_dir.exists():
        tuning_methods = [d.name for d in tuning_dir.iterdir() if d.is_dir()]

    def process_method_list(method_list, base_dir, is_tuning=False):
        if not method_list:
            return
            
        group_name = "Tuning Runs" if is_tuning else "Standard Methods"
        print(f"\nEvaluating {group_name} in: {base_dir}")
        
        metrics = {m: {"avg_nees": [], "rmse_pos": [], "avg_nis": [], "avg_iou": [], "nees_in_interval": [], "nis_in_interval": []} for m in method_list}
        
        for method in method_list:
            print(f"\n  -> Processing method: {method}")
            method_dir = base_dir / method
            if not method_dir.exists():
                 logging.warning(f"No results for method {method}")
                 continue
                 
            result_files = sorted(list(method_dir.glob("*_results.pkl")))
            print(f"     Found {len(result_files)} result files. Loading and computing metrics...")
            
            for i, res_file in enumerate(result_files, 1):
                 if i % 10 == 0 or i == 1 or i == len(result_files):
                     print(f"       - Analyzing run {i}/{len(result_files)}...", end='\r')
                 with open(res_file, "rb") as f:
                     sim_result = pickle.load(f)
                     
                 try:
                     consistency_analyzer = create_consistency_analysis_from_sim_result(sim_result)
                     nees_data = consistency_analyzer.get_nees(indices='all')
                     methods_nees = nees_data.a if nees_data else None
                     nees_in_interval = nees_data.in_interval * 100 if nees_data else None
                     
                     nis_data = consistency_analyzer.get_nis(indices='all')
                     methods_nis = nis_data.a if nis_data else None
                     nis_in_interval = nis_data.in_interval * 100 if nis_data else None
                     
                     rmse_pos = None
                     if hasattr(consistency_analyzer, 'x_err_gauss') and consistency_analyzer.x_err_gauss is not None:
                         pos_errs = np.array([e.mean[:2] for e in consistency_analyzer.x_err_gauss.values])
                         rmse_pos = np.sqrt(np.mean(np.sum(pos_errs**2, axis=1)))
                         
                     # Compute IoU 
                     iou_list = []
                     config = sim_result.config
                     pca_params = None
                     if hasattr(config.tracker, 'PCA_parameters_path') and config.tracker.PCA_parameters_path:
                         pca_path = Path(PROJECT_ROOT) / config.tracker.PCA_parameters_path
                         if pca_path.exists():
                             pca_params = np.load(pca_path)
                             
                     for t_idx in range(len(sim_result.tracker_results_ts.values)):
                         if sim_result.ground_truth_ts and t_idx < len(sim_result.ground_truth_ts.values):
                             gt_state = sim_result.ground_truth_ts.values[t_idx]
                             est_state = sim_result.tracker_results_ts.values[t_idx].state_posterior.mean
                             
                             gt_shape = compute_exact_vessel_shape_global(gt_state, config.extent.shape_coords_body)
                             est_shape = compute_estimated_shape_global(est_state, config, pca_params)
                             
                             if isinstance(gt_shape, tuple) and len(gt_shape) == 2:
                                 gt_x, gt_y = gt_shape
                             else:
                                 gt_x, gt_y = gt_shape[0, :], gt_shape[1, :]
                                 
                             if isinstance(est_shape, tuple) and len(est_shape) == 2:
                                 est_x, est_y = est_shape
                             else:
                                 est_x, est_y = est_shape[0, :], est_shape[1, :]
                                 
                             iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                             iou_list.append(iou)
                             
                     # Store config value for this method (assumes all runs in a method share the same config)
                     if "use_arc_length_residual" not in metrics[method]:
                         metrics[method]["use_arc_length_residual"] = getattr(config.tracker, 'use_arc_length_residual', None)
                     
                     if iou_list:
                         metrics[method]["avg_iou"].append(np.mean(iou_list))
                         
                     if methods_nees is not None: metrics[method]["avg_nees"].append(methods_nees)
                     if nees_in_interval is not None: metrics[method]["nees_in_interval"].append(nees_in_interval)
                     if rmse_pos is not None: metrics[method]["rmse_pos"].append(rmse_pos)
                     if methods_nis is not None: metrics[method]["avg_nis"].append(methods_nis)
                     if nis_in_interval is not None: metrics[method]["nis_in_interval"].append(nis_in_interval)
    
                 except Exception as e:
                     logging.error(f"Error analyzing {res_file}: {e}")
                     
        # Summarize, Plot, and Save to JSON
        title_tag = "Tuning" if is_tuning else ""
        print(f"\n--- Monte Carlo Analysis {title_tag}: {experiment_name} ---")
        
        summary_results = []
        
        for method in method_list:
             if metrics[method]["avg_nees"]:
                 mean_nees = np.mean(metrics[method]["avg_nees"])
                 std_nees = np.std(metrics[method]["avg_nees"])
                 mean_nees_in = np.mean(metrics[method]["nees_in_interval"]) if metrics[method]["nees_in_interval"] else 0.0
                 std_nees_in = np.std(metrics[method]["nees_in_interval"]) if metrics[method]["nees_in_interval"] else 0.0
                 
                 mean_rmse = np.mean(metrics[method]["rmse_pos"])
                 std_rmse = np.std(metrics[method]["rmse_pos"])
                 
                 mean_nis = np.mean(metrics[method]["avg_nis"])
                 std_nis = np.std(metrics[method]["avg_nis"]) if metrics[method]["avg_nis"] else 0.0
                 mean_nis_in = np.mean(metrics[method]["nis_in_interval"]) if metrics[method]["nis_in_interval"] else 0.0
                 std_nis_in = np.std(metrics[method]["nis_in_interval"]) if metrics[method]["nis_in_interval"] else 0.0
                 
                 mean_iou = np.mean(metrics[method]["avg_iou"]) if metrics[method]["avg_iou"] else 0.0
                 std_iou = np.std(metrics[method]["avg_iou"]) if metrics[method]["avg_iou"] else 0.0
                 
                 print(f"\nMethod: {method}")
                 print(f"  Runs Analyzed: {len(metrics[method]['avg_nees'])}")
                 print(f"  Avg NEES: {mean_nees:.4f} \u00B1 {std_nees:.4f} ({mean_nees_in:.1f}% \u00B1 {std_nees_in:.1f}% in 95% CI)")
                 print(f"  Avg NIS: {mean_nis:.4f} \u00B1 {std_nis:.4f} ({mean_nis_in:.1f}% \u00B1 {std_nis_in:.1f}% in 95% CI)")
                 print(f"  Pos RMSE: {mean_rmse:.4f} \u00B1 {std_rmse:.4f}")
                 print(f"  Avg IoU: {mean_iou:.4f} \u00B1 {std_iou:.4f}")
                 
                 summary_results.append({
                     "method": method,
                     "use_arc_length_residual": metrics[method].get("use_arc_length_residual", None),
                     "runs_analyzed": len(metrics[method]['avg_nees']),
                     "avg_nees": float(mean_nees),
                     "std_nees": float(std_nees),
                     "nees_in_interval_95": float(mean_nees_in),
                     "std_nees_in_interval_95": float(std_nees_in),
                     "avg_nis": float(mean_nis),
                     "std_nis": float(std_nis),
                     "nis_in_interval_95": float(mean_nis_in),
                     "std_nis_in_interval_95": float(std_nis_in),
                     "rmse_pos": float(mean_rmse),
                     "std_rmse_pos": float(std_rmse),
                     "avg_iou": float(mean_iou),
                     "std_iou": float(std_iou)
                 })
                 
        output_file = "tuning_summary_metrics.json" if is_tuning else "mc_summary_metrics.json"
        with open(base_dir / output_file, "w") as f:
            json.dump(summary_results, f, indent=4)
    
        # Plot Boxplots for RMSE
        plt.figure(figsize=(10, 6))
        data_to_plot = [metrics[m]["rmse_pos"] for m in method_list if metrics[m]["rmse_pos"]]
        labels = [m for m in method_list if metrics[m]["rmse_pos"]]
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel("Positional RMSE")
            plt.title(f"RMSE Distribution across Monte Carlo {title_tag} Runs ({experiment_name})")
            plot_prefix = "tuning_" if is_tuning else ""
            plot_path = base_dir / f"{plot_prefix}rmse_boxplot_{experiment_name}.png"
            plt.savefig(plot_path)
            print(f"\nSaved plot to {plot_path}")
            plt.close()
        else:
            print(f"No RMSE data to plot for {title_tag}.")

    process_method_list(methods, experiment_dir, is_tuning=False)
    if tuning_methods:
        process_method_list(tuning_methods, tuning_dir, is_tuning=True)


if __name__ == "__main__":
    EXPERIMENT_NAME = "exp1_linear_noise015"
    METHODS = []
    
    analyze_mc_experiment(EXPERIMENT_NAME, METHODS)