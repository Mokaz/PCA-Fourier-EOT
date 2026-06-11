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

def analyze_mc_experiment(experiment_name, methods, ignore_tuning=True):
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
    if not ignore_tuning and tuning_dir.exists():
        tuning_methods = [d.name for d in tuning_dir.iterdir() if d.is_dir()]

    def process_method_list(method_list, base_dir, is_tuning=False):
        if not method_list:
            return
            
        group_name = "Tuning Runs" if is_tuning else "Standard Methods"
        print(f"\nEvaluating {group_name} in: {base_dir}")
        
        metrics = {m: {"rmse_pos": [], "avg_iou": [], "track_lost_flags": [], "failed_runs": [], "run_mean_nees": [], "run_mean_anis": []} for m in method_list}
        
        for method in method_list:
            print(f"\n  -> Processing method: {method}")
            method_dir = base_dir / method
            if not method_dir.exists():
                 logging.warning(f"No results for method {method}")
                 continue
                 
            result_files = sorted(list(method_dir.glob("*_results.pkl")))
            print(f"     Found {len(result_files)} result files. Loading and computing metrics...")
            
            # --- NEW: Lists to hold the full time-series arrays for ensemble averaging ---
            all_nees_ts = []
            all_nis_ts = []
            nx_dof = None
            nz_dof_ts = None
            
            for i, res_file in enumerate(result_files, 1):
                 if i % 10 == 0 or i == 1 or i == len(result_files):
                     print(f"       - Analyzing run {i}/{len(result_files)}...", end='\r')
                 with open(res_file, "rb") as f:
                     sim_result = pickle.load(f)
                     
                 try:
                     consistency_analyzer = create_consistency_analysis_from_sim_result(sim_result)
                     
                     # 1. Collect NEES Time Series
                     nees_data = consistency_analyzer.get_nees(indices='all')
                     if nees_data:
                         all_nees_ts.append(nees_data.mahal_dist_tseq.values)
                         if nx_dof is None:
                             nx_dof = nees_data.dofs[0] # Usually 12 for your state
                             
                     # 2. Collect NIS Time Series
                     nis_data = consistency_analyzer.get_nis(indices='all')
                     if nis_data:
                         all_nis_ts.append(nis_data.mahal_dist_tseq.values)
                         if nz_dof_ts is None:
                             nz_dof_ts = nis_data.dofs # Number of LiDAR hits varies per frame!
                     
                     # 3. Collect RMSE
                     rmse_pos = None
                     if hasattr(consistency_analyzer, 'x_err_gauss') and consistency_analyzer.x_err_gauss is not None:
                         pos_errs = np.array([e.mean[:2] for e in consistency_analyzer.x_err_gauss.values])
                         rmse_pos = np.sqrt(np.mean(np.sum(pos_errs**2, axis=1)))
                         metrics[method]["rmse_pos"].append(rmse_pos)
                         
                     # 4. Collect IoU 
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
                             
                     if "use_arc_length_residual" not in metrics[method]:
                         metrics[method]["use_arc_length_residual"] = getattr(config.tracker, 'use_arc_length_residual', None)
                     
                     if iou_list:
                         metrics[method]["avg_iou"].append(np.mean(iou_list))
                         
                         consecutive_low_iou = 0
                         track_lost = False
                         for iou_val in iou_list:
                             if iou_val < 0.5:
                                 consecutive_low_iou += 1
                                 if consecutive_low_iou >= 30:
                                     track_lost = True
                                     break
                             else:
                                 consecutive_low_iou = 0
                                 
                         metrics[method]["track_lost_flags"].append(track_lost)
                         if track_lost:
                             metrics[method]["failed_runs"].append(res_file.name)
    
                 except Exception as e:
                     logging.error(f"Error analyzing {res_file}: {e}")
                     
            print(f"\n     Finished loading. Computing Ensemble Metrics...")
            from scipy.stats import chi2
            M = len(all_nees_ts)
            
            # --- PROPER ENSEMBLE ANEES CALCULATION ---
            if M > 0 and nx_dof is not None:
                nees_matrix = np.array(all_nees_ts) # Shape: (M, K_frames)
                anees_ts = np.mean(nees_matrix, axis=0) # Average over M runs
                
                # Calculate M=100 Tightened Bounds
                lower_chi2, upper_chi2 = chi2.interval(0.95, df=M * nx_dof)
                lower_bound_nees = lower_chi2 / M
                upper_bound_nees = upper_chi2 / M
                
                nees_in_interval_pct = np.mean((anees_ts >= lower_bound_nees) & (anees_ts <= upper_bound_nees)) * 100
                overall_avg_nees = np.mean(anees_ts)
                metrics[method]["ensemble_anees"] = overall_avg_nees
                metrics[method]["ensemble_nees_in_interval"] = nees_in_interval_pct
                
                run_mean_nees = np.mean(nees_matrix, axis=1)
                metrics[method]["run_mean_nees"] = run_mean_nees.tolist()
                
                # (Optional) Save the ANEES line and bounds to disk here so you can plot it easily later!
                # np.savez(method_dir / "anees_plot_data.npz", anees=anees_ts, lower=lower_bound_nees, upper=upper_bound_nees)

            # --- PROPER ENSEMBLE ANIS CALCULATION ---
            if M > 0 and nz_dof_ts is not None:
                try:
                    nis_matrix = np.array(all_nis_ts)
                    anis_ts = np.mean(nis_matrix, axis=0)
                    
                    # NIS degrees of freedom changes every frame (based on # of LiDAR hits)
                    in_interval_count = 0
                    for k, nz in enumerate(nz_dof_ts):
                        if k >= len(anis_ts): break
                        l_chi2, u_chi2 = chi2.interval(0.95, df=M * nz)
                        if (l_chi2 / M) <= anis_ts[k] <= (u_chi2 / M):
                            in_interval_count += 1
                            
                    nis_in_interval_pct = (in_interval_count / len(nz_dof_ts)) * 100
                    metrics[method]["ensemble_anis"] = np.mean(anis_ts)
                    metrics[method]["ensemble_nis_in_interval"] = nis_in_interval_pct
                    
                    run_mean_anis = np.mean(nis_matrix, axis=1)
                    metrics[method]["run_mean_anis"] = run_mean_anis.tolist()
                except ValueError as ve:
                    print(f"Warning: Could not process NIS matrix: {ve}")
        
        # Summarize, Plot, and Save to JSON
        title_tag = "Tuning" if is_tuning else ""
        print(f"\n--- Monte Carlo Analysis {title_tag}: {experiment_name} ---")
        
        summary_results = []
        
        for method in method_list:
             if metrics[method].get("ensemble_anees"):
                 mean_nees = metrics[method]["ensemble_anees"]
                 nees_in_interval = metrics[method]["ensemble_nees_in_interval"]
                 median_nees = np.median(metrics[method]["run_mean_nees"]) if metrics[method].get("run_mean_nees") else 0.0
                 
                 mean_rmse = np.mean(metrics[method]["rmse_pos"])
                 std_rmse = np.std(metrics[method]["rmse_pos"])
                 median_rmse = np.median(metrics[method]["rmse_pos"])
                 
                 mean_nis = metrics[method].get("ensemble_anis", 0.0)
                 nis_in_interval = metrics[method].get("ensemble_nis_in_interval", 0.0)
                 median_nis = np.median(metrics[method]["run_mean_anis"]) if metrics[method].get("run_mean_anis") else 0.0
                 
                 mean_iou = np.mean(metrics[method]["avg_iou"]) if metrics[method]["avg_iou"] else 0.0
                 std_iou = np.std(metrics[method]["avg_iou"]) if metrics[method]["avg_iou"] else 0.0
                 median_iou = np.median(metrics[method]["avg_iou"]) if metrics[method]["avg_iou"] else 0.0
                 
                 track_loss_flags = metrics[method].get("track_lost_flags", [])
                 track_loss_rate = (sum(track_loss_flags) / len(track_loss_flags)) * 100 if track_loss_flags else 0.0
                 failed_runs = metrics[method].get("failed_runs", [])
                 
                 print(f"\nMethod: {method}")
                 print(f"  Avg ANEES: {mean_nees:.4f} (Median: {median_nees:.4f}), {nees_in_interval:.1f}% in 95% CI")
                 print(f"  Avg ANIS: {mean_nis:.4f} (Median: {median_nis:.4f}), {nis_in_interval:.1f}% in 95% CI")
                 print(f"  Pos RMSE: {mean_rmse:.4f} \u00B1 {std_rmse:.4f} (Median: {median_rmse:.4f})")
                 print(f"  Avg IoU: {mean_iou:.4f} \u00B1 {std_iou:.4f} (Median: {median_iou:.4f})")
                 print(f"  Track Loss Rate: {track_loss_rate:.1f}% ({len(failed_runs)} runs)")
                 if failed_runs:
                     print(f"  Failed runs: {failed_runs[:5]}{'...' if len(failed_runs) > 5 else ''}")
                 
                 summary_results.append({
                     "method": method,
                     "use_arc_length_residual": metrics[method].get("use_arc_length_residual", None),
                     "runs_analyzed": len(metrics[method]["rmse_pos"]),
                     "track_loss_rate": float(track_loss_rate),
                     "failed_runs": failed_runs,
                     "avg_anees": float(mean_nees),
                     "median_anees": float(median_nees),
                     "anees_in_interval_95": float(nees_in_interval),
                     "avg_anis": float(mean_nis),
                     "median_anis": float(median_nis),
                     "anis_in_interval_95": float(nis_in_interval),
                     "rmse_pos": float(mean_rmse),
                     "median_rmse_pos": float(median_rmse),
                     "std_rmse_pos": float(std_rmse),
                     "avg_iou": float(mean_iou),
                     "median_iou": float(median_iou),
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
    # EXPERIMENT_NAME = "exp1_linear_noise015"
    # EXPERIMENT_NAME = "exp2_complex_maneuvers_noise015"
    EXPERIMENT_NAME = "exp3_progressive_noise015"
    METHODS = []
    
    analyze_mc_experiment(EXPERIMENT_NAME, METHODS, ignore_tuning=True)
