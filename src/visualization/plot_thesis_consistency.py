import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
from scipy.stats import chi2

# Add project root to path
SRC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.global_project_paths import SIMDATA_PATH, FIGURES_PATH
from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result

FILTER_COLORS = {
    'EKF': 'C0',
    'IEKF': 'C1',
    'IMPLICIT_EKF': 'C2',
    'IMPLICIT_IEKF': 'C3'
}

FILTER_LABELS = {
    'EKF': 'Explicit EKF',
    'IEKF': 'Explicit IEKF',
    'IMPLICIT_EKF': 'Implicit EKF',
    'IMPLICIT_IEKF': 'Implicit IEKF'
}

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
})

def main():
    parser = argparse.ArgumentParser(description="Plot NEES/NIS or ANEES/ANIS")
    parser.add_argument('--mode', type=str, choices=['single', 'mc'], default='single', help="Plotting mode: single run or MC runs")
    parser.add_argument('--exp_prefix', type=str, default='exp1_linear', help="Prefix for the experiment (e.g. exp1_linear, exp2_linear)")
    args = parser.parse_args()
    
    print(f"Starting consistency plot generation...")
    print(f"Mode: {args.mode}")
    print(f"Experiment prefix: {args.exp_prefix}")
    
    # filters in desired legend order
    filter_keys = ['EKF', 'IEKF', 'IMPLICIT_EKF', 'IMPLICIT_IEKF']
    
    sim_data = {}
    
    if args.mode == 'single':
        # Single run code (existing logic)
        for key in filter_keys:
            print(f"Processing single run for filter: {key}...")
            if "real_data" in args.exp_prefix:
                target_name = f"{args.exp_prefix}_{key.lower()}.pkl"
            else:
                target_name = f"{args.exp_prefix}_noiseless_{key.lower()}.pkl"
            paths = list(SIMDATA_PATH.rglob(target_name))
            
            selected_path = None
            for p in paths:
                if 'single_runs' in p.parts:
                    selected_path = p
                    break
            if selected_path is None and len(paths) > 0:
                selected_path = paths[0]
                
            if selected_path is None:
                print(f"  -> File not found for {key}")
                continue
            
            print(f"  -> Loading {selected_path.name}")
                
            with open(selected_path, 'rb') as f:
                sim = pickle.load(f)
                analysis = create_consistency_analysis_from_sim_result(sim)
                nees = analysis.get_nees()
                nis = analysis.get_nis()
                
                sim_data[key] = {
                    'nees': nees.mahal_dist_tseq.values_as_array(),
                    'nis': nis.mahal_dist_tseq.values_as_array(),
                    'lower_nees': nees.low_med_upp_tseq.values_as_array()[:, 0] if nees.aconf else None,
                    'upper_nees': nees.low_med_upp_tseq.values_as_array()[:, 2] if nees.aconf else None,
                    'lower_nis': nis.low_med_upp_tseq.values_as_array()[:, 0] if nis.aconf else None,
                    'upper_nis': nis.low_med_upp_tseq.values_as_array()[:, 2] if nis.aconf else None,
                }
    else: # mode == 'mc'
        experiment_dir = PROJECT_ROOT / "data" / "results" / "mc_experiments" / f"{args.exp_prefix}_noise015"
        print(f"Looking for MC results in: {experiment_dir}")
        for key in filter_keys:
            print(f"Processing MC runs for filter: {key}...")
            method_dir = experiment_dir / key.lower()
            if not method_dir.exists():
                print(f"  -> Directory not found for {key}")
                continue
            
            result_files = list(method_dir.glob("*_results.pkl"))
            if not result_files:
                 print(f"  -> No result files found for {key}")
                 continue
            
            print(f"  -> Found {len(result_files)} result files. Processing...")
                 
            nees_list = []
            nis_list = []
            nx = None
            nz_list = []
            
            for res_file in result_files:
                with open(res_file, 'rb') as f:
                    sim = pickle.load(f)
                    analysis = create_consistency_analysis_from_sim_result(sim)
                    nees = analysis.get_nees()
                    nis = analysis.get_nis()
                    
                    nees_arr = nees.mahal_dist_tseq.values_as_array()
                    nis_arr = nis.mahal_dist_tseq.values_as_array()
                    
                    nees_list.append(nees_arr)
                    nis_list.append(nis_arr)
                    
                    if nx is None:
                        nx = nees.dofs[0] if isinstance(nees.dofs, list) else nees.dofs
                    if not nz_list:
                        nz_list = nis.dofs
                        
            try:
                nees_stack = np.array(nees_list)
                nis_stack = np.array(nis_list)
            except ValueError:
                print(f"Skipping {key} because runs have different lengths.")
                continue
                
            anees = np.mean(nees_stack, axis=0)
            anis = np.mean(nis_stack, axis=0)
            
            M = len(result_files)
            alpha = 0.05
            lower_anees = np.full(len(anees), chi2.ppf(alpha / 2, M * nx) / M)
            upper_anees = np.full(len(anees), chi2.ppf(1 - alpha / 2, M * nx) / M)
            
            lower_anis = np.zeros(len(anis))
            upper_anis = np.zeros(len(anis))
            for i, nz in enumerate(nz_list):
                if i < len(anis):
                    lower_anis[i] = chi2.ppf(alpha / 2, M * nz) / M
                    val_upper = chi2.ppf(1 - alpha / 2, M * nz) / M
                    upper_anis[i] = val_upper
            
            sim_data[key] = {
                'nees': anees,
                'nis': anis,
                'lower_nees': lower_anees,
                'upper_nees': upper_anees,
                'lower_nis': lower_anis,
                'upper_nis': upper_anis,
            }

    out_dir = FIGURES_PATH / 'consistency_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if "real_data" in args.exp_prefix:
        title_suffix = 'Real Data'
    else:
        title_suffix = 'Noiseless' if args.mode == 'single' else 'Noisy (MC)'
        
    metric_nees_name = 'NEES' if args.mode == 'single' else 'ANEES'
    metric_nis_name = 'NIS' if args.mode == 'single' else 'ANIS'
    
    # --- Plot Combined NEES & NIS ---
    fig, (ax_nees, ax_nis) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    exp_display = args.exp_prefix.replace('_', ' ').title()
    fig.suptitle(f'{metric_nees_name} and {metric_nis_name}, {title_suffix}')
    
    bounds_plotted = False
    for key in filter_keys:
        if key not in sim_data:
            continue
        data = sim_data[key]
        val = data['nees']
        steps = np.arange(len(val))
        ax_nees.plot(steps, val, label=FILTER_LABELS[key], color=FILTER_COLORS[key])
        
        if not bounds_plotted and data['lower_nees'] is not None:
            ax_nees.plot(steps, data['lower_nees'], color='k', linestyle='--', alpha=0.9, linewidth=1.5, label='95% Confidence Bounds')
            ax_nees.plot(steps, data['upper_nees'], color='k', linestyle='--', alpha=0.9, linewidth=1.5)
            bounds_plotted = True
            
    ax_nees.set_ylabel(metric_nees_name)
    ax_nees.set_yscale('log')
    ax_nees.grid(True, which='both', linestyle=':', alpha=0.6)
    
    bounds_plotted = False
    for key in filter_keys:
        if key not in sim_data:
            continue
        data = sim_data[key]
        val = data['nis']
        steps = np.arange(len(val))
        ax_nis.plot(steps, val, label=FILTER_LABELS[key], color=FILTER_COLORS[key])
        
        if not bounds_plotted and data['lower_nis'] is not None:
            ax_nis.plot(steps, data['lower_nis'], color='k', linestyle='--', alpha=0.9, linewidth=1.5, label='95% Confidence Bounds')
            ax_nis.plot(steps, data['upper_nis'], color='k', linestyle='--', alpha=0.9, linewidth=1.5)
            bounds_plotted = True
            
    ax_nis.set_xlabel('Timestep')
    ax_nis.set_ylabel(metric_nis_name)
    ax_nis.set_yscale('log')
    ax_nis.grid(True, which='both', linestyle=':', alpha=0.6)
    
    handles, labels = ax_nees.get_legend_handles_labels()
    if '95% Confidence Bounds' in labels:
        idx = labels.index('95% Confidence Bounds')
        handles.insert(0, handles.pop(idx))
        labels.insert(0, labels.pop(idx))

    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=True)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    
    if "real_data" in args.exp_prefix:
        out_file = f'{args.exp_prefix}_consistency.pdf'
    else:
        out_file = f'{args.exp_prefix}_noiseless_consistency.pdf' if args.mode == 'single' else f'{args.exp_prefix}_noisy_mc_consistency.pdf'
    
    fig.savefig(out_dir / out_file)
    plt.close(fig)
    print(f"Saved figure to {out_dir / out_file}")

if __name__ == '__main__':
    main()
