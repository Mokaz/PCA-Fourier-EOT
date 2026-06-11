# holoviz_dashboard.py
import panel as pn
import plotly.graph_objects as go
import numpy as np
import pickle
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import hvplot.pandas
import pandas as pd

from bokeh.util.serialization import make_globally_unique_id

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.analysis.analysis_utils import create_consistency_analysis_from_sim_result
from src.global_project_paths import SIMDATA_PATH
from src.senfuslib.plotting import (
    interactive_show_consistency, 
    show_consistency, 
    interactive_show_error,
    matplotlib_show_consistency,
    matplotlib_show_error
)
from src.visualization.plotly_offline_generator import (
    generate_plotly_fig_for_frame,
    generate_initial_plotly_fig,
)
from src.visualization.cost_landscape_component import CostLandscapeComponent
from src.states.states import State_PCA, State_GP 
from src.utils.tools import calculate_body_angles
from src.utils.geometry_utils import compute_estimated_shape_global, compute_exact_vessel_shape_global, calculate_iou

ASSETS_DIR = Path(__file__).parent / 'assets'
MC_ROOT = PROJECT_ROOT / "data" / "results" / "mc_experiments"

js_files = {
    'jquery': 'assets/jquery-1.11.1.min.js',
    'goldenlayout': 'assets/goldenlayout.min.js',
}
css_files = [
    'assets/goldenlayout-base.css',
    'assets/goldenlayout-light-theme.css',
]
pn.extension('plotly', 'tabulator', js_files=js_files, css_files=css_files)

@pn.cache(max_items=5)
def load_data(target_path):
    """Loads simulation result and performs initial analysis. Cached for performance."""
    if not target_path:
        return None
    print(f"Loading and processing {target_path}...")
    
    target = Path(target_path)
    # Check if target_path is an absolute path (from MC selection)
    if target.is_absolute() and target.exists():
        pkl_path = target
    else:
        base_path = Path(SIMDATA_PATH) / "single_runs"
        if not base_path.exists():
            base_path = Path(SIMDATA_PATH)
            
        pkl_path = None
        sim_name = target_path
        
        # Support backward compatibility and nested runs
        pkl_matches = list(base_path.rglob(f"{sim_name}.pkl"))
        if pkl_matches:
            pkl_path = pkl_matches[0]
        elif (base_path / sim_name).exists() and sim_name.endswith('.pkl'):
            pkl_path = base_path / sim_name
        else:
            # Fallback
            pkl_path = base_path / f"{sim_name}.pkl"

    if not pkl_path or not pkl_path.exists():
        print(f"Could not find file: {target_path}")
        return None

    with open(pkl_path, "rb") as f:
        sim_result = pickle.load(f)
    
    consistency_analyzer = create_consistency_analysis_from_sim_result(sim_result)
    config = sim_result.config
    
    # Guard PCA parameter loading
    pca_params = None
    if hasattr(config.tracker, 'PCA_parameters_path') and config.tracker.PCA_parameters_path:
        pca_path = Path(config.tracker.PCA_parameters_path)
        # handle relative paths relative to project root if they don't exist as absolute
        if not pca_path.is_absolute():
            pca_path = PROJECT_ROOT / pca_path
        if pca_path.exists():
            pca_params = np.load(pca_path)

    return {
        "sim_result": sim_result,
        "consistency_analyzer": consistency_analyzer,
        "config": config,
        "pca_params": pca_params,
    }

# --- Data Loading Logic ---
def load_summary_dataframe():
    single_runs_dir = Path(SIMDATA_PATH) / "single_runs"
    if single_runs_dir.exists():
        json_paths = sorted(single_runs_dir.rglob("*.json"))
    else:
        json_paths = []
        
    data = []
    for jp in json_paths:
        if jp.name.endswith('_config.json'):
            continue
        try:
            with open(jp, 'r') as f:
                json_data = json.load(f)
                
                # Add modification date
                try:
                    mtime = jp.stat().st_mtime
                    from datetime import datetime
                    json_data["date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
                
                data.append(json_data)
        except Exception:
            continue
            
    if not data:
        # Fallback empty dataframe
        return pd.DataFrame([{"name": "No data found"}])
        
    df = pd.DataFrame(data)
    if "name" not in df.columns:
        if "filename" in df.columns:
            df["name"] = df["filename"].apply(lambda x: Path(x).stem)
            
    cols = df.columns.tolist()
    first_cols = []
    if 'date' in cols:
        first_cols.append('date')
    if 'name' in cols:
        first_cols.append('name')
        
    other_cols = [c for c in cols if c not in first_cols]
    df = df[first_cols + other_cols]
            
    return df

summary_df = load_summary_dataframe()
results_table = pn.widgets.Tabulator(
    summary_df, 
    pagination='local',
    page_size=30,
    selectable=1,
    sizing_mode='stretch_both',
    hidden_columns=['filename']
)

available_columns = summary_df.columns.tolist()
results_column_selector = pn.widgets.MultiChoice(
    name='Results Columns (Metrics)',
    options=available_columns,
    value=[c for c in available_columns if c != 'filename'],
    sizing_mode='stretch_width'
)

def update_results_columns(*events):
    new_cols = results_column_selector.value
    hidden_cols = [c for c in results_column_selector.options if c not in new_cols]
    if 'filename' not in hidden_cols and 'filename' in summary_df.columns: 
        hidden_cols.append('filename')
    results_table.hidden_columns = hidden_cols

results_column_selector.param.watch(update_results_columns, 'value')

results_column_accordion = pn.Accordion(
    ('Results Table Columns', results_column_selector),
    active=[],
    sizing_mode='stretch_width'
)

# --- Widgets ---
# --- Status Indicator ---
global_loading_spinner = pn.indicators.LoadingSpinner(value=False, width=30, height=30, sizing_mode='fixed')
pn.state.sync_busy(global_loading_spinner)

global_status_text = pn.pane.Markdown("**Ready**", sizing_mode='stretch_width', margin=(10, 0, 0, 5), min_height=30)

status_row = pn.Row(
    global_loading_spinner,
    global_status_text,
    sizing_mode='stretch_width',
    min_height=40,
    align='center'
)

def set_status(msg):
    global_status_text.object = f"**{msg}**"
    print(f"---> [Status] {msg}")

def _clear_status_on_idle(e):
    if not e.new:  # not busy
        global_status_text.object = "**Ready**"

pn.state.param.watch(_clear_status_on_idle, 'busy')

# --- Main file routing param ---
active_file_path = pn.widgets.TextInput(value='', visible=False)

data_source_mode = pn.widgets.RadioButtonGroup(
    name='Data Source', options=['Single Run', 'Monte Carlo'], value='Single Run', sizing_mode='stretch_width'
)

file_selector = pn.widgets.Select(
    name='Select Simulation File', 
    options=[None] + summary_df["name"].tolist() if "name" in summary_df else [None], 
    visible=True,
    sizing_mode='stretch_width'
)

refresh_files_button = pn.widgets.Button(
    name='Refresh Files',
    button_type='primary',
    sizing_mode='stretch_width'
)

# --- Monte Carlo Data Selectors ---
mc_config_mode_selector = pn.widgets.Select(name='MC Config Type', options=['Standard Methods', 'Tuning Runs'], sizing_mode='stretch_width')
mc_experiment_selector = pn.widgets.Select(name='MC Experiment', options=[], sizing_mode='stretch_width')
mc_method_selector = pn.widgets.Select(name='MC Method', options=[], sizing_mode='stretch_width')
mc_run_selector = pn.widgets.Select(name='MC Run', options=[], sizing_mode='stretch_width')

def update_mc_experiments(*events):
    if not MC_ROOT.exists():
        mc_experiment_selector.options = []
        return
    exps = sorted([d.name for d in MC_ROOT.iterdir() if d.is_dir()])
    mc_experiment_selector.options = exps
    if exps and mc_experiment_selector.value not in exps:
        mc_experiment_selector.value = exps[0]
    if exps:
        update_mc_methods()
        if data_source_mode.value == 'Monte Carlo':
            update_file_list()

def update_mc_methods(*events):
    exp = mc_experiment_selector.value
    config_mode = mc_config_mode_selector.value
    if not exp:
        mc_method_selector.options = []
        return
        
    exp_dir = MC_ROOT / exp
    
    if config_mode == 'Standard Methods':
        methods = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and d.name != 'tuning' and d.name != 'results'])
    else:  # Tuning Runs
        tuning_dir = exp_dir / 'tuning'
        if tuning_dir.exists():
            methods = sorted([d.name for d in tuning_dir.iterdir() if d.is_dir()])
        else:
            methods = []
            
    mc_method_selector.options = methods
    if methods and mc_method_selector.value not in methods:
        mc_method_selector.value = methods[0]
    elif not methods:
        mc_method_selector.value = None
        
    update_mc_runs()

def update_mc_runs(*events):
    exp = mc_experiment_selector.value
    method = mc_method_selector.value
    config_mode = mc_config_mode_selector.value
    if not exp or not method:
        mc_run_selector.options = []
        return
        
    if config_mode == 'Standard Methods':
        runs_dir = MC_ROOT / exp / method
    else:
        runs_dir = MC_ROOT / exp / 'tuning' / method
        
    runs = sorted([f.name for f in runs_dir.glob("*.pkl")])
    mc_run_selector.options = runs
    if runs and mc_run_selector.value not in runs:
        mc_run_selector.value = runs[0]

mc_config_mode_selector.param.watch(update_mc_methods, 'value')
mc_config_mode_selector.param.watch(lambda event: update_file_list() if data_source_mode.value == 'Monte Carlo' else None, 'value')
mc_experiment_selector.param.watch(update_mc_methods, 'value')
mc_experiment_selector.param.watch(lambda event: update_file_list() if data_source_mode.value == 'Monte Carlo' else None, 'value')
mc_method_selector.param.watch(update_mc_runs, 'value')

update_mc_experiments()

@pn.depends(data_source_mode.param.value, file_selector.param.value, mc_experiment_selector.param.value, mc_config_mode_selector.param.value, mc_method_selector.param.value, mc_run_selector.param.value, watch=True)
def sync_active_file(mode, single_run_val, exp_val, config_mode_val, method_val, mc_run_val):
    if mode == 'Single Run':
        active_file_path.value = str(single_run_val) if single_run_val else ''
    else:
        exp = mc_experiment_selector.value
        method = mc_method_selector.value
        run = mc_run_selector.value
        config_mode = mc_config_mode_selector.value
        if exp and method and run:
            if config_mode == 'Standard Methods':
                path = MC_ROOT / exp / method / run
            else:
                path = MC_ROOT / exp / 'tuning' / method / run
            active_file_path.value = str(path)
        else:
            active_file_path.value = ''

sync_active_file(data_source_mode.value, file_selector.value, mc_experiment_selector.value, mc_config_mode_selector.value, mc_method_selector.value, mc_run_selector.value)

single_run_controls = pn.Column(refresh_files_button, file_selector, sizing_mode='stretch_width')
mc_controls = pn.Column(mc_config_mode_selector, mc_experiment_selector, mc_method_selector, mc_run_selector, sizing_mode='stretch_width', visible=False)

def toggle_source_mode(event):
    if event.new == 'Single Run':
        single_run_controls.visible = True
        mc_controls.visible = False
    else:
        single_run_controls.visible = False
        mc_controls.visible = True
        sync_active_file('Monte Carlo', file_selector.value, mc_experiment_selector.value, mc_config_mode_selector.value, mc_method_selector.value, mc_run_selector.value)
    
    update_file_list()

data_source_mode.param.watch(toggle_source_mode, 'value')


def on_table_click(event):
    if len(event.new) > 0:
        selected_idx = event.new[0]
        try:
            selected_name = results_table.value.iloc[selected_idx]['name']
            if data_source_mode.value == 'Single Run':
                file_selector.value = selected_name
            else:
                # In Monte Carlo mode, selected_name is the "method" / "tuning configuration"
                if selected_name in mc_method_selector.options:
                    mc_method_selector.value = selected_name
        except Exception as e:
            print(f"Error selecting row: {e}")

results_table.param.watch(on_table_click, 'selection')

def on_dropdown_select(event):
    selected_name = event.new
    if selected_name:
        # Find index in the dataframe where name matches
        df = results_table.value
        matches = df.index[df['name'] == selected_name].tolist()
        if matches:
            results_table.selection = matches

file_selector.param.watch(on_dropdown_select, 'value')

def load_mc_summary_dataframe():
    exp = mc_experiment_selector.value
    config_mode = mc_config_mode_selector.value
    if not exp:
        return pd.DataFrame([{"name": "No MC Experiment selected"}])
    
    if config_mode == 'Standard Methods':
        summary_path = MC_ROOT / exp / "mc_summary_metrics.json"
    else:
        summary_path = MC_ROOT / exp / "tuning" / "tuning_summary_metrics.json"
        
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if "method" in df.columns:
            df["name"] = df["method"]
        return df
    else:
        return pd.DataFrame([{"name": f"No {summary_path.name} found. Run analyze_mc_experiments.py first."}])

def update_file_list(event=None):
    if data_source_mode.value == 'Single Run':
        new_df = load_summary_dataframe()
    else:
        new_df = load_mc_summary_dataframe()
        
    results_table.value = new_df
    
    current_val = file_selector.value
    
    if data_source_mode.value == 'Single Run':
        if "name" in new_df:
            file_selector.options = [None] + new_df["name"].tolist()
        else:
            file_selector.options = [None]
            
        if current_val in file_selector.options:
            file_selector.value = current_val
    else:
        # In MC mode, we don't strictly use file_selector, but we can keep it consistent
        file_selector.options = [None]
        file_selector.value = None

    # Update available columns for metrics table
    new_cols = new_df.columns.tolist()
    results_column_selector.options = new_cols
    current_selected_cols = results_column_selector.value
    # Keep previously selected cols that still exist, add new cols by default (or just keep what's valid)
    valid_cols = [c for c in current_selected_cols if c in new_cols]
    if len(valid_cols) == 0:
        valid_cols = [c for c in new_cols if c not in ('filename', 'method')]
    results_column_selector.value = valid_cols

# Initialize list
update_file_list()

refresh_files_button.on_click(update_file_list)

iterate_selector = pn.widgets.Select(
    name='Filter Iterate',
    options=['Final'],
    value='Final',
    visible=False,
    sizing_mode='stretch_width'
)

frame_player = pn.widgets.Player(
    name='Frame', start=0, end=0, step=1,
    visible=False, loop_policy='loop', interval=50,
    sizing_mode='stretch_width'
)

frame_input = pn.widgets.IntInput(
    name='Jump to Frame', value=0, start=0, step=1,
    visible=False, sizing_mode='stretch_width'
)

# Link player and input
frame_player.link(frame_input, value='value')
frame_input.link(frame_player, value='value')

# --- NEES Widgets ---
nees_group_selector = pn.widgets.MultiSelect(
    name='NEES Consistency Groups', 
    visible=False, 
    sizing_mode='stretch_width'
)

# --- Error Widgets ---
error_group_selector = pn.widgets.MultiSelect(
    name='Error Groups', 
    visible=False, 
    sizing_mode='stretch_width'
)

custom_states_selector = pn.widgets.MultiSelect(
    name='Custom States for NEES/Error',
    visible=False,
    sizing_mode='stretch_width'
)

# --- Covariance Widgets ---
COV_MATRIX_MAPPING = {
    'Posterior Covariance (P_k|k)': 'state_posterior',
    'Prior Covariance (P_k|k-1)': 'state_prior',
    'Innovation Covariance (S_k)': 'predicted_measurement',
}
cov_matrix_selector = pn.widgets.Select(
    name='Covariance Matrix to Display',
    options=list(COV_MATRIX_MAPPING.keys()),
    visible=False,
    sizing_mode='stretch_width'
)

# --- NIS Widgets ---
nis_field_selector = pn.widgets.Select(
    name='NIS Consistency',
    options=['all'],
    value='all',
    visible=False,
    sizing_mode='stretch_width'
)
# --- Cost Breakdown Widgets ---
cost_breakdown_toggle = pn.widgets.Toggle(
    name='Log Scale', 
    value=True, 
    width=100
)


# --- Plotting Control Widgets ---
plotting_divider = pn.layout.Divider(visible=False)
plotting_header = pn.pane.Markdown("### Plotting & Saving", visible=False)

x_min_slider = pn.widgets.FloatSlider(name='X min (East)', value=-60.0, start=-200.0, end=200.0, step=1.0, sizing_mode='stretch_width')
x_min_input = pn.widgets.FloatInput(value=-60.0, step=1.0, width=80)
x_min_slider.link(x_min_input, value='value')
x_min_input.link(x_min_slider, value='value')

x_max_slider = pn.widgets.FloatSlider(name='X max (East)', value=60.0, start=-200.0, end=200.0, step=1.0, sizing_mode='stretch_width')
x_max_input = pn.widgets.FloatInput(value=60.0, step=1.0, width=80)
x_max_slider.link(x_max_input, value='value')
x_max_input.link(x_max_slider, value='value')

y_min_slider = pn.widgets.FloatSlider(name='Y min (North)', value=-70.0, start=-200.0, end=200.0, step=1.0, sizing_mode='stretch_width')
y_min_input = pn.widgets.FloatInput(value=-70.0, step=1.0, width=80)
y_min_slider.link(y_min_input, value='value')
y_min_input.link(y_min_slider, value='value')

y_max_slider = pn.widgets.FloatSlider(name='Y max (North)', value=70.0, start=-200.0, end=200.0, step=1.0, sizing_mode='stretch_width')
y_max_input = pn.widgets.FloatInput(value=70.0, step=1.0, width=80)
y_max_slider.link(y_max_input, value='value')
y_max_input.link(y_max_slider, value='value')

keep_zoom_checkbox = pn.widgets.Checkbox(
    name='Keep zoom across frames/settings', 
    value=True, 
    sizing_mode='stretch_width'
)

preset_radio = pn.widgets.RadioButtonGroup(
    name='Presets', options=['linear', 'waypoints', 'waypoints2'], value='waypoints2', button_type='default', sizing_mode='stretch_width'
)

def set_preset_from_radio(event):
    # This function handles the standard change event.
    val = event.new if hasattr(event, 'new') else event
    if val in ('linear', 'waypoints'):
        x_min_input.value = -60.0
        x_max_input.value = 60.0
        y_min_input.value = -10.0
        y_max_input.value = 70.0
    elif val == 'waypoints2':
        x_min_input.value = -60.0
        x_max_input.value = 60.0
        y_min_input.value = -70.0
        y_max_input.value = 70.0

preset_radio.param.watch(set_preset_from_radio, 'value')

reset_preset_button = pn.widgets.Button(
    name='Reset to Current Preset', 
    button_type='default', 
    sizing_mode='stretch_width'
)

reset_counter = pn.widgets.IntInput(value=0, visible=False)

def reset_to_current_preset(event):
    set_preset_from_radio(preset_radio.value)
    # Increment counter to force a new uirevision hash and trigger an update
    reset_counter.value += 1

reset_preset_button.on_click(reset_to_current_preset)

show_angle_bounds_checkbox = pn.widgets.Checkbox(name='Show Angle Constraints', value=True, sizing_mode='stretch_width')
show_front_wall_checkbox = pn.widgets.Checkbox(name='Show Front Wall Constraints', value=True, sizing_mode='stretch_width')
show_centroid_depth_checkbox = pn.widgets.Checkbox(name='Show Centroid Depth Constraints', value=True, sizing_mode='stretch_width')

plot_settings_controls = pn.Accordion(
    (
        '2D View Settings',
        pn.Column(
            pn.pane.Markdown("**Constraint Visibility**"),
            show_angle_bounds_checkbox,
            show_front_wall_checkbox,
            show_centroid_depth_checkbox,
            pn.layout.Divider(),
            pn.pane.Markdown("**Plot Axis Ranges**"),
            preset_radio,
            reset_preset_button,
            keep_zoom_checkbox,
            pn.Row(x_min_slider, x_min_input, sizing_mode='stretch_width'),
            pn.Row(x_max_slider, x_max_input, sizing_mode='stretch_width'),
            pn.Row(y_min_slider, y_min_input, sizing_mode='stretch_width'),
            pn.Row(y_max_slider, y_max_input, sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        )
    ),
    active=[],
    visible=False,
    sizing_mode='stretch_width'
)

plot_backend_selector = pn.widgets.Select(
    name='Plotting Backend', 
    options=['Bokeh', 'Matplotlib'], 
    value='Bokeh',
    sizing_mode='stretch_width',
    visible=False
)

matplotlib_subplot_height_slider = pn.widgets.FloatSlider(
    name='Matplotlib Subplot Height', value=2.0, start=0.2, end=3.0, step=0.1,
    sizing_mode='stretch_width', visible=False
)

def update_backend_visibility(event):
    matplotlib_subplot_height_slider.visible = (event.new == 'Matplotlib')

plot_backend_selector.param.watch(update_backend_visibility, 'value')

save_filename_input = pn.widgets.TextInput(
    name='Save Filename (no ext)', 
    value='plot_output',
    placeholder='Enter filename...',
    sizing_mode='stretch_width',
    visible=False
)

save_button = pn.widgets.Button(
    name='Save Matplotlib Plots', 
    button_type='primary',
    sizing_mode='stretch_width',
    visible=False
)

save_status = pn.pane.Markdown("", sizing_mode='stretch_width', visible=False)

# --- Data Browser Widgets ---
data_browser_mode = pn.widgets.Select(
    name='Data Browser Mode',
    options=[
        'Current Frame Tracker Result', 
        'Current Frame GT', 
        'Current Frame State Error', 
        'Current Frame Measurement Error',
        'Consistency Analysis (Summary)',
        'Config', 
        'Copy Config Script',
        'Full Simulation Result (Summary)'
    ],
    value='Current Frame Tracker Result',
    sizing_mode='stretch_width',
    visible=False
)

# --- Interactive functions ---
@pn.depends(active_file_path.param.value, watch=True)
def update_widgets(target_path):
    if target_path:
        set_status(f"Loading '{target_path}' data...")

    loaded_data = load_data(target_path)
    if not loaded_data:
        frame_player.visible = False
        frame_input.visible = False
        plot_settings_controls.visible = False
        nees_group_selector.visible = False
        error_group_selector.visible = False
        cov_matrix_selector.visible = False
        nis_field_selector.visible = False
        custom_states_selector.visible = False
        
        data_browser_mode.visible = False
        plotting_divider.visible = False
        plotting_header.visible = False
        plot_backend_selector.visible = False
        save_filename_input.visible = False
        save_button.visible = False
        save_status.visible = False
        return

    sim_result = loaded_data["sim_result"]
    config = sim_result.config
    
    # 1. Inspect the first posterior state to determine type
    first_state_posterior = sim_result.tracker_results_ts.values[0].state_posterior.mean

    # 2. Define Groups and Labels dynamically based on State Type
    if isinstance(first_state_posterior, State_GP):
        # --- GP Configuration ---
        n_radii = len(first_state_posterior.radii)
        
        # NEES Groups
        dynamic_nees_mapping = {
            'Total NEES': 'all',
            'Position (x, y)': ['x', 'y'],
            'Heading (yaw)': ['yaw'],
            'Velocity (vx, vy)': ['vel_x', 'vel_y'],
            'Heading rate (yaw_rate)': ['yaw_rate'],
            'Shape (All Radii)': ['radii'] # Uses the 'radii' slice from State_GP
        }
        
        # Individual State Labels (for Custom Selector)
        custom_state_options = {
            'x': 'x', 'y': 'y', 'yaw': 'yaw', 
            'vel_x': 'vel_x', 'vel_y': 'vel_y', 'yaw_rate': 'yaw_rate'
        }
        # Add radii indices dynamically
        for i in range(n_radii):
            custom_state_options[f'radius_{i}'] = 6 + i

    elif isinstance(first_state_posterior, State_PCA):
        # --- PCA Configuration ---
        n_pca = config.tracker.N_pca
        
        # NEES Groups
        dynamic_nees_mapping = {
            'Total NEES': 'all',
            'Position (x, y)': ['x', 'y'],
            'Heading (yaw)': ['yaw'],
            'Velocity (vx, vy)': ['vel_x', 'vel_y'],
            'Heading rate (yaw_rate)': ['yaw_rate'],
            'Extent (L, W)': ['length', 'width'],
            'PCA Components': ['pca_coeffs'] # Uses the 'pca_coeffs' slice
        }
        
        # Individual State Labels
        custom_state_options = {
            'x': 'x', 'y': 'y', 'yaw': 'yaw', 
            'vel_x': 'vel_x', 'vel_y': 'vel_y', 'yaw_rate': 'yaw_rate',
            'length': 'length', 'width': 'width'
        }
        # Add PCA indices dynamically (assuming they map to the slice start)
        base_pca_idx = 8
        for i in range(n_pca):
            custom_state_options[f'pca_coeff_{i}'] = base_pca_idx + i

    else:
        # Fallback / Error safety
        dynamic_nees_mapping = {'Total NEES': 'all'}
        custom_state_options = {}

    # Create a copy for error mapping and remove 'Total NEES' if present
    error_mapping = dynamic_nees_mapping.copy()
    if 'Total NEES' in error_mapping:
        del error_mapping['Total NEES']

    # 3. Update Widgets
    frame_player.end = sim_result.config.sim.num_frames
    frame_player.value = 1
    frame_player.visible = True
    
    frame_input.end = sim_result.config.sim.num_frames
    frame_input.value = 1
    frame_input.visible = True
    plot_settings_controls.visible = True

    # Update NEES Group Selector
    nees_group_selector.options = dynamic_nees_mapping
    nees_group_selector.value = ['all']
    nees_group_selector.visible = True
    nees_group_selector.size = len(dynamic_nees_mapping)
    nees_group_selector.height = min(len(dynamic_nees_mapping) * 20, 400)

    # Update Error Group Selector
    error_group_selector.options = error_mapping
    error_group_selector.value = []
    error_group_selector.visible = True
    error_group_selector.size = len(error_mapping)
    error_group_selector.height = min(len(error_mapping) * 20, 400)

    # Update Custom NEES Selector
    custom_states_selector.options = custom_state_options
    custom_states_selector.value = []

    # Visbility
    cov_matrix_selector.visible = True
    nis_field_selector.visible = True

    data_browser_mode.visible = True
    plotting_divider.visible = True
    plotting_header.visible = True
    plot_backend_selector.visible = True
    save_filename_input.visible = True
    save_button.visible = True
    save_status.visible = True
    custom_states_selector.visible = True
    cost_breakdown_toggle.visible = True

def calculate_detailed_cost_breakdown(sim_result, consistency_analyzer):
    """
    Computes measurement cost and prior cost for the posterior state at each time step.
    Note: This re-evaluates the cost function, which might be expensive for long simulations.
    """
    measurement_costs = []
    prior_costs = []
    total_costs = []
    frames = []

    config = sim_result.config
    project_root = PROJECT_ROOT
    
    # Load PCA params if needed
    pca_params = None
    if hasattr(config.tracker, 'PCA_parameters_path') and config.tracker.PCA_parameters_path:
        pca_path = Path(config.tracker.PCA_parameters_path)
        if not pca_path.is_absolute():
            pca_path = project_root / pca_path
        if pca_path.exists():
            pca_params = np.load(pca_path)

    from src.dynamics.process_models import Model_PCA_CV
    from src.sensors.LidarModel import LidarMeasurementModel
    from src.tracker.BFGS import BFGS

    filter_dyn_model = Model_PCA_CV(
        x_pos_std_dev=config.tracker.pos_north_std_dev,
        y_pos_std_dev=config.tracker.pos_east_std_dev,
        yaw_std_dev=config.tracker.heading_std_dev,
        N_pca=config.tracker.N_pca,
        length_std_dev=config.tracker.length_std_dev,
        width_std_dev=config.tracker.width_std_dev
    )
    
    pca_eigenvectors = pca_params['eigenvectors'][:, :config.tracker.N_pca].real if pca_params else None
    pca_mean = pca_params['mean'] if pca_params else None

    sensor_model = LidarMeasurementModel(
        lidar_position=np.array(config.lidar.lidar_position),
        lidar_std_dev=config.tracker.lidar_std_dev,
        pca_mean=pca_mean,
        pca_eigenvectors=pca_eigenvectors,
        extent_cfg=config.extent
    )
    
    tracker = BFGS(dynamic_model=filter_dyn_model, lidar_model=sensor_model, config=config)

    for i, res in enumerate(sim_result.tracker_results_ts.values):
        try:
            state = res.state_posterior.mean
            state_pred = res.state_prior.mean
            P_pred = res.state_prior.cov
            
            meas_global = sim_result.measurements_global_ts.values[i]
            z_flat = meas_global.flatten('F')
            
            body_angles = calculate_body_angles(meas_global, state)
            tracker.body_angles = body_angles

            meas_cost, prior_cost = tracker.objective_function(
                state, state_pred, P_pred, z_flat, return_components=True
            )
            
            measurement_costs.append(meas_cost)
            prior_costs.append(prior_cost)
            total_costs.append(meas_cost + prior_cost)
            frames.append(i)
        except Exception as e:
            # Skip frames where calculation fails
            pass

    return pd.DataFrame({
        'Frame': frames,
        'Measurement Cost': measurement_costs,
        'Prior Cost': prior_costs,
        'Total Cost': total_costs
    }).set_index('Frame')

cost_auto_calc_checkbox = pn.widgets.Checkbox(name='Auto-Calculate Cost', value=False, margin=(5, 10, 5, 0))
cost_calc_button = pn.widgets.Button(name='Calculate Now', button_type='primary', width=120, margin=(5, 10, 5, 10))
cost_content_pane = pn.Column(pn.pane.Markdown("### Select a file to view Cost Breakdown"), sizing_mode="stretch_both")

def update_cost_breakdown_view(event=None, force=False):
    target_path = active_file_path.value
    log_scale = cost_breakdown_toggle.value

    if not target_path:
        cost_content_pane.objects = [pn.pane.Markdown("### Select a file to view Cost Breakdown")]
        return

    if not force and not cost_auto_calc_checkbox.value:
        cost_content_pane.objects = [pn.pane.Markdown(f"### Auto-calculation disabled for '{target_path}'. Click 'Calculate Now' below.")]
        return

    set_status("Calculating Cost Breakdown...")
    loaded_data = load_data(target_path)
    if not loaded_data:
        cost_content_pane.objects = [pn.pane.Markdown("### Error loading data.")]
        return

    if "cost_df" not in loaded_data:
        df = calculate_detailed_cost_breakdown(loaded_data["sim_result"], loaded_data["consistency_analyzer"])
        loaded_data["cost_df"] = df
    else:
        df = loaded_data["cost_df"]

    if df.empty:
        cost_content_pane.objects = [pn.pane.Markdown("### Could not calculate costs.")]
        return

    if log_scale:
        plot_df = np.log1p(df.clip(lower=0))
        y_label = "Log Cost"
    else:
        plot_df = df
        y_label = "Cost"

    plot = plot_df.hvplot.line(
        title=f"Cost Components over Time ({'Log Scale' if log_scale else 'Linear'})",
        ylabel=y_label,
        responsive=True,
        height=400,
        grid=True,
        line_width=2
    )

    cost_content_pane.objects = [pn.pane.HoloViews(plot, sizing_mode="stretch_both")]

cost_calc_button.on_click(lambda e: update_cost_breakdown_view(force=True))
active_file_path.param.watch(lambda e: update_cost_breakdown_view(force=False), 'value')
cost_breakdown_toggle.param.watch(lambda e: update_cost_breakdown_view(force=False), 'value')
cost_auto_calc_checkbox.param.watch(lambda e: update_cost_breakdown_view(force=True) if e.new else None, 'value')

def serialize_state_object(state_obj):
    """Helper to convert State_PCA/State_GP objects into a labeled dictionary."""
    cls_name = state_obj.__class__.__name__
    
    # Check for GP characteristics (radii)
    if hasattr(state_obj, 'radii'):
        return {
            "__type__": cls_name,
            "x": float(state_obj.x),
            "y": float(state_obj.y),
            "yaw": float(state_obj.yaw),
            "vel_x": float(state_obj.vel_x),
            "vel_y": float(state_obj.vel_y),
            "yaw_rate": float(state_obj.yaw_rate),
            "radii": state_obj.radii.tolist()
        }
        
    # Check for PCA characteristics (pca_coeffs)
    if hasattr(state_obj, 'pca_coeffs'):
        return {
            "__type__": cls_name,
            "x": float(state_obj.x),
            "y": float(state_obj.y),
            "yaw": float(state_obj.yaw),
            "vel_x": float(state_obj.vel_x),
            "vel_y": float(state_obj.vel_y),
            "yaw_rate": float(state_obj.yaw_rate),
            "length": float(state_obj.length),
            "width": float(state_obj.width),
            "pca_coeffs": state_obj.pca_coeffs.tolist()
        }
    
    return None

def safe_serialize(obj, max_depth=3, current_depth=0):
    """Recursively converts objects to JSON-serializable dicts/lists."""
    if current_depth > max_depth:
        return str(obj)
    
    if obj is None:
        return None
    
    if hasattr(obj, 'radii') or hasattr(obj, 'pca_coeffs') or obj.__class__.__name__ in ['State_PCA', 'State_GP']:
        res = serialize_state_object(obj)
        if res is not None:
            return res
        
    if isinstance(obj, (str, int, float, bool)):
        return obj
        
    if isinstance(obj, (np.integer, int)):
        return int(obj)
        
    if isinstance(obj, (np.floating, float)):
        return float(obj)
        
    if isinstance(obj, np.ndarray):
        if obj.size <= 2500:
            if obj.ndim > 1:
                matrix_str = np.array2string(
                    obj, 
                    separator=', ', 
                    precision=8, 
                    suppress_small=True, 
                    max_line_width=10000,
                    threshold=np.inf
                )
                return {"_matrix_data": matrix_str}
            return obj.tolist()
        return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
    
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x, max_depth, current_depth + 1) for x in obj]
    
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v, max_depth, current_depth + 1) for k, v in obj.items()}
        
    if hasattr(obj, '__dataclass_fields__'):
        return {k: safe_serialize(getattr(obj, k), max_depth, current_depth + 1) for k in obj.__dataclass_fields__}
        
    if hasattr(obj, '__dict__'):
        return {k: safe_serialize(v, max_depth, current_depth + 1) for k, v in obj.__dict__.items() if not k.startswith('_')}
        
    return str(obj)

def create_empty_figure(message="Select a file to begin"):
    """Helper to create a blank figure with a centered message."""
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text=message, 
        xref="paper", yref="paper", 
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="gray")
    )
    empty_fig.update_layout(
        xaxis={'visible': False}, yaxis={'visible': False},
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return empty_fig

persistent_plotly_pane = pn.pane.Plotly(
    object=create_empty_figure(),
    sizing_mode='stretch_both', 
    config={'responsive': True}
)

def update_plotly_view(frame_idx, target_path, iterate_sel, x_min, x_max, y_min, y_max, keep_zoom, reset_count, show_angle_bounds, show_front_wall, show_centroid_depth):
    loaded_data = load_data(target_path)

    if not loaded_data:
        persistent_plotly_pane.object = create_empty_figure()
        iterate_selector.visible = False
        return
    
    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]

    tracker_results = list(sim_result.tracker_results_ts.values)
    ground_truth_states = []
    if sim_result.ground_truth_ts is not None and len(sim_result.ground_truth_ts.values) > 0:
        ground_truth_states = list(sim_result.ground_truth_ts.values)

    if frame_idx >= len(tracker_results):
        return

    gt_state = ground_truth_states[frame_idx] if frame_idx < len(ground_truth_states) else None
    tracker_result = tracker_results[frame_idx]
    est_state = tracker_result.state_posterior.mean

    if frame_idx == 0:
        fig = generate_initial_plotly_fig(gt_state, est_state, config, pca_params)
    else:
        fig = generate_plotly_fig_for_frame(
            frame_idx=frame_idx,
            gt_state=gt_state,
            est_state=tracker_result.state_posterior.mean,
            z_lidar_cart=tracker_result.measurements.reshape((-1, 2)),
            ground_truth_history=ground_truth_states[:frame_idx] if ground_truth_states else [],
            config=config,
            pca_params=pca_params,
            virtual_constraints=None
        )

    if iterate_sel == 'All (Highlight EKF Update)':
        fig.update_traces(name='Estimated Extent (IEKF)', line_color='orangered', selector=dict(name='Estimated Extent'))

    def draw_vci(vci_list, label_suffix, is_first_iteration=True):
        if not vci_list:
            return
        lidar_pos = config.lidar.lidar_position
        
        virt_pts_x, virt_pts_y = [], []
        virt_pred_rays_x, virt_pred_rays_y = [], []
        virt_rays_x, virt_rays_y = [], []
        fw_arcs_x, fw_arcs_y = [], []
        cd_arcs_x, cd_arcs_y = [], []

        for vc in vci_list:
            c_type = vc.get('type', 'min_angle')
            
            if c_type in ['min_angle', 'max_angle'] or 'predicted_point' in vc:
                if not show_angle_bounds:
                    continue
                
                pt_global = vc['predicted_point']
                virt_pts_x.append(pt_global[0])
                virt_pts_y.append(pt_global[1])
                
                p_angle = np.arctan2(pt_global[1] - lidar_pos[1], pt_global[0] - lidar_pos[0])
                p_ray_end = np.array(lidar_pos) + config.lidar.max_distance * np.array([np.cos(p_angle), np.sin(p_angle)])
                virt_pred_rays_x.extend([lidar_pos[0], p_ray_end[0], None])
                virt_pred_rays_y.extend([lidar_pos[1], p_ray_end[1], None])
                
                if is_first_iteration:
                    angle = vc.get('measured_val', vc.get('measured_angle'))
                    if angle is not None:
                        ray_end = np.array(lidar_pos) + config.lidar.max_distance * np.array([np.cos(angle), np.sin(angle)])
                        virt_rays_x.extend([lidar_pos[0], ray_end[0], None])
                        virt_rays_y.extend([lidar_pos[1], ray_end[1], None])
            
            elif c_type == 'front_wall':
                if not show_front_wall:
                    continue
                radius = vc.get('measured_val')
                if radius is not None and is_first_iteration:
                    arc_angles = np.linspace(-np.pi, np.pi, 50)
                    for a in arc_angles:
                        fw_arcs_x.append(lidar_pos[0] + radius * np.cos(a))
                        fw_arcs_y.append(lidar_pos[1] + radius * np.sin(a))
                    fw_arcs_x.append(None)
                    fw_arcs_y.append(None)
                        
            elif c_type == 'centroid_depth':
                if not show_centroid_depth:
                    continue
                radius = vc.get('measured_val')
                if radius is not None and is_first_iteration:
                    arc_angles = np.linspace(-np.pi, np.pi, 50)
                    for a in arc_angles:
                        cd_arcs_x.append(lidar_pos[0] + radius * np.cos(a))
                        cd_arcs_y.append(lidar_pos[1] + radius * np.sin(a))
                    cd_arcs_x.append(None)
                    cd_arcs_y.append(None)
            
        if virt_pts_x:
            fig.add_trace(go.Scatter(x=virt_pts_y, y=virt_pts_x, mode='markers', name=f'Pred. Pts {label_suffix}', marker=dict(color='saddlebrown', size=8, symbol='diamond')))
            fig.add_trace(go.Scatter(x=virt_pred_rays_y, y=virt_pred_rays_x, mode='lines', name=f'Pred. Rays {label_suffix}', line=dict(color='saddlebrown', width=1)))
        
        if virt_rays_x:
            fig.add_trace(go.Scatter(x=virt_rays_y, y=virt_rays_x, mode='lines', name=f'Meas. Bounds {label_suffix}', line=dict(color='saddlebrown', width=1, dash='dash')))

        if fw_arcs_x:
            fig.add_trace(go.Scatter(x=fw_arcs_y, y=fw_arcs_x, mode='lines', name=f'Front Wall {label_suffix}', line=dict(color='red', width=1.5, dash='dot')))
            
        if cd_arcs_x:
            fig.add_trace(go.Scatter(x=cd_arcs_y, y=cd_arcs_x, mode='lines', name=f'Centroid Max {label_suffix}', line=dict(color='blue', width=1.5, dash='dashdot')))

    try:
        if hasattr(tracker_result, 'virtual_constraints_info') and tracker_result.virtual_constraints_info:
            vci = tracker_result.virtual_constraints_info
            
            if isinstance(vci, list) and len(vci) > 0 and isinstance(vci[0], list):
                if iterate_sel == 'Final':
                    draw_vci(vci[-1], '(Final)', is_first_iteration=True)
                elif iterate_sel.startswith('Iterate'):
                    idx = int(iterate_sel.split(' ')[1])
                    if idx < len(vci):
                        draw_vci(vci[idx], f'({iterate_sel})', is_first_iteration=True)
                elif iterate_sel in ['All', 'All (Highlight EKF Update)']:
                    for i, cycle_vci in enumerate(vci):
                        draw_vci(cycle_vci, f'(It {i})', is_first_iteration=(i == 0))
            else:
                draw_vci(vci, '(Final)')
    except Exception as e:
        print(f"Error drawing virtual constraints: {e}")

    has_iterates = hasattr(tracker_result, 'predicted_measurements_iterates') and tracker_result.predicted_measurements_iterates is not None and len(tracker_result.predicted_measurements_iterates) > 0
    if has_iterates:
        opts = ['Final', 'All', 'All (Highlight EKF Update)'] + [f'Iterate {i}' for i in range(len(tracker_result.predicted_measurements_iterates))]
        if list(iterate_selector.options) != opts:
            iterate_selector.options = opts
        iterate_selector.visible = True
    else:
        iterate_selector.visible = False

    if tracker_result.predicted_measurement is not None:
        if has_iterates and iterate_sel != 'Final':
            if iterate_sel in ['All', 'All (Highlight EKF Update)']:
                for i, z_pred in enumerate(tracker_result.predicted_measurements_iterates):
                    z_pred_cart = z_pred.reshape((-1, 2))
                    fig.add_trace(go.Scatter(
                        x=z_pred_cart[:, 1], 
                        y=z_pred_cart[:, 0], 
                        mode='markers', 
                        name=f'Predicted Meas. (It {i})', 
                        marker=dict(color='orange', symbol='cross', size=6, opacity=0.5)
                    ))
            elif iterate_sel.startswith('Iterate') and iterate_sel in iterate_selector.options:
                idx = int(iterate_sel.split(' ')[1])
                z_pred = tracker_result.predicted_measurements_iterates[idx]
                z_pred_cart = z_pred.reshape((-1, 2))
                fig.add_trace(go.Scatter(
                    x=z_pred_cart[:, 1], 
                    y=z_pred_cart[:, 0], 
                    mode='markers', 
                    name=f'Predicted Meas. ({iterate_sel})', 
                    marker=dict(color='orange', symbol='cross', size=8)
                ))
        else:
            z_pred_cart = tracker_result.predicted_measurement.mean.reshape((-1, 2))
            fig.add_trace(go.Scatter(
                x=z_pred_cart[:, 1], 
                y=z_pred_cart[:, 0], 
                mode='markers', 
                name='Predicted Meas. (Final)', 
                marker=dict(color='orange', symbol='cross', size=8)
            ))

    if has_iterates and iterate_sel != 'Final':
        if iterate_sel in ['All', 'All (Highlight EKF Update)']:
            if hasattr(tracker_result, 'iterates') and tracker_result.iterates is not None:
                for i, state_it in enumerate(tracker_result.iterates):
                    prior_shape_x, prior_shape_y = compute_estimated_shape_global(state_it, config, pca_params)
                    
                    if iterate_sel == 'All (Highlight EKF Update)' and i == 1:
                        fig.add_trace(go.Scatter(
                            x=prior_shape_y, 
                            y=prior_shape_x, 
                            mode='lines', 
                            name='Prior/Iterate 1 (EKF Update)', 
                            line=dict(color='green', dash='solid', width=2.5),
                            opacity=0.9
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=prior_shape_y, 
                            y=prior_shape_x, 
                            mode='lines', 
                            name=f'Prior/Iterate {i}', 
                            line=dict(color='purple', dash='dot'),
                            opacity=0.4
                        ))
        elif iterate_sel.startswith('Iterate'):
            idx = int(iterate_sel.split(' ')[1])
            if hasattr(tracker_result, 'iterates') and tracker_result.iterates is not None and len(tracker_result.iterates) > idx:
                state_it = tracker_result.iterates[idx]
                prior_shape_x, prior_shape_y = compute_estimated_shape_global(state_it, config, pca_params)
                fig.add_trace(go.Scatter(
                    x=prior_shape_y, 
                    y=prior_shape_x, 
                    mode='lines', 
                    name=f'Iterate {idx} Extent', 
                    line=dict(color='purple', dash='dot')
                ))
                fig.add_trace(go.Scatter(
                    x=[state_it.y], 
                    y=[state_it.x], 
                    mode='markers', 
                    name=f'Iterate {idx} Centroid', 
                    marker=dict(color='purple', size=3, symbol='diamond')
                ))
    else:
        if tracker_result.state_prior is not None:
            prior_state = tracker_result.state_prior.mean
            prior_shape_x, prior_shape_y = compute_estimated_shape_global(prior_state, config, pca_params)
            
            fig.add_trace(go.Scatter(
                x=prior_shape_y, 
                y=prior_shape_x, 
                mode='lines', 
                name='Prior Extent', 
                line=dict(color='purple', dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=[prior_state.y], 
                y=[prior_state.x], 
                mode='markers', 
                name='Prior Centroid', 
                marker=dict(color='purple', size=3, symbol='diamond')
            ))

    x_range = [float(min(x_min, x_max)), float(max(x_min, x_max))]
    y_range = [float(min(y_min, y_max)), float(max(y_min, y_max))]

    if keep_zoom:
        ui_rev = f"{x_min}_{x_max}_{y_min}_{y_max}_{reset_count}"
    else:
        ui_rev = str(make_globally_unique_id())

    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=40, pad=4),
        title=f"Frame: {frame_idx}", plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(range=x_range, constrain='domain',
                   gridcolor='rgb(200, 200, 200)', zerolinecolor='rgb(200, 200, 200)',
                   title='East [m]'),
        yaxis=dict(range=y_range, scaleanchor="x", scaleratio=1,
                   gridcolor='rgb(200, 200, 200)', zerolinecolor='rgb(200, 200, 200)',
                   title='North [m]'),
        legend=dict(x=1.05, y=1),
        uirevision=ui_rev
    )
    
    persistent_plotly_pane.object = fig

pn.bind(
    update_plotly_view,
    frame_player,
    active_file_path,
    iterate_selector,
    x_min_input,
    x_max_input,
    y_min_input,
    y_max_input,
    keep_zoom_checkbox,
    reset_counter,
    show_angle_bounds_checkbox,
    show_front_wall_checkbox,
    show_centroid_depth_checkbox,
    watch=True
)

@pn.depends(nees_group_selector.param.value, custom_states_selector.param.value, active_file_path.param.value, plot_backend_selector.param.value, matplotlib_subplot_height_slider.param.value)
def get_nees_view(selected_groups, custom_states, target_path, backend, height):
    loaded_data = load_data(target_path)
    if not loaded_data or (not selected_groups and not custom_states):
        return pn.pane.Markdown("### Select pre-defined groups or custom states to show NEES plot.")
    
    fields_to_plot = []
    fields_to_plot.extend(selected_groups)
    
    if custom_states:
        fields_to_plot.append(list(custom_states))
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    if consistency_analyzer.x_gts is None:
        return pn.pane.Markdown("### No ground truth available to compute NEES for this dataset.")
    
    if backend == 'Matplotlib':
        fig = matplotlib_show_consistency(
            analysis=consistency_analyzer, 
            fields_nees=fields_to_plot,
            subplot_height=height
        )
        if fig is None:
             return pn.pane.Markdown("### No data to plot for the selected fields.")
        return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')
    else:
        bokeh_plot = interactive_show_consistency(
            analysis=consistency_analyzer, 
            fields_nees=fields_to_plot
        )

        if bokeh_plot is None:
            return pn.pane.Markdown("### No data to plot for the selected fields.")
        
        return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')

@pn.depends(cov_matrix_selector.param.value, frame_player.param.value, active_file_path.param.value)
def get_covariance_view(matrix_name, frame_idx, target_path):
    loaded_data = load_data(target_path)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to begin.")

    tracker_result = loaded_data["sim_result"].tracker_results_ts.values[frame_idx]
    
    attr_name = COV_MATRIX_MAPPING[matrix_name]
    gauss_obj = getattr(tracker_result, attr_name)
    
    if gauss_obj is None:
        return pn.pane.Markdown(f"### {matrix_name} not available for Frame {frame_idx}")

    cov_matrix = gauss_obj.cov

    if 'state' in attr_name:
        state_mean = tracker_result.state_posterior.mean
        
        if isinstance(state_mean, State_GP):
            n_radii = len(state_mean.radii)
            labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate'] + [f'radius_{i}' for i in range(n_radii)]
            
        elif isinstance(state_mean, State_PCA):
            n_pca = loaded_data["config"].tracker.N_pca
            labels = ['x', 'y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'length', 'width'] + [f'pca_{i}' for i in range(n_pca)]
            
        else:
            N = cov_matrix.shape[0]
            labels = [f'state_{i}' for i in range(N)]
            
    else: 
        num_rays = cov_matrix.shape[0] // 2
        labels = [f'x{i}' for i in range(num_rays)] + [f'y{i}' for i in range(num_rays)]

    if len(labels) != cov_matrix.shape[0]:
        return pn.pane.Markdown(f"### Dimension Mismatch: Covariance is {cov_matrix.shape}, but generated {len(labels)} labels.")

    df = pd.DataFrame(cov_matrix, index=labels, columns=labels)
    df = df.iloc[::-1]
    
    try:
        cond_number = np.linalg.cond(cov_matrix)
        title_text = f"{matrix_name} at Frame {frame_idx} (cond={cond_number:.2e})"
    except np.linalg.LinAlgError:
        title_text = f"{matrix_name} at Frame {frame_idx} (Singular Matrix)"

    heatmap = df.hvplot.heatmap(
        cmap='viridis',
        rot=90,
        title=title_text
    ).opts(responsive=True, xrotation=90)
    
    return pn.pane.HoloViews(heatmap, sizing_mode="stretch_both")

cond_auto_calc_checkbox = pn.widgets.Checkbox(name='Auto-Calculate Condition Number', value=False, margin=(5, 10, 5, 0))
cond_calc_button = pn.widgets.Button(name='Calculate Now', button_type='primary', width=120, margin=(5, 10, 5, 10))
cond_content_pane = pn.Column(pn.pane.Markdown("### Select a file to view Condition Number over Time"), sizing_mode="stretch_both")

def update_cond_view(event=None, force=False):
    matrix_name = cov_matrix_selector.value
    target_path = active_file_path.value
    
    if not target_path:
        cond_content_pane.objects = [pn.pane.Markdown("### Select a file to view Condition Number over Time")]
        return
        
    if not force and not cond_auto_calc_checkbox.value:
        cond_content_pane.objects = [pn.pane.Markdown(f"### Auto-calculation disabled for Condition Number. Click 'Calculate Now' below.")]
        return

    loaded_data = load_data(target_path)
    if not loaded_data:
        cond_content_pane.objects = [pn.pane.Markdown("### Error loading data.")]
        return

    sim_result = loaded_data["sim_result"]
    attr_name = COV_MATRIX_MAPPING[matrix_name]
    
    cond_numbers = []
    frames = []
    
    for i, res in enumerate(sim_result.tracker_results_ts.values):
        gauss_obj = getattr(res, attr_name)
        if gauss_obj is None or getattr(gauss_obj, 'cov', None) is None:
            continue
            
        try:
            cond_number = np.linalg.cond(gauss_obj.cov)
            cond_numbers.append(cond_number)
            frames.append(i)
        except np.linalg.LinAlgError:
            pass
            
    if not frames:
        cond_content_pane.objects = [pn.pane.Markdown(f"### No valid covariance data to compute condition numbers for {matrix_name}.")]
        return
        
    df = pd.DataFrame({'Frame': frames, 'Condition Number': cond_numbers}).set_index('Frame')
    
    plot = df.hvplot.line(
        title=f"Condition Number over Time: {matrix_name}",
        ylabel="Condition Number",
        logy=True,
        responsive=True,
        height=400,
        grid=True,
        line_width=2
    )

    cond_content_pane.objects = [pn.pane.HoloViews(plot, sizing_mode="stretch_both")]

cond_calc_button.on_click(lambda e: update_cond_view(force=True))
active_file_path.param.watch(lambda e: update_cond_view(force=False), 'value')
cov_matrix_selector.param.watch(lambda e: update_cond_view(force=False), 'value')
cond_auto_calc_checkbox.param.watch(lambda e: update_cond_view(force=True) if e.new else None, 'value')

nis_auto_calc_checkbox = pn.widgets.Checkbox(name='Auto-Calculate NIS', value=False, margin=(5, 10, 5, 0))
nis_calc_button = pn.widgets.Button(name='Calculate Now', button_type='primary', width=120, margin=(5, 10, 5, 10))
nis_content_pane = pn.Column(pn.pane.Markdown("### Select a file to view NIS"), sizing_mode="stretch_both")

def update_nis_view(event=None, force=False):
    selected_field = nis_field_selector.value
    target_path = active_file_path.value
    backend = plot_backend_selector.value
    height = matplotlib_subplot_height_slider.value
    
    if not target_path:
        nis_content_pane.objects = [pn.pane.Markdown("### Select a file to view NIS")]
        return
        
    if not force and not nis_auto_calc_checkbox.value:
        nis_content_pane.objects = [pn.pane.Markdown(f"### Auto-calculation disabled for NIS. Click 'Calculate Now' below.")]
        return

    loaded_data = load_data(target_path)
    if not loaded_data:
        nis_content_pane.objects = [pn.pane.Markdown("### Error loading data.")]
        return
    
    consistency_analyzer = loaded_data["consistency_analyzer"]
    fields_to_plot = ["all"]
    
    if backend == 'Matplotlib':
        fig = matplotlib_show_consistency(
            analysis=consistency_analyzer, 
            fields_nis=fields_to_plot,
            subplot_height=height
        )
        if fig is None:
             nis_content_pane.objects = [pn.pane.Markdown("### No NIS data available.")]
             return
        nis_content_pane.objects = [pn.pane.Matplotlib(fig, sizing_mode='stretch_both')]
    else:
        bokeh_plot = interactive_show_consistency(
            analysis=consistency_analyzer, 
            fields_nis=fields_to_plot,
        )

        if bokeh_plot is None:
            nis_content_pane.objects = [pn.pane.Markdown("### No NIS data available.")]
            return
        
        nis_content_pane.objects = [pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')]

nis_calc_button.on_click(lambda e: update_nis_view(force=True))
active_file_path.param.watch(lambda e: update_nis_view(force=False), 'value')
nis_field_selector.param.watch(lambda e: update_nis_view(force=False), 'value')
plot_backend_selector.param.watch(lambda e: update_nis_view(force=False), 'value')
matplotlib_subplot_height_slider.param.watch(lambda e: update_nis_view(force=False), 'value')
nis_auto_calc_checkbox.param.watch(lambda e: update_nis_view(force=True) if e.new else None, 'value')


@pn.depends(error_group_selector.param.value, custom_states_selector.param.value, active_file_path.param.value, plot_backend_selector.param.value, matplotlib_subplot_height_slider.param.value)
def get_error_view(selected_groups, custom_states, target_path, backend, height):
    loaded_data = load_data(target_path)
    if not loaded_data or (not selected_groups and not custom_states):
        return pn.pane.Markdown("### Select pre-defined groups or custom states to show Error plot.")
    
    fields_to_plot = []
    for group in selected_groups:
        if isinstance(group, list):
            fields_to_plot.extend(group)
        else:
            fields_to_plot.append(group)
    
    if custom_states:
        fields_to_plot.extend(custom_states)
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    if consistency_analyzer.x_gts is None:
        return pn.pane.Markdown("### No ground truth available to compute Error for this dataset.")
    
    if backend == 'Matplotlib':
        fig = matplotlib_show_error(
            analysis=consistency_analyzer, 
            fields_err=fields_to_plot,
            subplot_height=height
        )
        if fig is None:
             return pn.pane.Markdown("### No data to plot for the selected fields.")
        return pn.pane.Matplotlib(fig, sizing_mode='stretch_both')
    else:
        bokeh_plot = interactive_show_error(
            analysis=consistency_analyzer, 
            fields_err=fields_to_plot
        )

        if bokeh_plot is None:
            return pn.pane.Markdown("### No data to plot for the selected fields.")
        
        return pn.pane.Bokeh(bokeh_plot, sizing_mode='stretch_both')


data_browser_depth_slider = pn.widgets.IntSlider(
    name='Auto-Expand Depth (-1 for all)', start=-1, end=5, value=2, sizing_mode='stretch_width'
)

data_browser_json_pane = pn.pane.JSON({}, sizing_mode='stretch_width', depth=2, theme='light')
data_browser_code_pane = pn.widgets.CodeEditor(value="", sizing_mode='stretch_both', readonly=True, language='python', visible=False, theme='monokai')

def _format_config_value(v, indent=4):
    if hasattr(v, '__dataclass_fields__'):
        return generate_config_code(v, indent)
    elif isinstance(v, str):
        return f"'{v}'"
    elif isinstance(v, np.ndarray):
        if v.size <= 16:
            return f"np.array({v.tolist()})"
        return f"np.array([]) # Omitted massive array of shape {v.shape}"
    else:
        return repr(v)

def generate_config_code(obj, indent=4):
    if not hasattr(obj, '__dataclass_fields__'):
        return repr(obj)
    cls_name = obj.__class__.__name__
    lines = [f"{cls_name}("]
    ind_str = " " * indent
    for field_name, f_def in obj.__dataclass_fields__.items():
        if not f_def.init:
            continue
        val = getattr(obj, field_name)
        val_str = _format_config_value(val, indent + 4)
        lines.append(f"{ind_str}{field_name}={val_str},")
    lines.append(" " * (indent - 4) + ")")
    return "\n".join(lines)

@pn.depends(data_browser_mode.param.value, frame_player.param.value, active_file_path.param.value, data_browser_depth_slider.param.value, watch=True)
def update_data_browser_view(mode, frame_idx, target_path, depth):
    data_browser_json_pane.depth = depth
    loaded_data = load_data(target_path)
    if not loaded_data:
        data_browser_json_pane.object = {"info": "Select a file to begin."}
        data_browser_json_pane.visible = True
        data_browser_code_pane.visible = False
        data_browser_depth_slider.visible = True
        return
    
    sim_result = loaded_data["sim_result"]
    consistency_analyzer = loaded_data["consistency_analyzer"]
    
    if mode == 'Copy Config Script':
        data_browser_json_pane.visible = False
        data_browser_depth_slider.visible = False
        data_browser_code_pane.visible = True
        try:
            imports = (
                "import sys\n"
                "from pathlib import Path\n\n"
                "PROJECT_ROOT = Path(__file__).resolve().parent.parent\n"
                "sys.path.append(str(PROJECT_ROOT))\n\n"
                "from src.utils.config_classes import Config, LidarConfig, ExtentConfig, TrackerConfig, SimulationConfig, TrajectoryConfig\n"
                "from src.states.states import State_PCA\n"
                "import numpy as np\n\n"
                "from src.experiment_runner import run_single_simulation\n\n"
            )
            code_str = imports + f"config = {generate_config_code(sim_result.config, indent=4)}\n\nif __name__ == '__main__':\n    # Run the simulation\n    run_single_simulation(config)\n"
            data_browser_code_pane.value = code_str
        except Exception as e:
            data_browser_code_pane.value = f"# Error generating code:\n# {e}"
        return
        
    data_browser_json_pane.visible = True
    data_browser_depth_slider.visible = True
    data_browser_code_pane.visible = False
    
    data_to_show = {}
    
    if mode == 'Current Frame Tracker Result':
        if frame_idx < len(sim_result.tracker_results_ts.values):
            res = sim_result.tracker_results_ts.values[frame_idx]
            data_to_show = safe_serialize(res, max_depth=4)
        else:
            data_to_show = {"info": "Frame index out of range"}
            
    elif mode == 'Current Frame GT':
        if frame_idx < len(sim_result.ground_truth_ts.values):
            res = sim_result.ground_truth_ts.values[frame_idx]
            data_to_show = safe_serialize(res, max_depth=4)
        else:
            data_to_show = {"info": "Frame index out of range"}

    elif mode == 'Current Frame State Error':
        if hasattr(consistency_analyzer, 'x_err_gauss') and frame_idx < len(consistency_analyzer.x_err_gauss.values):
            err_gauss = consistency_analyzer.x_err_gauss.values[frame_idx]
            
            err_mean = err_gauss.mean
            if len(sim_result.tracker_results_ts.values) > 0:
                ref_state = sim_result.tracker_results_ts.values[0].state_posterior.mean
                if not isinstance(err_mean, (State_PCA, State_GP)) and isinstance(err_mean, np.ndarray):
                    try:
                        err_mean = err_mean.view(ref_state.__class__)
                    except Exception:
                        pass

            data_to_show = {
                "error_mean": safe_serialize(err_mean, max_depth=4),
                "covariance_diag": safe_serialize(np.diag(err_gauss.cov), max_depth=4)
            }
        else:
            data_to_show = {"info": "Frame index out of range or no state error data."}

    elif mode == 'Current Frame Measurement Error':
        if hasattr(consistency_analyzer, 'z_err_gauss') and frame_idx < len(consistency_analyzer.z_err_gauss.values):
            err_gauss = consistency_analyzer.z_err_gauss.values[frame_idx]
            data_to_show = {
                "error_mean": safe_serialize(err_gauss.mean, max_depth=4),
                "covariance_diag": safe_serialize(np.diag(err_gauss.cov), max_depth=4)
            }
        else:
            data_to_show = {"info": "Frame index out of range or no measurement error data."}

    elif mode == 'Consistency Analysis (Summary)':
        data_to_show = {
            "num_frames_analyzed": len(consistency_analyzer.x_err_gauss) if hasattr(consistency_analyzer, 'x_err_gauss') else 0,
            "has_ground_truth": consistency_analyzer.x_gts is not None,
            "average_nees": safe_serialize(consistency_analyzer.get_nees(indices='all').a) if (consistency_analyzer.x_gts is not None and consistency_analyzer.get_nees(indices='all') is not None) else "N/A",
            "average_nis": safe_serialize(consistency_analyzer.get_nis(indices='all').a) if consistency_analyzer.get_nis(indices='all') is not None else "N/A",
        }
            
    elif mode == 'Config':
        data_to_show = safe_serialize(sim_result.config, max_depth=5)
        
    elif mode == 'Full Simulation Result (Summary)':
        data_to_show = {
            "config": "See Config mode",
            "num_frames": len(sim_result.tracker_results_ts.values),
            "static_covariances": safe_serialize(sim_result.static_covariances),
            "tracker_results_ts": f"TimeSequence with {len(sim_result.tracker_results_ts.values)} items",
            "ground_truth_ts": f"TimeSequence with {len(sim_result.ground_truth_ts.values)} items",
        }

    data_browser_json_pane.object = data_to_show

@pn.depends(active_file_path.param.value)
def get_cost_landscape_view(target_path):
    loaded_data = load_data(target_path)
    if not loaded_data:
        return pn.pane.Markdown("### Select a file to view Cost Landscape")
    
    explorer = CostLandscapeComponent(
        sim_result=loaded_data["sim_result"],
        tracker_config_path=None, 
        project_root=PROJECT_ROOT
    )
    
    pn.bind(explorer.update_frame, frame_player, watch=True)
    explorer.update_frame(frame_player.value)
    
    return explorer


iou_auto_calc_checkbox = pn.widgets.Checkbox(name='Auto-Calculate IoU', value=False, margin=(5, 10, 5, 0))
iou_calc_button = pn.widgets.Button(name='Calculate Now', button_type='primary', width=120, margin=(5, 10, 5, 10))
iou_content_pane = pn.Column(pn.pane.Markdown("### Select a file to view IoU over Time"), sizing_mode="stretch_both")

def update_iou_view(event=None, force=False):
    target_path = active_file_path.value

    if not target_path:
        iou_content_pane.objects = [pn.pane.Markdown("### Select a file to view IoU over Time")]
        return

    if not force and not iou_auto_calc_checkbox.value:
        iou_content_pane.objects = [pn.pane.Markdown(f"### Auto-calculation disabled for '{target_path}'. Click 'Calculate Now' below.")]
        return

    set_status("Calculating IoU (This may take a while)...")

    loaded_data = load_data(target_path)
    if not loaded_data:
        iou_content_pane.objects = [pn.pane.Markdown("### Error loading data.")]
        return

    sim_result = loaded_data["sim_result"]
    config = loaded_data["config"]
    pca_params = loaded_data["pca_params"]

    ground_truth_ts = sim_result.ground_truth_ts.values
    tracker_results_ts = sim_result.tracker_results_ts.values
    
    if len(ground_truth_ts) == 0 or len(tracker_results_ts) == 0:
        iou_content_pane.objects = [pn.pane.Markdown("### Not enough data to compute IoU.")]
        return

    extent_cfg = config.extent

    frames = []
    ious = []

    for i, res in enumerate(tracker_results_ts):
        if i < len(ground_truth_ts):
            gt_state = ground_truth_ts[i]
            est_state = res.state_posterior.mean

            try:
                gt_x, gt_y = compute_exact_vessel_shape_global(gt_state, extent_cfg.shape_coords_body)
                est_x, est_y = compute_estimated_shape_global(est_state, config, pca_params)
                iou = calculate_iou(gt_x, gt_y, est_x, est_y)
                frames.append(i)
                ious.append(iou)
            except Exception as e:
                pass
    
    if not frames:
        iou_content_pane.objects = [pn.pane.Markdown("### Could not calculate IoU.")]
        return

    import pandas as pd
    df = pd.DataFrame({
        'Frame': frames,
        'IoU': ious
    }).set_index('Frame')

    avg_iou = np.mean(ious) if ious else 0.0

    plot = df.hvplot.line(
        title=f"Intersection over Union (IoU) over Time (Avg: {avg_iou:.3f})",
        ylabel="IoU",
        ylim=(0, 1.05),
        responsive=True,
        height=400,
        grid=True,
        line_width=2
    )

    iou_content_pane.objects = [pn.pane.HoloViews(plot, sizing_mode="stretch_both")]
    set_status("Ready")

iou_calc_button.on_click(lambda e: update_iou_view(force=True))
active_file_path.param.watch(lambda e: update_iou_view(force=False), 'value')
iou_auto_calc_checkbox.param.watch(lambda e: update_iou_view(force=True) if e.new else None, 'value')


def get_constraints_view(target_path):
    print(f"DEBUG: get_constraints_view called with filename={target_path}", flush=True)
    import sys
    sys.stderr.write(f"\\n---> [Constraints View Triggered] filename={target_path}\\n")
    try:
        loaded_data = load_data(target_path)
        if not loaded_data:
            return pn.pane.Markdown(f"### Select a file to view Constraints History")
            
        sim_result = loaded_data["sim_result"]
        tracker_results = sim_result.tracker_results_ts.values
        
        frames = []
        constraints = []
        details = []
        
        for i, res in enumerate(tracker_results):
            if hasattr(res, 'clamped_length') and res.clamped_length is not None:
                frames.append(i)
                constraints.append("Length Clamped")
                details.append(f"{res.clamped_length[0]:.3f} -> {res.clamped_length[1]:.3f}")
                
            if hasattr(res, 'clamped_width') and res.clamped_width is not None:
                frames.append(i)
                constraints.append("Width Clamped")
                details.append(f"{res.clamped_width[0]:.3f} -> {res.clamped_width[1]:.3f}")
                
            if hasattr(res, 'mahalanobis_projection') and res.mahalanobis_projection is not None:
                frames.append(i)
                constraints.append("Mahalanobis Projection")
                dist = res.mahalanobis_projection[2] if len(res.mahalanobis_projection) > 2 else "unknown"
                if isinstance(dist, float):
                    details.append(f"dist: {dist:.2f}")
                else:
                    details.append(f"dist: {dist}")
                
            has_new_vci = False
            if hasattr(res, 'virtual_constraints_info') and res.virtual_constraints_info:
                vci = res.virtual_constraints_info
                final_vci = vci[-1] if isinstance(vci, list) and len(vci) > 0 and isinstance(vci[0], list) else vci
                if isinstance(final_vci, list) and len(final_vci) > 0:
                    has_new_vci = True
                    for vc in final_vci:
                        c_type = vc.get('type')
                        if c_type in ['min_angle', 'max_angle']:
                            frames.append(i)
                            constraints.append("Neg Info: Angular")
                            details.append(f"{c_type}, angle: {vc.get('body_angle', 0):.2f}")
                        elif c_type == 'front_wall':
                            frames.append(i)
                            constraints.append("Neg Info: Front Wall")
                            details.append(f"val: {vc.get('measured_val', 0):.2f}")
                        elif c_type == 'centroid_depth':
                            frames.append(i)
                            constraints.append("Neg Info: Centroid Depth")
                            details.append(f"rho_c: {vc.get('rho_c', 0):.2f}")

            if not has_new_vci and hasattr(res, 'negative_info_used') and res.negative_info_used is not None:
                val = res.negative_info_used
                if isinstance(val, bool) and val:
                    frames.append(i)
                    constraints.append("Negative Information Used")
                    details.append(f"active: {val}")
                elif isinstance(val, (int, float, np.number)) and val > 0:
                    frames.append(i)
                    constraints.append("Negative Information Used")
                    details.append(f"count: {val}")

        if not frames:
            return pn.pane.Markdown("### No constraints were triggered during this simulation.")
            
        df = pd.DataFrame({
            'Frame': frames,
            'Constraint Type': constraints,
            'Details': details
        })
        
        plot = df.hvplot.scatter(
            x='Frame', 
            y='Constraint Type', 
            by='Constraint Type',
            hover_cols=['Details'],
            size=75,
            alpha=0.85,
            line_color='black',
            line_width=0.5,
            title="Tracker Constraints Trigger History",
            height=400,
            responsive=True,
        ).opts(
            xlim=(0, len(tracker_results)),
            show_grid=True,
            framewise=True
        )
        
        return pn.pane.HoloViews(plot, sizing_mode="stretch_both")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return pn.pane.Markdown(f"### Error rendering constraints: {e}")


def save_plots(event):
    filename = save_filename_input.value
    if not filename:
        save_status.object = "Please enter a filename."
        return
        
    loaded_data = load_data(active_file_path.value)
    if not loaded_data:
        save_status.object = "No data loaded."
        return
        
    consistency_analyzer = loaded_data["consistency_analyzer"]
    saved_files = []
    
    # Save NEES
    nees_fields = []
    nees_fields.extend(nees_group_selector.value)
    if custom_states_selector.value:
        nees_fields.append(list(custom_states_selector.value))
        
    if nees_fields:
        fig = matplotlib_show_consistency(consistency_analyzer, fields_nees=nees_fields)
        if fig:
            path = Path(PROJECT_ROOT) / 'figures' / f'{filename}_nees.pdf'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved_files.append(path.name)
            plt.close(fig)

    # Save Error
    error_fields = []
    for group in error_group_selector.value:
        if isinstance(group, list):
            error_fields.extend(group)
        else:
            error_fields.append(group)
    if custom_states_selector.value:
        error_fields.extend(custom_states_selector.value)
        
    if error_fields:
        fig = matplotlib_show_error(consistency_analyzer, fields_err=error_fields)
        if fig:
            path = Path(PROJECT_ROOT) / 'figures' / f'{filename}_error.pdf'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved_files.append(path.name)
            plt.close(fig)

    # Save NIS
    if nis_field_selector.value:
        fig = matplotlib_show_consistency(consistency_analyzer, fields_nis=['all'])
        if fig:
            path = Path(PROJECT_ROOT) / 'figures' / f'{filename}_nis.pdf'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            saved_files.append(path.name)
            plt.close(fig)
            
    if saved_files:
        save_status.object = f"Saved: {', '.join(saved_files)} in figures/"
    else:
        save_status.object = "No plots to save (select groups first)."

save_button.on_click(save_plots)

# --- Build Panel objects ---
controls = pn.Column(
    pn.pane.Markdown("## Controls"),
    status_row,
    data_source_mode,
    single_run_controls,
    mc_controls,
    frame_player,
    frame_input,
    iterate_selector,
    plot_settings_controls,
    results_column_accordion,
    data_browser_mode,
    nees_group_selector,
    error_group_selector,
    custom_states_selector,
    cov_matrix_selector,
    nis_field_selector,
    plotting_divider,
    plotting_header,
    plot_backend_selector,
    matplotlib_subplot_height_slider,
    save_filename_input,
    save_button,
    save_status,
    sizing_mode="stretch_width",
)

plotly_view = pn.Column(persistent_plotly_pane, sizing_mode="stretch_both")
nees_view = pn.Column(get_nees_view, sizing_mode="stretch_both")
error_view = pn.Column(get_error_view, sizing_mode="stretch_both")
covariance_view = pn.Column(get_covariance_view, sizing_mode="stretch_both")

condition_number_view = pn.Column(
    pn.Row(cond_calc_button, cond_auto_calc_checkbox, sizing_mode="stretch_width", align='center'),
    cond_content_pane, 
    sizing_mode="stretch_both"
)
nis_view = pn.Column(
    pn.Row(nis_calc_button, nis_auto_calc_checkbox, sizing_mode="stretch_width", align='center'),
    nis_content_pane, 
    sizing_mode="stretch_both"
)

data_browser_view = pn.Column(
    data_browser_depth_slider, 
    data_browser_json_pane, 
    data_browser_code_pane,
    sizing_mode="stretch_both", 
    styles={'overflow-x': 'auto', 'overflow-y': 'auto'}
)
def get_csv_export():
    import io
    sio = io.BytesIO()
    results_table.current_view.to_csv(sio, index=False)
    sio.seek(0)
    return sio

results_table_view = pn.Column(
    pn.Row(pn.widgets.FileDownload(callback=get_csv_export, filename='simulation_results.csv', label='Export CSV', button_type='primary', sizing_mode='stretch_width'), sizing_mode="stretch_width"),
    results_table, 
    sizing_mode="stretch_both", 
    styles={'overflow-x': 'auto', 'overflow-y': 'auto'}
)
cost_landscape_view = pn.Column(get_cost_landscape_view, sizing_mode="stretch_both")

cost_breakdown_view = pn.Column(
    pn.Row(cost_breakdown_toggle, cost_calc_button, cost_auto_calc_checkbox, sizing_mode="stretch_width", align='center'), 
    cost_content_pane, 
    sizing_mode="stretch_both"
)
iou_view = pn.Column(
    pn.Row(iou_calc_button, iou_auto_calc_checkbox, sizing_mode="stretch_width", align='center'),
    iou_content_pane, 
    sizing_mode="stretch_both"
)
constraints_view = pn.Column(pn.bind(get_constraints_view, active_file_path.param.value), sizing_mode="stretch_both")

# --- Custom GoldenLayout Template ---
template_file = ASSETS_DIR / 'golden_template.html'

with open(template_file, 'r') as f:
    template_str = f.read()

tmpl = pn.Template(template_str)

tmpl.nb_template.globals['get_id'] = make_globally_unique_id

tmpl.add_panel('controls', controls)
tmpl.add_panel('plotly_view', plotly_view)
tmpl.add_panel('nees_view', nees_view)
tmpl.add_panel('error_view', error_view)
tmpl.add_panel('covariance_view', covariance_view)
tmpl.add_panel('condition_number_view', condition_number_view)
tmpl.add_panel('nis_view', nis_view)
tmpl.add_panel('data_browser_view', data_browser_view)
tmpl.add_panel('results_table_view', results_table_view)
tmpl.add_panel('cost_breakdown', cost_breakdown_view)
tmpl.add_panel('cost_landscape', cost_landscape_view)
tmpl.add_panel('iou_view', iou_view)
tmpl.add_panel('constraints_view', constraints_view)
tmpl.servable(title="GP-PCA-EOT Simulation Analysis Dashboard")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5007, help='Port to run the dashboard on')
    args = parser.parse_args()
    
    pn.serve(tmpl, port=args.port, show=True, static_dirs={'assets': str(ASSETS_DIR)}, title="GP-PCA-EOT Simulation Analysis Dashboard")