import os
import json
import math
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "./data/stl_ships"
CSV_FILE = os.path.join(DATA_DIR, "source.csv")
OUTPUT_JSON = "./data/processed_ships.json"
OUTPUT_IMAGE = "./data/ship_inspection.pdf"
OUTPUT_IMAGE_POLAR = "./data/ship_inspection_polar.pdf"
OUTPUT_IMAGE_RESCALED = "./data/ship_inspection_rescaled.pdf"
DEBUG_DIR = "./data/debug_ships"
NUM_RAYS = 360
DOWNSAMPLE = 1
ROTATED_SHIP_IDS = ["103", "143", "144", "145", "149", "82"]

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def safe_int(value):
    """Safely converts string to int, returning 0 if empty or invalid."""
    try:
        if pd.isna(value) or str(value).strip() == '':
            return 0
        return int(float(value))
    except:
        return 0

def load_metadata(filepath):
    """Robust metadata loader."""
    if not os.path.exists(filepath):
        print(f"ERROR: CSV file not found at {filepath}")
        return {}

    try:
        df = pd.read_csv(filepath, sep=';', dtype=str, encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        if 'BoatID' not in df.columns:
            print("CRITICAL ERROR: 'BoatID' column not found in CSV.")
            return {}

        df['BoatID'] = df['BoatID'].str.strip()
        df = df[df['BoatID'].notna() & (df['BoatID'] != '')]
        df = df.drop_duplicates(subset=['BoatID'], keep='first')
        
        meta_dict = df.set_index('BoatID').to_dict('index')
        print(f"Loaded {len(meta_dict)} metadata entries.")
        return meta_dict
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}

def get_ray_intersection(poly_points, angle):
    """
    Finds the intersection of a ray from (0,0) at 'angle' 
    with the polygon defined by 'poly_points'.
    """
    rd = np.array([np.cos(angle), np.sin(angle)])
    best_t = None
    
    for i in range(len(poly_points) - 1):
        p1 = poly_points[i]
        p2 = poly_points[i+1]
        
        v1 = p1 
        v2 = p2 - p1 
        
        # Cross product in 2D is determinant
        det = rd[0] * (-v2[1]) - rd[1] * (-v2[0])
        
        if abs(det) < 1e-8: continue # Parallel
        
        t = (p1[0] * (-v2[1]) - p1[1] * (-v2[0])) / det
        u = (rd[0] * p1[1] - rd[1] * p1[0]) / det
        
        if t > 0 and 0 <= u <= 1:
            if best_t is None or t < best_t:
                best_t = t
                
    return best_t if best_t is not None else 0.0

def generate_debug_plot(boat_id, mesh_vertices, points_2d, hull_points, aligned_pts, radii, output_dir):
    """
    Generates a 6-panel debug plot.
    """
    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"Step-by-Step Processing of ID: {boat_id}", fontsize=16)

    # --- STEP 1: RAW 3D MESH ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    step = max(1, len(mesh_vertices) // 1000)
    v = mesh_vertices[::step]
    ax1.scatter(v[:,0], v[:,1], v[:,2], c=v[:,2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_title("1. Raw 3D Mesh")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    
    # Keep aspect ratio
    max_range = np.array([v[:,0].max()-v[:,0].min(), v[:,1].max()-v[:,1].min(), v[:,2].max()-v[:,2].min()]).max() / 2.0
    mid_x = (v[:,0].max()+v[:,0].min()) * 0.5
    mid_y = (v[:,1].max()+v[:,1].min()) * 0.5
    mid_z = (v[:,2].max()+v[:,2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- STEP 2: PROJECTION ---
    ax2 = fig.add_subplot(2, 3, 2)
    step_2d = max(1, len(points_2d) // 1000)
    ax2.scatter(points_2d[::step_2d, 0], points_2d[::step_2d, 1], s=1, c='gray', alpha=0.5)
    ax2.set_title("2. Top-Down Projection")
    ax2.set_aspect('equal')
    ax2.grid(True)

    # --- STEP 3: CONVEX HULL ---
    ax3 = fig.add_subplot(2, 3, 3)
    hull_viz = np.vstack((hull_points, hull_points[0]))
    ax3.plot(hull_viz[:,0], hull_viz[:,1], 'r-', linewidth=2)
    ax3.set_title("3. Convex Hull")
    ax3.set_aspect('equal')
    ax3.grid(True)

    # --- STEP 4: ALIGNMENT & NORMALIZATION ---
    ax4 = fig.add_subplot(2, 3, 4)
    aligned_viz = np.vstack((aligned_pts, aligned_pts[0]))
    ax4.plot(aligned_viz[:,0], aligned_viz[:,1], 'b-', linewidth=2, label='Unit Shape')
    ax4.fill(aligned_viz[:,0], aligned_viz[:,1], 'b', alpha=0.1)
    ax4.set_title("4. Unit Square Normalized")
    ax4.set_aspect('equal')
    ax4.grid(True)
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_ylim(-0.6, 0.6)

    # --- STEP 5: POLAR PROFILE ---
    ax5 = fig.add_subplot(2, 3, 5, projection='polar')
    target_angles = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
    ax5.plot(target_angles, radii, 'g-', label='Radii')
    ax5.set_title("5. Polar Profile")

    # --- STEP 6: RECONSTRUCTION CHECK ---
    ax6 = fig.add_subplot(2, 3, 6)
    rec_x = np.array(radii) * np.cos(target_angles)
    rec_y = np.array(radii) * np.sin(target_angles)
    rec_x = np.append(rec_x, rec_x[0])
    rec_y = np.append(rec_y, rec_y[0])
    
    ax6.plot(aligned_viz[:,0], aligned_viz[:,1], 'b-', linewidth=1, label='Hull', alpha=0.5)
    ax6.plot(rec_x, rec_y, 'r--', linewidth=2, label='Reconstructed')
    ax6.set_title("6. Reconstruction Check")
    ax6.set_aspect('equal')
    ax6.grid(True)
    ax6.set_xlim(-0.6, 0.6)
    ax6.set_ylim(-0.6, 0.6)

    output_file = os.path.join(output_dir, f"debug_{boat_id}.png")
    plt.savefig(output_file)
    plt.close(fig)

metadata = load_metadata(CSV_FILE)
processed_list = []
failed_log = []

stl_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.stl')]
stl_files.sort()

print(f"Found {len(stl_files)} STL files. Processing...")

for fname in tqdm(stl_files):
    try:
        boat_id = os.path.splitext(fname)[0]
        meta = metadata.get(boat_id, {})
        
        full_path = os.path.join(DATA_DIR, fname)
        mesh = trimesh.load(full_path)
        
        if mesh.is_empty:
            raise ValueError("Mesh is empty")

        # --- STEP 2: Projection ---
        points_2d = mesh.vertices[:, :2]

        # --- STEP 3: Convex Hull ---
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        
        # --- STEP 4: Alignment & Normalization ---
        # 1. Centering
        min_xy = np.min(hull_points, axis=0) 
        max_xy = np.max(hull_points, axis=0)
        centroid = (min_xy + max_xy) / 2.0
        aligned_pts = hull_points - centroid

        # 1.5. Manual Rotation (180 deg)
        if boat_id in ROTATED_SHIP_IDS:
            aligned_pts = -aligned_pts
        
        # 2. Get Dimensions
        min_x, max_x = np.min(aligned_pts[:, 0]), np.max(aligned_pts[:, 0])
        min_y, max_y = np.min(aligned_pts[:, 1]), np.max(aligned_pts[:, 1])
        
        original_length = max_x - min_x
        original_width = max_y - min_y
        
        # 3. Anisotropic Scaling (Normalize BOTH axes to ~1.0)
        if original_length > 1e-4:
            aligned_pts[:, 0] /= original_length
        else:
            print(f"Warning: Boat {boat_id} has near-zero length.")

        if original_width > 1e-4:
            aligned_pts[:, 1] /= original_width
        else:
            print(f"Warning: Boat {boat_id} has near-zero width.")

        # Close the loop
        aligned_pts_closed = np.vstack((aligned_pts, aligned_pts[0]))

        # --- STEP 5: Ray Casting ---   
        # Now rays intersect the SQUASHED (unit square) shape
        target_angles = np.linspace(-np.pi, np.pi, NUM_RAYS, endpoint=False)
        radii = []

        for angle in target_angles:
            r = get_ray_intersection(aligned_pts_closed, angle)
            radii.append(r)
        
        radii = np.array(radii)

        # Generate Debug Plot
        generate_debug_plot(boat_id, mesh.vertices, points_2d, hull_points, aligned_pts, radii, DEBUG_DIR)
        
        entry = {
            "id": boat_id,
            "filename": fname,
            "original_length_m": float(original_length),
            "original_width_m": float(original_width), # Added width
            "name": meta.get('ProjectName', 'Unknown'),
            "designer": meta.get('Designer', 'Unknown'),
            "is_hull": safe_int(meta.get('Is Hull', 0)),
            "is_solid": safe_int(meta.get('Is Solid', 0)),
            "is_boat": safe_int(meta.get('Is Boat', 0)),
            "is_kayak": safe_int(meta.get('Is Kayak', 0)),
            "must_be_closed": safe_int(meta.get('Must be closed', 0)),
            "radii": radii.tolist()
        }
        
        processed_list.append(entry)
        
    except Exception as e:
        print(f"FAILED {fname}: {e}")
        failed_log.append(f"{fname}: {str(e)}")

# Save JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(processed_list, f, indent=2)

print(f"\nSaved {len(processed_list)} processed ships to {OUTPUT_JSON}")

# --- VISUALIZATION SECTION ---
print("Generating inspection images...")

N = len(processed_list)
if N > 0:
    cols = 6
    rows = math.ceil(N / cols)
    
    # === 1. Normalized Unit Shape Plot (ship_inspection.pdf) ===
    # This shows the data exactly as the PCA sees it (roughly square)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    fig.suptitle("Normalized Unit Shapes (Input to PCA)", fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes_flat = axes.flatten() if N > 1 else [axes]
    
    for i, ax in enumerate(axes_flat):
        if i < N:
            data = processed_list[i]
            r = np.array(data['radii'])
            theta = np.linspace(-np.pi, np.pi, len(r), endpoint=False)
            
            x = np.append(r * np.cos(theta), r[0] * np.cos(theta[0]))
            y = np.append(r * np.sin(theta), r[0] * np.sin(theta[0]))
            
            color = '#d62728'
            if data['is_boat'] == 1: color = '#1f77b4'
            if data['is_kayak'] == 1: color = '#2ca02c'
            
            ax.fill(x, y, alpha=0.3, fc=color)
            ax.plot(x, y, color=color, lw=1.5)
            
            # Plot Green Centroid
            ax.plot(0, 0, 'go', markersize=2)

            # Styling
            name_clean = data['name'][:20] if data['name'] else "Unknown"
            title_text = f"ID: {data['id']}\n{name_clean}"
            ax.set_title(title_text, fontsize=7)
            
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 0.6 x 0.6 Borders
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.6, 0.6)
        else:
            ax.axis('off')
    
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close(fig)
    print(f"Normalized plot saved to {OUTPUT_IMAGE}")

    # === 2. Rescaled Real-World Shape Plot (ship_inspection_rescaled.pdf) ===
    # This shows the data restored to original L/W aspect ratio
    
    fig_res, axes_res = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    fig_res.suptitle("Rescaled Real-World Shapes (Verification)", fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes_res_flat = axes_res.flatten() if N > 1 else [axes_res]

    for i, ax in enumerate(axes_res_flat):
        if i < N:
            data = processed_list[i]
            r = np.array(data['radii'])
            theta = np.linspace(-np.pi, np.pi, len(r), endpoint=False)
            
            # Unit shape
            x_unit = r * np.cos(theta)
            y_unit = r * np.sin(theta)
            
            # Rescale to Real Dimensions
            L = data.get('original_length_m', 1.0)
            W = data.get('original_width_m', 1.0)
            
            x_real = x_unit * L
            y_real = y_unit * W
            
            # Close loop
            x_real = np.append(x_real, x_real[0])
            y_real = np.append(y_real, y_real[0])

            color = '#d62728'
            if data['is_boat'] == 1: color = '#1f77b4'
            if data['is_kayak'] == 1: color = '#2ca02c'
            
            ax.fill(x_real, y_real, alpha=0.3, fc=color)
            ax.plot(x_real, y_real, color=color, lw=1.5)
            
            # Plot Green Centroid
            ax.plot(0, 0, 'go', markersize=2)
            
            ax.set_title(f"ID: {data['id']}\nL={L:.1f}, W={W:.1f}", fontsize=7)
            ax.set_aspect('equal')
            
            # Set consistent square limits individually per boat
            local_max = max(np.max(np.abs(x_real)), np.max(np.abs(y_real)))
            local_limit = local_max * 1.1
            ax.set_xlim(-local_limit, local_limit)
            ax.set_ylim(-local_limit, local_limit)
            
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.tick_params(labelsize=6)
        else:
            ax.axis('off')

    plt.savefig(OUTPUT_IMAGE_RESCALED, dpi=150)
    plt.close(fig_res)
    print(f"Rescaled verification plot saved to {OUTPUT_IMAGE_RESCALED}")

    # === 3. Polar Plot ===
    fig_polar, axes_polar = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    fig_polar.suptitle("Polar Profiles (Normalized Radii)", fontsize=16)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes_polar_flat = axes_polar.flatten() if N > 1 else [axes_polar]
    
    for i, ax in enumerate(axes_polar_flat):
        if i < N:
            data = processed_list[i]
            r = np.array(data['radii'])
            theta = np.linspace(-np.pi, np.pi, len(r), endpoint=False)
            
            color = '#d62728'
            if data['is_boat'] == 1: color = '#1f77b4'
            if data['is_kayak'] == 1: color = '#2ca02c'

            ax.plot(theta, r, color=color, lw=1.5)
            
            name_clean = data['name'][:20] if data['name'] else "Unknown"
            ax.set_title(f"ID: {data['id']}\n{name_clean}", fontsize=7)
            
            ax.set_xlim(-np.pi, np.pi)
            # Since normalized to fit square, max radius is ~0.5-0.71
            ax.set_ylim(0, 0.8) 
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.axis('off')

    plt.savefig(OUTPUT_IMAGE_POLAR, dpi=150)
    plt.close(fig_polar)
    print(f"Polar inspection image saved to {OUTPUT_IMAGE_POLAR}")