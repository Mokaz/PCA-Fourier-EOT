import os
import json
import math
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_STL = "data/test_stl/Havfruen.stl"
OUTPUT_DIR = "data/test_stl"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "Havfruen_processed.json")
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "Havfruen_inspection.pdf")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug_havfruen")
NUM_RAYS = 360

if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

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

def main():
    print(f"Loading {INPUT_STL} ...")
    mesh = trimesh.load(INPUT_STL)
    
    if mesh.is_empty:
        raise ValueError("Mesh is empty")

    boat_id = "Havfruen"
    
    # Check extents and optionally rotate so the primary axis points along +X
    # For boats, we typically want the bow pointing in +X direction.
    # In Havfruen.stl, extents are roughly X: 2.8, Y: 7.5. Thus length is along Y.
    # We should rotate by 90 degrees around Z so it aligns with X.
    
    # Let's rotate -90 degrees around Z
    # We can do this on the 2d points directly or on the mesh. Let's do it on the mesh for viz.
    angle_rad = -np.pi / 2
    rot_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
    mesh.apply_transform(rot_matrix)

    # --- STEP 2: Projection ---
    points_2d = mesh.vertices[:, :2]

    # --- STEP 3: Convex Hull ---
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    
    # --- STEP 4: Alignment & Normalization ---
    min_xy = np.min(hull_points, axis=0) 
    max_xy = np.max(hull_points, axis=0)
    centroid = (min_xy + max_xy) / 2.0
    aligned_pts = hull_points - centroid
    
    # 2. Get Dimensions
    min_x, max_x = np.min(aligned_pts[:, 0]), np.max(aligned_pts[:, 0])
    min_y, max_y = np.min(aligned_pts[:, 1]), np.max(aligned_pts[:, 1])
    
    original_length = max_x - min_x
    original_width = max_y - min_y
    
    # 3. Anisotropic Scaling (Normalize BOTH axes to ~1.0)
    if original_length > 1e-4:
        aligned_pts[:, 0] /= original_length
        
    if original_width > 1e-4:
        aligned_pts[:, 1] /= original_width

    # Close the loop
    aligned_pts_closed = np.vstack((aligned_pts, aligned_pts[0]))

    # --- STEP 5: Ray Casting ---   
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
        "filename": os.path.basename(INPUT_STL),
        "original_length_m": float(original_length),
        "original_width_m": float(original_width),
        "name": boat_id,
        "radii": radii.tolist()
    }
    
    # Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump([entry], f, indent=2)

    print(f"Saved processed data to {OUTPUT_JSON}")

    # Plot final inspection
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    theta = np.linspace(-np.pi, np.pi, len(radii), endpoint=False)
    x = np.append(radii * np.cos(theta), radii[0] * np.cos(theta[0]))
    y = np.append(radii * np.sin(theta), radii[0] * np.sin(theta[0]))
            
    x_real = x * original_length
    y_real = y * original_width
    
    ax.fill(x_real, y_real, alpha=0.3, fc='blue')
    ax.plot(x_real, y_real, color='blue', lw=1.5)
    ax.plot(0, 0, 'go', markersize=4)
    ax.set_title(f"ID: {boat_id}\nL={original_length:.1f}, W={original_width:.1f}", fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close(fig)
    print(f"Saved inspection plot to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
