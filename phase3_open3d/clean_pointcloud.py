"""
Phase 3, Exercise 2: Clean up a sparse point cloud using outlier removal.

Takes the noisy COLMAP sparse reconstruction (object + scattered outlier
points from reflections/background) and removes the outliers using
Open3D's statistical outlier removal.

HOW IT WORKS:
For every point, Open3D looks at its k nearest neighbors and computes the
average distance to them. Points whose average neighbor-distance is much
higher than the norm (i.e., isolated, sparse points -- exactly what stray
noise looks like) get flagged and removed. Points that are part of a dense
cluster (like your dinosaur) survive, since their neighbors are all close by.

USAGE:
    python phase3_open3d/clean_pointcloud.py <path_to_sparse_export_folder>

Example:
    python phase3_open3d/clean_pointcloud.py phase2_colmap/object_01/sparse_export

OUTPUT:
    Shows a before/after visualization, and saves the cleaned point cloud to
    phase3_open3d/cleaned_pointcloud.ply
"""

import numpy as np
import open3d as o3d
import sys
import os


def read_points3D_text(path):
    points = []
    colors = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points), np.array(colors)


def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_pointcloud.py <path_to_sparse_export_folder>")
        sys.exit(1)

    export_dir = sys.argv[1]
    points_path = os.path.join(export_dir, "points3D.txt")

    if not os.path.exists(points_path):
        print(f"Could not find {points_path}")
        sys.exit(1)

    points, colors = read_points3D_text(points_path)
    print(f"Loaded {len(points)} points.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- Statistical outlier removal ---
    # nb_neighbors: how many nearest neighbors to consider per point
    # std_ratio: how many standard deviations above the mean distance
    #            counts as "too far" (lower = more aggressive removal)
    print("\nRunning statistical outlier removal...")
    cleaned_pcd, inlier_indices = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=1.5
    )

    n_removed = len(points) - len(inlier_indices)
    pct_removed = 100 * n_removed / len(points)
    print(f"Removed {n_removed} points ({pct_removed:.1f}%)")
    print(f"Remaining: {len(inlier_indices)} points")

    # --- Visualize: outliers in red, inliers keep their original color ---
    outlier_pcd = pcd.select_by_index(inlier_indices, invert=True)
    outlier_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red = removed points

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    print("\nShowing BEFORE/comparison view: red = points that will be removed.")
    print("Close this window to continue.\n")
    o3d.visualization.draw_geometries(
        [cleaned_pcd, outlier_pcd, coord_frame],
        window_name="Red = outliers being removed",
        width=1280,
        height=800,
    )

    print("Showing AFTER: cleaned point cloud only.")
    print("Close this window to finish.\n")
    o3d.visualization.draw_geometries(
        [cleaned_pcd, coord_frame],
        window_name="Cleaned point cloud",
        width=1280,
        height=800,
    )

    # --- Save the cleaned result ---
    out_dir = "phase3_open3d"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cleaned_pointcloud.ply")
    o3d.io.write_point_cloud(out_path, cleaned_pcd)
    print(f"Saved cleaned point cloud to {out_path}")


if __name__ == "__main__":
    main()
