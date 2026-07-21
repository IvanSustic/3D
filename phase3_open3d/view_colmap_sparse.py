"""
Phase 3, Exercise 1: Load a COLMAP sparse reconstruction and visualize it
with Open3D.

This reads the points3D.txt file COLMAP exported (the sparse point cloud
from your reconstruction) and displays it interactively.

USAGE:
    python phase3_open3d/view_colmap_sparse.py <path_to_sparse_export_folder>

Example:
    python phase3_open3d/view_colmap_sparse.py phase2_colmap/object_01/sparse_export

CONTROLS (Open3D viewer window):
    Left-click + drag  : rotate
    Right-click + drag  : pan
    Scroll wheel        : zoom
    Press 'Q' or close window: exit
"""

import numpy as np
import open3d as o3d
import sys
import os


def read_points3D_text(path):
    """
    Parses COLMAP's points3D.txt format.
    Each valid line looks like:
        POINT3D_ID X Y Z R G B ERROR TRACK[...]
    Lines starting with '#' are comments/headers and are skipped.
    """
    points = []
    colors = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # parts[0] = POINT3D_ID, [1:4] = X Y Z, [4:7] = R G B, [7] = ERROR, rest = track
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points), np.array(colors)


def main():
    if len(sys.argv) != 2:
        print("Usage: python view_colmap_sparse.py <path_to_sparse_export_folder>")
        sys.exit(1)

    export_dir = sys.argv[1]
    points_path = os.path.join(export_dir, "points3D.txt")

    if not os.path.exists(points_path):
        print(f"Could not find {points_path}")
        print("Make sure you exported as TXT format from COLMAP (File -> Export model).")
        sys.exit(1)

    print(f"Reading {points_path} ...")
    points, colors = read_points3D_text(points_path)
    print(f"Loaded {len(points)} 3D points.")

    if len(points) == 0:
        print("No points found -- check the export actually contains a points3D.txt with data.")
        sys.exit(1)

    # Build an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # A coordinate frame helps orient yourself (red=X, green=Y, blue=Z)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    print("\nOpening viewer...")
    print("  Left-click + drag  : rotate")
    print("  Right-click + drag : pan")
    print("  Scroll wheel        : zoom")
    print("  Close window to exit\n")

    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="COLMAP Sparse Reconstruction",
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
