"""Quick diagnostic: check point cloud bounding box size and average point spacing."""
import open3d as o3d
import numpy as np
import sys

pcd = o3d.io.read_point_cloud(sys.argv[1])
points = np.asarray(pcd.points)
bbox = pcd.get_axis_aligned_bounding_box()
extent = bbox.get_extent()
print(f"Points: {len(points)}")
print(f"Bounding box extent (X, Y, Z): {extent}")
print(f"Bounding box diagonal: {np.linalg.norm(extent):.4f}")

# Estimate average nearest-neighbor distance as a proxy for point spacing
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
sample_size = min(500, len(points))
idxs = np.random.choice(len(points), sample_size, replace=False)
dists = []
for i in idxs:
    [_, _, d] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)
    dists.append(np.sqrt(d[1]))
avg_spacing = np.mean(dists)
print(f"Estimated average point spacing: {avg_spacing:.6f}")
print(f"Suggested normal estimation radius (~5-10x spacing): {avg_spacing*7:.6f}")
