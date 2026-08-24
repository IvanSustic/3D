"""
Phase 3, Exercise 3: Mesh reconstruction from a point cloud.

Takes your cleaned sparse point cloud and reconstructs a solid triangle
mesh surface from it, using Poisson Surface Reconstruction.

STEPS:
1. Estimate normals for every point (which way is "outward")
2. Orient normals consistently (so they don't point randomly in/out)
3. Run Poisson reconstruction to fit a smooth surface
4. Crop away low-density mesh regions (Poisson tends to "balloon out" and
   guess/fill in areas with no real data -- we trim that back)

USAGE:
    python phase3_open3d/mesh_reconstruction.py <path_to_cleaned_pointcloud.ply>

Example:
    python phase3_open3d/mesh_reconstruction.py phase3_open3d/cleaned_pointcloud.ply

OUTPUT:
    Shows the point cloud with estimated normals, then the resulting mesh.
    Saves the mesh to phase3_open3d/mesh_output.ply
"""

import numpy as np
import open3d as o3d
import sys
import os


def main():
    if len(sys.argv) != 2:
        print("Usage: python mesh_reconstruction.py <path_to_cleaned_pointcloud.ply>")
        sys.exit(1)

    pcd_path = sys.argv[1]
    if not os.path.exists(pcd_path):
        print(f"Could not find {pcd_path}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded {len(pcd.points)} points.")

    if len(pcd.points) < 50:
        print("Very few points -- mesh quality will likely be poor.")
        print("Consider using the uncleaned/less-aggressively-filtered point cloud instead.")

    # --- Step 1: Estimate normals ---
    # For each point, look at its local neighborhood and fit a plane;
    # the normal is perpendicular to that local plane.
    # NOTE: every fresh COLMAP reconstruction can land on a different
    # arbitrary scale (no absolute real-world units unless calibrated).
    # ALWAYS re-run check_scale.py on new point clouds and update this
    # radius accordingly -- don't assume a previous value still applies.
    # Current value (0.025) tuned for the masked dense cloud, ~0.0036 avg spacing.
    print("\nEstimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=50)
    )

    # --- Step 2: Orient normals consistently ---
    # Normal estimation alone doesn't know which way is "outward" vs "inward" --
    # each normal could point either way. This step propagates a consistent
    # orientation across neighboring points using a minimum spanning tree,
    # so all normals broadly agree on which side is "outside."
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print("Showing point cloud with normals (each point now has an orientation).")
    print("Close this window to continue to mesh reconstruction.\n")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point cloud with normals (press N to toggle normal display)",
        width=1280,
        height=800,
        point_show_normal=True,
    )

    # --- Step 3: Poisson surface reconstruction ---
    # depth controls mesh resolution/detail -- higher = more detail but
    # needs denser input data to avoid noise. With a dense cloud (~180k
    # points), depth 10 is reasonable; drop to 9 if the result looks noisy.
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10
    )
    densities = np.asarray(densities)
    print(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # --- Step 4: Trim low-density regions ---
    # Poisson reconstruction extrapolates a smooth surface everywhere, even
    # in areas with little/no real point support (e.g., the missing #8-22
    # gap, or the underside you never photographed). Those regions have low
    # "density" values -- this step removes the least-supported 10% of the
    # mesh, which are usually the most speculative/inaccurate parts.
    density_threshold = np.quantile(densities, 0.1)
    low_density_vertices = densities < density_threshold
    mesh.remove_vertices_by_mask(low_density_vertices)
    print(f"After trimming low-density regions: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles")

    mesh.compute_vertex_normals()

    print("\nShowing final mesh. Close this window to finish.\n")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Reconstructed mesh",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )

    out_dir = "phase3_open3d"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mesh_output.ply")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved mesh to {out_path}")


if __name__ == "__main__":
    main()
