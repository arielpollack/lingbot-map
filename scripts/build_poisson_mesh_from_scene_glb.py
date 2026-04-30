"""Build a Poisson surface-reconstruction mesh from a `scene.glb` point cloud.

This is the EXACT recipe that produced the `proto-poisson-mesh` run that the
user signed off on. The worker's `poc/worker/mesh.py` mirrors this recipe;
keep this script as the canonical reference if anything ever drifts.

Inputs:
    --scene-glb /path/to/scene.glb   (the dense colored point cloud GLB,
                                       e.g. downloaded from R2 runs/<id>/scene.glb)
    --output /path/to/mesh.glb       (output GLB with per-vertex colors)

The point cloud in `scene.glb` has already had `apply_scene_alignment` applied
during `lingbot_map.vis.glb_export`, so the resulting mesh shares the same
coord frame as the point-cloud viewer.

Defaults match the worker mesh defaults — change them to test new tunings.

Usage:
    .venv/bin/python scripts/build_poisson_mesh_from_scene_glb.py \\
        --scene-glb /tmp/proto-scene.glb --output /tmp/proto-poisson.glb
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-glb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--normal-radius", type=float, default=0.10)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--density-quantile", type=float, default=0.05)
    parser.add_argument("--target-faces", type=int, default=0,
                        help="0 disables decimation (recommended).")
    args = parser.parse_args()

    print(f"loading {args.scene_glb}...")
    scene = trimesh.load(str(args.scene_glb))
    pc = max(
        (g for g in scene.geometry.values() if hasattr(g, "vertices")),
        key=lambda g: len(g.vertices),
    )
    pts = np.asarray(pc.vertices, dtype=np.float64)
    cols = np.asarray(pc.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
    print(f"  {len(pts)} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    print(f"voxel_down_sample(voxel_size={args.voxel_size})...")
    t = time.time()
    pcd_ds = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"  {len(pcd_ds.points)} points after voxel ds ({time.time() - t:.1f}s)")

    print(f"estimate_normals(radius={args.normal_radius}, max_nn={args.normal_max_nn})...")
    t = time.time()
    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius, max_nn=args.normal_max_nn
        )
    )
    pcd_ds.orient_normals_consistent_tangent_plane(k=15)
    print(f"  done ({time.time() - t:.1f}s)")

    print(f"create_from_point_cloud_poisson(depth={args.poisson_depth})...")
    t = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_ds, depth=args.poisson_depth
    )
    densities = np.asarray(densities)
    print(f"  {len(mesh.vertices)} verts / {len(mesh.triangles)} faces ({time.time() - t:.1f}s)")
    print(f"  density range: {densities.min():.2f}..{densities.max():.2f}, "
          f"p10={np.percentile(densities, 10):.2f}")

    if args.density_quantile > 0:
        keep = densities > np.quantile(densities, args.density_quantile)
        mesh.remove_vertices_by_mask(~keep)
        print(f"  after density crop ({args.density_quantile}): "
              f"{len(mesh.vertices)} verts / {len(mesh.triangles)} faces")

    if args.target_faces > 0 and len(mesh.triangles) > args.target_faces:
        print(f"decimating to {args.target_faces} faces...")
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
        print(f"  {len(mesh.vertices)} verts / {len(mesh.triangles)} faces")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    vc = np.asarray(mesh.vertex_colors)
    if vc.size:
        rgba = np.empty((len(verts), 4), dtype=np.uint8)
        rgba[:, :3] = (vc * 255).clip(0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        visual = trimesh.visual.color.ColorVisuals(vertex_colors=rgba)
    else:
        visual = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, process=False, visual=visual
    )
    out_mesh.export(str(args.output))
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nwrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
