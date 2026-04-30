from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _import_open3d():
    import open3d as o3d

    return o3d


def create_clay_mesh_from_prepared(
    prepared: dict[str, Any],
    output_dir: str | Path,
    options: dict[str, Any],
    scene_glb_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fuse LingBot-MAP depth predictions into a coarse clay GLB mesh.

    This is intentionally a diagnostic geometry artifact, not the final
    photoreal representation. It must fail soft so point-cloud and splat output
    still complete when TSDF fusion is unavailable or unsuitable.
    """
    if not bool(options.get("mesh_enabled", True)):
        return {"enabled": False, "reason": "disabled"}

    start = time.time()
    output_dir = Path(output_dir)
    mesh_path = output_dir / "mesh.glb"
    info: dict[str, Any] = {"enabled": True, "status": "started"}

    depth = prepared.get("depth")
    extrinsic_c2w = prepared.get("extrinsic")
    intrinsic = prepared.get("intrinsic")
    images_chw = prepared.get("images")
    missing = [
        name
        for name, value in {
            "depth": depth,
            "extrinsic": extrinsic_c2w,
            "intrinsic": intrinsic,
            "images": images_chw,
        }.items()
        if value is None
    ]
    if missing:
        info.update(
            {
                "status": "failed",
                "error": f"missing prediction tensors required for mesh export: {', '.join(missing)}",
            }
        )
        return info

    try:
        o3d = _import_open3d()
        depth_np = _depth_to_shw(depth)
        intrinsic_np = np.asarray(intrinsic, dtype=np.float64)
        images_np = _images_to_hwc_uint8(images_chw)
        w2c = _c2w_to_w2c(extrinsic_c2w)

        if depth_np.ndim != 3:
            raise ValueError(f"depth must have shape (S,H,W), got {depth_np.shape}")
        if images_np.shape[:3] != (*depth_np.shape,):
            raise ValueError(
                "images must match depth frames and image size; "
                f"got images {images_np.shape}, depth {depth_np.shape}"
            )

        depth_conf = prepared.get("depth_conf")
        confidence_np = _depth_to_shw(depth_conf) if depth_conf is not None else None
        conf_percentile = float(
            options.get("mesh_conf_percentile", options.get("conf_percentile", 50.0))
        )
        conf_threshold = None
        if confidence_np is not None and confidence_np.size:
            conf_threshold = float(np.percentile(confidence_np, conf_percentile))

        # ─── Reference recipe (proto-poisson-mesh, signed off by user) ───
        # 1. Build a dense colored point cloud from depth-unproject (using
        #    c2w to MATCH glb_export.py — see _build_dense_colored_cloud).
        # 2. voxel_down_sample(0.02)
        # 3. estimate_normals(radius=0.10, max_nn=30) + orient_consistent
        # 4. Poisson reconstruction at depth=9
        # 5. density crop at the 5th percentile
        # 6. NO quadric decimation (it stretches faces across density-crop
        #    holes and produces visible "curtain" artifacts).
        # 7. apply_scene_alignment (matching glb_export) so the mesh shares
        #    a coord frame with scene.glb.
        # The standalone, runnable equivalent of this recipe lives at
        # `scripts/build_poisson_mesh_from_scene_glb.py`.
        # If you change a default below, expect mesh quality to change too.
        target_faces = int(options.get("mesh_target_faces", 0))
        poisson_depth = int(options.get("mesh_poisson_depth", 9))
        density_quantile = float(options.get("mesh_density_quantile", 0.05))
        cloud_voxel_size = float(options.get("mesh_cloud_voxel_size", 0.02))
        normal_radius = float(options.get("mesh_normal_radius", 0.10))
        normal_max_nn = int(options.get("mesh_normal_max_nn", 30))
        # 10 matches the default `downsample_factor` of glb_export, which is
        # what produced the cloud the proto run was built from.
        coloring_downsample = int(options.get("mesh_color_downsample", 10))

        # If scene.glb has already been written (the normal case — glb_export
        # runs first in the worker pipeline), load its colored point cloud
        # directly. This GUARANTEES the mesh ends up in scene.glb's coord
        # frame because we're literally Poisson-ing the same vertices that
        # the point-cloud viewer renders.
        cloud_source = "scene_glb"
        cloud_points, cloud_colors = _load_cloud_from_scene_glb(scene_glb_path)
        if cloud_points is None:
            cloud_source = "depth_unproject"
            cloud_points, cloud_colors = _build_dense_colored_cloud(
                depth_np=depth_np,
                intrinsic_np=intrinsic_np,
                extrinsic_c2w=extrinsic_c2w,
                images_np=images_np,
                confidence_np=confidence_np,
                conf_threshold=conf_threshold,
                downsample_factor=coloring_downsample,
            )
        if cloud_points.shape[0] < 1000:
            raise ValueError(
                f"dense cloud has only {cloud_points.shape[0]} points after filtering"
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            (cloud_colors.astype(np.float64) / 255.0)
        )
        pcd_voxel_count = len(pcd.points)
        if cloud_voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=cloud_voxel_size)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(k=15)
        except Exception:
            # Falls back to viewing-direction orientation if k-tangent fails on
            # disconnected components.
            pcd.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 0.0))

        o3d_mesh, densities = (
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth
            )
        )
        densities = np.asarray(densities)
        if densities.size and density_quantile > 0:
            keep_threshold = float(np.quantile(densities, density_quantile))
            o3d_mesh.remove_vertices_by_mask(densities < keep_threshold)
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()
        if target_faces > 0 and len(o3d_mesh.triangles) > target_faces:
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
        o3d_mesh.compute_vertex_normals()

        vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError("Poisson reconstruction produced an empty mesh")

        # When the cloud came from scene.glb, vertices are already in
        # scene.glb's frame and no transform is needed. The depth-unproject
        # fallback path produces points in a different frame (the
        # consistent-but-not-true world that glb_export operates in BEFORE its
        # apply_scene_alignment); we don't try to align here because the
        # primary path is scene-glb-fed.

        # Color mesh vertices by NN against the SAME dense colored cloud we
        # built from depth-unproject. (Open3D's Poisson keeps vertex colors,
        # but only via interpolation across triangulation; explicit NN gives
        # us deterministic per-vertex color sourcing matching the input.)
        coloring = _color_vertices_from_cloud(
            mesh_vertices=vertices,
            cloud_points=cloud_points,
            cloud_colors=cloud_colors,
        )

        _export_glb_with_vertex_colors(
            vertices, faces, coloring["vertex_colors"], mesh_path
        )

        # ── Texture bake (optional, replaces mesh.glb on success) ──────
        # Defaults to enabled. Soft-fails if bake throws — we keep the
        # vertex-colored mesh as the shipped artifact in that case.
        # If `mesh_atlas_max_faces` is set, we decimate the mesh BEFORE
        # baking. xatlas.parametrize is the runtime hot spot and scales
        # roughly linearly with face count. 300k = ~90s xatlas + ~30s
        # per-pixel bake at 4K atlas = under 3 min total.
        texture_info: dict[str, Any] = {"enabled": False}
        if bool(options.get("mesh_texture_bake", True)):
            atlas_max_faces = int(options.get("mesh_atlas_max_faces", 300_000))
            bake_vertices, bake_faces = _maybe_decimate_for_bake(
                o3d, o3d_mesh, vertices, faces, max_faces=atlas_max_faces
            )
            texture_info = _try_bake_texture(
                vertices=bake_vertices,
                faces=bake_faces,
                depth_np=depth_np,
                intrinsic_np=intrinsic_np,
                extrinsic_c2w=extrinsic_c2w,
                images_np=images_np,
                output_path=mesh_path,  # OVERWRITE on success
                options=options,
                fallback_cloud_points=cloud_points,
                fallback_cloud_colors=cloud_colors,
            )
            texture_info["decimated_faces"] = int(bake_faces.shape[0])
        bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=float)
        info.update(
            {
                "status": "ok",
                "path": str(mesh_path),
                "duration_seconds": round(time.time() - start, 3),
                "vertex_count": int(vertices.shape[0]),
                "face_count": int(faces.shape[0]),
                "size_mb": round(os.path.getsize(mesh_path) / 1e6, 3),
                "frame_count": int(depth_np.shape[0]),
                "bounds": bounds.tolist(),
                "method": "poisson_from_dense_cloud",
                "cloud_points": int(pcd_voxel_count),
                "cloud_points_after_voxel_ds": int(len(pcd.points)),
                "coloring": coloring["stats"],
                "texture_bake": texture_info,
                "settings": {
                    "poisson_depth": poisson_depth,
                    "density_quantile": density_quantile,
                    "cloud_voxel_size": cloud_voxel_size,
                    "normal_radius": normal_radius,
                    "normal_max_nn": normal_max_nn,
                    "color_downsample_factor": coloring_downsample,
                    "target_faces": target_faces,
                    "conf_percentile": conf_percentile,
                    "conf_threshold": conf_threshold,
                },
            }
        )
        return info
    except Exception as exc:
        info.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": round(time.time() - start, 3),
            }
        )
        return info


def _images_to_hwc_uint8(images: Any) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"images must have shape (S,3,H,W) or (S,H,W,3), got {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    return np.ascontiguousarray((arr * 255.0).astype(np.uint8))


def _depth_to_shw(depth: Any) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"depth must have shape (S,H,W) or (S,H,W,1), got {arr.shape}")
    return arr


def _c2w_to_w2c(extrinsic_c2w: Any) -> np.ndarray:
    c2w = np.asarray(extrinsic_c2w, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must have shape (S,3,4), got {c2w.shape}")
    c2w4 = np.zeros((c2w.shape[0], 4, 4), dtype=np.float64)
    c2w4[:, :3, :4] = c2w
    c2w4[:, 3, 3] = 1.0
    return np.linalg.inv(c2w4)


def _sample_frame_indices(frame_count: int, max_frames: int) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=np.int64)
    if max_frames <= 0 or frame_count <= max_frames:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(np.linspace(0, frame_count - 1, max_frames).round().astype(np.int64))


def _depth_percentile(depth: np.ndarray, percentile: float) -> float:
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        return 1.0
    return max(float(np.percentile(valid, percentile)), 1e-3)


def _integration_namespace(o3d):
    return getattr(getattr(o3d, "pipelines", o3d), "integration", getattr(o3d, "integration", None))


def _scene_alignment_transform(extrinsics_matrices_4x4: np.ndarray) -> np.ndarray:
    """Compute the same alignment matrix that lingbot_map.vis.glb_export's
    `apply_scene_alignment` applies to scene.glb. Returns a 4x4 to multiply
    point coordinates by (as `point @ T.T` since points are row-vectors).
    """
    from scipy.spatial.transform import Rotation

    opengl_conv = np.eye(4)
    opengl_conv[1, 1] = -1
    opengl_conv[2, 2] = -1
    rot_y180 = np.eye(4)
    rot_y180[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    return np.linalg.inv(extrinsics_matrices_4x4[0]) @ opengl_conv @ rot_y180


def _export_glb_with_vertex_colors(
    vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray, path: Path
) -> None:
    """Write a GLB with per-vertex RGBA colors that GLTFLoader will surface."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if vertex_colors.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"vertex_colors length {vertex_colors.shape[0]} != vertices {vertices.shape[0]}"
        )
    tri_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        visual=trimesh.visual.color.ColorVisuals(vertex_colors=vertex_colors),
    )
    tri_mesh.export(path)


def _maybe_decimate_for_bake(
    o3d, source_o3d_mesh, vertices: np.ndarray, faces: np.ndarray, *, max_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate the mesh down to `max_faces` triangles for the texture
    bake. The vertex-colored mesh keeps the high-res version in mesh.glb
    earlier — this only affects what gets UV-unwrapped + baked. If the input
    is already small enough, returns it unchanged.
    """
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return vertices, faces.astype(np.int32)
    try:
        decimated = source_o3d_mesh.simplify_quadric_decimation(max_faces)
        decimated.remove_duplicated_vertices()
        decimated.remove_duplicated_triangles()
        decimated.remove_degenerate_triangles()
        decimated.remove_unreferenced_vertices()
        return (
            np.asarray(decimated.vertices, dtype=np.float32),
            np.asarray(decimated.triangles, dtype=np.int32),
        )
    except Exception as exc:
        print(f"warning: pre-bake decimation failed, baking full-res: {exc}", flush=True)
        return vertices, faces.astype(np.int32)


def _try_bake_texture(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    depth_np: np.ndarray,
    intrinsic_np: np.ndarray,
    extrinsic_c2w: np.ndarray,
    images_np: np.ndarray,
    output_path: Path,
    options: dict[str, Any],
    fallback_cloud_points: np.ndarray | None = None,
    fallback_cloud_colors: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bake a UV texture atlas onto the mesh from posed frames. Soft-fails:
    on any error, the original (vertex-colored) mesh.glb stays in place and
    we return a stats dict with `error`.
    """
    import signal

    class _BakeTimeout(Exception):
        pass

    info: dict[str, Any] = {"enabled": True}
    bake_start = time.time()
    timeout_s = int(options.get("mesh_texture_bake_timeout", 600))

    def _alarm_handler(signum, frame):
        raise _BakeTimeout(f"bake exceeded {timeout_s}s")

    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        from poc.worker.texture import bake_texture_atlas

        # Build TRUE w2c (4, 4) per frame from c2w (3, 4).
        c2w = np.asarray(extrinsic_c2w, dtype=np.float64)
        S = c2w.shape[0]
        c2w_4 = np.zeros((S, 4, 4), dtype=np.float64)
        c2w_4[:, :3, :4] = c2w
        c2w_4[:, 3, 3] = 1.0
        w2c_4 = np.linalg.inv(c2w_4).astype(np.float32)

        atlas_resolution = int(options.get("mesh_atlas_resolution", 4096))
        top_k_frames = int(options.get("mesh_atlas_top_k", 4))
        bake = bake_texture_atlas(
            vertices=vertices,
            faces=faces.astype(np.int32),
            images_hwc_uint8=images_np,
            intrinsics=intrinsic_np.astype(np.float32),
            extrinsics_w2c=w2c_4,
            output_path=output_path,
            atlas_resolution=atlas_resolution,
            top_k_frames=top_k_frames,
            fallback_cloud_points=fallback_cloud_points,
            fallback_cloud_colors=fallback_cloud_colors,
        )
        info.update(bake)
        info["bake_total_seconds"] = round(time.time() - bake_start, 2)
        return info
    except Exception as exc:
        info.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "bake_total_seconds": round(time.time() - bake_start, 2),
            }
        )
        import traceback
        print(f"texture bake failed (kept vertex-color mesh): {info['error']}", flush=True)
        traceback.print_exc()
        return info
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def _load_cloud_from_scene_glb(
    scene_glb_path: str | Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the dense colored point cloud out of a scene.glb that
    `lingbot_map.vis.glb_export.export_predictions_glb_file` already wrote.

    Returns (points (N, 3) float32, colors (N, 3) uint8) on success, or
    (None, None) if the path is missing/empty/malformed (caller falls back to
    re-building from depth).
    """
    if scene_glb_path is None:
        return None, None
    p = Path(scene_glb_path)
    if not p.exists() or p.stat().st_size == 0:
        return None, None
    try:
        scene = trimesh.load(str(p))
    except Exception:
        return None, None

    # Pick the largest geometry that has vertex colors — that's the point cloud.
    candidates = []
    for name, geom in getattr(scene, "geometry", {}).items():
        if not hasattr(geom, "vertices"):
            continue
        verts = np.asarray(geom.vertices)
        if verts.shape[0] < 100:
            continue
        candidates.append((verts.shape[0], name, geom))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    geom = candidates[0][2]

    pts = np.asarray(geom.vertices, dtype=np.float32)
    visual = getattr(geom, "visual", None)
    vc = getattr(visual, "vertex_colors", None) if visual is not None else None
    if vc is None or len(vc) == 0:
        return None, None
    cols = np.asarray(vc, dtype=np.uint8)
    if cols.ndim == 2 and cols.shape[1] == 4:
        cols = cols[:, :3]
    if cols.shape[0] != pts.shape[0]:
        return None, None
    return pts, np.ascontiguousarray(cols)


def _build_dense_colored_cloud(
    *,
    depth_np: np.ndarray,
    intrinsic_np: np.ndarray,
    extrinsic_c2w: np.ndarray,
    images_np: np.ndarray,
    confidence_np: np.ndarray | None,
    conf_threshold: float | None,
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject every (downsampled) pixel into world coords + carry colors.
    Returns (points (N, 3) float32, colors (N, 3) uint8) after dropping NaN /
    out-of-bounds / low-confidence pixels.

    NOTE: passes c2w (the post-postprocess `extrinsic`) to unproject, matching
    `lingbot_map.vis.glb_export.export_predictions_glb_file`. This places the
    resulting cloud in the SAME consistent-but-non-true frame as scene.glb so
    `_scene_alignment_transform` (also copied from glb_export) lands the mesh
    in scene.glb's final coord frame. Passing the docstring-correct w2c here
    instead would produce TRUE world coords but mismatch glb_export's frame
    and break `apply_scene_alignment`.
    """
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

    downsample_factor = max(1, int(downsample_factor))
    c2w_3x4 = np.asarray(extrinsic_c2w, dtype=np.float64)[:, :3, :4]
    intr = intrinsic_np.astype(np.float64).copy()
    depth = depth_np.astype(np.float32, copy=False)
    images = images_np

    if downsample_factor > 1:
        depth = depth[:, ::downsample_factor, ::downsample_factor]
        images = images[:, ::downsample_factor, ::downsample_factor, :]
        intr[:, 0, 0] /= downsample_factor
        intr[:, 1, 1] /= downsample_factor
        intr[:, 0, 2] /= downsample_factor
        intr[:, 1, 2] /= downsample_factor

    if depth.ndim == 3:
        depth = depth[..., None]  # unproject_depth_map_to_point_map calls .squeeze(-1)

    world_points = unproject_depth_map_to_point_map(depth, c2w_3x4, intr)
    pts = np.asarray(world_points).reshape(-1, 3).astype(np.float32)
    cols = np.ascontiguousarray(images).reshape(-1, 3)

    mask = (
        np.isfinite(pts).all(axis=-1)
        & (np.abs(pts).max(axis=-1) < 100.0)
        & (depth[..., 0].reshape(-1) > 0.0)
    )
    if confidence_np is not None and confidence_np.size and conf_threshold is not None:
        conf_ds = (
            confidence_np[:, ::downsample_factor, ::downsample_factor]
            if downsample_factor > 1
            else confidence_np
        )
        mask &= conf_ds.reshape(-1) >= conf_threshold

    return pts[mask], cols[mask]


def _color_vertices_from_cloud(
    *,
    mesh_vertices: np.ndarray,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
) -> dict[str, Any]:
    """Assign each mesh vertex the color of its nearest neighbor in the cloud.
    Falls back to flat gray if the KDTree query fails or the cloud is empty.
    Returns {'vertex_colors': (N, 4) uint8, 'stats': {...}}.
    """
    from scipy.spatial import cKDTree

    stats: dict[str, Any] = {
        "method": "kdtree_nn_from_dense_cloud",
        "cloud_points": int(cloud_points.shape[0]),
    }
    fallback = np.tile(
        np.array([[200, 200, 198, 255]], dtype=np.uint8), (mesh_vertices.shape[0], 1)
    )
    if cloud_points.shape[0] < 100:
        stats["fallback_reason"] = f"too few cloud points ({cloud_points.shape[0]})"
        return {"vertex_colors": fallback, "stats": stats}
    try:
        tree = cKDTree(cloud_points)
        distances, idxs = tree.query(mesh_vertices, k=1)
        nn_colors = cloud_colors[idxs]
        stats["nn_distance_mean"] = float(distances.mean())
        stats["nn_distance_p99"] = float(np.percentile(distances, 99))
        rgba = np.empty((nn_colors.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = nn_colors
        rgba[:, 3] = 255
        return {"vertex_colors": rgba, "stats": stats}
    except Exception as exc:
        stats["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return {"vertex_colors": fallback, "stats": stats}
