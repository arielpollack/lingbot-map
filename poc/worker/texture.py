"""Bake a UV-mapped texture atlas onto a triangle mesh from posed RGB frames.

The output is a GLB with a `baseColorTexture` PBR material — Three.js'
GLTFLoader handles it natively, no custom shader needed.

Algorithm:
1. UV-unwrap the mesh with xatlas → triangle UVs in [0, 1]² atlas space.
2. Build an Open3D RaycastingScene of the mesh for per-face visibility tests.
3. For each frame: rasterize the mesh from that camera's viewpoint to get a
   FACE-ID buffer (which face is visible at each pixel).
4. For each (face, frame) pair: visibility = fraction of the face's pixels
   in that frame that show this face. View-angle weight = max(0, dot(face
   normal, -view_direction)). Score = visibility * view_angle.
5. For each atlas pixel inside a UV triangle:
   - Compute barycentric → 3D point on the face.
   - For each top-K best frame (by score): project the 3D point to the
     frame, sample bilinearly, accumulate weighted average.
   - Write the blended color to the atlas pixel.
6. Save mesh + UV + atlas image as a GLB.

Coordinate frame: assumes mesh vertices and frame extrinsics share a world
frame. We've verified this empirically for the worker's mesh.glb +
COLMAP-style w2c poses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def _dilate_atlas(
    atlas: np.ndarray, filled_mask: np.ndarray, iterations: int = 2
) -> np.ndarray:
    """Grow filled UV islands outward by a few pixels using nearest-filled
    color. Prevents mipmapping from sampling the gray fallback color along
    UV seams at distance.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return atlas

    if filled_mask.all():
        return atlas

    # For each unfilled pixel, find the index of the nearest filled pixel.
    distance, indices = distance_transform_edt(
        ~filled_mask, return_distances=True, return_indices=True
    )
    grow_mask = (~filled_mask) & (distance <= iterations)
    if not grow_mask.any():
        return atlas

    out = atlas.copy()
    out[grow_mask] = atlas[indices[0][grow_mask], indices[1][grow_mask]]
    return out


def bake_texture_atlas(
    *,
    vertices: np.ndarray,         # (V, 3) float32
    faces: np.ndarray,            # (F, 3) int32
    images_hwc_uint8: np.ndarray, # (S, H, W, 3) RGB uint8
    intrinsics: np.ndarray,       # (S, 3, 3) float
    extrinsics_w2c: np.ndarray,   # (S, 4, 4) world → camera
    output_path: str | Path,
    atlas_resolution: int = 2048,
    top_k_frames: int = 4,
    visibility_check: bool = True,
    fallback_color: tuple[int, int, int] = (200, 200, 198),
    fallback_cloud_points: np.ndarray | None = None,  # (N, 3)
    fallback_cloud_colors: np.ndarray | None = None,  # (N, 3) uint8
) -> dict[str, Any]:
    """Bake a UV-textured GLB. Returns metadata dict."""
    import open3d as o3d
    import trimesh
    import xatlas
    from PIL import Image

    start = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info: dict[str, Any] = {
        "status": "started",
        "atlas_resolution": int(atlas_resolution),
        "top_k_frames": int(top_k_frames),
        "frames": int(images_hwc_uint8.shape[0]),
        "vertices_in": int(vertices.shape[0]),
        "faces_in": int(faces.shape[0]),
    }

    # ── 1. UV unwrap ─────────────────────────────────────────────────────
    t0 = time.time()
    vmapping, atlas_indices, uvs = xatlas.parametrize(
        vertices.astype(np.float32), faces.astype(np.uint32)
    )
    info["uv_unwrap_seconds"] = round(time.time() - t0, 2)
    info["atlas_vertices"] = int(uvs.shape[0])
    # `vmapping[i]` = original vertex index for atlas vertex i.
    # `atlas_indices` = (F, 3) face indices into uvs/vmapping (atlas verts).
    # `uvs` = (atlas_V, 2) UV coords in [0, 1].
    atlas_verts_3d = vertices[vmapping]               # (atlas_V, 3) — 3D positions
    atlas_faces = atlas_indices.astype(np.int64)      # (F, 3) into atlas_V

    # Per-face normals + centroids (3D space).
    v0 = atlas_verts_3d[atlas_faces[:, 0]]
    v1 = atlas_verts_3d[atlas_faces[:, 1]]
    v2 = atlas_verts_3d[atlas_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(norms, 1e-12)
    face_centroids = (v0 + v1 + v2) / 3.0
    F = atlas_faces.shape[0]

    # ── 2. Per-frame face visibility via Open3D raycasting ───────────────
    t0 = time.time()
    rays_scene = o3d.t.geometry.RaycastingScene()
    o3d_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(atlas_verts_3d.astype(np.float32), o3d.core.Dtype.Float32),
        o3d.core.Tensor(atlas_faces.astype(np.uint32), o3d.core.Dtype.UInt32),
    )
    rays_scene.add_triangles(o3d_mesh)
    info["raycasting_setup_seconds"] = round(time.time() - t0, 2)

    S = extrinsics_w2c.shape[0]
    H, W = images_hwc_uint8.shape[1], images_hwc_uint8.shape[2]
    cam_centers = np.zeros((S, 3), dtype=np.float32)
    for i in range(S):
        R = extrinsics_w2c[i, :3, :3]
        t = extrinsics_w2c[i, :3, 3]
        cam_centers[i] = -R.T @ t  # camera position in world

    # ── 3. Score (face × frame): visibility + view-angle ─────────────────
    # Precompute scores. For each face, for each frame:
    #   1. Project face centroid to frame
    #   2. If centroid is outside frame, score = 0
    #   3. Else cast a ray from camera through pixel; if it hits THIS face
    #      (within tolerance), visible. Else occluded.
    #   4. Weight by max(0, cos(view-angle)).
    t0 = time.time()
    scores = np.zeros((F, S), dtype=np.float32)
    project_K = intrinsics.astype(np.float32)
    project_W = extrinsics_w2c.astype(np.float32)

    # Vectorized projection: for each frame, project all face centroids.
    centroids_h = np.concatenate(
        [face_centroids, np.ones((F, 1), dtype=np.float32)], axis=1
    )  # (F, 4)
    for i in range(S):
        cam = (project_W[i] @ centroids_h.T).T[:, :3]  # (F, 3)
        z = cam[:, 2]
        in_front = z > 1e-3
        u = project_K[i, 0, 0] * cam[:, 0] / np.maximum(z, 1e-3) + project_K[i, 0, 2]
        v = project_K[i, 1, 1] * cam[:, 1] / np.maximum(z, 1e-3) + project_K[i, 1, 2]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front

        # View angle: |cos(angle)| between face normal and direction TO
        # camera. ABS because Poisson reconstruction sometimes produces
        # inverted winding for indoor scenes (cameras inside the surface);
        # we don't care which way the normal points, only that the face
        # roughly faces the camera.
        view_dir = cam_centers[i] - face_centroids
        view_dir_norm = view_dir / np.maximum(
            np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-9
        )
        cos_angle = np.abs(np.sum(face_normals * view_dir_norm, axis=1))
        front_facing = cos_angle > 0.05  # exclude grazing-angle faces

        candidate_mask = in_bounds & front_facing
        if not candidate_mask.any():
            continue

        # Visibility via raycast: rays from camera through projected pixel.
        cand_idx = np.where(candidate_mask)[0]
        ray_origins = np.tile(cam_centers[i], (cand_idx.size, 1))
        # Direction = centroid - cam_center
        directions = face_centroids[cand_idx] - cam_centers[i]
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
        if visibility_check:
            rays_t = o3d.core.Tensor(
                np.concatenate([ray_origins, directions], axis=1).astype(np.float32),
                o3d.core.Dtype.Float32,
            )
            hits = rays_scene.cast_rays(rays_t)
            primitive_ids = hits["primitive_ids"].numpy()  # (N,) int32
            visible_mask = primitive_ids == cand_idx
        else:
            visible_mask = np.ones(cand_idx.size, dtype=bool)

        scores[cand_idx[visible_mask], i] = cos_angle[cand_idx[visible_mask]]
    info["scoring_seconds"] = round(time.time() - t0, 2)

    # ── 4. Pick top-K frames per face ────────────────────────────────────
    t0 = time.time()
    # Argsort along axis=1 (frames), take last K (highest scores).
    top_k = min(top_k_frames, S)
    topk_frames = np.argsort(-scores, axis=1)[:, :top_k]   # (F, K)
    topk_weights = np.take_along_axis(scores, topk_frames, axis=1)  # (F, K)
    info["topk_seconds"] = round(time.time() - t0, 2)

    # ── 5. PER-PIXEL bake (sharp textures, not flat per-face) ────────────
    # (a) Rasterize face-id image so each atlas pixel knows which face it
    #     belongs to. cv2.fillPoly per face into a uint8x3 buffer encoding
    #     face_id+1 as 24-bit RGB — handles up to 16M faces.
    # (b) Vectorized barycentric → 3D point per pixel.
    # (c) Vectorized projection through each top-K frame, bilinear sample.
    # (d) Pixels with no visible frame get NN color from the dense cloud
    #     fallback (no more clay-gray).
    t0 = time.time()
    import cv2

    uvs_px = np.empty_like(uvs)
    uvs_px[:, 0] = uvs[:, 0] * (atlas_resolution - 1)
    uvs_px[:, 1] = (1.0 - uvs[:, 1]) * (atlas_resolution - 1)

    images_f32 = images_hwc_uint8.astype(np.float32)
    project_K_arr = intrinsics.astype(np.float32)
    project_W_arr = extrinsics_w2c.astype(np.float32)

    # (a) Rasterize face-id image. Encode face_id+1 as 3-byte RGB.
    face_id_img = np.zeros((atlas_resolution, atlas_resolution, 3), dtype=np.uint8)
    triangles_int = uvs_px[atlas_faces].astype(np.int32)  # (F, 3, 2)
    for fi in range(F):
        fid = fi + 1  # 0 means empty
        color = ((fid >> 16) & 0xFF, (fid >> 8) & 0xFF, fid & 0xFF)
        cv2.fillPoly(face_id_img, [triangles_int[fi]], color=color)
    face_ids = (
        (face_id_img[..., 0].astype(np.uint32) << 16)
        | (face_id_img[..., 1].astype(np.uint32) << 8)
        | face_id_img[..., 2].astype(np.uint32)
    )
    filled = face_ids > 0
    info["pixels_filled"] = int(filled.sum())
    info["pixels_total"] = int(atlas_resolution * atlas_resolution)
    info["fill_percent"] = round(100.0 * info["pixels_filled"] / info["pixels_total"], 1)

    # (b) Per-pixel barycentric + 3D point.
    pixel_yx = np.argwhere(filled)        # (N, 2) [y, x]
    pixel_face_idx = face_ids[filled] - 1 # (N,) into atlas_faces
    N = pixel_yx.shape[0]
    if N == 0:
        atlas_normalized = np.full((atlas_resolution, atlas_resolution, 3), fallback_color, dtype=np.uint8)
        info["faces_with_color"] = 0
    else:
        # Triangle vertices for each pixel (UV pixel space + 3D world space).
        v_uv = uvs_px[atlas_faces[pixel_face_idx]]  # (N, 3, 2)
        a_uv, b_uv, c_uv = v_uv[:, 0], v_uv[:, 1], v_uv[:, 2]
        denom = (b_uv[:, 1] - c_uv[:, 1]) * (a_uv[:, 0] - c_uv[:, 0]) + (
            c_uv[:, 0] - b_uv[:, 0]
        ) * (a_uv[:, 1] - c_uv[:, 1])
        denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        xs_f = pixel_yx[:, 1].astype(np.float32) + 0.5
        ys_f = pixel_yx[:, 0].astype(np.float32) + 0.5
        w_a = ((b_uv[:, 1] - c_uv[:, 1]) * (xs_f - c_uv[:, 0]) + (c_uv[:, 0] - b_uv[:, 0]) * (ys_f - c_uv[:, 1])) / denom_safe
        w_b = ((c_uv[:, 1] - a_uv[:, 1]) * (xs_f - c_uv[:, 0]) + (a_uv[:, 0] - c_uv[:, 0]) * (ys_f - c_uv[:, 1])) / denom_safe
        w_c = 1.0 - w_a - w_b
        v_3d = atlas_verts_3d[atlas_faces[pixel_face_idx]]  # (N, 3, 3)
        pts3d = (
            w_a[:, None] * v_3d[:, 0]
            + w_b[:, None] * v_3d[:, 1]
            + w_c[:, None] * v_3d[:, 2]
        ).astype(np.float32)
        pts3d_h = np.concatenate([pts3d, np.ones((N, 1), dtype=np.float32)], axis=1)

        # (c) Per top-K frame: project all pixels, bilinear sample, accumulate.
        pixel_color = np.zeros((N, 3), dtype=np.float32)
        pixel_weight = np.zeros(N, dtype=np.float32)
        for k in range(top_k):
            frame_per_pixel = topk_frames[pixel_face_idx, k]  # (N,)
            weight_per_pixel = topk_weights[pixel_face_idx, k]  # (N,)
            valid_w = weight_per_pixel > 0
            if not valid_w.any():
                continue
            K_per = project_K_arr[frame_per_pixel]
            W_per = project_W_arr[frame_per_pixel]
            cam = np.einsum("nij,nj->ni", W_per, pts3d_h)[:, :3]
            z = cam[:, 2]
            valid = valid_w & (z > 1e-3)
            u = K_per[:, 0, 0] * cam[:, 0] / np.maximum(z, 1e-3) + K_per[:, 0, 2]
            v = K_per[:, 1, 1] * cam[:, 1] / np.maximum(z, 1e-3) + K_per[:, 1, 2]
            in_image = valid & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
            if not in_image.any():
                continue
            sel = np.where(in_image)[0]
            u_v = u[sel]
            v_v = v[sel]
            u0 = np.floor(u_v).astype(np.int32); u1 = u0 + 1
            v0_ = np.floor(v_v).astype(np.int32); v1_ = v0_ + 1
            du = u_v - u0; dv = v_v - v0_
            f_sel = frame_per_pixel[sel]
            rgb = (
                (1 - du)[:, None] * (1 - dv)[:, None] * images_f32[f_sel, v0_, u0]
                + du[:, None] * (1 - dv)[:, None] * images_f32[f_sel, v0_, u1]
                + (1 - du)[:, None] * dv[:, None] * images_f32[f_sel, v1_, u0]
                + du[:, None] * dv[:, None] * images_f32[f_sel, v1_, u1]
            )
            wt = weight_per_pixel[sel][:, None]
            pixel_color[sel] += wt * rgb
            pixel_weight[sel] += weight_per_pixel[sel]

        # (d) Pixels with no frame contribution: NN fallback from cloud.
        no_frame = pixel_weight <= 0
        if no_frame.any() and fallback_cloud_points is not None and fallback_cloud_points.shape[0] > 100:
            from scipy.spatial import cKDTree
            tree = cKDTree(fallback_cloud_points)
            _, nn_idx = tree.query(pts3d[no_frame], k=1)
            pixel_color[no_frame] = fallback_cloud_colors[nn_idx].astype(np.float32)
            pixel_weight[no_frame] = 1.0
            info["pixels_from_cloud_nn"] = int(no_frame.sum())
        info["pixels_from_frames"] = int(N - no_frame.sum())

        # Normalize and write to atlas
        final_pixel = (pixel_color / np.maximum(pixel_weight[:, None], 1e-6)).clip(0, 255).astype(np.uint8)
        atlas = np.full((atlas_resolution, atlas_resolution, 3), fallback_color, dtype=np.uint8)
        atlas[pixel_yx[:, 0], pixel_yx[:, 1]] = final_pixel
        atlas_normalized = atlas
        info["faces_with_color"] = int(np.unique(pixel_face_idx).size)
    weight_acc = filled.astype(np.float32)

    # Dilate UV islands by a few pixels so mipmapping doesn't bleed the
    # background fallback color into the rendered surface near UV-seam edges.
    filled_mask = weight_acc > 0
    atlas_normalized = _dilate_atlas(atlas_normalized, filled_mask, iterations=2)

    # Fill any remaining unfilled pixels with fallback color
    if not filled_mask.all():
        # Re-check after dilation
        try:
            from scipy.ndimage import binary_dilation
            still_unfilled = ~binary_dilation(filled_mask, iterations=2)
        except Exception:
            still_unfilled = ~filled_mask
        atlas_normalized[still_unfilled] = fallback_color

    info["bake_seconds"] = round(time.time() - t0, 2)

    # ── 6. Export GLB with PBR baseColorTexture ──────────────────────────
    t0 = time.time()
    atlas_image = Image.fromarray(atlas_normalized)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=atlas_image, baseColorFactor=(255, 255, 255, 255)
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uvs, material=material)
    out_mesh = trimesh.Trimesh(
        vertices=atlas_verts_3d,
        faces=atlas_faces,
        process=False,
        visual=visual,
    )
    out_mesh.export(str(output_path))
    info["export_seconds"] = round(time.time() - t0, 2)
    info["size_mb"] = round(output_path.stat().st_size / 1e6, 2)
    info["status"] = "ok"
    info["total_seconds"] = round(time.time() - start, 2)
    return info
