"""Tier 1 (LiDAR) pipeline.

Phone capture bundle → fused colored point cloud → mesh + texture + 3DGS.
Skips lingbot-map entirely: ARKit + LiDAR already give us ground-truth poses
and depth, so we plug those directly into the same downstream code that the
raw-video pipeline runs after its own depth/pose inference step.

Bundle layout (zipped, written by ios_app):
    manifest.json
    frames/{000000..N}.jpg          # RGB
    poses/{000000..N}.json           # intrinsics, c2w transform, image/depth res
    depth/{000000..N}.bin            # float32 LE meters, depth_h * depth_w
    depth/{000000..N}.conf           # uint8 ARConfidence (0/1/2)

The "prepared" dict format consumed by glb_export / mesh.py / gsplat.py:
    depth         (S, H, W)        float32  LiDAR meters
    depth_conf    (S, H, W)        float32  0..2 (ARConfidence)
    intrinsic     (S, 3, 3)        float32  scaled to depth resolution
    extrinsic     (S, 3, 4)        float32  c2w
    images        (S, 3, H, W)     float32  in [0, 1], resized to depth res
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from poc.app.config import require_env
from poc.app.r2 import R2Client


def process_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = payload["run_id"]
    input_bundle_key = payload["input_bundle_key"]
    output_prefix = (payload.get("output_prefix") or f"runs/{run_id}/").rstrip("/")
    options = dict(payload.get("options") or {})

    work_dir = Path(tempfile.mkdtemp(prefix=f"lingbot-bundle-{run_id}-"))
    r2 = R2Client(require_env())

    try:
        bundle_zip = work_dir / "bundle.zip"
        r2.download_file(input_bundle_key, bundle_zip)

        bundle_dir = work_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip) as z:
            z.extractall(bundle_dir)

        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Bundle missing manifest.json")
        manifest = json.loads(manifest_path.read_text())
        tier = manifest.get("tier", "lidar")

        if tier == "lidar_mesh":
            return _process_lidar_mesh(
                run_id=run_id,
                bundle_dir=bundle_dir,
                work_dir=work_dir,
                output_prefix=output_prefix,
                options=options,
                manifest=manifest,
                r2=r2,
            )
        if tier == "lidar":
            return _process_lidar(
                run_id=run_id,
                bundle_dir=bundle_dir,
                work_dir=work_dir,
                output_prefix=output_prefix,
                options=options,
                manifest=manifest,
                r2=r2,
            )
        if tier == "vio":
            raise NotImplementedError(
                "Tier 2 (VIO without depth) is not implemented yet — "
                "phone needs a LiDAR-capable device, or use the raw-video upload."
            )
        raise ValueError(f"Unknown bundle tier: {tier}")
    finally:
        if os.getenv("KEEP_WORKDIR") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)


def _process_lidar(
    *,
    run_id: str,
    bundle_dir: Path,
    work_dir: Path,
    output_prefix: str,
    options: dict[str, Any],
    manifest: dict[str, Any],
    r2: R2Client,
) -> dict[str, Any]:
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _bundle_to_prepared(bundle_dir, manifest)

    # Tier 1 default options — override the raw-video defaults that assume
    # noisy lingbot-map predictions.
    options.setdefault("downsample_factor", 1)         # LiDAR is already low-res
    options.setdefault("conf_percentile", 50.0)        # works fine on AR conf {0,1,2}
    options.setdefault("splat_init_conf_threshold", 1.5)  # medium+high
    options.setdefault("mesh_enabled", True)

    # Scene point cloud (GLB) — same exporter the raw-video path uses.
    from lingbot_map.vis.glb_export import export_predictions_glb_file

    conf_arr = prepared["depth_conf"]
    conf_pct = float(options["conf_percentile"])
    conf_threshold = float(np.percentile(conf_arr, conf_pct)) if conf_arr.size else 1.0

    glb_path = output_dir / "scene.glb"
    export = export_predictions_glb_file(
        prepared,
        str(glb_path),
        conf_threshold=conf_threshold,
        downsample_factor=int(options["downsample_factor"]),
    )
    export["conf_threshold_used"] = conf_threshold
    export["conf_percentile_used"] = conf_pct

    # Mesh (Poisson + texture). Reuses the exact same code path as the raw-video
    # pipeline; we just hand it ground-truth depth + poses instead of predicted.
    from poc.worker.mesh import create_clay_mesh_from_prepared

    mesh_info = create_clay_mesh_from_prepared(
        prepared, output_dir, options, scene_glb_path=glb_path
    )

    # 3D Gaussian Splatting. Reuses the existing splat phase verbatim.
    from poc.worker.run_reconstruction import _run_splat_phase

    splat_info = _run_splat_phase(prepared, output_dir, options)

    # Upload artifacts.
    metadata = {
        "tier": "lidar",
        "manifest": manifest,
        "options": dict(options),
        "frame_count": int(prepared["depth"].shape[0]),
        "depth_resolution": list(prepared["depth"].shape[1:]),
        "export": export,
        "mesh": mesh_info,
        "splat": splat_info,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    scene_key = f"{output_prefix}/scene.glb"
    metadata_key = f"{output_prefix}/metadata.json"
    r2.upload_file(glb_path, scene_key, content_type="model/gltf-binary")
    r2.upload_file(metadata_path, metadata_key, content_type="application/json")

    mesh_key: str | None = None
    mesh_path = mesh_info.get("path")
    if mesh_path and Path(mesh_path).exists():
        mesh_key = f"{output_prefix}/mesh.glb"
        r2.upload_file(mesh_path, mesh_key, content_type="model/gltf-binary")

    splat_key: str | None = None
    splat_gz_path = splat_info.get("gz_path")
    if splat_gz_path and Path(splat_gz_path).exists():
        splat_key = f"{output_prefix}/splat.ply"
        r2.upload_file(
            splat_gz_path,
            splat_key,
            content_type="application/octet-stream",
            content_encoding="gzip",
        )

    diagnostic_keys: list[str] = []
    for artifact in splat_info.get("diagnostic_artifacts", []):
        artifact_path = artifact.get("path")
        if not artifact_path or not Path(artifact_path).exists():
            continue
        name = Path(artifact.get("name") or Path(artifact_path).name).name
        artifact_key = f"{output_prefix}/diagnostics/{name}"
        r2.upload_file(
            artifact_path,
            artifact_key,
            content_type=artifact.get("content_type") or "application/octet-stream",
            content_encoding=artifact.get("content_encoding"),
        )
        diagnostic_keys.append(artifact_key)

    return {
        "run_id": run_id,
        "scene_key": scene_key,
        "mesh_key": mesh_key,
        "splat_key": splat_key,
        "diagnostic_keys": diagnostic_keys,
        "metadata_key": metadata_key,
        "metadata": metadata,
    }


def _process_lidar_mesh(
    *,
    run_id: str,
    bundle_dir: Path,
    work_dir: Path,
    output_prefix: str,
    options: dict[str, Any],
    manifest: dict[str, Any],
    r2: R2Client,
) -> dict[str, Any]:
    """Tier `lidar_mesh`: ARKit fused the LiDAR depth into a mesh on-device,
    so the bundle ships `mesh.obj` + RGB keyframes + ARKit poses. Server
    just bakes a texture atlas — no depth fusion, no Poisson reconstruction,
    no lingbot-map.

    Bundle layout (`tier=lidar_mesh`):
        manifest.json        # tier, frame_count, image_resolution, mesh meta
        mesh.obj             # ARMeshAnchor union, transformed to world
        frames/{N}.jpg       # RGB keyframes (~2 fps)
        poses/{N}.json       # intrinsics @ image_res + transform_c2w (ARKit)
    """
    import time
    import cv2
    import trimesh

    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    timings: dict[str, float] = {}

    def _phase(name: str, t0: float) -> None:
        elapsed = round(time.time() - t0, 2)
        timings[name] = elapsed
        print(f"[lidar_mesh] {name}: {elapsed}s", flush=True)

    mesh_path_in = bundle_dir / "mesh.obj"
    if not mesh_path_in.exists():
        raise ValueError("Bundle missing mesh.obj — phone did not produce a fused mesh")

    # ── Load mesh (vertices + faces only — texture baker computes the rest). ──
    t0 = time.time()
    raw_mesh = trimesh.load(mesh_path_in, process=False, force="mesh")
    if not isinstance(raw_mesh, trimesh.Trimesh):
        raise ValueError(f"mesh.obj loaded as {type(raw_mesh).__name__}, expected Trimesh")
    vertices = np.asarray(raw_mesh.vertices, dtype=np.float32)
    faces = np.asarray(raw_mesh.faces, dtype=np.int32)

    # Keep the original ARKit (Y-up) vertices for splat init / sampling. The
    # rotated copy below is what we feed into the texture baker.
    vertices_arkit = vertices.copy()
    mesh_arkit = trimesh.Trimesh(vertices=vertices_arkit, faces=faces, process=False)

    # ARKit world is Y-up; the existing /mesh viewer hard-rotates the GLB by
    # 180° around X (`gltf.scene.rotation.x = Math.PI`) on the assumption
    # that meshes are Y-down. To keep that viewer code tier-agnostic we
    # pre-rotate the ARKit mesh by 180° around X: viewer's rotation undoes
    # ours, the mesh ends up oriented correctly.
    #
    # Critically, this is a real ROTATION (det=+1), not a Y reflection
    # (det=-1). A single-axis flip mirrors the geometry — letters appear
    # backwards. A 180° rotation around X flips both Y AND Z, keeps
    # handedness, and produces the correct view.
    rot_x_180 = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ], dtype=np.float32)
    verts_h = np.hstack([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)])
    vertices = (verts_h @ rot_x_180.T)[:, :3].astype(np.float32)
    raw_mesh.vertices = vertices  # used later for scene.glb point sample
    # No face winding flip — proper rotations preserve triangle orientation.
    _phase("load_mesh", t0)
    print(
        f"[lidar_mesh] mesh: {vertices.shape[0]} verts, {faces.shape[0]} faces",
        flush=True,
    )

    # ── Load keyframes + poses + depth (if any). ──
    t0 = time.time()
    pose_files = sorted((bundle_dir / "poses").glob("*.json"))
    if not pose_files:
        raise ValueError("Bundle has no pose files for texture baking")

    # Look up depth_resolution / depth files. Older bundles (manifest
    # version <= 2) don't ship depth at all; the splat trainer still works
    # but skips depth supervision in that case.
    depth_dir = bundle_dir / "depth"
    depth_files_by_idx: dict[str, Path] = {}
    conf_files_by_idx: dict[str, Path] = {}
    if depth_dir.exists():
        for p in depth_dir.glob("*.bin"):
            depth_files_by_idx[p.stem] = p
        for p in depth_dir.glob("*.conf"):
            conf_files_by_idx[p.stem] = p

    # Cap the frames we feed to the texture baker. Per-frame visibility
    # raycasting is O(frames * faces * atlas_pixels); top_k=4 means more
    # frames don't actually improve the atlas after ~30 well-spread views.
    max_frames = int(options.get("texture_max_frames", 30))
    if len(pose_files) > max_frames:
        idxs = np.linspace(0, len(pose_files) - 1, max_frames).astype(int)
        pose_files = [pose_files[i] for i in idxs]
        print(
            f"[lidar_mesh] subsampled keyframes {len(idxs)}/"
            f"{len(list((bundle_dir / 'poses').glob('*.json')))}",
            flush=True,
        )

    images: list[np.ndarray] = []
    intrinsics_list: list[np.ndarray] = []
    # Two pose sets:
    #   _texture: poses live in the rot_x_180-rotated world frame so the mesh
    #     viewer's `gltf.scene.rotation.x = Math.PI` ends up putting the
    #     scene right-side up.
    #   _splat: poses live in the ARKit-native world (Y-up) so the splat
    #     viewer (which doesn't apply any compensating rotation) renders the
    #     trained Gaussians right-side up.
    extrinsics_w2c_for_texture: list[np.ndarray] = []
    extrinsics_w2c_for_splat: list[np.ndarray] = []
    depths_per_frame: list[np.ndarray | None] = []
    confs_per_frame: list[np.ndarray | None] = []

    image_w, image_h = manifest["image_resolution"]
    manifest_depth_res = manifest.get("depth_resolution")
    depth_w_canon = depth_h_canon = None
    if isinstance(manifest_depth_res, (list, tuple)) and len(manifest_depth_res) == 2:
        depth_w_canon, depth_h_canon = int(manifest_depth_res[0]), int(manifest_depth_res[1])

    # Two transforms compose:
    #   gl_to_cv: ARKit camera frame (X right, Y up, Z back, GL-style)
    #             → OpenCV camera frame (X right, Y down, Z forward).
    #   rot_x_180: same 180° X rotation we applied to the mesh vertices, so
    #             the cameras stay in the same world frame as the geometry.
    gl_to_cv_4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    # rot_x_180 is identical to the matrix used for the mesh above; recompute
    # locally so this function reads top-to-bottom.
    rot_x_180 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    for pose_file in pose_files:
        idx = pose_file.stem
        pose = json.loads(pose_file.read_text())
        K_image = np.asarray(pose["intrinsics"], dtype=np.float32)  # at image res
        T_c2w = np.asarray(pose["transform_c2w"], dtype=np.float32)  # 4×4 ARKit

        # OpenCV camera → ARKit world (T_c2w @ gl_to_cv).
        T_c2w_arkit = T_c2w @ gl_to_cv_4
        # ... → rotated world for the texture-baked mesh.
        T_c2w_rotated = rot_x_180 @ T_c2w_arkit
        extrinsics_w2c_for_texture.append(np.linalg.inv(T_c2w_rotated).astype(np.float32))
        extrinsics_w2c_for_splat.append(np.linalg.inv(T_c2w_arkit).astype(np.float32))

        frame_file = bundle_dir / "frames" / f"{idx}.jpg"
        bgr = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Frame {frame_file} could not be decoded")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Frames may already be downsampled by the iOS app (jpegLongSideCap=1280).
        # Scale intrinsics to whatever resolution the actual JPEG carries —
        # the texture baker uses (intrinsic, image) shape consistently.
        sx = w / float(image_w)
        sy = h / float(image_h)
        K = K_image.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        intrinsics_list.append(K)
        images.append(rgb)

        depth_arr = None
        conf_arr = None
        depth_path = depth_files_by_idx.get(idx)
        if depth_path is not None:
            raw = np.fromfile(depth_path, dtype="<f4")
            if depth_w_canon is None or depth_h_canon is None:
                # Infer from first depth file when manifest doesn't declare it.
                # Most LiDAR sensors are 192×256 (LiDAR Pro) or 256×192;
                # iPhone ARKit reports (w, h) so use that ordering.
                # Fall back to 192×256 if the size doesn't match either.
                total = raw.size
                for cand_w, cand_h in [(256, 192), (192, 256), (320, 240), (240, 320)]:
                    if cand_w * cand_h == total:
                        depth_w_canon, depth_h_canon = cand_w, cand_h
                        break
            if depth_w_canon and depth_h_canon and raw.size == depth_w_canon * depth_h_canon:
                depth_arr = raw.reshape(depth_h_canon, depth_w_canon).astype(np.float32)
                conf_path = conf_files_by_idx.get(idx)
                if conf_path is not None:
                    conf_raw = np.fromfile(conf_path, dtype=np.uint8)
                    if conf_raw.size == depth_w_canon * depth_h_canon:
                        conf_arr = conf_raw.reshape(depth_h_canon, depth_w_canon)
            else:
                print(
                    f"[lidar_mesh] depth file {depth_path.name} size {raw.size} "
                    f"doesn't match expected dims — skipping depth supervision for it",
                    flush=True,
                )

        depths_per_frame.append(depth_arr)
        confs_per_frame.append(conf_arr)

    images_hwc = np.stack(images, axis=0)  # (S, H, W, 3) uint8
    intrinsics_arr = np.stack(intrinsics_list, axis=0)  # (S, 3, 3)
    extrinsics_w2c_arr = np.stack(extrinsics_w2c_for_texture, axis=0)  # rotated world
    extrinsics_w2c_arkit_arr = np.stack(extrinsics_w2c_for_splat, axis=0)  # ARKit world
    n_with_depth = sum(1 for d in depths_per_frame if d is not None)
    _phase("load_frames", t0)
    print(
        f"[lidar_mesh] frames={images_hwc.shape[0]} @ {images_hwc.shape[1:3]} "
        f"· depth_maps={n_with_depth} @ "
        f"{(depth_h_canon, depth_w_canon) if depth_h_canon else 'n/a'}",
        flush=True,
    )

    # ── Bake texture. ──
    # 2048 atlas with the 30-frame cap is the sweet spot: per-face visibility
    # raycasting is O(frames * faces * face_pixels) and we capped frames; the
    # atlas pixel-fill is the dominant phase and 2048 → sharp textures.
    # Drop to 1024 via options.texture_atlas_resolution=1024 if you need
    # faster turnaround.
    t0 = time.time()
    from poc.worker.texture import bake_texture_atlas

    mesh_glb_path = output_dir / "mesh.glb"
    bake_info = bake_texture_atlas(
        vertices=vertices,
        faces=faces,
        images_hwc_uint8=images_hwc,
        intrinsics=intrinsics_arr,
        extrinsics_w2c=extrinsics_w2c_arr,
        output_path=mesh_glb_path,
        atlas_resolution=int(options.get("texture_atlas_resolution", 2048)),
        top_k_frames=int(options.get("texture_top_k_frames", 4)),
    )
    _phase("bake_texture", t0)

    # ── Sample the mesh for scene.glb point-cloud view. ──
    t0 = time.time()
    sample_count = int(options.get("scene_point_count", 200_000))
    sampled = raw_mesh.sample(min(sample_count, max(1, faces.shape[0]) * 3))
    scene_glb_path = output_dir / "scene.glb"
    point_cloud = trimesh.points.PointCloud(sampled)
    point_cloud.export(scene_glb_path)
    _phase("sample_scene_glb", t0)

    # ── 3DGS training (the photoreal pass). ──
    # Initialized from a sample of the ARKit-Y-up mesh; trained with the
    # ARKit-world poses (different from the texture-bake poses, which are in
    # the rot_x_180-rotated world). The splat viewer doesn't apply any axis
    # rotation, so the result is rendered correctly Y-up.
    #
    # The new gsplat-library trainer (replacing graphdeco's subprocess
    # train.py) takes per-camera depth + conf when present, prunes the
    # output Gaussians against `mesh_arkit` to remove floaters, and emits
    # the antimatter15-style `.splat` quantized format directly. See
    # `_splat_phase_lidar_mesh` for the full sequence.
    splat_info: dict[str, Any] = {"enabled": False}
    splat_path: str | None = None
    if bool(options.get("splat_enabled", True)):
        t0 = time.time()
        splat_info = _splat_phase_lidar_mesh(
            mesh_arkit=mesh_arkit,
            images_hwc=images_hwc,
            intrinsics_arr=intrinsics_arr,
            extrinsics_w2c_arkit=extrinsics_w2c_arkit_arr,
            depths_per_frame=depths_per_frame,
            confs_per_frame=confs_per_frame,
            depth_resolution=(depth_h_canon, depth_w_canon),
            output_dir=output_dir,
            options=options,
            timings=timings,
        )
        splat_path = splat_info.get("splat_path")
        _phase("splat_total", t0)
    else:
        print("[lidar_mesh] splat disabled by options.splat_enabled=false", flush=True)

    # ── Upload + metadata. ──
    t0 = time.time()
    metadata = {
        "tier": "lidar_mesh",
        "manifest": manifest,
        "options": dict(options),
        "frame_count": int(images_hwc.shape[0]),
        "mesh": {
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
        },
        "texture": bake_info,
        "splat": splat_info,
        "timings": timings,
        "duration_seconds": round(time.time() - start, 3),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    scene_key = f"{output_prefix}/scene.glb"
    metadata_key = f"{output_prefix}/metadata.json"
    mesh_key = f"{output_prefix}/mesh.glb"
    r2.upload_file(scene_glb_path, scene_key, content_type="model/gltf-binary")
    r2.upload_file(mesh_glb_path, mesh_key, content_type="model/gltf-binary")
    r2.upload_file(metadata_path, metadata_key, content_type="application/json")

    splat_key: str | None = None
    if splat_path and Path(splat_path).exists():
        splat_key = f"{output_prefix}/splat.splat"
        r2.upload_file(
            splat_path,
            splat_key,
            content_type="application/octet-stream",
        )

    _phase("upload", t0)
    print(
        f"[lidar_mesh] DONE in {round(time.time() - start, 2)}s — "
        f"timings={timings}",
        flush=True,
    )

    return {
        "run_id": run_id,
        "scene_key": scene_key,
        "mesh_key": mesh_key,
        "splat_key": splat_key,
        "diagnostic_keys": [],
        "metadata_key": metadata_key,
        "metadata": metadata,
    }


def _splat_phase_lidar_mesh(
    *,
    mesh_arkit,
    images_hwc: "np.ndarray",
    intrinsics_arr: "np.ndarray",
    extrinsics_w2c_arkit: "np.ndarray",
    depths_per_frame: list,
    confs_per_frame: list,
    depth_resolution: tuple[int | None, int | None],
    output_dir: Path,
    options: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    """Train 3D Gaussian Splatting on top of the ARKit fused mesh.

    Three-stage pipeline:

      1. **train** — `gsplat_trainer.train_lidar_mesh_splat`.
         Init points sampled uniformly from the ARKit-Y-up mesh surface,
         mid-gray init colors. RGB + (optional) LiDAR depth supervision at
         depth resolution. sh_degree=0 (no SH).
      2. **prune** — `splat_prune.prune_gaussians_with_mesh`.
         Drop Gaussians > τ meters from the nearest mesh face. The single
         most effective lever against floaters / haze.
      3. **encode** — `splat_format.encode_splat_file`.
         Pack the pruned Gaussians into the antimatter15 `.splat` binary
         format (32 B/Gaussian, no SH, viewer-native).

    Returns a dict with `splat_path` set on success, `error` on failure.
    Failures are logged but never raised — the run still completes with
    the textured mesh artifact.
    """
    import time
    import cv2
    import trimesh

    info: dict[str, Any] = {"enabled": True}
    try:
        from poc.worker.gsplat_trainer import Camera, train_lidar_mesh_splat
        from poc.worker.splat_format import encode_splat_file
        from poc.worker.splat_prune import prune_gaussians_with_mesh

        # ── 1. Sample mesh for init points ────────────────────────────────
        n_init = int(options.get("splat_init_points", 200_000))
        sample_count = max(n_init, 1)
        sampled, _face_idx = trimesh.sample.sample_surface(mesh_arkit, sample_count)
        init_xyz = np.asarray(sampled, dtype=np.float32)
        # Mid-gray seed colors — gsplat overrides them within the first few
        # hundred iterations from the gt_images, so seed quality doesn't
        # matter for final appearance.
        init_rgb = np.full((init_xyz.shape[0], 3), 128, dtype=np.uint8)

        # ── 2. Build training cameras at depth resolution ─────────────────
        # Determine the training render resolution. Prefer LiDAR depth
        # resolution when present (matches the depth supervision natively),
        # otherwise downsample to a fixed budget — rendering at full
        # 1280×720 makes each gsplat step ~50× more expensive.
        depth_h, depth_w = depth_resolution if depth_resolution[0] else (None, None)
        n_with_depth = sum(1 for d in depths_per_frame if d is not None)
        S, H_img, W_img = images_hwc.shape[:3]
        if depth_h and depth_w and n_with_depth == S:
            train_h, train_w = depth_h, depth_w
        else:
            # No depth (or partial) → no depth supervision; pick a small
            # render resolution similar to LiDAR spatial resolution.
            train_w = int(options.get("splat_render_width", 256))
            train_h = max(1, int(round(H_img * (train_w / W_img))))

        sx = train_w / float(W_img)
        sy = train_h / float(H_img)

        cameras: list[Camera] = []
        for i in range(S):
            img_resized = cv2.resize(
                images_hwc[i], (train_w, train_h), interpolation=cv2.INTER_AREA
            )
            K = intrinsics_arr[i].copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            cam = Camera(
                image=np.ascontiguousarray(img_resized),
                intrinsic=K.astype(np.float32),
                extrinsic_w2c=extrinsics_w2c_arkit[i].astype(np.float32),
                depth=depths_per_frame[i],
                depth_conf=confs_per_frame[i],
            )
            cameras.append(cam)

        iterations = int(options.get("splat_iterations", 7000))
        depth_lambda = float(options.get("splat_depth_lambda", 0.2))
        densify_grad = float(options.get("splat_densify_grad", 0.0004))

        # ── 3. Train ──────────────────────────────────────────────────────
        t0 = time.time()
        train_out = train_lidar_mesh_splat(
            init_xyz=init_xyz,
            init_rgb=init_rgb,
            cameras=cameras,
            iterations=iterations,
            depth_lambda=depth_lambda,
            densify_grad_threshold=densify_grad,
        )
        timings["splat_train"] = round(time.time() - t0, 2)
        print(f"[lidar_mesh] splat_train: {timings['splat_train']}s", flush=True)

        # ── 4. Prune against mesh ─────────────────────────────────────────
        t0 = time.time()
        threshold = float(options.get("splat_prune_threshold", 0.12))
        try:
            pruned, prune_stats = prune_gaussians_with_mesh(
                means=train_out["means"],
                quats_wxyz=train_out["quats_wxyz"],
                scales=train_out["scales"],
                opacities=train_out["opacities"],
                rgb=train_out["rgb"],
                mesh=mesh_arkit,
                threshold=threshold,
            )
        except Exception as prune_exc:
            # Pruning failure shouldn't kill the run — fall back to the
            # un-pruned splat. Floaters will be visible but the model still
            # exists.
            import traceback
            traceback.print_exc()
            print(
                f"[lidar_mesh] splat_prune FAILED: {prune_exc} — using unpruned splat",
                flush=True,
            )
            pruned = {
                "means": train_out["means"],
                "quats_wxyz": train_out["quats_wxyz"],
                "scales": train_out["scales"],
                "opacities": train_out["opacities"],
                "rgb": train_out["rgb"],
            }
            prune_stats = {"error": f"{type(prune_exc).__name__}: {prune_exc}"}
        timings["splat_prune"] = round(time.time() - t0, 2)
        print(
            f"[lidar_mesh] splat_prune: {timings['splat_prune']}s · "
            f"kept {prune_stats.get('kept', '?')}/{prune_stats.get('input', '?')} "
            f"(pruned {prune_stats.get('pruned', '?')})",
            flush=True,
        )

        # ── 5. Encode .splat ──────────────────────────────────────────────
        t0 = time.time()
        splat_path = output_dir / "splat.splat"
        # Sort by distance from mesh centroid so that even before the
        # viewer's GPU sort kicks in, the initial appearance is sane.
        centroid = mesh_arkit.centroid.astype(np.float32)
        encode_info = encode_splat_file(
            means=pruned["means"],
            quats_wxyz=pruned["quats_wxyz"],
            scales=pruned["scales"],
            opacities=pruned["opacities"],
            rgb=pruned["rgb"],
            output_path=str(splat_path),
            sort_by_distance_to=tuple(float(c) for c in centroid),
        )
        timings["splat_encode"] = round(time.time() - t0, 2)
        print(
            f"[lidar_mesh] splat_encode: {timings['splat_encode']}s · "
            f"{encode_info['gaussians']} gaussians · "
            f"{round(encode_info['bytes'] / 1e6, 2)} MB",
            flush=True,
        )

        info.update(
            {
                "splat_path": str(splat_path),
                "iterations": iterations,
                "depth_lambda": depth_lambda,
                "densify_grad_threshold": densify_grad,
                "prune_threshold": threshold,
                "render_resolution": [int(train_h), int(train_w)],
                "depth_supervision": train_out["stats"]["depth_supervision"],
                "train_stats": train_out["stats"],
                "prune_stats": prune_stats,
                "encode": encode_info,
                "size_mb": round(encode_info["bytes"] / 1e6, 2),
                "format": "splat",
            }
        )
        return info
    except Exception as exc:
        import traceback
        traceback.print_exc()
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info


def _bundle_to_prepared(bundle_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Read frames + poses + depth from the extracted bundle into the prepared
    dict format the existing pipeline functions consume.

    Resizing convention: RGB frames are downsampled to depth resolution and
    intrinsics are rescaled accordingly. This matches the existing pipeline
    where lingbot-map outputs everything at a single working resolution
    (`predictions["images"]` and `predictions["depth"]` already share H×W).
    """
    import cv2

    frames_dir = bundle_dir / "frames"
    poses_dir = bundle_dir / "poses"
    depth_dir = bundle_dir / "depth"

    pose_files = sorted(poses_dir.glob("*.json"))
    if not pose_files:
        raise ValueError("Bundle has no pose files")

    image_w, image_h = manifest["image_resolution"]
    depth_w, depth_h = manifest["depth_resolution"]
    if depth_w == 0 or depth_h == 0:
        raise ValueError("Manifest reports zero depth resolution — likely a non-LiDAR capture")

    # Note: lingbot_map's `unproject_depth_map_to_point_map` indexes per-frame
    # then unconditionally calls .squeeze(-1) on the depth, so depth must have
    # a trailing singleton dim (its docstring's (S,H,W) alternative doesn't
    # actually work). Same trailing dim convention for depth_conf to match.
    S = len(pose_files)
    images = np.zeros((S, 3, depth_h, depth_w), dtype=np.float32)
    depths = np.zeros((S, depth_h, depth_w, 1), dtype=np.float32)
    confs = np.zeros((S, depth_h, depth_w, 1), dtype=np.float32)
    intrinsics = np.zeros((S, 3, 3), dtype=np.float32)
    extrinsics = np.zeros((S, 3, 4), dtype=np.float32)

    sx = depth_w / float(image_w)
    sy = depth_h / float(image_h)

    # ARKit camera frame is GL-style (X right, Y up, Z back). The downstream
    # unproject + projection code (lingbot_map.utils.geometry, gsplat) expects
    # OpenCV convention (X right, Y down, Z forward). Compose the c2w with a
    # diag(1,-1,-1) flip on the right so a point given in OpenCV camera frame
    # ends up in world correctly. Intrinsics need no flip — both conventions
    # use top-left pixel origin with X right, Y down in image space.
    gl_to_cv = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

    for i, pose_file in enumerate(pose_files):
        idx = pose_file.stem  # "000000"
        pose = json.loads(pose_file.read_text())

        K_image = np.asarray(pose["intrinsics"], dtype=np.float32)  # 3x3 at image res
        K = K_image.copy()
        K[0, 0] *= sx  # fx
        K[0, 2] *= sx  # cx
        K[1, 1] *= sy  # fy
        K[1, 2] *= sy  # cy
        intrinsics[i] = K

        T_c2w = np.asarray(pose["transform_c2w"], dtype=np.float32)  # 4x4 ARKit
        R = T_c2w[:3, :3] @ gl_to_cv  # OpenCV camera frame → world
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = T_c2w[:3, 3]

        frame_file = frames_dir / f"{idx}.jpg"
        bgr = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Frame {frame_file} could not be decoded")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        images[i] = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

        depth_file = depth_dir / f"{idx}.bin"
        depth_arr = np.fromfile(depth_file, dtype="<f4")
        if depth_arr.size != depth_h * depth_w:
            raise ValueError(
                f"Depth file {depth_file} has {depth_arr.size} floats, "
                f"expected {depth_h * depth_w}"
            )
        depths[i, ..., 0] = depth_arr.reshape(depth_h, depth_w)

        conf_file = depth_dir / f"{idx}.conf"
        if conf_file.exists():
            conf_arr = np.fromfile(conf_file, dtype=np.uint8)
            if conf_arr.size == depth_h * depth_w:
                confs[i, ..., 0] = conf_arr.reshape(depth_h, depth_w).astype(np.float32)
            else:
                confs[i, ..., 0] = 2.0
        else:
            confs[i, ..., 0] = 2.0

    return {
        "depth": depths,
        "depth_conf": confs,
        "intrinsic": intrinsics,
        "extrinsic": extrinsics,
        "images": images,
    }
