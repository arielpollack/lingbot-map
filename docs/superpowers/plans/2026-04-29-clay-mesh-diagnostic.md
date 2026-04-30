# Clay Mesh Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a coarse clay-surface mesh artifact and browser viewer to validate whether LingBot-MAP depth/pose geometry can produce a walkable property shell before relying on GSplat display quality.

**Architecture:** Keep the locked point-cloud export untouched. Add `poc.worker.mesh` as an optional post-processing stage fed by the already prepared depth, intrinsics, extrinsics, confidence, and frames. Upload `mesh.glb` beside `scene.glb` and `splat.ply`, expose `/mesh?run=...`, and link it from completed runs when present.

**Tech Stack:** Python, NumPy, Open3D TSDF fusion on Modal, Trimesh GLB export, FastAPI artifact streaming, Three.js GLTF viewer, pytest.

---

### Task 1: Mesh Artifact Contract

**Files:**
- Create: `tests/poc/test_mesh.py`
- Modify: `tests/poc/test_worker_handler.py`

- [x] Write tests for mesh stage metadata and upload behavior.
- [x] Run the new tests and confirm they fail because `poc.worker.mesh` and `mesh_key` do not exist yet.

### Task 2: Mesh Generation Module

**Files:**
- Create: `poc/worker/mesh.py`
- Modify: `poc/worker/run_reconstruction.py`
- Modify: `poc/worker/modal_app.py`

- [x] Implement Open3D TSDF fusion from depth + camera poses.
- [x] Export a simplified grey/clay GLB via trimesh.
- [x] Return structured metadata: enabled, status/error, vertex_count, face_count, size_mb, settings.
- [x] Install `open3d` in the Modal image layer.

### Task 3: Upload And API Surface

**Files:**
- Modify: `poc/worker/pipeline.py`
- Modify: `poc/app/main.py`

- [x] Upload `mesh.glb` to `runs/{run_id}/mesh.glb`.
- [x] Return `mesh_key` in run output.
- [x] Serve `.glb` artifacts with `model/gltf-binary`.

### Task 4: Browser Viewer

**Files:**
- Create: `poc/app/static/mesh.html`
- Create: `poc/app/static/mesh.js`
- Modify: `poc/app/main.py`
- Modify: `poc/app/static/app.js`

- [x] Add `/mesh` route.
- [x] Load `mesh.glb` with Three.js, force a clay material, add first-person controls, and show mesh stats from metadata.
- [x] Link completed mesh-enabled runs to `view mesh`.

### Task 5: Verification And Real Run

**Files:**
- Update: `docs/superpowers/plans/2026-04-29-clay-mesh-diagnostic.md`

- [x] Run focused tests and the full pytest suite.
- [x] Deploy Modal.
- [x] Probe the deployed worker.
- [x] Submit or reuse a real run with mesh enabled.
- [x] Add the completed local run record if needed so the browser URL works.
- [x] Verify the local mesh viewer URL loads a non-empty GLB artifact.

## Verification Results

- Local tests: `.venv/bin/python -m pytest -q` -> `19 passed`.
- Deployed worker probe: imports `poc.worker.mesh`, Open3D `0.19.0`, R2 connection OK.
- Real run: `meshdiag-68ac9baf0903`, Modal call `fc-01KQDCJNWEJRA755D2SVJ4SKAQ`.
- Mesh artifact: `runs/meshdiag-68ac9baf0903/mesh.glb`, 5,226,228 bytes, `model/gltf-binary`.
- Mesh metadata: `status=ok`, `139077` vertices, `249999` faces, `60` fused frames.
- Local viewer: `http://127.0.0.1:7860/mesh?run=meshdiag-68ac9baf0903`.
- Viewer dependencies: mesh viewer uses local Three.js vendor files under `poc/app/static/vendor/three/`, not CDN imports.
