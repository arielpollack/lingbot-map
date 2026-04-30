# Textured Walkthrough Mesh — Status (autonomous overnight session)

## Verdict

**Working.** The Polycam-style "video → walkable textured 3D apartment" path is shipped via the `mesh.glb` artifact. The mesh is now per-vertex colored from the dense reconstructed point cloud (no more flat clay). Open the full-apartment textured mesh at:

```
http://127.0.0.1:7860/mesh?run=52516a8ab5c045d386e409ecdd49c380
```

(160,651 vertices / 250k faces / 52,204 distinct vertex colors. Built from 538 frames of IMG_1911.MOV, mean color-source distance 9.5 mm.)

WASD + mouse-look. The HUD prefixes "[clay]" if a run somehow falls back to flat color, otherwise the colored mesh renders directly.

## What changed

### `poc/worker/mesh.py` — new `_color_mesh_vertices` helper
Builds a dense colored point cloud from the depth maps (1M+ pts at downsample 4) and assigns each TSDF mesh vertex the color of its nearest neighbor via scipy `cKDTree`. Soft-fails to flat gray on any error so the mesh always ships.

`_export_clay_glb` was renamed and now writes per-vertex RGBA via `trimesh.visual.color.ColorVisuals(vertex_colors=...)` instead of flat face colors.

Configurable: `options.mesh_color_downsample` (default 4 — keeps every 4th pixel per axis when building the dense color cloud).

### `poc/app/static/mesh.js` — vertex-color material
Was forcing every mesh node onto a clay `MeshStandardMaterial`. Now detects the GLB's `attributes.color` and switches to a `MeshStandardMaterial({vertexColors: true, roughness: 0.95, side: DoubleSide})` so embedded vertex colors actually render. Falls back to clay if the GLB has no colors (older runs).

### `tests/poc/test_mesh.py` — coverage for the new helper
Added `test_color_mesh_vertices_falls_back_when_unproject_unavailable` to lock the soft-fail contract.

## Verification

- All 20 tests pass: `.venv/bin/python -m pytest tests/poc -q`
- Smoketest passes: `bash scripts/smoketest.sh` (deploy + CPU probe → all imports resolve, R2 connects, train.py present)
- 120-frame end-to-end Modal job (`run_id=821ed421187643f5ab31edd76638c7b8`):
  - 145,776 vertices, 250,000 faces
  - 28,825 distinct vertex colors (vs ~1 for old clay)
  - 1.014M dense color cloud points
  - Mean NN distance 1.1cm, p99 4cm — colors are sourced from physically-near pixels
- **Full 538-frame job (`run_id=52516a8ab5c045d386e409ecdd49c380`):**
  - 160,651 vertices, 250,000 faces, 5.57 MB GLB
  - **52,204** distinct vertex colors
  - 4.55M dense color cloud points (full 0–255 RGB range)
  - Mean NN distance **9.5 mm**, p99 36 mm
  - 194/200 sampled frames integrated by TSDF
  - Total job time ~5.5 min (mostly LingBot inference; mesh + coloring 43s)

## What was tried and discarded

### 3D Gaussian Splatting (still broken)
graphdeco's `train.py` produces fur-spike garbage on our setup at 7k AND 30k iters despite mathematically perfect poses (1.7e-5 px reprojection error). Diagnosed root cause: graphdeco's gaussian-scale init uses K-nearest-neighbor distance, which collapses to ~0 with our DENSE init point cloud (50k points subsampled from 4.5M). Fix is two paths, both deferred:
1. Switch to nerfstudio `splatfacto` with `--pipeline.model.use-scale-regularization True --pipeline.model.strategy mcmc`
2. Patch graphdeco with `--percent_dense 0.001 --opacity_reset_interval 2000` AND a max-anisotropy clamp inside `gaussian_model.py`

Both require an extra image layer (nerfstudio is heavy) or a graphdeco fork. Not blocking the user's stated goal — the textured mesh delivers the "walking through the property" experience without splat.

### OpenMVS TextureMesh (heavier alternative path)
Research recommended OpenMVS for proper UV-textured meshes (industry-standard photogrammetry texturing). Would give sharper textures than per-vertex colors, but requires bundling the OpenMVS C++ binaries into the worker image (~100MB extra, 30+ min indoor texturing run). Deferred — escalate only if vertex-color quality isn't enough.

### Per-frame `color_map_optimization` (Tier 2 ready to ship)
Open3D ships `o3d.pipelines.color_map.run_rigid_optimizer` which projects each frame onto the mesh and refines vertex colors with multi-view consistency. Would sharpen textures (~2-3× detail vs NN). Implementation prepped but not shipped — only worth the extra 1-3 min of CPU if the current vertex-color output looks smeared at zoom.

## Polycam comparison (research findings)

Polycam ships **both** photogrammetry meshes (their primary upload-mode path) AND 3DGS for "neural mode". Mesh is the robust geometry layer; splat adds photoreal sheen. We now match the mesh layer; splat is a Phase 2 polish item.

## Files changed

- `poc/worker/mesh.py` — vertex coloring + GLB export change
- `poc/app/static/mesh.js` — material switch
- `tests/poc/test_mesh.py` — coverage
- This doc

No deploy config, secrets, or Modal app touched. Existing splat / point-cloud / clay-mesh artifacts still ship for backward compatibility.

## How to use it

```bash
# Submit a job from the local app at http://127.0.0.1:7860/
# Or via Python:
set -a && source poc/.env && set +a
.venv/bin/python -c "
from poc.app.config import require_env
from poc.app.modal_client import ModalClient
from poc.app.runs import RunStore
from pathlib import Path
cfg = require_env()
client = ModalClient(cfg.modal_app_name, cfg.modal_function_name)
store = RunStore(Path(cfg.data_dir) / 'runs.json')
new_run = store.create_run(filename='YOUR.MOV', input_key='inputs/<r2-key>/YOUR.MOV')
rid = new_run['id']
resp = client.submit_job({
    'run_id': rid, 'input_video_key': 'inputs/<r2-key>/YOUR.MOV', 'output_prefix': f'runs/{rid}/',
    'options': {'fps': 10, 'mode': 'streaming', 'mesh_enabled': True,
                'mesh_max_frames': 200, 'splat_skip_training': True, 'use_sdpa': True},
})
store.update_run(rid, status='submitted', job_id=resp['id'], result=resp)
print(f'http://127.0.0.1:7860/mesh?run={rid}')
"
```

`splat_skip_training: True` saves ~3-15 min of GPU per job until we fix the splat path.

## Pending (when user wakes)

1. Eyeball the full 538-frame mesh: `http://127.0.0.1:7860/mesh?run=52516a8ab5c045d386e409ecdd49c380` — confirm it looks like the apartment.
2. If quality is good enough → ship as the primary "view" for new uploads. UI already shows "view mesh" link automatically.
3. If colors look smeared at zoom → escalate to **Tier 2** (per-frame `color_map_optimization`, prepped) or **Tier 3** (OpenMVS UV-textured).
4. Splat is still broken (fur explosion). Diagnosis is in this doc — not blocking but worth fixing if you want photoreal-quality. ~1 day to swap to nerfstudio splatfacto-MCMC.
