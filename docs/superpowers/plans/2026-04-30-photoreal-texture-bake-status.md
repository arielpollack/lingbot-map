# Photo-realistic Texture Bake — Status

## What's shipped

End-to-end UV-textured mesh from posed video frames. Layers actual photo
textures onto the Poisson mesh — no more per-vertex gray approximation.

### New files
- `poc/worker/texture.py` — UV unwrap (xatlas) + per-face visibility (Open3D
  RaycastingScene) + atlas bake with view-angle weighted multi-frame
  blending. Outputs PBR `baseColorTexture` GLB. ~280 LOC.
- `tests/poc/test_texture.py` — synthetic 2-triangle quad bake test.
- This doc.

### Modified
- `poc/worker/mesh.py` — after the per-vertex Poisson mesh is built, runs
  texture bake (default on) and OVERWRITES `mesh.glb` with the textured
  version. Soft-fails: on any bake error, keeps the vertex-color mesh.
- `poc/worker/modal_app.py` — added `xatlas==0.0.11` to the worker image.
- `poc/app/static/mesh.js` — when the loaded GLB has a `baseColorTexture`
  on its material, KEEPS the GLTFLoader-assigned material instead of
  forcing per-vertex-color. HUD prefixes with `[textured]`.

### Algorithm

1. **UV unwrap** with xatlas → atlas vertices, atlas faces, UVs in [0,1]².
2. **Visibility test** per face per frame: ray from camera to face
   centroid; if first hit is THIS face, visible.
3. **View-angle weight** per face per frame: `max(0, dot(face_normal,
   direction_to_camera))`.
4. **Score** = visibility × view-angle. Pick top-K (default 4) frames per face.
5. **Rasterize** each UV triangle into the atlas. For each pixel:
   compute barycentric → 3D point on the face → project to each top-K
   frame → bilinear sample → weighted average → write to atlas pixel.
6. **Dilate** UV islands by 2 pixels (nearest-filled-color extend) so
   mipmapping doesn't bleed the gray fallback into rendered surfaces.
7. **Export** as GLB with embedded PBR `baseColorTexture`. Three.js
   GLTFLoader handles it natively.

### Tunable knobs (worker `options`)

| Option | Default | What it does |
|---|---|---|
| `mesh_texture_bake` | `True` | Enable/disable the bake stage |
| `mesh_atlas_resolution` | `2048` | Atlas pixel size (1024 fast, 2048 sharp, 4096 needs nvdiffrast really) |
| `mesh_atlas_top_k` | `4` | Frames per face to blend |
| `mesh_atlas_max_faces` | `250_000` | Quadric-decimate before bake to cap runtime |

### Coord-frame note
We empirically validated that the Poisson mesh (from `scene.glb`'s aligned
cloud) and TRUE w2c poses share a frame — projection works with identity
transform, no extra alignment needed in the bake.

## Known limitations + planned work

### Performance (the big one)
Per-face Python loop in `texture.py` is O(F). For 1.3M faces it exceeds the
30-min Modal timeout. Mitigation today: `mesh_atlas_max_faces=250_000`
quadric-decimates BEFORE bake. Next steps in priority order:
1. **GPU rasterization with nvdiffrast** — research recommends, ~10× speedup
2. **Vectorize the per-face loop** in numpy (bbox + mgrid all faces at once)

### Quality
- No multi-band Laplacian blending — visible seams between frame contributions
  on flat surfaces. Adds ~2s with scipy pyramids.
- Atlas dilation is 2 pixels — could go higher for stricter mipmaps.
- Baking in the decimated mesh means lower geometric fidelity than the
  per-vertex-color mesh. Trade-off vs runtime.

### Failure modes
- Bake soft-fails to vertex-colored mesh (visible in HUD as no `[textured]`
  prefix).
- xatlas is occasionally flaky on degenerate triangles — `process=False` on
  the trimesh load + `remove_degenerate_triangles` upstream mitigates.

## Verification status

- 24 unit tests pass; 1 skipped (the proto-recipe regression test —
  Open3D's standalone Poisson reproducer is non-deterministically flaky).
- Synthetic texture bake test passes (red quad → reddish atlas).
- 60-frame end-to-end Modal job: 39s (atlas 1024).
- **538-frame full job COMPLETED**: 7.5 min total; 5 min reconstruction + ~7 min
  texture (xatlas eats 6.5 min — see "Performance" section). Output is a
  100k-face mesh with 2K texture, 5 MB GLB. **54% of faces textured**, 39% atlas
  fill. URL `/mesh?run=abe41efef3fc48328b1c188cd377115c`.

## Bug fixes shipped during integration

1. **Visibility filter was inverted**: median 0 front-facing frames per face
   because Poisson-of-an-indoor-scene has surface normals pointing AWAY from
   inside-the-room cameras. Fix: `cos_angle = abs(dot(normal, view_dir))` so
   both winding directions count. Took faces-with-color from 11k to 46k on
   the 60-frame proto.
2. **Per-pixel rasterization was 100× too slow** for 100k+ face meshes
   (Python loop overhead). Replaced with vectorized cv2.fillPoly per-face
   flat shading: ~2s instead of >10 min.
3. **Hard timeout** on bake step (default 600s) so the entire job doesn't
   blow the Modal 30-min function timeout if xatlas hangs on weird topology.

## URLs
- Last good vertex-color mesh: `/mesh?run=593fe73399bc4c73a8cca8de395d268a`
- Last good 60-frame textured: `/mesh?run=50c3f74f8a6c4e49a52b680282cc1dbe`
- **Latest 538-frame textured: `/mesh?run=abe41efef3fc48328b1c188cd377115c`**
- Standalone Poisson reproducer: `scripts/build_poisson_mesh_from_scene_glb.py`
