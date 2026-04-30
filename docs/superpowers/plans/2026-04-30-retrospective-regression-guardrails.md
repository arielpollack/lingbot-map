# Retrospective: mesh regressions + guardrails

## What happened

Between the "proto-poisson-mesh" run that the user signed off on and the current
worker output, the mesh regressed three times:

1. **112 MB curtain artifacts** — removed decimation but the worker's input
   cloud was 6× denser than the proto's, producing a much larger raw mesh.
2. **Spiky sphere** — added a `scene_alignment` transform copied from
   `glb_export.py`. The math was identical to glb_export's, but the worker's
   input cloud was in a *different starting frame* than glb_export's expects,
   so the same transform applied to different inputs produced a frame
   rotated by ~180° around Z relative to the proto.
3. **Stats-perfect but upside down** — bit-for-bit metadata match (vertex
   count, face count, bbox extents) but bbox **signs** flipped. Declared
   "match!" without checking sign or visual geometry.

## Root cause

I rebuilt a proven result via a *different code path* and assumed parameter
matching would make it converge. The proto loaded `scene.glb` from R2 and
Poisson-meshed it; the worker built its own cloud from depth-unproject and
Poisson-meshed that. Both depended on `c2w[0]` which differs across LingBot
runs, so the "same recipe" produced different output frames.

## The principle

**When a result works, the worker should use the same code path that produced
it, not "an equivalent recipe."** Equivalence under matrix math doesn't survive
real-world divergence in inputs.

## Guardrails

1. **Source-of-truth scene.glb** (shipped). Worker `mesh.py` now loads the
   freshly-written `scene.glb` and Poisson-meshes those exact vertices, with
   depth-unproject only as a fallback. By construction, mesh and point cloud
   share a frame because they share vertices.

2. **Geometric parity check** (shipped — `tests/poc/test_mesh_recipe.py`).
   Asserts that the worker's mesh output, given a known scene.glb, matches
   the canonical proto recipe's vertex count, face count, AND bbox SIGN
   within tolerance. Bbox sign would have caught regression #3.

3. **Recipe defaults are tagged**. The constants in `mesh.py` are documented
   as the "proto-poisson-mesh recipe" defaults. Any default change requires
   a comment block explaining what's different and why — see the existing
   block in `mesh.py` lines ~84-106. Do not silently bump.

4. **Standalone reproducer**. `scripts/build_poisson_mesh_from_scene_glb.py`
   is the canonical recipe as a runnable script. If the worker output ever
   diverges from this script's output on the same input, that's a bug.

5. **Compare visually, not by metadata alone**. Vertex count + face count +
   bbox extents can all match while the geometry is upside down or mirrored.
   When validating a new mesh, look at the actual rendered output (via the
   browser viewer or a headless screenshot) before declaring success.

## Anti-patterns to flag

- "I copied the math from glb_export so it should match" — math equivalence
  doesn't preserve under different inputs. Use the glb_export OUTPUT, not
  its code.
- "Vertex count matches, ship it" — orientation/sign isn't in vertex count.
- "Tweak one default to fix one thing" — defaults compound. If a tweak is
  needed because a result regressed, first revert to the last good state,
  then add ONE knob with a regression test.
