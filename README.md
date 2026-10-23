# Seosoyoung Motion Portrait

A static WebGL portrait viewer with a closed 3D head sphere, rigid child eyeballs, layered artwork and hanging hair physics.

![Front view](docs/captures/viewer_1440_front.png)

## Run

Serve the repository with Python 3, then open `http://localhost:8000/bundle/`:

```sh
python3 -m http.server 8000
```

The viewer has no remote runtime dependencies. Use the pose buttons, sliders, automatic motion, part selection, before/after comparison and sphere grid controls.

## Geometry

The head is a complete sphere of radius 262, centered at `(502, 326, 0)` in the artwork coordinate system. Yaw, pitch and roll rotate XYZ vectors with orthogonal matrices before orthographic projection. Rotating the sphere preserves its center, radius and circular silhouette. Depth testing and face culling select the visible surface.

Artwork inside the front disk is lifted onto the sphere. Artwork extending beyond its rim continues as vertical attached strips. These strips rotate in 3D with their attachment rather than receiving a screen-space shear. The rear surface is textured from the supplied rear-hair layers. It is a reconstruction from the available front artwork, not a separately captured rear view.

Each eye is a radius-54 sphere attached in head-local coordinates. Gaze rotates the child sphere first, and the head transform rotates the entire child afterward. The eyelid opening follows the head and clips the iris, pupil and highlights during blinking. Head rotation does not change local gaze.

The neck stays stationary during pure yaw. Pitch and roll blend the hidden upper attachment into the body. One moving neck owns the skin above the collar; the shirt's duplicate upper neck is removed. Hair ties and hanging tails follow head attachment points and damped gravity motion.

This is a custom WebGL renderer and JSON format. It does not load or produce Cubism binary models.

## Rebuild

Python 3.12+ and Node.js 20+ are used by the build and checks. Install the Python dependencies in a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python pipeline/stage2_inpaint_and_clean.py
.venv/bin/python pipeline/stage3_build_atlas_and_rig.py
node pipeline/test_sphere_3d.mjs
.venv/bin/python pipeline/verify_viewer.py
```

Stage 2 reconstructs the 40 cropped RGBA parts from `source/character/` and the cached masks in `sam3_masks/`. Restoration uses deterministic pixel interpolation and alpha geometry; it does not call an external generative inpainting service. Stage 3 produces the paired 2048-square atlases, front/back sphere textures, meshes and rig data under `bundle/assets/`.

The optional `stage1_sam3_segment.py` uses the existing cached masks without network calls. Generating a missing mask calls the paid fal.ai SAM 3 endpoint and requires `FAL_KEY` in the environment. No credentials are included. Keep all masks to reproduce the provided build offline.

## Validation

The browser check starts and stops its own local HTTP server. It writes screenshots and `viewer_metrics.json` under ignored `debug/`.

Checks cover the rendered sphere silhouette at yaw, pitch and combined extremes; radius preservation for the head and child eyes; stationary neck under pure yaw; neck cap occlusion; eye clipping; independent gaze; before/after sphere textures; image and console errors; image enlargement; and narrow layout overflow.

The checked-in [metrics](docs/viewer_metrics.json) and [capture index](docs/validation.md) describe the current implementation. Sphere-grid and normal renders are available for front, left, right, up and down poses.

## Files

- `bundle/`: standalone static viewer and generated runtime assets.
- `source/character/`: supplied Seosoyoung artwork and minimal placement metadata.
- `sam3_masks/`: cached segmentation masks and non-secret generation metadata.
- `parts/`: restored transparent artwork with canvas bounding boxes.
- `pipeline/`: segmentation, restoration, atlas generation and validation.
- `docs/`: visual evidence and measured results.
