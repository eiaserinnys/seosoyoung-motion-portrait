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

The head is a complete sphere of radius 262, centered at `(502, 326, 0)` in the artwork coordinate system. Yaw, pitch and roll rotate XYZ vectors with orthogonal matrices before orthographic projection. Yaw and pitch preserve the sphere center. Roll pivots the complete head around the fixed neck joint `(502, 568, 0)`, so its center follows an arc while the radius and circular silhouette remain unchanged. Depth testing and face culling select the visible surface.

The yellow guide remains a closed sphere. Portrait artwork has an independently tunable ellipsoid depth surface. The recommended preset uses a shallow face, a vertical jaw ribbon starting before the south pole, local nose/mouth depth, far-eye aperture compression and an adjustable diagonal residual. The pure preset sets the ratios to 1 and all correction amounts to 0. Both presets preserve the original artwork XY coordinates in the neutral pose.

The upper backing uses a sampled hair strip from our own crown drawing, with an open face and jaw region. It does not write depth over the illustration layers. Artwork ribbons are two-sided; the closed diagnostic sphere uses face culling. This avoids hiding ears or cutting the lower face at a sphere pole. The reconstructed rear is inferred from front artwork, not a captured rear view.

The diagnostic child eyes are radius-54 spheres. The recommended portrait keeps measured yaw anchors, adds curved socket depth and pitch foreshortening, and rotates radius-42 corneal caps with the head. Independent gaze is applied before that rotation. The projected eyelid opening clips the iris, pupil and highlights during blinking.

The neck stays stationary during pure yaw. Pitch and roll blend the hidden upper attachment into the body. Breath and torso sway move the head and upper neck together, then fade into the torso. One moving neck owns the skin above the collar; the shirt's duplicate upper neck is removed. Hair ties and hanging tails follow head attachment points and damped gravity motion.

All six front hair layers share a rounded dome, protected in front of the forehead by a smooth depth maximum. The upper backing meets that dome without a depth inset. The rear cranium retains the complete elliptical shell, with a narrow upper seam blend and yaw-only visibility. Crown, bangs and hairpin retain their illustration order. The hair controls adjust volume and far-side occlusion.

The face uses a silhouette mesh with jointly optimized shared depth, pose translations and smooth 2D residuals. `pipeline/optimize_3d_2d.py` solves the graph-weighted least-squares system with NumPy and Schur elimination, then pins the measured rim and six separate feature anchors. Neutral yaw keeps the r11 skin; its hair-occluded rows above restY465 use the r11 prior. Skin queries interpolate mesh/rim controls, while feature centers query their dedicated solved nodes. Eyelid, gaze, brow and mouth motion stays in the existing local evaluator. Downward diagonals use the registered down-right measurements as their canonical target; down-left mirrors them. The legacy pitch field remains the hair attachment contract.

The three r15 hair prior modules stay byte identical to `0b75a02`. `hair-solve.mjs` adds a smooth 12×16 residual chart through the existing outline correction weight. It is exactly zero at neutral pitch and below restY435, with a smooth fade from 400 to435. Each of the six pitch keys fits silhouette normals and hairline observations with ridge and second-difference regularization. The residual budget is RMS 6px / maximum22px. The release check compares actual WebGL frames against the same uncorrected prior: every pitch pose must improve or remain equal, and the residual chart must retain determinant above 0.25. Absolute reference mismatches remain visible in the metrics.

`source/viewer/face-solve.json` records depth, translations, residuals, source hashes and eight leave-one-pose-out measurements. `explained3D` is one minus weighted residual SSE divided by weighted displacement SSE from the depth-free rotated rest plane, including translation in the 3D explanation; `duRms` is the unweighted vector RMS over all solve nodes. `stats.rms` measures the pinned rim/anchor controls; `observationRms` also includes the softly fitted mesh priors. `bundle/face-solve-data.mjs` is the compact runtime copy. Updating the downward `face-0` targets preserves neutral XY and the approved `family45` fit. `family45` adds separate `ZLeft`/`ZRight` depths and four relative-yaw poses on the same node ordering. Trigonometric collocation reproduces the 67.5° and 90° solutions, with zero displacement at the ±45° seam for every pitch, followed by neck-pivot roll. Eyes and brows use the measured additional horizontal compression around their solved centers, including a 0.02 far-eye width ratio at 90°.

`ParamAngleX` spans `[-90, 90]`, including identity keys at ±90 in both head and neck deformer chains. Set the existing rotation strength (`CorrTurn`) to 1 for a direct degree-to-degree mapping; the recommended preset retains its original 0.65 strength and existing pose buttons. The four side masks use the same quartic smoothing below restY465 as the original 45° contours. The additional horizontal displacement fades by `1 - 0.15 * min(1, abs(pitch) / 30)` on diagonals, preserving every horizontal key and the ±45° seam. Accepted determinant floors are 0.35 at 67.5°, 0.22 at 90°, and 0.15 at intermediate/diagonal poses, with zero reversed triangles throughout. `pipeline/capture_face_turnaround.py` captures the seven actual angles from -90° to +90° in `docs/captures/face_turnaround_7angle.png`, alongside the unchanged nine-direction capture workflow.

This is a custom WebGL renderer and JSON format. It does not load or produce Cubism binary models.

## Rebuild

Python 3.12+ and Node.js 20+ are used by the build and checks. Install the Python dependencies in a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python pipeline/stage2_inpaint_and_clean.py
.venv/bin/python pipeline/stage3_build_viewer.py
python3 pipeline/optimize_3d_2d.py
python3 pipeline/solve_to_bundle.py
python3 pipeline/bake_face90.py
python3 pipeline/capture_hair_solve_prior.py
python3 pipeline/fit_hair_solve.py
python3 pipeline/accept_hair_solve.py
python3 pipeline/probe_hair_only.py
node --test --test-timeout=60000 pipeline/test_face_solve.mjs
python3 -m pytest --timeout=60 pipeline/test_face_solve.py
node pipeline/test_sphere_3d.mjs
node pipeline/test_corrections.mjs
node pipeline/test_attachment.mjs
node pipeline/test_hair_shell.mjs
node --test pipeline/test_outline.mjs
node --test tests/head-turn.test.mjs
.venv/bin/python pipeline/verify_viewer.py
.venv/bin/python pipeline/verify_outline.py
```

Stage 2 reconstructs the 40 cropped RGBA parts from `source/character/` and the cached masks in `sam3_masks/`. Restoration uses deterministic pixel interpolation and alpha geometry; it does not call an external generative inpainting service. The optional one-time face-depth fit uses `.venv/bin/python -m pip install -r requirements-depth.txt` followed by `.venv/bin/python pipeline/fit_depth.py`; it downloads the pinned Depth Anything V2 Small model. Stage 3 adds two independently segmented 45° cranium parts, fits the dome and face outlines from cached masks, and produces the paired 2048-square atlases, front/back sphere textures, meshes and rig data under `bundle/assets/`.

The optional `stage1_sam3_segment.py` uses the existing cached masks without network calls. Generating a missing mask calls the paid fal.ai SAM 3 endpoint and requires `FAL_KEY` in the environment. No credentials are included. Keep all masks to reproduce the provided build offline.

## Seedance skin registration

`source/references/seedance_9dir/` contains camera-normalized representatives of the same 361-frame Seedance clip. Manually inspected seed patches, frame IDs and image hashes are recorded in `source/viewer/seedance-reference-frames.json`; independent generated side portraits are excluded from fitting. The canonical front input fixes the rest coordinates. The tracker combines [OpenCV NCC](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html), bidirectional [pyramidal optical flow](https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html) between seeds and skin segmentation. Occluded ear controls follow the corresponding facial side boundary.

SciPy alternates shared 3D depths, weak-perspective cameras and cubic image residuals with temporal velocity/acceleration priors. Representatives are selected from the final training trajectory, checking nose-side and eye-to-ear-height direction cues. Near-frontal selection also checks neutral feature consistency so a drifting mouth patch cannot win solely on a near-zero inferred angle. [Clarabel](https://clarabel.org/stable/python/getting_started_py/) fits the final ten-term cubic coefficients with key/video weights 10:1, image-space RMS cones and polar material-area inequalities. The final report includes training error as well as the held-out error; the trajectory figure shows every frame.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipeline/solve_ponytail_seedance_registration.py --inputs ../../.local/artifacts/20261025-ponytail-seedance-reg
PATH="$PWD/.venv/bin:$PATH" node build.mjs
.venv/bin/python pipeline/capture_ponytail_registration.py --artifacts ../../.local/artifacts/20261025-ponytail-seedance-reg
```

On the shared host, run each command through the configured `heavy_verify.py` runner. An observations cache may be supplied with `--observations` to repeat fitting without decoding/tracking the clip. Three contiguous 18-frame blocks are excluded from shape fitting and representative selection. Held-out camera poses are inferred from tracked image points, initialized from adjacent training poses; the reported error measures conditional shape reconstruction. Every gate recomputes predictions from saved coefficients.

The skin solve changes the previously unobserved forehead/temple region, with the six feature controls and lower rim fixed. Its additional continuous correction vanishes at all nine authored poses, throughout neutral pitch, and below rest Y=410. A separate upper skin ribbon fills the original mesh's narrow upper UV cap. Its 97 angular columns and six rows follow the learned polar silhouette, with a skirt overlapping the original face. Build-time mirror parity and centered two-eye alignment keep the frontal dome symmetric; a six-degree Hermite fade and normalized periodic angles keep mouse tracking continuous across the center; authored direction controls map to measured video angles. A filled upper silhouette mask clips the old narrow cap while passing the existing face below the ear attachment line. Bright, eroded canonical skin samples supply the separate RGBA texture, and a 24px RGB repair blends the original upper edge without altering atlas bytes or hard alpha. The renderer draws the ribbon after back hair/ears and before face/front hair. Hair geometry, bead attachments, the 673 original skin vertices, rig and original atlas remain unchanged.

The left and up-left near ear is already separated from the original face in a hairless inspection; its unchanged bounding box and area were checked with the new fill and clip disabled. The upper ribbon preserves that existing lower-ear geometry. Stage 4 pixel gates measure new exposed skin below the ear line, frontal dome symmetry and forehead seam luminance. `docs/stage4_registration_metrics.json` includes conditional held-out camera reconstruction, a diagnostic with linearly interpolated held-out cameras, and point-split validation (eyes/nose/chin estimate each camera; eyebrows/mouth/ears are prediction-only). It also records five center-tracking samples, bidirectional microsteps and the achieved/measured/applied angle table.

The legacy `explained3D` and cross-validation diagnostics in `face-solve.json` describe the preserved base solve before the upper-skin step. Integrated displacement and triangle-orientation statistics are refreshed after that step; Stage 4 has its own independently recomputed registration errors.

## Validation

The browser check starts and stops its own local HTTP server. It writes screenshots and `viewer_metrics.json` under ignored `debug/`.

Checks cover the rendered sphere silhouette at yaw, pitch and combined extremes; radius preservation for the head and child eyes; stationary neck under pure yaw; composited ear, nose, mouth and hair-tie visibility; eye clipping; independent gaze; before/after portrait textures and live correction controls; image and console errors; image enlargement; and narrow layout overflow.

The checked-in [metrics](docs/viewer_metrics.json) and [capture index](docs/validation.md) describe the current implementation. Sphere-grid and normal renders are available for front, left, right, up and down poses.

## Side parts beyond 45 degrees

`pipeline/bake_face90.py` samples the approved left no-rimlight 67.5°/90° references and their clean mirrored RGB through the four exact solved `face-skin` poses. It reconstructs occluded skin from a smooth 2D color field, removes stationary eyes/hair, and retains the anterior nose and lip detail. The additional `face-profile.png` contains four RGB tiles in rest-canvas UVs; the original atlas, skin alpha, mesh topology and depth remain unchanged. RGB blending starts strictly above 45°, reaching the 67.5° tile before blending to the 90° tile with `smoothstep(60,85,abs(yaw))`.

Canvas-side labels determine occlusion: negative yaw hides the left eye/brow, positive yaw the right. Their alpha fades at 50°–66°; frontal nose and closed-mouth decals fade at 50°–65° to avoid doubling the profile artwork. Near eyes transition to a shared asymmetric eyelid field at 65°–80°, retaining independent gaze and blink controls. Near brow width reaches 0.70/0.48 of its 45° width at 67.5°/90°. Above 45°, all facial details also clip to the skin silhouette while retaining the existing iris aperture mask. Ears, side locks and hanging roots have additional reference-calibrated displacement strictly beyond 45°. Distant side locks disappear by 60° and tails by 62°; upper near locks inherit the first measured row's displacement with a broad 240–400 transition. Hair shell/dome/pitch priors are untouched.

Profile detail is restricted to a face mask eroded inward by three pixels; the right nose/lip RGB mirrors the clean left source about x502. Reference-sheet right images are shifted left 20px to match their registered masks. The viewer preserves its original slider mapping through 45°, then maps the remaining travel continuously to an actual 90°, including under the recommended 0.65 gain. Four side-pose buttons reach actual ±67.5°/±90° directly.

Run `python3 pipeline/bake_face90.py` again after changing the face solution or reference images. `python3 pipeline/capture_side90.py` produces [face](docs/captures/face_turnaround_7angle.png), [full portrait](docs/captures/full_turnaround_7angle.png) and [blink](docs/captures/profile_blink.png) sheets. It also renders the pinned pre-change renderer from Git and requires zero changed pixels for all nine original poses and ±44.99°. Results are in [side90-webgl.json](docs/side90-webgl.json). `pipeline/test_side90.mjs` independently compares every original mesh vertex and alpha against that baseline.

## Files

- `bundle/`: standalone static viewer and generated runtime assets.
- `source/character/`: supplied Seosoyoung artwork and minimal placement metadata.
- `sam3_masks/`: cached segmentation masks and non-secret generation metadata.
- `parts/`: restored transparent artwork with canvas bounding boxes.
- `pipeline/`: segmentation, restoration, atlas generation and validation.
- `docs/`: visual evidence and measured results.

## Correction study

The generic [deformation module](src/head-turn.mjs), [model](docs/deformation-model.md), [measurements](docs/measurements.md) and [surface comparison](docs/surface-comparison.md) were integrated from analysis commit `230782b`. They contain aggregate measurements and generic code, without third-party model files or per-vertex data. Viewer defaults are authored for this portrait, not recovered coefficients from those models.

`source/viewer/index.html` and `source/viewer/corrections.json` are the HTML and correction-schema sources. The viewer builder generates `bundle/index.html`, `bundle/corrections.mjs`, copies the shared deformation module, and writes the matching rig metadata. Local geometry and residual corrections in `bundle/geometry.mjs` are specific to this portrait. The guide sphere and child eye spheres stay rigid when art-surface settings change.

## Head base

The viewer loads `bundle/assets/head-base.bin` and `head-base.json` as a fitted head backing. `CorrHeadBase` is 1 in the recommended preset and 0 in the pure preset; the face-skin and ears use this surface when both `CorrHeadBase` and `CorrOutline` are 1. The toolbar can hide the backing or hide and restore all hair layers. Feature and hair evaluators remain unchanged.

The head and face outline follow the Seedance key frames with a fixed rest-frame translation; the pose-dependent displacement field remains active. At rest, points outside the fitted surface retain their original 2D offsets; those offsets fade with angular distance from rest. The backing extends around the neck opening, and the face edge fades only where the backing is behind it. Ears follow the attached surface's rotation relative to rest. The fitted head and the existing hair attachment field remain separate. The calibrated lower row turns backward between effective yaw 25° and 45° because the keys themselves differ; the viewer retains those key values.
