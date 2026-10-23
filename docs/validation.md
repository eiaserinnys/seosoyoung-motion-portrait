# r11: spherical side hair and measured face morph

The viewer uses independent left and right 45° reference images, preserving the original asymmetric lighting and hairpin. SAM 3 separately segmented both faces and both near-side cranium regions. Source hashes, prompts and scores are recorded in [the segmentation manifest](../sam3_masks/references/manifest.json). The 40 original parts plus two cranium parts produce 42 atlas meshes.

## Reference and projection

![Left 45°, front and right 45° references](../source/references/ref_45_triptych.png)

For a reference pixel at height y, the fitted horizontal radius is `R(y) = rx * sqrt(max(0, 1 - ((y-cy)/ry)^2))`. The intrinsic longitude is `asin(clamp((x-cx)/R(y), -1, 1)) - thetaRef`, where thetaRef is independently -45° or +45°. At runtime the longitude plus effective yaw is clamped to the visible hemisphere. Coordinates are `x=cx+R*sin(longitude)`, `z=R*cos(longitude)`; pitch and roll follow this projection. Thus the surface unfolds at the horizon with cosine compression, rather than translating a flat strip.

A small reference-coordinate residual restores points outside the fitted ellipse at the reference angle. Both cranium meshes recover their registered reference x/y coordinates within **1e-8 canvas pixels** at ±45°. This is a coordinate test, not an assertion that the assembled portrait exactly reproduces the reference image. Clamped hidden triangles can collapse at the horizon; projected backfaces are culled. The zero-fold regression applies to the original face and hair ribbons, not these intentionally hidden spherical triangles.

The dome fit uses 129 silhouette samples from the original front crown mask, left-reference crown and right-reference cranium outer boundary. Center x is 502; fitted center y is 399.405, rx 282.421 and ry 366.605. Silhouette fit RMS is **11.113 px**, maximum **23.259 px**. The right-reference full-crown SAM response contained only 32 pixels and is retained as a rejected diagnostic; it is excluded from the fit. [Fit samples and coefficients](../source/viewer/dome-fit.json) make that selection explicit. The upper added depth depends on both x and y; its horizontal second difference is negative in the dome regression.

At the texture join, clean alpha and an overlapping feather blend the independently generated cranium into the original hair. The existing hair and auxiliary patch share dome parameters, but this implementation does not constrain both meshes to an identical seam longitude or prove equal seam velocity. Texture continuity remains part of the separate visual review.

## Face residual fit

Front boundaries come from the restored face alpha; both rotated boundaries come from their own largest connected SAM face component. References are registered with canvas offset (152,20). The front chin at (502.5,616) maps toward (441.5,615) and (586,615), respectively. The two targets are independently measured, not mirrored.

For each side, a quartic least-squares curve smooths the visible cheek and jaw samples from y=465 through 616. Across four target boundaries, fit RMS is **1.953 px**, maximum **5.930 px** against sampled SAM edges. Cubic Hermite interpolation connects those fitted rows. At ±45° the runtime edges agree with the fitted targets within **0.001 px**. [Raw rows, fitted rows and error](../source/viewer/outline-traces.json) are retained separately.

The correction is the target boundary minus its base 3D projection. Internal vertices interpolate left/right residual by normalized face width. Angle interpolation uses the odd basis `sin(theta)/sin(45°)` and even basis `(1-cos(theta))/(1-cos(45°))`, recovering both asymmetric endpoints and zero residual at front. Above the visible cheek, the y=465 residual is held constant and faded near the forehead to preserve eye shape beneath the hair. This upper occluded region is not claimed as an exact fit to every SAM contour point. The chin also receives its measured vertical shift.

`45° 얼굴형 모핑` reuses the existing correction slider component: recommended 1, pure-sphere preset 0. Recommended rotation gain stays 0.65, so input ±50 means effective ±32.5°. Dedicated actual-45° captures use gain 1. Face fitting clamps beyond ±45°; spherical hair continues to the requested angle.

## Occlusion and scope

- Both ears receive negative relief and a face/side-hair coverage mask. Front ear overlap with those visible surfaces is zero. The far ear fades smoothly, reaching zero by effective 25°.
- Left/right back hair and the corresponding half of center back hair receive stronger inward contraction than the far side hair. The renderer also clips the far rear contribution to the face/side-hair coverage. This covers both yaw directions, inputs ±25/±50 and gains 0.65/1.
- Both cranium parts are invisible at front and on the far side. The near part unfolds with its own reference texture. White background is removed before atlas packing; alpha feathering softens the join.
- Fixed neck, body, gaze, outfit controls and existing panel structure remain in the regression scope. No new CSS dimensions were introduced; the slider and two tree rows reuse existing components, typography and spacing.

## Captures and visual inspection

![Recommended yaw sweep](captures/viewer_yaw_sweep.png)

| Pose | Capture |
|:--|:--|
| Front | [Front](captures/viewer_1440_front.png) |
| Input yaw -50 / +50 | [Left](captures/viewer_1440_turn_left_50.png), [right](captures/viewer_1440_turn_right_50.png) |
| Input pitch -20 / +20 | [Down](captures/viewer_1440_pitch_down_20.png), [up](captures/viewer_1440_pitch_up_20.png) |
| Actual yaw -45 / +45 | [Left](captures/viewer_actual_45_left.png), [right](captures/viewer_actual_45_right.png) |
| Narrow layout | [1024×768](captures/viewer_1024_narrow.png) |

The six requested final captures and both actual-45° captures were opened directly. Front ears remain behind the hair. The far rear silhouette no longer forms a detached exposed strip, while the near cranium shows round volume and asymmetric hair color. The crown has a curved shoulder under yaw and pitch. The cheek narrows toward the shifted chin without the earlier eye shearing. Independently generated hair textures retain a visible tonal transition at some near-side joins; specialist review must judge that blend in motion. These images are evidence for that review, not its verdict.

## Executed verification

| Check | Result |
|:--|:--|
| `node --test pipeline/test_outline.mjs` | 8 tests passed: ear depth, far visibility, rear contraction, dome curvature, chin, atlas/unfold, sampled face endpoints, inverse longitude |
| `node pipeline/test_attachment.mjs` | 200 original ribbon/pose/gain cases; zero reversed triangles |
| `node pipeline/test_hair_shell.mjs` | 92 checks; original face contract with outline toggle off |
| `node pipeline/test_corrections.mjs` | 401 checks passed |
| `node pipeline/test_sphere_3d.mjs` | closed sphere, rigid eyes, stationary neck passed |
| `node --test tests/head-turn.test.mjs` | 12 tests passed |
| `python3 pipeline/verify_outline.py` | 10 actual WebGL pose/gain cases; front ear overlap 0, far rear outside pixels 0, far cranium pixels 0 |
| `python3 pipeline/check_outline_metrics.py` | measured results pass; four injected violations rejected |
| `python3 pipeline/verify_viewer.py` | full browser regression passed; 72 attachment cases, zero folds, zero errors/failed requests, no wide/narrow overflow |

[Outline raster measurements](outline_metrics.json) and [full browser measurements](viewer_metrics.json) were generated on Linux Chromium software WebGL. At gain 0.65/input ±25 the effective 16.25° far ear is still fading; at input ±50 it is zero. No native platform is involved in this static web viewer. Independent code review passed. UI/UX specialist review, merge and publication belong to the coordinating session.
