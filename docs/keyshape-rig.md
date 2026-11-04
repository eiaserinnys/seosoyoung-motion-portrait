# Directional face data

The build extracts nine reference poses into the original artwork coordinate system. Runtime controls blend the four neighboring keys in each quadrant. The front key has zero vertex displacement and an identity eye transform.

## Mesh groups and draw order

| Order | Meshes | Motion | Viewer stage 1 |
|---|---|---|---|
| 80 | `still-fill-*` | Vertex displacement; neck ramp | Drawn |
| 81 | `still-fill-hair-*` | Vertex displacement | Hidden; retained for the next stage |
| 90 | `skull-plate` | Directional silhouette | Triangulated skin-color fill plus one-pixel alpha band |
| 91, 100–120 | `hair-plate`, back hair | Similarity transform | Hidden |
| 150 | `face-backing` | Vertex displacement and forehead blend | Drawn and silhouette-clipped |
| 200 | `neck` | Vertex displacement and neck ramp | Drawn |
| 220–240 | Skirt, sweater, necklace | Fixed | Drawn |
| 280, 281, 285, 286 | Ears and hair-off covers | Four-point similarity transform and near-edge blend | Drawn as applicable |
| 291–295 | Ties, bun, top hair | Directional transform | Hidden |
| 300–301 | `face-skin`, `face-scalp` | Vertex displacement, contour motion and forehead blend | Drawn and silhouette-clipped |
| 400–461 | Eyes and closed-eye artwork | Eye transform and blink curves | Drawn |
| 500, 510 | Brows | Three-point similarity transform | Drawn |
| 520, 530 | Nose, mouth | Translation / three-point similarity transform | Drawn |
| 700–740 | Side hair, bangs, hairpin | Next-stage motion | Hidden |

The pipeline builds all 84 meshes. The viewer hides hair-only meshes and the meshes listed in `bareOnlyParts` whenever hair is enabled.

## Controls

| Control | Range | Meaning |
|---|---:|---|
| `u` | −1…1 | Left to right; reference frames 34 and 240 anchor the ends |
| `v` | −1…1 | Down to up; reference frames 133 and 89 anchor the ends |
| Roll | −30…30° | Rotation around `(502, 568)` |
| Eye opening, left and right | 0…1 | Open-eye parts fade from 1 to 0 between 0.30 and 0.22; closed-eye artwork turns on at 0.28 |
| Gaze X/Y | −1…1 | Moves iris, pupil and highlights by up to 11 pixels |
| Breathing, body tilt | Existing control ranges | Small upper-body motion |

## Direction interpolation and movement

Vertex displacement, eye transforms, silhouettes and face polygons use bilinear interpolation among the four surrounding keys. Eye blink and gaze are applied in front coordinates before the direction transform. At the front key the displacement is zero and the eye transform is the identity.

The measured movement ratios below use eye movement as 1.0:

| Part | Horizontal | Vertical |
|---|---:|---:|
| Eye | 1.00 | 1.01 |
| Brow | 0.93 | 1.13 |
| Nose | 1.45 | 1.34 |
| Mouth | 1.22 | 1.24 |
| Chin tip | 0.85 | 0.83 |
| Face contour | 0.58 | 1.24 |
| Hairline | 1.00 | 1.10 |
| Skull contour | 0.08 | 0.61 |
| Left / right ear | 0.26 / 0.37 | 0.03 / 0.04 |

The far eye narrows to 0.59 of its front width at the left endpoint and 0.44 at the right endpoint. Nose movement is larger than mouth movement.

## Eye clipping and blink

Iris, pupil and highlights are clipped by the corresponding sclera alpha. Blink curves store per-column sclera bounds, closed-line centers, lash bounds, lower-line bounds and closed-line bounds. For each vertex column, the lashes and lower line move toward the measured closed-line range; sclera and iris elements compress vertically toward the closed-line center. The eye transform is applied after this motion. Visible eye parts and closed-eye artwork cross-fade from 0.30 to 0.22.

The far ear is moved behind the cheek and uses no opacity fade. The viewer triangulates the interpolated 160-point silhouette and fills it with the configured skin color. Its outer one-pixel triangle band interpolates alpha from one to zero. Face skin, scalp cover and backing use the silhouette triangles plus the outer clip-margin band as a binary clipping mask.

## Viewer rendering

The front reconstruction cuts each part by front ownership and solves its color against the original composite. Front mesh positions remain integer-valued. The offline mask rasterizer and warp sample at pixel centers; the browser applies no half-pixel offset to screen positions or UVs. Atlas storage remains straight 8-bit RGBA; the shader premultiplies each of four fetched texels before bilinear interpolation.

Rendering uses an antialias-disabled 1024 × 1536 target. When available, the main target is `RGBA32F` with `EXT_color_buffer_float` and `EXT_float_blend`; otherwise the viewer draws directly into the default 8-bit target. The blend function is `ONE, ONE_MINUS_SRC_ALPHA`, and fragment shaders do not discard. The face clipping mask is the silhouette triangulation plus an outer triangle band extending `clipMarginPx + maskRasterMarginPx` (0.5 px) from the edge. The front bake uses `clipMarginPx` without the raster margin.

The hair-off silhouette backing uses `alpha = clip(1 + signedDistance)`. The interior is opaque; the outer one-pixel band fades linearly to zero. The 6판 viewer comparison checks the eight non-front directions against the CPU reference: value delta >2 is at most 0.6% of visible pixels, alpha delta >8 is at most 110 pixels, and value delta >2 outside the silhouette edge band is at most 4,000 pixels. Hole limits come from `source/keyshapes/v6-checks.json`.

## Next stage: hair

All hair meshes and hair fill tiles remain in the atlas and generated data. Stage 1 does not draw them. Far-side hair ordering, far-wipe blending and hair controls belong to the later stage.
