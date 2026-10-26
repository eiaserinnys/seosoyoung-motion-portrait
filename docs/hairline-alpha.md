# Forehead backing alpha

The fuzzy horizontal hairline was already present in r23. Turning left or down-left uncovered the hair-top backing's 20px horizontal fade; the same edge made a pale step above the far temple on rightward poses. This change applies the approved C texture boundary without changing geometry, depth or the eye renderer.

## Boundary and preserved pixels

`prepare_front_hair.taper_added` takes strand endpoints from the immutable bangs alpha, removes its smooth trend, and places those endpoints on a curve rising toward the temples. The boundary has 1.5px antialiasing. It modifies only hair-top pixels whose immutable source alpha is zero.

`source/front-hair-inputs/hair-top-backing-alpha.png` preserves the existing backing's support and visible alpha. It is a pipeline input, alongside the immutable original part; it replaces recalculating the discarded horizontal fade. Behind bangs alpha ≥250, C uses A's approved strand boundary (including A's opaque fringe interior). Elsewhere it preserves this input mask. The approved central patch x519–566, y226–244 also uses the new boundary: its rendered RGB error against the original illustration decreased.

Both production callers, `build` and `fill_seam`, use the same function. The delivered atlas was patched locally: 6,915 alpha texels changed, all inside the hair-top added area. RGB, authored hair-top alpha and every other atlas texel are identical to a29bb35. The full build was not run.

## Verification

The permanent tests compare the current atlas against the preserved alpha input using the current rig and renderer, without reading a historical Git commit. They check the added-pixel restriction, pipeline/atlas agreement, central-patch improvement against `source/character/still.png`, and ≤1 LSB outside the patch. A synthetic control detects a 2 LSB violation and missing improvement. The old temple test now checks original notch pixels locally rather than constraining future changes to the entire atlas.

```bash
PYTHONPATH=.:pipeline python3 -m pytest pipeline/test_hair_pixels.py --timeout=60 -q -s
```

Run this command from the worktree through the shared heavy runner (`--timeout 2400`). Python, requirements.txt dependencies, pytest, pytest-timeout and Playwright Chromium are required. To open the viewer, serve `bundle/` with `python3 -m http.server 8765 --directory bundle`.

The neutral WebGL test freshly measured RGB MAE 36.244152 → 15.858918 in the patch. Outside it, five pixels at bangs alpha 250–254 changed by at most one channel step: (669,185), (672,239), (672,240), (518,244), (568,244). This meets the assignee's accepted 1 LSB tolerance. Inside the patch, 497 pixels changed.

The prior C trial covers 13 poses at CorrTurn 1.00. All poses have 0px skin-fill exposure below the ear attachment line and unchanged geometry. Direct sheet review preserved hair/ear/tie connections. The final atlas is byte-identical to that trial, and every runtime JavaScript file and rig is unchanged, so those renders remain valid. The neutral pair was rerendered by the permanent test. SHA-256 values and pose measurements are in [hairline-alpha-validation.json](hairline-alpha-validation.json).

## Capture sheets

- [Representative comparison](captures/hairline-representative-3col-2x.png): a29bb35 | C | Seedance scalp reference.
- [13 poses, two columns with 2× forehead strips](captures/hairline-13poses-2col.png).
- [Neutral original | a29bb35 | C](captures/hairline-front-original-baseline-C.png).
- Five symptom poses, three columns at 2×: [left and down-left](captures/hairline-five-poses-left-3col-2x.png), [right, up-right and down-right](captures/hairline-five-poses-right-3col-2x.png).

Seedance frames are registered by the existing eye-center transform; their scalp boundary is a visual reference, with different bangs and turn angles. S3's far-side right-angle side-hair bend remains outside this change. Merge, deployment and specialist review belong to the assignee.
