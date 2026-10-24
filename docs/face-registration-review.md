# Unified face registration and shared pitch field

The approved s3d follow-up replaces the four-landmark vertical mapping from
PR #9's initial revision. It keeps the r11 yaw geometry and the shared skin /
feature residual field while sampling the silhouette at its own 3D height.

## Implementation

| Contract | Implementation |
| :--- | :--- |
| Vertical fractions | `outline_vertical_registration.mjs` measures the mean left/right `faceWorld` Y at yaw 0, pitch ±30 and CorrTurn 1. Eye Y431.5 and chin Y616 normalize the sixteen s3d fractions. Each pitch shares exactly the same fractions across all three yaw poses. |
| Reference sampling | `fit_outline.py` uses raw eye-to-chin spans independently for each edge. Upward samples have a 0.06 floor to avoid hair-occluded temple rows; downward samples have a zero floor. `mapping.clampedRows` records every affected rest row. |
| Canvas Y | Unclamped s3d maps each registered eye Y to the canvas chin. The chin is the pitch-corrected mouth plus `CHIN_CANVAS_SPAN` (up 64px / down 50px), a stated compromise without a separate chin-underside part. |
| Chin width | Symmetry uses `d=max(measured_d,5.5-(r-l)/2)` because output width is `(r-l)+2*d`. This approved sign correction guarantees at least 11px and prevents the quartic endpoint crossing. |
| Existing paths | Shared camera translation, binocular metadata, sixteen independent edge rows, feature pitch X and relative eye width, near-cheek r11 bounds, the shared six-anchor field and side-hair attachment / drawing policy remain intact. |

Quartic X fits still use all sixteen rest rows. The mapping metadata records
`restY`, `s3d`, `sFloor`, `clampedRows`, `rawYSpan` and `canvasYSpan`. There is no
nose landmark offset in the silhouette mapping. Nose and mouth remain controls
of the internal shared residual field.

## Approved acceptance criteria

The initial absolute 2px / 10px checks are superseded by these approved contracts.
They preserve the accepted yaw drawing and distinguish geometry from final
raster coverage.

* Internal anchors: nose and mouth shift error <1e-6px; all six controls <0.02px
  to accommodate fixed band Y versus the authored eye and eyebrow Y values.
* Nine keyforms: frontal eyes have at least 2px skin margin. Turned eyes either
  have at least 2px, or preserve their same-yaw r11 margin within 1.5px and stay
  above -4.75px. Actual alpha >127 boundaries exclude transparent mesh padding.
* Sixteen intermediate poses: yaw ±11.25 eyes retain 2px margin. Yaw ±33.75 eyes
  remain inside the corresponding r11 ±45 endpoint envelope, with zero outward
  overshoot. Intermediate yaw is compared to the approved endpoint envelope.
* Near cheeks: every diagonal row through rest Y570 retains the r11 endpoint
  within 1e-6px. Turned ear/skin overlap at source Y440, 465 and 490 stays within
  2px of the neutral baseline; final raster coverage must have zero gaps.
* Fit quality: up RMS <=4.5px; other pitch fits <=5.5px. Upper diagonal far edges
  converge monotonically through rest Y465, 510, 550, 570, 590 and 616.

The eye predicate has permanent injected-failure checks for protrusion,
frontal coverage, the absolute floor and intermediate endpoint overshoot.
The raster scan reuses `rowAlphaGaps`, whose existing mutation checks reject
injected transparent pixels. The chin regression rejects crossed endpoints.

## Results and evidence

Pipeline regeneration and viewer build passed. The registration suite passed
12 tests with 0 failed. Existing 14 raster poses and all eight new pitched
key/intermediate poses passed; every `pitchedCheekEarGapPixels` value is zero.
The existing `debug/outline_metrics.json` list format remains compatible with
its standalone validator; new pitch metrics are stored separately.

Final verification: all Node suites 110 passed / 0 failed, Python 9 passed /
0 failed, full viewer passed with zero console, HTTP and WebGL errors. Canopy
continuity has zero error across 972 checks; skin projected backfaces are zero
and the viewer's minimum skin 3D area ratio is 0.468906. Controls, eye clipping,
neck coverage, automatic motion, lightbox and narrow layout checks passed.

The side-hair regression uses the sample's nonzero skin Y residual to verify
that the hanging lock keeps its own Y, while retaining the X attachment and Z
assertions. The new r11-constrained s3d contour can legitimately have zero X
residual there. The final independent Sol code review passed and directly read
the final test logs without repeating verification.

Capture SHA256: `c1d5e096b8acaa95130655662077a87621f168bc401ef61c076513f00b1adc59`.
`bundle/index.html` SHA256: `ea46ddeb87634f7f2b63ee9fa3870677cbfc758519548d1b74a4c4fec6b67ab4`.

`face_registration_metrics.json` records the actual alpha-boundary margins,
same-yaw and endpoint baselines, ear overlaps and six-anchor shift errors for
all 25 poses, plus fit RMS, clamped rows and chin widths. It contains only data
from this portrait's own artwork.

| Measured quantity | Final value |
| :--- | :--- |
| Up / up-left / up-right fit RMS | 4.1273 / 5.0255 / 3.5611px |
| Frontal up / down left-eye margin | +4.1842 / +3.6315px |
| Left-down far-eye margin / loss from r11 | -4.6850px / 1.4462px |
| Left-down intermediate far-eye margin / r11 endpoint | -1.5849px / -3.2388px |
| Largest turned ear-overlap loss | 1.8414px |
| Up / down chin width | 11.0000 / 14.7469px |
| Maximum six-anchor shift error | 0.013262px |

The previous revision and current nine-direction captures were opened directly
by the implementer and independent Sol reviewer. Upper diagonal far cheeks
converge toward the chin; no reversed chin endpoint or transparent ear/cheek
gap is visible. Incremental code and visual evidence review passed with no
remaining blocking defect. Professional UI review remains with the coordinator.

The generated +45 r11 midpoint is `[576.1,429.525]`, calculated from the actual
symmetrized centers. Source artwork, alpha, atlases, rig and UI controls are
unchanged. No commercial model assets or vertices are included.
