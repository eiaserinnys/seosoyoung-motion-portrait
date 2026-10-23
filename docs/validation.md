# Hair shell rotation validation

The r10 hair surface joins the unchanged face surface at the forehead and cheek. Crown and rear artwork share the same depth sign, with only a layer relief between them. The upper surface reaches zero depth at y=64. Its rounded pole transition avoids a square-root cliff in the sampled artwork mesh. The trailing outer contour retains volume during yaw; the far ear and outer side strands narrow toward the cheek and move behind it in camera depth.

## Captures

| Pose | Image |
|:--|:--|
| Front | [Front](captures/viewer_1440_front.png) |
| Yaw -25 / +25 | [Left](captures/viewer_1440_turn_left_25.png), [right](captures/viewer_1440_turn_right_25.png) |
| Yaw -50 / +50 | [Left](captures/viewer_1440_turn_left_50.png), [right](captures/viewer_1440_turn_right_50.png) |
| Pitch -20 / +20 | [Down](captures/viewer_1440_pitch_down_20.png), [up](captures/viewer_1440_pitch_up_20.png) |
| Gain 1, yaw -50 / +50, pitch -28 | [Left](captures/viewer_hair_extreme_left.png), [right](captures/viewer_hair_extreme_right.png) |
| Continuous yaw | [Seven poses](captures/viewer_yaw_sweep.png) |
| Hair controls | [Existing control rows](captures/viewer_hair_controls.png) |
| Narrow layout | [1024 by 768](captures/viewer_1024_narrow.png) |

The yaw strip, right-50, down-20 and up-20 captures were opened and visually compared. The crown parting remains single, the cheek-side gap is closed, and the face crop no longer cuts a skin-colored strip through the front hair. Downward pitch expands the upper cap vertically; upward pitch compresses it. At positive yaw the right ear recedes while the left remains visible; the relationship reverses at negative yaw. The original face proportions and fixed neck remain intact.

The default gain remains 0.65, so input yaw 50 and pitch 20 correspond to actual angles 32.5 and 13 degrees. The separate extreme captures use gain 1. The two new controls reuse the existing parameter-row component, font, spacing and panel scroll behavior; no new CSS dimensions or explanatory copy were added.

## Verification

- `node pipeline/test_hair_shell.mjs`: 92 checks passed, including shared crown depth, exact cheek-boundary coordinates, north-pole depth, pitch exposure and preserved face geometry.
- `node pipeline/test_attachment.mjs`: 200 part/pose/gain cases passed with no reversed triangles, including gain 1 at yaw +/-50 and pitch +/-28.
- `node pipeline/test_corrections.mjs`: 401 checks passed for face corrections, embedded eye spheres, neck yaw independence and fixed roll pivot.
- `node pipeline/test_sphere_3d.mjs`: closed sphere, rigid eyes and zero neck-yaw displacement passed.
- `node --test tests/head-turn.test.mjs`: 12 tests passed.
- `python3 pipeline/verify_viewer.py`: passed on Linux Chromium software WebGL. Both new inputs update their parameters, image enlargement and automatic motion work, eye clipping and neck masking pass, and neither 1440 by 900 nor 1024 by 768 overflows. Console errors, failed requests and WebGL errors are zero.

[Measured results](viewer_metrics.json) include hair bounds, visible ear contributions and the crown layer's pixel contribution. A single layer's contribution is affected by overlapping bangs and is not a measurement of total visible scalp area. Pitch exposure is also checked from the transformed upper-cap boundaries and the actual captures. The guide sphere remains rigid and independent of the artwork corrections.

The geometric tests were retained after the renderer-only overlap fix; the affected browser checks and captures were regenerated. Independent code review passed. These captures are the handoff for a separate UI/UX review, not its verdict.
