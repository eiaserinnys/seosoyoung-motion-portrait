# Unified face registration and shared pitch field

The implementation follows the approved M1–M6 formulas. It is a reviewable
candidate with unresolved acceptance gates, not a merge-ready revision.

| Requirement | Implementation |
| :--- | :--- |
| M1 | `fit_features.py` symmetrizes the approved yaw row before passing it to `reference_registration.mjs`; registration keeps its midpoint X and adds only relative 3D pitch Y. The outline fitter reuses the same delta. `binocularRatio` is recorded without scaling. |
| M2 | `fit_outline.py` independently maps raw sampling Y and canvas Y through four landmarks. Each of the nine poses contains sixteen `[restY, yL, xL, yR, xR]` rows. Quartic X fits, far-cheek mirror means and mirrored edge heights replace normalized spans. |
| M3 | `face-features.mjs` uses measured pitch X and applies only the eye width ratio relative to the neutral yaw row. |
| M4 | `face-outline.mjs` computes one cached field per frame, combining sixteen edge controls and six feature controls. Interior residuals vanish at both outline edges. Neutral feature evaluation uses a separate cache. |
| M5 | Near cheeks retain at least the approved r11 edge through rest Y570; the less occluded near displacement is mirrored. |
| M6 | Temple joins inherit only skin X. Strands preserve yaw XY while pitching at their inner attachment depth. Only far side strands use depth testing. |

## Acceptance conflicts

The thresholds have not been lowered and the approved yaw geometry has not been
changed to force these checks to pass. Measurements are recorded for all nine
key poses and sixteen intermediate poses in `face_registration_metrics.json`.

* Actual sclera alpha boundaries use alpha >127, excluding transparent ArtMesh
  padding. At yaw -45 / pitch 0, the left margin is -3.2388px rather than the
  required +2px. At yaw 0 / pitch +30 the minimum is -9.1356px.
* The near ear's source inner edge is X626 on the right and X379 on the left.
  At yaw -45 / pitch 0, the overlap at source Y440 is -4.9047px rather than
  the required +10px. Skin boundaries are intersected at the deformed ear Y.
* Fixed interpolation bands Y347 and Y431.5 differ from the actual eyebrow
  and eye anchor Y values. The largest measured feature pitch shift error is
  0.013307px. Nose and mouth anchors agree to floating point precision.

The first two conflicts already exist in the approved pitch-zero row. Resolving
them requires a decision about the preservation requirement or measurement
contract. The fixed-band precision requirement also needs a stated tolerance
or revised interpolation contract.

The generated +45 r11 midpoint is `[576.1, 429.525]`, computed from the actual
symmetrized eye centers. The supplied `[576.15, 429.5]` values are approximate.

## Evidence and scope

The modified production paths are reference registration, skin deformation,
all six feature controls and the shared six-layer front canopy. The two strand
sides and both near/far renderer branches are included. Hair artwork, alpha,
atlases, source references, the rig and UI controls remain unchanged.

`verify_outline.py` passed all fourteen existing raster poses: internal holes,
forehead intrusion, cheek/ear gaps and far-back leakage are zero. This raster
coverage check is distinct from the requested 10px skin/ear geometry overlap.

`stage3_build_viewer.py` passed after the worktree environment acquired the
existing pinned Triangle and Shapely dependencies. Full `verify_viewer.py`
passed: canopy continuity 972 cases with zero error, projected skin backfaces
zero, minimum skin 3D area ratio 0.40721, eye clipping and neck cap coverage
passed, controls/lightbox/automatic motion/narrow layout passed, and
console/HTTP/WebGL errors were zero. Its first execution ended before the
full result; the successful rerun adds per-frame cache maps to direct geometry
checks and retains all original assertions.

The old and new nine-direction captures were opened directly. The four diagonal
cheeks surround the eyes more naturally, and the near side strands cover the
previous vertical skin cuts. Upward central pitch has a wider jaw, consistent
with directly matching the measured target rather than suppressing width.

Independent Sol 6.1 high review found the M1–M6 implementation consistent with
the specification, with no additional blocking defect or overengineering.
It retained the known acceptance conflicts as merge blockers. The verification
harness cache update also passed independent review.

Final affected Node regression run: 50 passed, 0 failed, including 200 mesh
attachment cases with zero projected skin backfaces and minimum 3D area ratio
0.006831. The separate new registration suite retains 7 passing checks and
3 failing acceptance checks (feature-anchor exactness, sclera margin and ear
overlap). The original full run had 96 passing, 11 failing and 1 cancelled
tests, including duplicate imported tests and obsolete assertions; the affected
50-test rerun replaces those obsolete-contract and cache-timeout results.
Python outline mapping, near-cheek envelope, source alpha and fixture checks:
9 passed in 0.42 seconds.

Reproduction commands use Node's 60-second test timeout and one worker. Builds,
browser checks and regression suites run through the shared heavy-work runner.
The repository contains no TypeScript package requiring a separate type check.

All committed masks, fixture points and captures belong to this portrait.
No commercial model assets or vertices are included.
