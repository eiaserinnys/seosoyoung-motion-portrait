# Hair depth and attachments on the r15 baseline

PR #8 is based on `d6a58ba5d0b409389ac1baf4ffdae6a0695c5364`. The face, facial features, renderer, side strands and rig retain that baseline byte for byte. The 19-file manifest is [hair-r15-protected-sha256.txt](hair-r15-protected-sha256.txt). The separate `earWorld` implementation is also unchanged.

## Hair geometry

The existing 45-degree crown, strand and bead fits constrain the absolute depth. A cached Depth Anything V2 Small inference contributes internal crown relief; ordinary fitting uses the checked-in cache without model dependencies. Its model revision, source image digest and calibration are recorded in [hair-depth-prior.json](../source/viewer/hair-depth-prior.json) and [dome-fit.json](../source/viewer/dome-fit.json).

The new rotation and six-reference residual act inside the crown. Their envelope is zero on the approved perimeter and below source Y310. Rear cranium and backing use the r15 reference surface throughout. Removing the extra downward horizon turn, rear seam transfer and positive crop-width expansion eliminates the inflated down-diagonal helmet, brown patch bands and up-diagonal step notches. No texture, alpha, atlas, face or side-strand changes are included.

## Far upward tie attachment

In r15 the upward far tie and tail remain below the lifted skull. Only positive pitch and the far side receive the shared lower-rim Y bound, smoothly weighted by yaw. The original bead rotation remains intact for neutral, downward, frontal and near-side poses. Both tail layers inherit the corrected root. The frame cache stores an unrolled scalar displacement; roll rotates that displacement after lookup. Ear geometry is unchanged.

Measured blue tie coordinates remain observations. Their biased facial-registration offsets are not applied to cheek locks or tail roots. The C correction leaves the approved A/B crown and rear geometry unchanged.

## Verification

| Gate | Result |
|:--|:--|
| Node tests | 118 passed, 0 failed, including all 12 face-registration tests |
| Python tests | 16 passed |
| Outline raster | 22 poses passed; every pitched cheek/ear gap count is 0 |
| Geometry preservation | Nine-direction vertex hashes preserve the r15 faces, rear surfaces, lower side locks and attachments, except the two authorized far/up tail corrections |
| Far/up attachment | Left/right tie overlap is 33.8094/32.0265 px; required minimum is 25 px |
| Full viewer | PASS; canopy 972/error 0, console/requests/WebGL errors 0, clipping/controls/auto/lightbox/narrow layout passed |
| Pixel preservation | Pitch-zero middle row equals r15 exactly; down row equals the approved A/B capture exactly |
| Protection | 19 baseline files and the original `earWorld` implementation are byte identical |

Full browser metrics and the refreshed nine-direction capture are linked below. Browser verification covers the actual WebGL entry point, canopy continuity, attachment geometry, feature clipping and visibility, controls, auto motion, lightbox and narrow layout. Projected back faces remain permitted and culled; 3D collapse checks remain active.

Evidence: [nine directions](captures/viewer_9dir_grid.png), [hair metrics](hair-r15-metrics.json), [viewer metrics](viewer_metrics.json), [outline metrics](outline_metrics.json), [pitched gaps](pitched_outline_metrics.json), [bundle SHA256](bundle-sha256.txt). Independent Sol 6.1 review passed on the source and final nine-direction capture with no P0/P1 findings. Direct comparison with r15 and the original up/down diagonal references shows a smooth rear silhouette without the inflated helmet, brown patch band or step notch; both upward far ties now join the lower hair. The face and near side-hair curves remain the r15 curves. Specialist UI/UX review, merge and publication remain with the coordinator.
