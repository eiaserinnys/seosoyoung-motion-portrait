# Rotation correction validation

The yellow guide is a closed 3D sphere. Its diagnostic raster uses an opaque material to measure the geometry, independently of the hair backing's deliberately open face and jaw alpha. The child eye guide spheres remain rigid. Artwork corrections are independent of those guides.

| Pose | Pure sphere | Recommended correction |
|:--|:--|:--|
| Front | [Image](captures/viewer_correction_pure_front.png) | [Image](captures/viewer_correction_recommended_front.png) |
| Yaw -25 | [Image](captures/viewer_correction_pure_left.png) | [Image](captures/viewer_correction_recommended_left.png) |
| Yaw +25 | [Image](captures/viewer_correction_pure_right.png) | [Image](captures/viewer_correction_recommended_right.png) |
| Pitch +20 | [Image](captures/viewer_correction_pure_up.png) | [Image](captures/viewer_correction_recommended_up.png) |
| Pitch -20 | [Image](captures/viewer_correction_pure_down.png) | [Image](captures/viewer_correction_recommended_down.png) |
| Diagonal | [Image](captures/viewer_correction_pure_diagonal.png) | [Image](captures/viewer_correction_recommended_diagonal.png) |

The pure mode is an uncorrected geometric comparison, including its south-pole distortion. The recommended mode retains the jaw as a vertical ribbon, uses a shallow face and normal-aligned eye attachment, and exposes each authored adjustment as an independent control. The default coefficients are for this portrait, not parameters extracted from commercial models.

## Shared attachment and preserved outline

Previously, hair rotated on a deep sphere while the face used a shallow ellipsoid. At extreme poses, identical artwork coordinates separated by more than 80 pixels and outer hair triangles folded over themselves. Face, scalp, bangs, side hair, ears and tail roots now share one corrected attachment and rotation frame. Outside normalized radius 0.60, the illustrated surface continues along the cap tangent instead of falling down a square-root boundary. The contour-preservation control blends this extension and the jaw ribbon with the pure surface. At full preservation, the original hair layers own the visible outline; the opaque sphere reference backing fades out. The yellow diagnostic sphere remains a closed, rigid sphere.

The default rotation gain is 0.65. Auto inputs remain X 50, Y 28 and Z 14, while actual XYZ rotation amplitudes are 32.5, 18.2 and 14 degrees. The viewport reports the effective angles. The gain slider reaches 1.00, and the pure preset uses 1.00. This calibration retains the portrait's proportions while showing real depth, near/far-eye asymmetry and occlusion. [Continuous yaw comparison](captures/viewer_yaw_sweep.png) shows the outer silhouette and visible ears throughout the range. The recommended surface has no flipped triangles in 72 attachment/pose/gain cases, including gain 1.00 at all four extreme XY corners.

## Regression evidence

[Metrics](viewer_metrics.json) report the actual composited pixel difference when ears, nose, mouth and ties are toggled at seven poses. Eye clipping is measured for 72 layer/pose/blink cases. Neck yaw displacement stays zero; neck cap pixels outside the face mask are tested separately from raw attachment point diagnostics. The head/eye radius error is measured in XYZ relative to the transformed center. Yaw/pitch preserve the head center. Roll fixes the neck joint and moves the head center along an arc without resizing the sphere. Diagnostic raster size varies by at most one pixel. [Neck-pivot roll](captures/viewer_1440_neck_pivot_roll.png).

[Front grid](captures/viewer_1440_front_grid.png), [left grid](captures/viewer_1440_turn_left_25_grid.png), [right grid](captures/viewer_1440_turn_right_25_grid.png), [up grid](captures/viewer_1440_pitch_up_20_grid.png), [down grid](captures/viewer_1440_pitch_down_20_grid.png), [extreme left](captures/viewer_1440_range_left.png), [extreme right](captures/viewer_1440_range_right.png), [partial blink](captures/viewer_1440_half_blink.png), [narrow layout](captures/viewer_1024_narrow.png).

Captures use Linux Chromium software WebGL at 1440 by 900 and 1024 by 768. The reference sphere's rear texture is reconstructed from an unobstructed crown strip of the supplied artwork. This is an illustration rig, not an anatomical multi-view reconstruction.
