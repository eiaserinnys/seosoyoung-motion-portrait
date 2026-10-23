# 3D sphere validation

The grid uses complete latitude circles and longitude great circles in 3D. The outer ring is the orthographic silhouette of the same closed sphere. The browser check also renders the actual textured sphere alone and measures its alpha bounds, independently of that overlay ring.

| Pose | Artwork | Sphere grid |
|:--|:--|:--|
| Front | [Image](captures/viewer_1440_front.png) | [Image](captures/viewer_1440_front_grid.png) |
| Yaw -25 | [Image](captures/viewer_1440_turn_left_25.png) | [Image](captures/viewer_1440_turn_left_25_grid.png) |
| Yaw +25 | [Image](captures/viewer_1440_turn_right_25.png) | [Image](captures/viewer_1440_turn_right_25_grid.png) |
| Pitch +20 | [Image](captures/viewer_1440_pitch_up_20.png) | [Image](captures/viewer_1440_pitch_up_20_grid.png) |
| Pitch -20 | [Image](captures/viewer_1440_pitch_down_20.png) | [Image](captures/viewer_1440_pitch_down_20_grid.png) |

Combined extreme poses: [left](captures/viewer_1440_range_left.png), [right](captures/viewer_1440_range_right.png). Partial blink: [image](captures/viewer_1440_half_blink.png).

The figures are local Chromium software-WebGL captures at 1440 by 900. The numerical checks are recorded in [viewer_metrics.json](viewer_metrics.json). Mesh rasterization can change the silhouette's boundary by one pixel, while the underlying sphere radius remains invariant within floating-point precision.

The rear texture is reconstructed from existing rear hair. Attached strips and gravity-driven tails intentionally extend beyond the spherical head's silhouette. This implementation validates the geometric hierarchy and interactive viewer; it is not a multi-view reconstruction of an anatomical head.
