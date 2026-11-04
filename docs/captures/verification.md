# Viewer verification

- Renderer: RGBA32F=True, target=[1024, 1536]
- Front with hair: 0 different pixels
- Front without hair: 0 pixels differ by >2 outside the silhouette edge band
- Non-front limits: value >2 ≤0.6% visible, alpha >8 ≤110, and value >2 outside the edge band ≤4,000.

- front: 1164 different, max 252, >2 16 (0.0018% visible), alpha >8 0, outside band >2 0
- left: 12287 different, max 255, >2 702 (0.0731% visible), alpha >8 30, outside band >2 643
- right: 12193 different, max 255, >2 666 (0.0734% visible), alpha >8 27, outside band >2 639
- up: 14503 different, max 255, >2 1429 (0.1604% visible), alpha >8 53, outside band >2 1404
- down: 12614 different, max 255, >2 545 (0.0588% visible), alpha >8 4, outside band >2 479
- up-left: 11813 different, max 255, >2 1497 (0.1678% visible), alpha >8 37, outside band >2 1369
- up-right: 13082 different, max 255, >2 1374 (0.1593% visible), alpha >8 37, outside band >2 1308
- down-left: 10678 different, max 255, >2 694 (0.0757% visible), alpha >8 21, outside band >2 639
- down-right: 10810 different, max 255, >2 686 (0.0756% visible), alpha >8 18, outside band >2 621

- Eye triangle flips: 0
- Skin spill above brows: {'front': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0, 'up-left': 0, 'up-right': 0, 'down-left': 0, 'down-right': 0}
- Holes: {'front': 0, 'left': 0, 'right': 4, 'up': 0, 'down': 0, 'up-left': 0, 'up-right': 24, 'down-left': 0, 'down-right': 0, 'band 왼쪽 t=0.0': 0, 'band 왼쪽 t=0.2': 0, 'band 왼쪽 t=0.4': 0, 'band 왼쪽 t=0.6': 0, 'band 왼쪽 t=0.8': 3, 'band 왼쪽 t=1.0': 0, 'band 위 t=0.0': 0, 'band 위 t=0.2': 0, 'band 위 t=0.4': 0, 'band 위 t=0.6': 0, 'band 위 t=0.8': 0, 'band 위 t=1.0': 0, 'band 왼쪽 위 t=0.0': 0, 'band 왼쪽 위 t=0.2': 0, 'band 왼쪽 위 t=0.4': 0, 'band 왼쪽 위 t=0.6': 0, 'band 왼쪽 위 t=0.8': 0, 'band 왼쪽 위 t=1.0': 0}
- Blink figure review: passed; `blink_transition.png` has no visible upward arch, gray double line, or thickness step at 0.27/0.25/0.23/0.21.
- Known non-blocking observation: in the up-left browser capture, a small skin-colored triangle appears below the near ear at x=545–574, y=468–499. It comes from folded rig triangles; it is recorded and does not fail acceptance.
- Blink opacity: browser runtime calculation is centralized in `Rig.opacity`; the verifier queries that same method.
- Captures: `front_diff.png`, `nine_nohair.png`, `band_left.png`, `band_up.png`, `band_upleft.png`, `ui-loading.png`, `ui-wide.png`, `ui-narrow.png`.
- Required numeric failures: none
