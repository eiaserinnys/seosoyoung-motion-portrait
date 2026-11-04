# Viewer verification

- Renderer: RGBA32F=True, target=[1024, 1536]
- Front with hair: 0 different pixels
- Front without hair: 0 pixels differ by >2 outside the silhouette edge band
- Non-front limits: value >2 ≤0.6% visible, alpha >8 ≤110, and value >2 outside the edge band ≤4,000.

- front: 1164 different, max 252, >2 16 (0.0018% visible), alpha >8 0, outside band >2 0
- left: 12197 different, max 255, >2 569 (0.0593% visible), alpha >8 30, outside band >2 534
- right: 12438 different, max 255, >2 633 (0.0699% visible), alpha >8 14, outside band >2 572
- up: 13903 different, max 255, >2 622 (0.0644% visible), alpha >8 34, outside band >2 594
- down: 12339 different, max 255, >2 502 (0.0545% visible), alpha >8 3, outside band >2 447
- up-left: 10575 different, max 255, >2 585 (0.0598% visible), alpha >8 16, outside band >2 544
- up-right: 12080 different, max 255, >2 664 (0.0739% visible), alpha >8 19, outside band >2 620
- down-left: 11000 different, max 255, >2 672 (0.0747% visible), alpha >8 22, outside band >2 623
- down-right: 10486 different, max 255, >2 586 (0.0639% visible), alpha >8 37, outside band >2 546

- Eye triangle flips: 0
- Skin spill above brows: {'front': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0, 'up-left': 0, 'up-right': 0, 'down-left': 0, 'down-right': 0}
- Holes: {'front': 0, 'left': 0, 'right': 4, 'up': 0, 'down': 0, 'up-left': 0, 'up-right': 18, 'down-left': 0, 'down-right': 0, 'band 왼쪽 t=0.0': 0, 'band 왼쪽 t=0.2': 0, 'band 왼쪽 t=0.4': 0, 'band 왼쪽 t=0.6': 3, 'band 왼쪽 t=0.8': 0, 'band 왼쪽 t=1.0': 0, 'band 위 t=0.0': 0, 'band 위 t=0.2': 0, 'band 위 t=0.4': 0, 'band 위 t=0.6': 0, 'band 위 t=0.8': 0, 'band 위 t=1.0': 0, 'band 왼쪽 위 t=0.0': 0, 'band 왼쪽 위 t=0.2': 0, 'band 왼쪽 위 t=0.4': 0, 'band 왼쪽 위 t=0.6': 0, 'band 왼쪽 위 t=0.8': 0, 'band 왼쪽 위 t=1.0': 0}
- Blink figure review: `blink.png`, `blink_transition.png`; 0.36→0.20 at 0.01 steps, RGB8 mean maximum-channel deltas, max/median ratios: {'front': 1.358, 'left': 1.378, 'right': 1.404, 'up': 1.357, 'down': 1.376, 'up-left': 1.38, 'up-right': 1.37, 'down-left': 1.361, 'down-right': 1.391}
- Viewer per-step mean maximum RGB8 channel delta after white compositing:

  | From → to | Front | Right | Up-left |
  |---|---:|---:|---:|
  | 0.36 → 0.35 | 2.695 | 2.248 | 1.690 |
  | 0.35 → 0.34 | 2.646 | 2.175 | 1.681 |
  | 0.34 → 0.33 | 2.588 | 2.069 | 1.629 |
  | 0.33 → 0.32 | 2.432 | 2.002 | 1.521 |
  | 0.32 → 0.31 | 2.289 | 1.842 | 1.429 |
  | 0.31 → 0.30 | 2.271 | 1.858 | 1.449 |
  | 0.30 → 0.29 | 2.487 | 2.008 | 1.589 |
  | 0.29 → 0.28 | 2.966 | 2.500 | 1.906 |
  | **0.28 → 0.27** | **3.096** | **2.597** | **1.985** |
  | 0.27 → 0.26 | 0.401 | 0.345 | 0.270 |
  | 0.26 → 0.25 | 0.406 | 0.350 | 0.275 |
  | 0.25 → 0.24 | 0.412 | 0.354 | 0.280 |
  | 0.24 → 0.23 | 0.417 | 0.359 | 0.283 |
  | 0.23 → 0.22 | 0.420 | 0.363 | 0.287 |
  | 0.22 → 0.21 | 0.000 | 0.000 | 0.000 |
  | 0.21 → 0.20 | 0.000 | 0.000 | 0.000 |
- Blink comparison with v7 source measurements (median / maximum RGB8 levels):

  | Direction | Prototype | Viewer | 0.28→0.27 prototype / viewer | Viewer max / median |
  |---|---:|---:|---:|---:|
  | Front | 2.29 / 3.10 | 2.28 / 3.10 | 3.10 / 3.096 | 1.36× |
  | Right | 1.85 / 2.60 | 1.85 / 2.60 | 2.60 / 2.597 | 1.40× |
  | Up-left | 1.44 / 1.99 | 1.44 / 1.99 | 1.99 / 1.985 | 1.38× |
- Ear wedge screenshots: `ear_wedge_check.png` compares the up-left endpoint and the up-left 0.8 interpolation around x=545–574, y=468–499.
- Captures: `front_diff.png`, `nine_nohair.png`, `blink_transition.png`, `band_left.png`, `band_up.png`, `band_upleft.png`, `ear_wedge_check.png`, `ui-loading.png`, `ui-wide.png`, `ui-narrow.png`.
- Required numeric failures: none
