"""Small read-only coordinate audit for the falsification experiment."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent.parent
v = np.load(WORKSPACE / '.local/artifacts/ict-facekit/da5f95a607f5e6b37755b38d3385d7f2853732e5/ict_head.npz')['neutral_vertices']
for low in range(-18, 15, 2):
    q = v[(v[:, 1] >= low) & (v[:, 1] < low+2)]
    print('y', low, 'count', len(q), 'x range', q[:, 0].min(), q[:, 0].max(), 'z range', q[:, 2].min(), q[:, 2].max())
reg = json.loads((ROOT / 'source/viewer/ponytail-seedance-registration.json').read_text())
for k in reg['keys']:
    if k['sourceFrame'] in [3,42,264]:
        print('KEY', k['sourceFrame'], 'pose', reg['video'][k['sourceFrame']-1]['pose'], 'affine', k['normalization'])
