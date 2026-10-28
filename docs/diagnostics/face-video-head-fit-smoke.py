import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
sys.path.insert(0, str(Path('/home/eias/workspace/.projects/seosoyoung-motion-portrait--feat-face-video-refit-f0ed41f9/pipeline')))

import numpy as np
from face_video_contour import project
from face_video_contour_fit import edge_incidence, statistics
from face_video_head_fit import SCALE_UNIT, apply_head, evaluate, fit, pose_factor
from face_video_residual_fit import basis_transform

rest = np.array([[492., 316.], [512., 316.], [512., 336.], [492., 336.]])
triangles = np.array([[0, 1, 2], [0, 2, 3]])
z = np.zeros(4)
poses = [[10., 0., 0.], [0., 10., 0.]]
expected = np.array([[3., -2., np.log(1.05) * SCALE_UNIT], [-4., 2., np.log(.97) * SCALE_UNIT]])
rows = []
for i, pose in enumerate(poses):
    vertices = apply_head(project(np.column_stack((rest - [502, 326], z)), pose), expected[i])
    points = [(1 - ratio) * vertices[a] + ratio * vertices[b]
              for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
              for ratio in np.linspace(0, 1, 10, endpoint=False)]
    rows.append({'frame': i + 1, 'pose': pose, 'pointsArt': np.asarray(points).tolist(),
                 'keep': [True] * len(points)})
_, transform, precision = basis_transform([p[:2] for p in poses])
incidence, factor = edge_incidence(rest, triangles), pose_factor(precision)
initial = np.zeros((2, 3))
before = statistics(evaluate(rest, z, triangles, rows, np.zeros((2, 4, 3)), initial))
head = fit(rest, z, triangles, rows, initial, None, incidence, factor, transform)
corrected = fit(rest, z, triangles, rows, np.asarray(head['head']), .1, incidence, factor, transform)
assert all(np.isfinite([s['mean'], s['p95'], s['max']]).all() for s in head['stats'] + corrected['stats'])
assert all(s['overlap_pairs'] == 0 for s in head['stats'] + corrected['stats'])
assert sum(s['mean'] for s in head['stats']) < sum(s['mean'] for s in before)
print('PASS: two-frame synthetic head and field fit, finite, improves, zero overlap', flush=True)
