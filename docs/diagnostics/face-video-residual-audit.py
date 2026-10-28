import json
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))

import numpy as np

root = Path('/home/eias/workspace/.projects/seosoyoung-motion-portrait--feat-face-video-refit-f0ed41f9')
sys.path.insert(0, str(root / 'pipeline'))
from face_video_contour import centered_basis
from face_video_contour_fit import statistics
from face_video_residual_fit import evaluate_fields

path = Path('/home/eias/workspace/.local/artifacts/20261028-2333-face-refit/direct-residual-five.json')
result = json.loads(path.read_text())
data = json.loads((root / 'source/viewer/face-solve.json').read_text())
rows = json.loads((root / 'source/viewer/face-video-observations.json').read_text())['observations']
by_frame = {row['frame']: row for row in rows}
observations = [by_frame[f] for f in result['frames']]
n = data['meshCount']
rest, triangles, z = np.asarray(data['points'][:n]), np.asarray(data['triangles']), np.asarray(result['depth'])
basis = np.asarray([centered_basis(*row['pose'][:2]) for row in observations])
neutral = centered_basis(0, 0)
for solution in result['solutions']:
    ccoef = np.asarray(solution['C_coefficients'])
    tcoef = np.asarray(solution['T_coefficients'])
    field = (basis @ ccoef.reshape(35, -1)).reshape(len(observations), n, 3)
    translation = basis @ tcoef[:, :2]
    actual = statistics(evaluate_fields(rest, z, triangles, observations, field, translation))
    largest = max(abs(a[key] - b[key]) for a, b in zip(actual, solution['stats']) for key in ('mean', 'p95', 'max'))
    mean = float(np.max(np.abs(field.mean(axis=1))))
    neutral_c = float(np.max(np.abs(neutral @ ccoef.reshape(35, -1))))
    neutral_t = float(np.max(np.abs(neutral @ tcoef)))
    assert largest < 1e-6 and mean < 1e-8 and neutral_c < 1e-12 and neutral_t < 1e-12
    assert all(a['overlap_pairs'] == b['overlap_pairs'] == 0 and a['back_faces'] == b['back_faces']
               for a, b in zip(actual, solution['stats']))
    print('λ', solution['lambda_c'], 'distance max diff', largest,
          'field mean', mean, 'neutral', neutral_c, neutral_t, 'overlap 0', flush=True)
print('3 solutions × 5 frames: saved coefficients reproduce the recorded results', flush=True)
