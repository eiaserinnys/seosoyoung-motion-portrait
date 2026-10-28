"""Check whether one observed disconnected skin island can avoid overlap by depth only."""
import json
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))

import numpy as np
import shapely
from scipy import sparse
from scipy.sparse.csgraph import connected_components

root = Path('/home/eias/workspace/.projects/seosoyoung-motion-portrait--feat-face-video-refit-f0ed41f9')
sys.path.insert(0, str(root / 'pipeline'))
from face_video_contour import project, rotation


def overlapping(small, other):
    if not len(small) or not len(other):
        return 0
    pairs = shapely.STRtree(other).query(small, predicate='intersects')
    areas = shapely.area(shapely.intersection(small[pairs[0]], other[pairs[1]]))
    return int(np.count_nonzero(areas > 1e-7))


def signed_areas(points, tri):
    a, b, c = points[tri].transpose(1, 0, 2)
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


example = shapely.polygons(np.array([[[0., 0.], [3., 0.], [0., 3.]]]))
overlap = shapely.polygons(np.array([[[.5, .5], [2., .5], [.5, 2.]]]))
separate = shapely.polygons(np.array([[[5., 5.], [6., 5.], [5., 6.]]]))
assert overlapping(example, overlap) == 1 and overlapping(example, separate) == 0
if '--smoke' in sys.argv:
    print('2 synthetic overlap checks passed')
    sys.exit(0)

data = json.loads((root / 'source/viewer/face-solve.json').read_text())
observations = json.loads((root / 'source/viewer/face-video-observations.json').read_text())['observations']
rows = {row['frame']: row for row in observations}
frames = [1, 34, 88, 298, 256]
n, triangles = data['meshCount'], np.asarray(data['triangles'])
rest = np.asarray(data['points'][:n])
z = np.asarray(data['Z'][:n])
edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
adjacency = sparse.coo_matrix((np.ones(len(edges) * 2),
                               (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                              shape=(n, n)).tocsr()
_, components = connected_components(adjacency, directed=False)
island = components == components[0]
assert island.sum() == 4
island_tri = triangles[np.all(island[triangles], axis=1)]
other_tri = triangles[~np.all(island[triangles], axis=1)]
state = []
for frame in frames:
    pose = rows[frame]['pose']
    q = project(np.column_stack((rest - [502, 326], z)), pose)
    island_front = island_tri[signed_areas(q, island_tri) > 0]
    other_front = other_tri[signed_areas(q, other_tri) > 0]
    state.append((q, island_front, shapely.polygons(q[other_front]), rotation(*pose)[:2, 2]))

results = []
for offset in [0] + [s * v for v in (1, 2, 4, 8, 16, 32, 64, 128) for s in (-1, 1)]:
    counts = []
    for q, island_front, other, derivative in state:
        moved = q[island_front] + offset * derivative
        counts.append(overlapping(shapely.polygons(moved), other))
    results.append({'depth_offset': offset, 'overlap_pairs': dict(zip(map(str, frames), counts))})
    print(offset, counts, flush=True)

available = [r for r in results if not any(r['overlap_pairs'].values())]
out = Path('/home/eias/workspace/.local/artifacts/20261028-2333-face-refit/feasible-island-probe.json')
out.write_text(json.dumps({'frames': frames, 'island_vertices': np.flatnonzero(island).tolist(),
                           'candidates': results, 'feasible': available}, ensure_ascii=False, indent=2) + '\n')
print(f'SAVED {out}: {len(available)} feasible candidates', flush=True)
