"""Fixed-camera global fit and pixel-unit linear Laplacian surface editing."""
from __future__ import annotations
import numpy as np
from fit_core import FIT, ROOT, read_json

RESULTS = FIT/'results/fit2d'


def pose_affine(to720):
    """Video poses already use 720px; image transforms start in source 640px."""
    return np.asarray(to720, dtype=float)[:2] @ np.diag([640/720, 640/720, 1.])


def frame_sets():
    rows = read_json(ROOT/'pipeline/headmodel/data/frames.json')['frames']
    keys = [r['frame'] for r in rows if r['role'] == 'key']
    samples = [r['frame'] for r in rows if r['role'] == 'sample']
    assert len(keys) == 9 and len(samples) == 121
    return keys + samples[:120:5], [r['frame'] for r in rows], [samples[i] for i in (0,20,40,60,80,100)]


def template():
    with np.load(FIT/'results/fit2c/egg-template-v5.npz') as b:
        v, tri, modes = (b[k].copy() for k in ('neutral_vertices','triangles','identity_modes'))
        masks = {k:b[n].copy() for k,n in [('head','head_mask'),('face','face_mask'),
                    ('ear-left','ear_left_mask'),('ear-right','ear_right_mask')]}
    # A retains 2b's lower-Face loss definition; B uses the entire Face mask.
    faces = {'head':np.flatnonzero(masks['head'][tri].all(1)),
             'face':np.flatnonzero((masks['face'] & (v[:,1]<=1.5))[tri].all(1))}
    for k in ('ear-left','ear-right'):
        faces[k] = np.flatnonzero((masks[k][tri].sum(1)>=2) & masks['head'][tri].all(1))
    return v,tri,modes,masks,faces
