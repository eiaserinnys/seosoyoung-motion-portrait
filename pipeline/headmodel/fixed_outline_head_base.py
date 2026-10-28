#!/usr/bin/env python3
"""Fix the head outline around the vertical model axis using the rest surface."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

FIT = Path(__file__).resolve().parent / 'fit'


def smoothstep(edge0, edge1, value):
    t = np.clip((value-edge0)/(edge1-edge0), 0, 1)
    return t*t*(3-2*t)


def main():
    with np.load(FIT/'results/fit2f/fit-state-n3.npz') as data:
        source = {key: data[key].copy() for key in data.files}
    with np.load(FIT/'results/fit2f/surface-a.npz') as data:
        ear = data['ear_excluded'].copy()
    with np.load(FIT/'results/fit2e/final-state.npz') as data:
        fields, nodes, sigma = data['D_k'].copy(), data['theta_k'].copy(), float(data['sigma'])
    with np.load(FIT.parent/'cheek-field.npz') as data:
        cheek = data['displacement'].copy()
    poses = json.loads((FIT/'results/fit2e/poses-refined.json').read_text())['frames']
    theta = np.asarray(next(row['refined_pose'][:2] for row in poses if row['frame'] == 3))
    delta = (nodes-theta)*[1., 1.4]
    logits = -np.sum(delta*delta, axis=1)/(2*sigma*sigma)
    weights = np.exp(logits-logits.max()); weights /= weights.sum()
    rest_field = np.einsum('k,kij->ij', weights, fields)
    vertices, triangles = source['vertices'], source['triangles']
    rest = vertices+rest_field/float(source['s_ref'])+cheek
    a, b, c = triangles.T
    edges = np.vstack((np.c_[a, b], np.c_[b, c], np.c_[c, a]))
    adjacency = sparse.coo_matrix((np.ones(len(edges)*2),
                                   (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                                  shape=(len(vertices), len(vertices))).tocsr()
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    samples = []
    for height in np.linspace(rest[:, 1].min(), rest[:, 1].max(), 160):
        candidates = np.flatnonzero((np.abs(rest[:, 1]-height) <= .25) & ~ear)
        if len(candidates):
            samples.append((height, np.abs(rest[candidates, 0]).max()))
    samples = np.asarray(samples)
    radius = np.interp(rest[:, 1], samples[:, 0], samples[:, 1])
    psi = np.degrees(np.arctan2(np.abs(rest[:, 0]), rest[:, 2]))
    target = rest.copy()
    factor = radius/np.hypot(rest[:, 0], rest[:, 2])
    target[:, 0] *= factor
    target[:, 2] *= factor
    w = smoothstep(90, 105, psi)*smoothstep(150, 130, psi)*smoothstep(-3., -2., rest[:, 1])*(~ear)
    need = w[:, None]*(target-rest)
    displacement = need.copy()
    for _ in range(40):
        displacement = .5*displacement+.5*(adjacency @ displacement)/degree[:, None]
        displacement[ear] = 0
        displacement[rest[:, 1] < -3.] = 0
        displacement[psi < 90] = 0
        displacement[w > .999] = need[w > .999]
    source.update(vertices=vertices+displacement+w[:, None]*cheek, w=w)
    path = FIT/'results/fixed-outline/fit-state-fixed.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **source)
    print(json.dumps({'path': str(path.relative_to(FIT.parent.parent.parent)),
                      'vertices': len(vertices), 'full_weight': int(np.count_nonzero(w > .999)),
                      'partial_weight': int(np.count_nonzero((w > 0) & (w <= .999))),
                      'front_psi_lt_90_max_displacement_px': float(np.linalg.norm(displacement[psi < 90], axis=1).max()*float(source['s_ref'])),
                      'max_displacement_px': float(np.linalg.norm(displacement, axis=1).max()*float(source['s_ref']))},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
