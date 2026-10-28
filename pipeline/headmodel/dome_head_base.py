#!/usr/bin/env python3
"""Replace the fitted upper head with the approved fixed ellipsoid dome."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull

FIT = Path(__file__).resolve().parent / 'fit'


def smoothstep(edge0, edge1, value):
    t = np.clip((value - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def mvee(points):
    """Khachiyan minimum-volume enclosing ellipsoid."""
    count, dimensions = points.shape
    q = np.vstack((points.T, np.ones(count)))
    weights = np.full(count, 1. / count)
    while True:
        x = (q * weights) @ q.T
        m = np.sum(q * (np.linalg.inv(x) @ q), axis=0)
        index = int(np.argmax(m))
        if m[index] / (dimensions + 1) - 1 < 1e-3:
            break
        step = (m[index] - dimensions - 1) / ((dimensions + 1) * (m[index] - 1))
        weights *= 1 - step
        weights[index] += step
    center = points.T @ weights
    matrix = np.linalg.inv((points.T * weights) @ points - np.outer(center, center)) / dimensions
    return center, matrix


def main():
    with np.load(FIT / 'results/fit2f/fit-state-n3.npz') as data:
        source = {key: data[key].copy() for key in data.files}
    with np.load(FIT / 'results/fit2f/surface-a.npz') as data:
        ear = data['ear_excluded'].copy()
    vertices, triangles = source['vertices'], source['triangles']
    a, b, c = triangles.T
    edges = np.vstack((np.c_[a, b], np.c_[b, c], np.c_[c, a]))
    adjacency = sparse.coo_matrix((np.ones(len(edges) * 2),
                                   (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                                  shape=(len(vertices), len(vertices))).tocsr()
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    ear_dilated = ear.copy()
    for _ in range(4):
        ear_dilated = (adjacency @ ear_dilated.astype(float)) > 0
    selected = (vertices[:, 1] > 3.5) & ~ear_dilated
    hull_points = vertices[selected][ConvexHull(vertices[selected]).vertices]
    center, matrix = mvee(hull_points)
    q = vertices[selected] - center
    matrix /= np.einsum('ij,jk,ik->i', q, matrix, q).max()
    q = vertices - center
    ellipsoid = center + q / np.sqrt(np.einsum('ij,jk,ik->i', q, matrix, q))[:, None]
    phi = np.degrees(np.arctan2(np.abs(vertices[:, 0]), vertices[:, 2] - center[2]))
    face = smoothstep(75, 30, phi) * smoothstep(11.5, 10., vertices[:, 1])
    w = smoothstep(2.5, 6.5, vertices[:, 1]) * (1 - face) * (~ear_dilated)
    displacement = w[:, None] * (ellipsoid - vertices)
    for _ in range(80):
        displacement = .5 * displacement + .5 * (adjacency @ displacement) / degree[:, None]
        displacement[ear_dilated] = 0
        displacement[vertices[:, 1] < 2.] = 0
    source['vertices'] = vertices + displacement
    source.update(w=w, center=center, A=matrix)
    path = FIT / 'results/dome/fit-state-dome.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **source)
    scale = float(source['s_ref'])
    lengths = np.linalg.norm(displacement, axis=1) * scale
    print(json.dumps({'path': str(path), 'center': center.tolist(),
                      'principal_radii': (1 / np.sqrt(np.linalg.eigvalsh(matrix))).tolist(),
                      's_ref': scale, 'hull_points': len(hull_points),
                      'moved_vertices': int(np.count_nonzero(lengths > 1e-8)),
                      'max_displacement_px': float(lengths.max()),
                      'front_phi_max_degrees': 25,
                      'front_y_3.5_7.5_px': float(lengths[(phi < 25) & (vertices[:, 1] >= 3.5) & (vertices[:, 1] < 7.5)].max()),
                      'front_y_7.5_9.5_px': float(lengths[(phi < 25) & (vertices[:, 1] >= 7.5) & (vertices[:, 1] < 9.5)].max()),
                      'front_y_9.5_12_px': float(lengths[(phi < 25) & (vertices[:, 1] >= 9.5) & (vertices[:, 1] < 12)].max()),
                      'side_upper_px': float(lengths[(phi >= 60) & (phi < 120) & (vertices[:, 1] >= 6.5)].max()),
                      'back_px': float(lengths[phi >= 120].max())}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
