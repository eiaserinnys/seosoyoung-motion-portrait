#!/usr/bin/env python3
"""Restrict the rear side of the fitted head to the approved temple line."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

FIT = Path(__file__).resolve().parent / 'fit'
Z_CENTER = .5


def smoothstep(edge0, edge1, value):
    t = np.clip((value - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def temple_protrusion(vertices, original, ear, scale):
    """Compare the near silhouette with the original front-side vertices."""
    result = []
    for yaw in (-35, -25, -16, 16, 25, 35):
        a = np.radians(yaw)
        x = vertices[:, 0]*np.cos(a) + vertices[:, 2]*np.sin(a)
        z = -vertices[:, 0]*np.sin(a) + vertices[:, 2]*np.cos(a)
        near_x = -np.sign(yaw)*x
        for pitch in (0, 12):
            b = np.radians(pitch)
            y = vertices[:, 1]*np.cos(b) - z*np.sin(b)
            maximum = 0.
            bins = 0
            for height in np.arange(-2.3, 11.0001, .25):
                band = (np.abs(y-height) <= .15) & ~ear
                front = band & (original[:, 2] >= Z_CENTER)
                if band.any() and front.any():
                    maximum = max(maximum, float(near_x[band].max() - near_x[front].max()))
                    bins += 1
            result.append({'yaw': yaw, 'pitch': pitch, 'max_px': maximum*scale, 'bins': bins})
    return result


def main():
    with np.load(FIT/'results/fit2f/fit-state-n3.npz') as data:
        source = {key: data[key].copy() for key in data.files}
    with np.load(FIT/'results/fit2f/surface-a.npz') as data:
        ear = data['ear_excluded'].copy()
    vertices, triangles = source['vertices'], source['triangles']
    a, b, c = triangles.T
    edges = np.vstack((np.c_[a, b], np.c_[b, c], np.c_[c, a]))
    adjacency = sparse.coo_matrix((np.ones(len(edges)*2),
                                   (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                                  shape=(len(vertices), len(vertices))).tocsr()
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    ear_dilated = ear.copy()
    for _ in range(3):
        ear_dilated |= (adjacency @ ear_dilated.astype(float)) > 0
    samples = []
    for height in np.linspace(vertices[:, 1].min(), vertices[:, 1].max(), 120):
        candidates = np.flatnonzero((np.abs(vertices[:, 1]-height) <= .35) &
                                    (vertices[:, 2] >= Z_CENTER) & ~ear_dilated)
        if len(candidates):
            index = candidates[np.argmax(np.abs(vertices[candidates, 0]))]
            samples.append((height, abs(vertices[index, 0]), vertices[index, 2]))
    samples = np.asarray(samples)
    xt = np.interp(vertices[:, 1], samples[:, 0], samples[:, 1])
    zt = np.interp(vertices[:, 1], samples[:, 0], samples[:, 2])
    limit = np.maximum(xt-(zt-vertices[:, 2])*np.tan(np.radians(40)), 1.5)
    w = smoothstep(Z_CENTER, Z_CENTER-1., vertices[:, 2]) * smoothstep(-3., -1.5, vertices[:, 1])
    need = np.sign(vertices[:, 0])*(np.minimum(np.abs(vertices[:, 0]), limit)-np.abs(vertices[:, 0]))*w
    displacement = np.zeros_like(vertices)
    displacement[:, 0] = need
    for _ in range(40):
        displacement = .5*displacement + .5*(adjacency @ displacement)/degree[:, None]
        displacement[vertices[:, 2] > Z_CENTER+1.] = 0
        displacement[vertices[:, 1] < -3.5] = 0
        displacement[:, 0] = np.where(np.abs(need) > np.abs(displacement[:, 0]), need, displacement[:, 0])
    side = vertices+displacement
    g = smoothstep(-3., -1.5, vertices[:, 1])
    source.update(vertices=side, w=w, g=g)
    path = FIT/'results/side-fix/fit-state-side.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **source)
    scale = float(source['s_ref'])
    widths = {}
    for height in (0., 2.5, 5., 7.5):
        row = np.abs(vertices[:, 1]-height) <= .35
        rear = row & (vertices[:, 2] < -3)
        widths[str(height)] = {'all_before': float(np.max(np.abs(vertices[row, 0]))),
                               'all_after': float(np.max(np.abs(side[row, 0]))),
                               'rear_before': float(np.max(np.abs(vertices[rear, 0]))),
                               'rear_after': float(np.max(np.abs(side[rear, 0])))}
    report = {'path': str(path), 'max_displacement_px': float(np.max(np.abs(displacement[:, 0]))*scale),
              'widths_ict': widths, 'temple_protrusion': temple_protrusion(side, vertices, ear, scale)}
    (FIT/'results/side-fix/side-fix-metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
