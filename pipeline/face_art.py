"""Extend interior skin colors to the face mesh edge for directional blending."""

from collections import Counter

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def mesh_distances(mesh):
    x, y, width, height = mesh['bbox']
    points = np.asarray(mesh['positions'], dtype=float) - [x, y]
    triangles = np.asarray(mesh['triangles'], dtype=int)
    edges = Counter(tuple(sorted(edge)) for tri in triangles
                    for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
    yy, xx = np.mgrid[:height, :width]
    xx, yy = xx + .5, yy + .5
    distance2 = np.full((height, width), np.inf)
    for (a, b), count in edges.items():
        if count != 1:
            continue
        start, delta = points[a], points[b] - points[a]
        length2 = np.dot(delta, delta)
        if length2 == 0:
            continue
        t = np.clip(((xx-start[0])*delta[0] + (yy-start[1])*delta[1]) / length2, 0, 1)
        squared = (xx-start[0]-t*delta[0])**2 + (yy-start[1]-t*delta[1])**2
        np.minimum(distance2, squared, out=distance2)
    inside = np.zeros((height, width), np.uint8)
    for tri in triangles:
        polygon = np.rint((points[tri] - .5) * 256).astype(np.int32)
        cv2.fillPoly(inside, [polygon], 1, shift=8)
    return np.sqrt(distance2), inside.astype(bool)


def clean_face(art, distance, inside):
    art = np.asarray(art, dtype=np.uint8)
    interior = inside & (distance > 12) & (art[:, :, 3] > 0)
    nearest = distance_transform_edt(~interior, return_distances=False, return_indices=True)
    extended = distance <= 9
    result = art.copy()
    result[extended, :3] = art[nearest[0], nearest[1], :3][extended]
    return result, extended


def write_clean_assets(mesh, after, before, assets):
    ax, ay, width, height = mesh['atlas']
    distance, inside = mesh_distances(mesh)
    counts = {}
    for name, atlas in [('face-clean.png', after), ('face-clean-before.png', before)]:
        image = np.asarray(atlas.convert('RGBA'))[ay:ay+height, ax:ax+width]
        cleaned, extended = clean_face(image, distance, inside)
        Image.fromarray(cleaned).save(assets / name)
        counts[name] = {'extended_pixels': int(extended.sum()),
                        'changed_pixels': int(np.count_nonzero(np.any(cleaned[:, :, :3] != image[:, :, :3], axis=2)))}
    return counts
