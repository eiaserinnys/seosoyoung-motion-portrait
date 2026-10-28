"""Make the directional face texture without altering pixels outside its rim."""

from collections import Counter

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

STROKE_DARKNESS = 30
INPAINT_RADIUS = 3


def boundary_weights(mesh):
    points = np.asarray(mesh['positions'], dtype=float)
    edges = Counter(tuple(sorted(edge)) for tri in mesh['triangles']
                    for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
    boundary = np.asarray([edge for edge, count in edges.items() if count == 1], dtype=int)
    start, delta = points[boundary[:, 0]], points[boundary[:, 1]] - points[boundary[:, 0]]
    weights = []
    for point in points:
        t = np.clip(np.sum((point-start)*delta, axis=1) / np.sum(delta*delta, axis=1), 0, 1)
        distance = np.linalg.norm(point-start-t[:, None]*delta, axis=1).min()
        weights.append(np.clip(1-distance/12, 0, 1))
    return np.asarray(weights)


def rim_pixels(mesh):
    x, y, width, height = mesh['bbox']
    points = np.asarray(mesh['positions'], dtype=float) - [x, y]
    weights = boundary_weights(mesh)
    rim = np.zeros((height, width), dtype=np.float32)
    for tri in np.asarray(mesh['triangles'], dtype=int):
        p0, p1, p2 = points[tri]
        lo = np.maximum(np.floor(np.min([p0,p1,p2], axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(np.max([p0,p1,p2], axis=0)).astype(int), [width,height])
        if np.any(lo >= hi):
            continue
        yy, xx = np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
        xx, yy = xx+.5, yy+.5
        denom = (p1[1]-p2[1])*(p0[0]-p2[0]) + (p2[0]-p1[0])*(p0[1]-p2[1])
        if abs(denom) < 1e-10:
            continue
        a = ((p1[1]-p2[1])*(xx-p2[0]) + (p2[0]-p1[0])*(yy-p2[1]))/denom
        b = ((p2[1]-p0[1])*(xx-p2[0]) + (p0[0]-p2[0])*(yy-p2[1]))/denom
        c = 1-a-b
        inside = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        value = a*weights[tri[0]] + b*weights[tri[1]] + c*weights[tri[2]]
        region = rim[lo[1]:hi[1],lo[0]:hi[0]]
        region[inside] = np.maximum(region[inside], value[inside])
    return rim


def clean_face(art, rim):
    art = np.asarray(art, dtype=np.uint8)
    interior = (rim <= 1e-7) & (art[:, :, 3] >= 128)
    nearest = distance_transform_edt(~interior, return_distances=False, return_indices=True)
    skin = art[nearest[0], nearest[1], :3]
    median = cv2.medianBlur(cv2.cvtColor(skin, cv2.COLOR_RGB2GRAY), 15)
    luma = cv2.cvtColor(art[:, :, :3], cv2.COLOR_RGB2GRAY)
    stroke = (rim > 1e-7) & (art[:, :, 3] > 0) & (median.astype(int)-luma.astype(int) >= STROKE_DARKNESS)
    source = art[:, :, :3].copy()
    source[art[:, :, 3] == 0] = skin[art[:, :, 3] == 0]
    painted = cv2.inpaint(source, stroke.astype(np.uint8), INPAINT_RADIUS, cv2.INPAINT_TELEA)
    output = art.copy()
    output[stroke, :3] = painted[stroke]
    return output, stroke


def write_clean_assets(mesh, after, before, assets):
    ax, ay, width, height = mesh['atlas']
    rim = rim_pixels(mesh)
    counts = {}
    for name, atlas in [('face-clean.png', after), ('face-clean-before.png', before)]:
        image = np.asarray(atlas.convert('RGBA'))[ay:ay+height, ax:ax+width]
        cleaned, stroke = clean_face(image, rim)
        Image.fromarray(cleaned).save(assets / name)
        counts[name] = int(stroke.sum())
    return counts
