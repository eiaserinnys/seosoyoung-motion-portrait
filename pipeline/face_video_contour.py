"""Front-facing face-skin silhouette for fixed-angle video fitting."""
from collections import Counter

import numpy as np
import shapely
from scipy.interpolate import make_interp_spline
from scipy.spatial import cKDTree


YAW = np.array([-45, -30, -15, 0, 15, 30, 45], float)
PITCH = np.array([-34, -17, 0, 17, 34], float)
YAW_BASIS = make_interp_spline(YAW, np.eye(len(YAW)), k=3, axis=0)
PITCH_BASIS = make_interp_spline(PITCH, np.eye(len(PITCH)), k=3, axis=0)


def centered_basis(yaw, pitch):
    return (np.outer(YAW_BASIS(yaw), PITCH_BASIS(pitch)) -
            np.outer(YAW_BASIS(0), PITCH_BASIS(0))).reshape(-1)


def rotation(yaw, pitch, roll):
    a, b, c = np.deg2rad([yaw, pitch, roll])
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rigid = np.array([[ca, sa * sb, sa * cb], [0, cb, -sb], [-sa, ca * sb, ca * cb]])
    rolled = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rolled @ rigid


def project(local, pose):
    yaw, pitch, roll = pose
    point = np.asarray(local) @ rotation(yaw, pitch, roll).T
    angle = np.deg2rad(roll)
    pivot = np.array([0., 242.])
    offset = pivot - np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]]) @ pivot
    return point[:, :2] + [502, 326] + offset


def huber_weights(distance):
    return np.minimum(1., 2. / np.maximum(np.asarray(distance), 1e-12))


def valid_viewer_samples(samples, all_observations, keep):
    closest = cKDTree(all_observations).query(samples)[1]
    return np.asarray(keep)[closest]


def contour(points, triangles, spacing=1.):
    """Return union exterior samples and their front-mesh edge interpolation."""
    points, triangles = np.asarray(points), np.asarray(triangles)
    coordinates = points[triangles]
    a, b, c = coordinates.transpose(1, 0, 2)
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    front_ids = np.flatnonzero(area2 > 0)
    front = triangles[front_ids]
    polygons = shapely.polygons(points[front])
    tree = shapely.STRtree(polygons)
    pairs = tree.query(polygons, predicate='intersects')
    pairs = pairs[:, pairs[0] < pairs[1]]
    areas = shapely.area(shapely.intersection(polygons[pairs[0]], polygons[pairs[1]]))
    overlaps = areas > 1e-7
    surface = shapely.union_all(polygons)
    components = [surface] if surface.geom_type == 'Polygon' else list(surface.geoms)
    boundaries = [polygon.exterior for polygon in components if polygon.geom_type == 'Polygon']
    samples = []
    for line in boundaries:
        count = max(1, int(np.ceil(line.length / spacing)))
        samples.append(shapely.get_coordinates(shapely.line_interpolate_point(line, np.arange(count) * line.length / count)))
    samples = np.concatenate(samples)
    counts = Counter(tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in front for i in range(3))
    edges = np.asarray([edge for edge, count in counts.items() if count == 1])
    start, vector = points[edges[:, 0]], points[edges[:, 1]] - points[edges[:, 0]]
    delta = samples[:, None, :] - start[None, :, :]
    t = np.clip(np.sum(delta * vector[None, :, :], axis=2) / np.sum(vector * vector, axis=1), 0, 1)
    distance2 = np.sum((delta - t[:, :, None] * vector[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distance2, axis=1)
    weights = t[np.arange(len(samples)), nearest]
    return {'points': samples, 'edges': edges[nearest], 'edge_t': weights,
            'boundary_length': float(sum(line.length for line in boundaries)),
            'back_faces': int(np.count_nonzero(area2 <= 0)),
            'overlap_pairs': int(np.count_nonzero(overlaps)), 'overlap_area': float(areas[overlaps].sum()),
            'overlap_triangles': front_ids[pairs[:, overlaps].T], 'overlap_areas': areas[overlaps],
            'edge_distance_max': float(np.sqrt(distance2[np.arange(len(samples)), nearest].max()))}
