#!/usr/bin/env python3
"""Export the fitted head surface and attach the rig's rest-space queries."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT / 'pipeline/headmodel/fit'
sys.path.insert(0, str(FIT))
from fit_core import project
from fit2f_core import geometric_operator, raster_silhouette
from fit2d_sheets import image_for

KEYS = [(3, 0, 0), (42, -45, 0), (192, -45, 30), (206, 0, 30),
        (219, 45, 30), (240, 45, 0), (264, 45, -30),
        (270, 0, -30), (303, -45, -30)]


def project_np(vertices, pose):
    return project(torch.as_tensor(vertices, dtype=torch.float64),
                   torch.as_tensor(pose, dtype=torch.float64)).numpy()


def phi(theta, nodes, sigma):
    delta = (nodes - theta) * [1., 1.4]
    logits = -np.sum(delta * delta, axis=1) / (2 * sigma * sigma)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def flatten_ears(vertices, triangles, fields, ear):
    lap, mass = geometric_operator(vertices, triangles)
    unknown = ear.copy()
    for _ in range(2):
        unknown[np.unique(triangles[unknown[triangles].any(axis=1)])] = True
    bilap = lap @ sparse.diags(1. / mass) @ lap
    values = np.concatenate([vertices, fields.transpose(1, 0, 2).reshape(len(vertices), -1)], axis=1)
    inside, outside = np.flatnonzero(unknown), np.flatnonzero(~unknown)
    values[inside] = splu(bilap[inside][:, inside].tocsc()).solve(-(bilap[inside][:, outside] @ values[outside]))
    return values[:, :3], values[:, 3:].reshape(len(vertices), len(fields), 3).transpose(1, 0, 2)


def vertex_normals(vertices, triangles):
    a, b, c = vertices[triangles].transpose(1, 0, 2)
    faces = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for index in range(3):
        np.add.at(normals, triangles[:, index], faces)
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def neck_skirt(vertices, triangles, fields, edges, excluded, s_ref):
    neighbors = {}
    for a, b in edges:
        a, b = int(a), int(b)
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if any(len(items) != 2 for items in neighbors.values()):
        raise ValueError('STOP: open boundary is not a set of closed loops')
    remaining, loops = set(neighbors), []
    while remaining:
        first = next(iter(remaining))
        loop, previous, current = [], None, first
        while current not in loop:
            loop.append(current)
            remaining.discard(current)
            adjacent = neighbors[current]
            previous, current = current, adjacent[0] if adjacent[0] != previous else adjacent[1]
        if current != first:
            raise ValueError('STOP: open boundary does not form a closed loop')
        loops.append(loop)
    ring = max(loops, key=len)
    if len(loops) > 1:
        print(json.dumps({'open_loops': list(map(len, loops)), 'selected': len(ring)}), flush=True)
    directed = {}
    for tri in triangles:
        for a, b in zip(tri, np.roll(tri, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key in directed:
                directed[key] = None
            else:
                directed[key] = (int(a), int(b))
    if directed[tuple(sorted(ring[:2]))] != tuple(ring[:2]):
        ring = [ring[0]] + ring[:0:-1]
    if any(directed[tuple(sorted((a, b)))] != (a, b) for a, b in zip(ring, ring[1:] + ring[:1])):
        raise ValueError('STOP: open boundary winding is inconsistent')
    ring = np.asarray(ring, dtype=int)
    count, original = len(ring), len(vertices)
    center = vertices[ring][:, [0, 2]].mean(axis=0)
    rings = [ring]
    values = vertices[ring].copy()
    extras = []
    for step in range(2):
        values = values.copy()
        values[:, [0, 2]] = center + (values[:, [0, 2]] - center) * .95
        values[:, 1] -= 45 / s_ref
        extras.append(values)
        rings.append(np.arange(original + step * count, original + (step + 1) * count))
    additions = []
    for upper, lower in zip(rings, rings[1:]):
        for i in range(count):
            j = (i + 1) % count
            additions.extend([[upper[j], upper[i], lower[i]], [upper[j], lower[i], lower[j]]])
    vertices = np.concatenate([vertices, *extras])
    fields = np.concatenate([fields, fields[:, ring, :], fields[:, ring, :]], axis=1)
    triangles = np.concatenate([triangles, np.asarray(additions, dtype=triangles.dtype)])
    excluded = np.r_[excluded, np.ones(2 * count, dtype=bool)]
    excluded[ring] = True
    stats = {'ring_length': count, 'added_vertices': 2 * count, 'added_triangles': len(additions),
             'rings': [r.tolist() for r in rings], 'loop_lengths': list(map(len, loops))}
    return vertices, triangles, fields, excluded, stats


def ear_sheet(vertices, triangles):
    normal = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                      vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    light = np.array([-.32, .52, .79]); light /= np.linalg.norm(light)
    color = np.clip(180 * (.35 + .65 * np.maximum(normal @ light, 0)), 0, 255).astype(np.uint8)
    images = []
    for label, yaw in [('FRONT', 0.), ('SIDE 90', 90.)]:
        q = project_np(vertices, [yaw, 0, 0, np.log(23), 360, 350])
        _, owner = raster_silhouette(q, triangles)
        image = np.full((720, 720, 3), 248, np.uint8)
        mask = owner >= 0
        image[mask] = color[owner[mask], None]
        cv2.putText(image, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
        images.append(image)
    path = FIT / 'sheets/head-base-ears.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.hstack(images)):
        raise RuntimeError(f'Could not write {path}')
    return path


def neck_sheet(vertices, triangles):
    normal = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                      vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    light = np.array([-.32, .52, .79]); light /= np.linalg.norm(light)
    color = np.clip(180 * (.35 + .65 * np.maximum(normal @ light, 0)), 0, 255).astype(np.uint8)
    images = []
    for label, yaw, pitch in [('FRONT', 0., 0.), ('SIDE 90', 90., 0.), ('DOWN 30', 0., -30.)]:
        q = project_np(vertices, [yaw, pitch, 0, np.log(19), 360, 300])
        _, owner = raster_silhouette(q, triangles)
        image = np.full((720, 720, 3), 248, np.uint8)
        mask = owner >= 0
        image[mask] = color[owner[mask], None]
        cv2.putText(image, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
        images.append(image)
    path = FIT / 'sheets/head-base-neck.png'
    if not cv2.imwrite(str(path), np.hstack(images)):
        raise RuntimeError(f'Could not write {path}')
    return path


def outside_sheet(points):
    image = image_for(3)
    for point in points:
        pixel = np.rint(np.asarray(point) - [152, 20]).astype(int)
        cv2.circle(image, tuple(pixel), 3, (0, 0, 255), -1, cv2.LINE_AA)
    path = FIT / 'sheets/head-base-outside.png'
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Could not write {path}')
    return path


class Attachments:
    def __init__(self, q, triangles, owner):
        self.q = q
        self.triangles = triangles
        self.visible = np.unique(owner[owner >= 0])
        faces = triangles[self.visible]
        self.coords = q[faces]
        a, b, c = self.coords.transpose(1, 0, 2)
        self.a, self.b, self.c = a, b, c
        self.det = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0]) + (c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])

    def attach(self, point):
        x, y = point
        a, b, c = self.a, self.b, self.c
        with np.errstate(divide='ignore', invalid='ignore'):
            u = ((b[:, 1]-c[:, 1])*(x-c[:, 0]) + (c[:, 0]-b[:, 0])*(y-c[:, 1])) / self.det
            v = ((c[:, 1]-a[:, 1])*(x-c[:, 0]) + (a[:, 0]-c[:, 0])*(y-c[:, 1])) / self.det
        w = 1-u-v
        candidates = np.flatnonzero((self.det < -1e-8) & (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9))
        if len(candidates):
            z = u[candidates]*a[candidates, 2] + v[candidates]*b[candidates, 2] + w[candidates]*c[candidates, 2]
            choice = candidates[np.argmax(z)]
            return {'vertices': self.triangles[self.visible[choice]].tolist(),
                    'weights': [float(u[choice]), float(v[choice]), float(w[choice])],
                    'offset': [0., 0.]}, 0.
        distance2 = np.full(len(self.visible), np.inf)
        weights = np.zeros((len(self.visible), 3))
        for i, j, start, end in ((0, 1, a, b), (1, 2, b, c), (2, 0, c, a)):
            direction = end[:, :2] - start[:, :2]
            length2 = np.sum(direction * direction, axis=1)
            t = np.clip(np.sum((point-start[:, :2])*direction, axis=1)/length2, 0, 1)
            nearest = start[:, :2] + t[:, None]*direction
            current = np.sum((nearest-point)**2, axis=1)
            improve = current < distance2
            distance2[improve] = current[improve]
            weights[improve] = 0
            weights[improve, i] = 1-t[improve]
            weights[improve, j] = t[improve]
        minimum = float(distance2.min())
        tied = np.flatnonzero(np.isclose(distance2, minimum, atol=1e-12, rtol=0))
        choice = tied[np.argmax(np.abs(self.det[tied]))]
        nearest = weights[choice] @ self.coords[choice, :, :2]
        return {'vertices': self.triangles[self.visible[choice]].tolist(),
                'weights': weights[choice].tolist(), 'offset': (point-nearest).tolist()}, float(np.sqrt(minimum))


def main():
    with np.load(FIT / 'results/fit2f/fit-state-n3.npz') as source:
        vertices, triangles, s_ref = source['vertices'].copy(), source['triangles'].copy(), float(source['s_ref'])
        open_edges, excluded = source['open_edges'].copy(), source['excluded'].copy()
    with np.load(FIT / 'results/fit2f/surface-a.npz') as source:
        ear = source['ear_excluded'].copy()
    with np.load(FIT / 'results/fit2e/final-state.npz') as source:
        fields, nodes, sigma = source['D_k'].copy(), source['theta_k'].copy(), float(source['sigma'])
    poses = json.loads((FIT / 'results/fit2e/poses-refined.json').read_text())
    refined = {row['frame']: row['refined_pose'] for row in poses['frames']}
    rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
    skin = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'face-skin')
    raw_solve = (ROOT / 'bundle/face-solve-data.mjs').read_text().split('=', 1)[1].strip().removesuffix(';')
    solve = json.loads(raw_solve)

    vertices, fields = flatten_ears(vertices, triangles, fields, ear)
    vertices = vertices.astype('<f4')
    fields = fields.astype('<f4')
    sheet = ear_sheet(vertices, triangles)
    vertices, triangles, fields, excluded, skirt = neck_skirt(vertices, triangles, fields, open_edges, excluded, s_ref)
    vertices = vertices.astype('<f4')
    fields = fields.astype('<f4')
    normals = vertex_normals(vertices.astype(float), triangles).astype('<f4')
    neck = neck_sheet(vertices, triangles)
    front = np.array(refined[3])
    theta_rest = front[:2]
    rest_vertices = vertices.astype(float) + np.einsum('k,kij->ij', phi(theta_rest, nodes, sigma), fields.astype(float)) / s_ref
    rest_pose = [front[0], front[1], 0., front[3], front[4], front[5]]
    projected = project_np(rest_vertices, rest_pose)
    _, owner = raster_silhouette(projected, triangles)
    projected[:, :2] += [152, 20]
    projected[:, 2] *= np.exp(front[3])
    binder = Attachments(projected, triangles, owner)
    groups = {'skin': skin['positions'], 'solve': solve['points'], 'eye': [[501.9, 431.5]],
              'ear': [[369, 465], [636, 464.5]]}
    attachments, statistics, outside_points = {}, {}, set()
    for name, points in groups.items():
        outside = []
        all_weights = []
        for point in points:
            key = ','.join(map(str, point))
            entry, distance = binder.attach(np.asarray(point, dtype=float))
            attachments[key] = entry
            all_weights.extend(entry['weights'])
            if distance > 0:
                outside.append(distance)
                outside_points.add(tuple(point))
        statistics[name] = {'total': len(points), 'outside': len(outside),
                            'max_distance_px': max(outside, default=0.),
                            'min_weight': min(all_weights), 'max_weight': max(all_weights)}
    outside = outside_sheet(outside_points)
    print(json.dumps({'sheet': str(sheet), 'neck_sheet': str(neck), 'outside_sheet': str(outside),
                      'skirt': {k: v for k, v in skirt.items() if k != 'rings'}, 'attachments': statistics}, ensure_ascii=False), flush=True)
    if statistics['solve']['outside'] > 111 or statistics['solve']['max_distance_px'] > 15:
        raise SystemExit('STOP: solve points exceed the approved outside count or distance')
    if statistics['ear']['outside']:
        raise SystemExit('STOP: ear attachment is outside the visible surface')

    colors = np.asarray(Image.open(ROOT / 'bundle/assets/texture_00.png').convert('RGB'))
    uv = np.asarray(skin['uvs']) * np.array([colors.shape[1], colors.shape[0]])
    position = np.asarray(skin['positions'])
    nearest = np.argmin(np.linalg.norm(position - [502, 300], axis=1))
    center = np.rint(uv[nearest] + [502, 300] - position[nearest]).astype(int)
    base_color = (colors[center[1]-4:center[1]+5, center[0]-4:center[0]+5].mean(axis=(0, 1))/255).tolist()
    asset_path = ROOT / 'bundle/assets/head-base.bin'
    offsets, offset = {}, 0
    with asset_path.open('wb') as output:
        for name, array in [('vertices', vertices), ('normals', normals), ('fields', fields), ('triangles', triangles.astype('<u4'))]:
            payload = array.tobytes(order='C')
            output.write(payload)
            offsets[name] = {'offset': offset, 'length': len(payload)}
            offset += len(payload)
    metadata = {'offsets': offsets, 'vertexCount': len(vertices), 'triangleCount': len(triangles),
                's_ref': s_ref, 'sigma': sigma, 'theta_k': nodes.tolist(),
                'keys': [{'frame': frame, 'yaw': yaw, 'pitch': pitch, 'model': refined[frame][:2]}
                         for frame, yaw, pitch in KEYS],
                'logscale_front': front[3], 'rest_translation': [front[4]+152, front[5]+20],
                'rest_eye': [501.9, 431.5], 'attachments': attachments,
                'inset_px': 0, 'color': base_color, 'attachmentStatistics': statistics,
                'neck_skirt': skirt, 'excluded_vertices': np.flatnonzero(excluded).tolist()}
    (ROOT / 'bundle/assets/head-base.json').write_text(json.dumps(metadata, ensure_ascii=False, separators=(',', ':')))
    indices = np.linspace(0, len(vertices)-1, 100, dtype=int)
    fixture_poses = [refined[frame] for frame, _, _ in KEYS]
    for yaw, pitch in [(60, 0), (-60, 0), (0, 40), (0, -40)]:
        pose = list(front); pose[:3] = [yaw, pitch, 0]; fixture_poses.append(pose)
    fixtures = {'indices': indices.tolist(), 'poses': fixture_poses,
                'projections': [project_np(vertices[indices], pose).tolist() for pose in fixture_poses],
                'fields': [np.einsum('k,kij->ij', phi(node, nodes, sigma), fields[:, indices, :].astype(float)).tolist() for node in nodes]}
    (FIT / 'results/fit2e/head-base-fixture.json').write_text(json.dumps(fixtures, ensure_ascii=False, separators=(',', ':')))
    print(json.dumps({'bin_bytes': asset_path.stat().st_size, 'json_bytes': (ROOT / 'bundle/assets/head-base.json').stat().st_size}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
