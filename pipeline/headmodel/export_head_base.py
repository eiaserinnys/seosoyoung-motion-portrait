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


def root_rotation(reference, current):
    reference = reference - reference.mean(axis=0)
    current = current - current.mean(axis=0)
    u, _, vh = np.linalg.svd(reference.T @ current)
    signs = np.diag([1., 1., np.linalg.det(vh.T @ u.T)])
    return vh.T @ signs @ u.T


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


def boundary_loops(edges):
    neighbors = {}
    for a, b in edges:
        a, b = int(a), int(b)
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if any(len(items) != 2 for items in neighbors.values()):
        raise ValueError('STOP: boundary is not a set of closed loops')
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
            raise ValueError('STOP: boundary does not form a closed loop')
        loops.append(loop)
    return loops


def smooth_open_boundary(vertices, fields, edges):
    loops = boundary_loops(edges)
    ring = np.asarray(max(loops, key=len), dtype=int)
    if len(loops) > 1:
        print(json.dumps({'open_loops': list(map(len, loops)), 'selected': len(ring)}), flush=True)
    vertices = vertices.copy()
    fields = fields.copy()
    for _ in range(10):
        current = vertices[ring].copy()
        vertices[ring] = .5 * current + .25 * (np.roll(current, 1, axis=0) + np.roll(current, -1, axis=0))
        current = fields[:, ring, :].copy()
        fields[:, ring, :] = .5 * current + .25 * (np.roll(current, 1, axis=1) + np.roll(current, -1, axis=1))
    return vertices, fields, ring


def cap_open_boundary(vertices, fields, triangles, ring):
    """Close the existing boundary without moving its vertices or adding a skirt."""
    oriented = {}
    for face in triangles:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(a), int(b))))
            if key in oriented:
                del oriented[key]
            else:
                oriented[key] = (int(a), int(b))
    center = len(vertices)
    cap = []
    for a, b in zip(ring, np.roll(ring, -1)):
        edge = oriented[tuple(sorted((int(a), int(b))))]
        cap.append([edge[1], edge[0], center])
    vertices = np.concatenate([vertices, vertices[ring].mean(axis=0, keepdims=True)])
    fields = np.concatenate([fields, fields[:, ring, :].mean(axis=1, keepdims=True)], axis=1)
    cap = np.asarray(cap, dtype=triangles.dtype)
    normal = np.cross(vertices[cap[:, 1]] - vertices[cap[:, 0]],
                      vertices[cap[:, 2]] - vertices[cap[:, 0]]).sum(axis=0)
    if normal[1] >= 0:
        raise ValueError('STOP: neck cap winding does not face down')
    return vertices, fields, np.concatenate([triangles, cap])


def ear_root_rings(vertices, triangles, ear):
    rings = {}
    for side, selected in [('left', ear & (vertices[:, 0] < 0)), ('right', ear & (vertices[:, 0] >= 0))]:
        faces = triangles[np.all(selected[triangles], axis=1)]
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        loops = boundary_loops(unique[counts == 1])
        if len(loops) != 1:
            raise ValueError(f'STOP: {side} ear root has {len(loops)} boundary loops')
        rings[side] = np.asarray(loops[0], dtype=int)
    return rings


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
        ear_vertices = source['vertices'].copy()
        if not np.array_equal(source['triangles'], triangles):
            raise ValueError('STOP: ear root and fitted surface use different topology')
    with np.load(FIT / 'results/fit2e/final-state.npz') as source:
        fields, nodes, sigma = source['D_k'].copy(), source['theta_k'].copy(), float(source['sigma'])
    poses = json.loads((FIT / 'results/fit2e/poses-refined.json').read_text())
    refined = {row['frame']: row['refined_pose'] for row in poses['frames']}
    rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
    skin = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'face-skin')
    raw_solve = (ROOT / 'bundle/face-solve-data.mjs').read_text().split('=', 1)[1].strip().removesuffix(';')
    solve = json.loads(raw_solve)

    root_rings = ear_root_rings(ear_vertices, triangles, ear)
    vertices, fields = flatten_ears(vertices, triangles, fields, ear)
    sheet = ear_sheet(vertices.astype('<f4'), triangles)
    vertices, fields, open_ring = smooth_open_boundary(vertices, fields, open_edges)
    excluded[open_ring] = True
    attachment_triangles = triangles.copy()
    vertices, fields, triangles = cap_open_boundary(vertices, fields, triangles, open_ring)
    excluded = np.r_[excluded, True]
    vertices = vertices.astype('<f4')
    fields = fields.astype('<f4')
    normals = vertex_normals(vertices.astype(float), triangles).astype('<f4')
    neck = neck_sheet(vertices, triangles)
    front = np.array(refined[3])
    theta_rest = front[:2]
    rest_vertices = vertices.astype(float) + np.einsum('k,kij->ij', phi(theta_rest, nodes, sigma), fields.astype(float)) / s_ref
    rest_pose = [front[0], front[1], 0., front[3], front[4], front[5]]
    projected = project_np(rest_vertices, rest_pose)
    # The cap closes the neck opening; existing face attachments stay on the head surface.
    _, owner = raster_silhouette(projected, attachment_triangles)
    projected[:, :2] += [152, 20]
    projected[:, 2] *= np.exp(front[3])
    ear_roots = {}
    for side, anchor in [('left', [369, 465]), ('right', [636, 464.5])]:
        ring = root_rings[side]
        center = projected[ring].mean(axis=0)
        ear_roots[side] = {'ring': ring.tolist(), 'anchor': anchor, 'rest_center_ict': rest_vertices[ring].mean(axis=0).tolist(),
                           'rest_center': center.tolist(), 'delta': [anchor[0] - center[0], anchor[1] - center[1], 0.]}
    binder = Attachments(projected, attachment_triangles, owner)
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
                      'open_ring': len(open_ring), 'cap_vertices': 1, 'cap_triangles': len(open_ring),
                      'ear_roots': {side: {'count': len(data['ring']), 'center': data['rest_center'], 'delta': data['delta']} for side, data in ear_roots.items()},
                      'attachments': statistics}, ensure_ascii=False), flush=True)
    if statistics['solve']['outside'] > 111 or statistics['solve']['max_distance_px'] > 15:
        raise SystemExit('STOP: solve points exceed the approved outside count or distance')

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
                'open_boundary': open_ring.tolist(), 'ear_roots': ear_roots,
                'excluded_vertices': np.flatnonzero(excluded).tolist()}
    (ROOT / 'bundle/assets/head-base.json').write_text(json.dumps(metadata, ensure_ascii=False, separators=(',', ':')))
    indices = np.linspace(0, len(vertices)-1, 100, dtype=int)
    fixture_poses = [refined[frame] for frame, _, _ in KEYS]
    for yaw, pitch in [(60, 0), (-60, 0), (0, 40), (0, -40)]:
        pose = list(front); pose[:3] = [yaw, pitch, 0]; fixture_poses.append(pose)
    fixtures = {'indices': indices.tolist(), 'poses': fixture_poses,
                'projections': [project_np(vertices[indices], pose).tolist() for pose in fixture_poses],
                'fields': [np.einsum('k,kij->ij', phi(node, nodes, sigma), fields[:, indices, :].astype(float)).tolist() for node in nodes]}
    fixtures['ear_roots'] = []
    for frame, yaw, pitch in KEYS:
        theta = refined[frame][:2]
        posed_vertices = vertices.astype(float) + np.einsum('k,kij->ij', phi(theta, nodes, sigma), fields.astype(float)) / s_ref
        q = project_np(posed_vertices, [theta[0], theta[1], 0., front[3], front[4], front[5]])
        q[:, :2] += [152, 20]
        q[:, 2] *= np.exp(front[3])
        expected = {}
        for side, data in ear_roots.items():
            ring = data['ring']
            rotation = root_rotation(projected[ring], q[ring])
            anchor = q[ring].mean(axis=0) + rotation @ (np.asarray(data['delta']) + [0., 0., 5.])
            expected[side] = {'R': rotation.tolist(), 'anchor': anchor.tolist()}
        fixtures['ear_roots'].append({'frame': frame, 'yaw': yaw, 'pitch': pitch, 'expected': expected})
    (FIT / 'results/fit2e/head-base-fixture.json').write_text(json.dumps(fixtures, ensure_ascii=False, separators=(',', ':')))
    print(json.dumps({'bin_bytes': asset_path.stat().st_size, 'json_bytes': (ROOT / 'bundle/assets/head-base.json').stat().st_size}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
