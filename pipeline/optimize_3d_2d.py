"""Stage 1 joint depth / translation / smooth 2D residual least squares.

The mesh graph is sparse; its small Schur system is solved densely with NumPy.
No SciPy dependency. Exact observation pins follow the unconstrained depth solve.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAMBDA = {'2D': .02, '2s': .25, 'Z': .5, 'prior': .01}


def projection(points, yaw, pitch):
    a, b = np.radians([yaw, pitch])
    w = .65*min(1., abs(yaw/50*pitch/40))
    dx, dy = (points - [502., 326.]).T
    xy = np.column_stack((502.+dx*np.cos(a),
                          326.+dy*np.cos(b)+(1-w)*dx*np.sin(a)*np.sin(b)))
    depth = np.array([np.sin(a), -((1-w)*np.cos(a)+w)*np.sin(b)])
    return xy, depth


def laplacian(points, triangles, mesh_count):
    edges = set()
    for tri in triangles:
        for u, v in zip(tri, np.roll(tri, -1)):
            edges.add(tuple(sorted((int(u), int(v)))))
    # Sample and landmark nodes attach to the same local mesh, without a second field.
    for i in range(mesh_count, len(points)):
        distances = np.sum((points[:mesh_count]-points[i])**2, axis=1)
        for j in np.argsort(distances)[:3]:
            edges.add((int(j), i))
    result = np.zeros((len(points), len(points)))
    for i, j in edges:
        weight = 200./(np.sum((points[i]-points[j])**2)+4.)
        result[i, i] += weight
        result[j, j] += weight
        result[i, j] -= weight
        result[j, i] -= weight
    return result


def schur_pose(weights, regularizer):
    """Eliminate both XY residual vectors with a single symmetric factor solve."""
    w = np.diag(weights)
    inverse_w = np.linalg.solve(w+regularizer, w)
    effective = w-w@inverse_w
    return (effective+effective.T)*.5, inverse_w


def prepare(inputs):
    points = np.array(inputs['points'])
    lap = laplacian(points, inputs['triangles'], inputs['meshCount'])
    systems = []
    for pose in inputs['poses']:
        if not (pose['yaw'] or pose['pitch']):
            continue
        weights = np.array(pose['weights'])
        r = LAMBDA['2s']*lap+np.diag(LAMBDA['2D']*np.where(weights > 0, 1., 1e-4))
        effective, inverse_w = schur_pose(weights, r)
        ones = effective.sum(axis=1)
        total = ones.sum()
        centered = effective-np.outer(ones, ones)/total
        a, b = projection(points, pose['yaw'], pose['pitch'])
        d = np.array(pose['obs'])-a
        systems.append(dict(pose=pose, a=a, b=b, d=d, r=r, inverse_w=inverse_w,
                            ones=ones, total=total,
                            h=np.dot(b, b)*centered, g=centered@d@b))
    return lap, systems


def solve_depth(prior, lap, systems):
    h = LAMBDA['Z']*lap+LAMBDA['prior']*np.eye(len(prior))
    g = LAMBDA['prior']*prior
    for s in systems:
        h = h+s['h']
        g = g+s['g']
    return np.linalg.solve(h, g)


def recover(system, z):
    d = system['d']-z[:, None]*system['b']
    t = system['ones']@d/system['total']
    du = system['inverse_w']@(d-t)
    # Dirichlet pins keep the approved silhouette/landmarks and r11 yaw exact.
    # Unobserved vertices minimize the SAME residual regularizer with those pins.
    fixed = np.zeros(len(z), dtype=bool)
    fixed[system['pose']['pins']] = True
    du[fixed] = (d-t)[fixed]
    free = ~fixed
    if np.any(free):
        weights = np.array(system['pose']['weights'])
        m = system['r']+np.diag(weights)
        rhs = weights[:, None]*(d-t)
        du[free] = np.linalg.solve(m[np.ix_(free, free)], rhs[free]-m[np.ix_(free, fixed)]@du[fixed])
    return t, du


def determinants(rest, deformed, triangles):
    def area(p):
        a, b, c = p[np.array(triangles)].transpose(1, 0, 2)
        return (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])-(b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
    return area(deformed)/area(rest)


def solve_family45(inputs, z0):
    """Each side shares depth in its evaluated 45-degree rest frame."""
    points = np.array(inputs['points'])
    lap = laplacian(points, inputs['triangles'], inputs['meshCount'])
    family = inputs['family45']
    result = {}
    solved = {}
    for sign, side in [(-1, 'Left'), (1, 'Right')]:
        rest = np.array(family['p45'+side])
        angle = np.radians(sign*45)
        prior = -(points[:, 0]-502)*np.sin(angle)+np.array(z0)*np.cos(angle)
        systems = []
        for pose in family['poses']:
            if np.sign(pose['yaw']) != sign:
                continue
            weights = np.array(pose['weights'])
            r = .05*lap+np.diag(.2*np.where(weights > 0, 1., 1e-4))
            effective, inverse_w = schur_pose(weights, r)
            ones = effective.sum(axis=1)
            total = ones.sum()
            centered = effective-np.outer(ones, ones)/total
            a, b = projection(rest, pose['delta'], 0)
            d = np.array(pose['obs'])-a
            systems.append(dict(pose=pose, a=a, b=b, d=d, r=r, inverse_w=inverse_w,
                                ones=ones, total=total, h=np.dot(b, b)*centered, g=centered@d@b))
        h = .05*lap+.001*np.eye(len(points))
        g = .001*prior
        for s in systems:
            h = h+s['h']
            g = g+s['g']
        z = np.linalg.solve(h, g)
        result['Z'+side] = z.tolist()
        for s in systems:
            pose = s['pose']
            t, du = recover(s, z)
            actual = s['a']+z[:, None]*s['b']+t+du
            error = actual-np.array(pose['obs'])
            det = determinants(points, actual, inputs['triangles'])
            w = np.array(pose['weights'])[:, None]
            stats = dict(rms=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                         observationRms=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                         explained3D=float(1-np.sum(w*du**2)/np.sum(w*s['d']**2)),
                         duRms=float(np.sqrt(np.mean(np.sum(du**2, axis=1)))),
                         duMax=float(np.linalg.norm(du, axis=1).max()),
                         minDetJ=float(det.min()), reversedTriangles=int(np.count_nonzero(det <= 0)))
            solved[pose['name']] = dict(name=pose['name'], yaw=pose['yaw'], delta=pose['delta'], pitch=0,
                                       t=t.tolist(), du=du.tolist(), stats=stats, eyeWidthScale=pose['eyeWidthScale'])
    result['poses'] = [solved[p['name']] for p in family['poses']]
    return result


def solve(inputs):
    points = np.array(inputs['points'])
    lap, systems = prepare(inputs)
    z = solve_depth(np.array(inputs['prior']), lap, systems)
    poses = []
    for pose in inputs['poses']:
        if not (pose['yaw'] or pose['pitch']):
            poses.append(dict(name=pose['name'], yaw=0, pitch=0, t=[0, 0], s=1.,
                              du=np.zeros((len(z), 2)).tolist(), stats=dict(
                                  rms=0., explained3D=1., duRms=0., duMax=0., minDetJ=1., reversedTriangles=0)))
            continue
        system = next(s for s in systems if s['pose']['name'] == pose['name'])
        t, du = recover(system, z)
        actual = system['a']+z[:, None]*system['b']+t+du
        mask = np.array(pose['weights']) > 0
        gates = np.array(pose['pins'])
        w = np.array(pose['weights'])[mask, None]
        residual_energy = np.sum(w*du[mask]**2)
        # Fraction of measured displacement from the depth-free rotated plane
        # explained by the shared 3D depth plus pose translation (weighted SSE).
        baseline_energy = np.sum(w*system['d'][mask]**2)
        det = determinants(points, actual, inputs['triangles'])
        error = actual-np.array(pose['obs'])
        stats = dict(rms=float(np.sqrt(np.mean(np.sum(error[gates]**2, axis=1)))),
                     observationRms=float(np.sqrt(np.mean(np.sum(error[mask]**2, axis=1)))),
                     explained3D=float(1-residual_energy/baseline_energy),
                     duRms=float(np.sqrt(np.mean(np.sum(du**2, axis=1)))),
                     duMax=float(np.linalg.norm(du, axis=1).max()),
                     minDetJ=float(det.min()), reversedTriangles=int(np.count_nonzero(det <= 0)))
        poses.append(dict(name=pose['name'], yaw=pose['yaw'], pitch=pose['pitch'],
                          t=t.tolist(), s=1., du=du.tolist(), stats=stats))
    cv = []
    rim = [s[side] for s in inputs['samples'] for side in ['left', 'right']]
    anchors = list(inputs['anchors'].values())
    for held in systems:
        remaining = [s for s in systems if s is not held]
        zz = solve_depth(np.array(inputs['prior']), lap, remaining)
        p = held['pose']
        prediction = held['a']+zz[:, None]*held['b']
        # Add available cardinal residuals, never the held-out pose's observations.
        for s in remaining:
            q = s['pose']
            if (p['yaw'] and q['yaw'] == p['yaw'] and q['pitch'] == 0) or (
                    p['pitch'] and q['pitch'] == p['pitch'] and q['yaw'] == 0):
                t, du = recover(s, zz)
                prediction = prediction+t+du
        error = prediction-np.array(p['obs'])
        cv.append(dict(heldOut=p['name'], silhouetteRms=float(np.sqrt(np.mean(np.sum(error[rim]**2, axis=1)))),
                       featureRms=float(np.sqrt(np.mean(np.sum(error[anchors]**2, axis=1))))))
    sources = {key: hashlib.sha256((ROOT/f'source/viewer/{name}.json').read_bytes()).hexdigest()
               for key, name in [('outlineTraces', 'outline-traces'), ('featureFit', 'face-feature-fit'), ('depth', 'face-depth')]}
    # A query triangulation includes exact silhouette samples and feature anchors.
    # OpenCV is already used by the silhouette pipeline; the solve itself needs NumPy only.
    import cv2
    evaluation_points = np.array(inputs['evaluationPoints'])
    lower = np.floor(evaluation_points.min(axis=0)-1).astype(int)
    upper = np.ceil(evaluation_points.max(axis=0)+1).astype(int)
    subdiv = cv2.Subdiv2D((*lower.tolist(), *(upper-lower).tolist()))
    anchor_ids = set(inputs['anchors'].values())
    # Feature anchors are separate controls, not inserted skin vertices. Skin
    # queries interpolate the mesh/rim; features query their six solved nodes.
    skin_query_ids = [i for i in range(len(points)) if i not in anchor_ids]
    for pt in evaluation_points[skin_query_ids]:
        subdiv.insert(tuple(map(float, pt)))
    query_triangles = []
    for tri in subdiv.getTriangleList().reshape(-1, 3, 2):
        distances = np.sum((tri[:, None, :]-evaluation_points[None, :, :])**2, axis=2)
        indices = distances.argmin(axis=1)
        if np.max(distances[np.arange(3), indices]) < .001:
            query_triangles.append(indices.tolist())
    return dict(family='face-0', restMesh='face-skin', points=inputs['points'], triangles=inputs['triangles'],
                evaluationPoints=inputs['evaluationPoints'], evaluationTriangles=query_triangles,
                meshCount=inputs['meshCount'], samples=inputs['samples'], anchors=inputs['anchors'],
                Z=z.tolist(), poses=poses, **{'lambda': LAMBDA}, crossValidation=cv, sources=sources,
                explained3DDefinition='1 - weighted residual SSE / weighted displacement SSE from depth-free rotated rest plane; translation included in 3D explanation')


def main():
    inputs = json.loads(subprocess.check_output(['node', str(ROOT/'pipeline/face_solve_inputs.mjs')], timeout=60))
    result = solve(inputs)
    path = ROOT/'source/viewer/face-solve.json'
    # The 45..90 degree family is a separate approved fit. Updating face-0
    # observations must not refit its depths from a different face-0 prior.
    result['family45'] = json.loads(path.read_text())['family45']
    path.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
    for pose in result['poses']+result['family45']['poses']:
        print(pose['name'], json.dumps(pose['stats']))


if __name__ == '__main__':
    main()
