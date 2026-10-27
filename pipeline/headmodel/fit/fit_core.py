"""CPU contour ICP: numpy visibility, OpenCV distance fields, torch gradients.

All lengths in the objective are pixels at 720. Visibility/correspondences are
piecewise constant for five Adam updates, then rebuilt from the current mesh.
No vertex residual, image landmark, or differentiable rasterizer is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FIT = Path(__file__).resolve().parent
ROOT = FIT.parents[2]
WORKSPACE = ROOT.parent.parent
SIZE = 720
FRAMES = (3, 42, 264)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_model():
    manifest = read_json(ROOT/'pipeline/headmodel/model/ict_manifest.json')
    # Consume exactly the neutral/mode bundle made by ict_loader.py.
    path = WORKSPACE / manifest['artifact_directory'] / 'ict_head.npz'
    with np.load(path) as bundle:
        v = bundle['neutral_vertices'].copy()
        tri = bundle['triangles'].astype(np.int64)
        modes = bundle['identity_modes'].copy()
    return v, tri, modes


def define_regions(v, tri):
    x, y, z = v.T
    # Oblique submandibular plane: retain the chin but remove the nape/neck.
    head = y + .4*z >= -5.
    # Lateral pinnae behind the cheek; thresholds use neutral ICT units only.
    ear = (np.abs(x) > 7.6) & (y > -4.) & (y < 4.8) & (z < 3.5)
    face = (np.arange(len(v)) <= 9408) & head & ~ear
    masks = {'head': head, 'neck': ~head, 'face': face,
             'face_lower': face & (y <= 1.5),
             'ear-left': ear & (x < 0), 'ear-right': ear & (x > 0)}
    regions = {key: np.flatnonzero(val) for key, val in masks.items()}
    # No triangle containing an excluded neck vertex participates in any loss.
    faces = {'head': np.flatnonzero(head[tri].all(1)),
             'face': np.flatnonzero((face & (y <= 1.5))[tri].all(1))}
    # Assign a triangle by the majority of its vertices, then clip to head.
    for key in ('ear-left', 'ear-right'):
        faces[key] = np.flatnonzero((masks[key][tri].sum(1) >= 2) & head[tri].all(1))
    return regions, faces


def normalize_pose(pose, affine):
    a = np.asarray(affine); p = np.array(pose, dtype=float).copy()
    p[2] += np.degrees(np.arctan2(a[1, 0], a[0, 0]))
    p[3] += np.log(np.linalg.norm(a[:, 0]))
    p[4:6] = a[:, :2] @ p[4:6] + a[:, 2]
    return p


def project(vertices, pose):
    """ICT (right, up, forward) -> source (right, down, forward), Rx, Ry, roll.

    This is exactly seedance_temporal.project_frames' rigid rotation after
    flipping ICT Y. Returned third coordinate increases toward the camera.
    """
    yaw, pitch, roll = pose[:3] * (np.pi/180)
    cy, sy, cp, sp = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin()
    cr, sr = roll.cos(), roll.sin()
    x, y, z = vertices[:, 0], -vertices[:, 1], vertices[:, 2]
    yp, zp = cp*y-sp*z, sp*y+cp*z
    xp, depth = cy*x+sy*zp, -sy*x+cy*zp
    xy = torch.stack([cr*xp-sr*yp, sr*xp+cr*yp], -1)
    return torch.cat([xy*pose[3].exp()+pose[4:6], depth[:, None]], 1)


def clean_mask(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError('Empty required mask')
    mask = (labels == 1+np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    padded = np.pad(mask, 1)
    flooded = padded.copy(); cv2.floodFill(flooded, None, (0, 0), 1)
    return (padded | (1-flooded))[1:-1, 1:-1]


def boundary(mask, count=None):
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('Empty silhouette')
    pts = max(contours, key=cv2.contourArea)[:, 0].astype(np.float32)
    # Same topmost origin and winding across erosion levels, for t interpolation.
    start = np.lexsort((pts[:, 0], pts[:, 1]))[0]; pts = np.roll(pts, -start, axis=0)
    if count is None:
        return pts
    closed = np.r_[pts, pts[:1]]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    samples = np.arange(count)*arc[-1]/count
    return np.stack([np.interp(samples, arc, closed[:, dim]) for dim in (0, 1)], 1).astype(np.float32)


def lower_boundary(mask):
    """Downward envelope below the widest row: chin/lower cheek, no hairline."""
    mask = np.asarray(mask, bool)
    row = int(np.argmax(mask.sum(1)))
    xs = np.flatnonzero(mask.any(0))
    ys = mask.shape[0]-1-np.argmax(mask[::-1, xs], axis=0)
    return np.column_stack([xs[ys >= row], ys[ys >= row]]).astype(np.float32)


def eroded_target(head, thickness):
    inside = cv2.distanceTransform(np.asarray(head, np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (inside > thickness).astype(np.uint8)


def contour_dt(points, size=SIZE):
    image = np.ones((size, size), np.uint8)
    xy = np.rint(points).astype(int)
    image[xy[:, 1], xy[:, 0]] = 0
    return torch.from_numpy(cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))


def load_targets(frames=FRAMES, pose_affine=None):
    registration = read_json(ROOT/'source/viewer/ponytail-seedance-registration.json')
    keys = {int(k['sourceFrame']): k for k in registration['keys']}
    targets = []
    for frame in frames:
        directory = ROOT / f'pipeline/headmodel/data/frame_{frame:04d}'
        manifest = read_json(directory/'manifest.json')
        def mask(name):
            value = cv2.imread(str(directory / (name+'.png')), cv2.IMREAD_GRAYSCALE)
            if value is None or value.shape != (SIZE, SIZE):
                raise ValueError(f'Missing 720 mask: {directory / name}')
            return (value > 0).astype(np.uint8)
        head = mask('head')
        skin = clean_mask(mask('raw/face skin') & (1-mask('raw/neck')))
        lower = lower_boundary(skin)
        eroded = [eroded_target(head, float(t)) for t in range(3, 15)]
        ears = {key: mask(key) for key in ('ear-left', 'ear-right')
                if manifest['masks'][key]['visible']}
        target = {'frame': frame, 'head': head, 'skin': skin, 'lower': lower,
                  'head_dt': torch.stack([contour_dt(boundary(m)) for m in eroded]),
                  'head_points': torch.tensor(np.stack([boundary(m, 512) for m in eroded])),
                  'skin_points': torch.tensor(lower[::2]), 'skin_dt': contour_dt(lower),
                  'ears': ears, 'ear_points': {k: torch.tensor(boundary(m, 128)) for k,m in ears.items()},
                  'ear_dt': {k: contour_dt(boundary(m)) for k,m in ears.items()},
                  'raw_pose': registration['video'][frame-1]['pose'],
                  'affine': (keys[frame]['normalization'] if pose_affine is None
                             else pose_affine(manifest['to720'])), 'key': keys.get(frame)}
        target['normalized_pose'] = normalize_pose(target['raw_pose'], target['affine'])
        targets.append(target)
    return targets


def rasterize(q, triangles, size=SIZE, cull_backfaces=False):
    """Vectorized numpy barycentric z-buffer; no differentiable rasterizer."""
    depth = np.full(size*size, -np.inf, np.float32)
    owner = np.full(size*size, -1, np.int32)
    chunks = []
    for offset in range(0, len(triangles), 512):
        ids = np.arange(offset, min(offset+512, len(triangles)))
        tri = q[triangles[ids]]
        lo = np.maximum(np.floor(tri[:, :, :2].min(1)).astype(int), 0)
        hi = np.minimum(np.ceil(tri[:, :, :2].max(1)).astype(int), size-1)
        widths = np.maximum(hi[:, 0]-lo[:, 0]+1, 0)
        heights = np.maximum(hi[:, 1]-lo[:, 1]+1, 0)
        counts = widths*heights
        ix = np.repeat(np.arange(len(ids)), counts)
        if not len(ix):
            continue
        steps = np.arange(counts.sum())-np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
        px = lo[ix, 0]+steps%widths[ix]+.5
        py = lo[ix, 1]+steps//widths[ix]+.5
        a,b,c = tri[ix, 0],tri[ix, 1],tri[ix, 2]
        det = (b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        # Projection flips world Y. Outward, camera-facing +Z normals therefore
        # have clockwise (negative signed-area) winding in screen coordinates.
        valid = det < -1e-8 if cull_backfaces else np.abs(det)>1e-8
        det = np.where(valid, det, 1.)
        u = ((b[:,1]-c[:,1])*(px-c[:,0])+(c[:,0]-b[:,0])*(py-c[:,1]))/det
        v = ((c[:,1]-a[:,1])*(px-c[:,0])+(a[:,0]-c[:,0])*(py-c[:,1]))/det
        w = 1-u-v
        keep = valid & (u>=-1e-5) & (v>=-1e-5) & (w>=-1e-5)
        pixel = py[keep].astype(int)*size+px[keep].astype(int)
        z = (u*a[:,2]+v*b[:,2]+w*c[:,2])[keep].astype(np.float32)
        face = ids[ix[keep]].astype(np.int32)
        np.maximum.at(depth, pixel, z)
        chunks.append((pixel,z,face))
    for pixel,z,face in chunks:
        keep = z>=depth[pixel]-1e-6
        np.maximum.at(owner, pixel[keep], face[keep])
    return depth.reshape(size,size), owner.reshape(size,size)


def attach_contour(points, q, tri, owner):
    """Freeze visible triangle/edge ids, retain differentiable edge positions."""
    points = np.asarray(points)
    pixels = np.rint(points).astype(int)
    ids = owner[pixels[:,1], pixels[:,0]]
    valid = ids>=0; points = points[valid]+.5; ids = ids[valid]
    vertices = tri[ids]
    xy = q[vertices, :2]
    a = xy; b = np.roll(xy, -1, axis=1)
    d = b-a
    t = np.clip(((points[:,None]-a)*d).sum(2)/np.maximum((d*d).sum(2),1e-10), 0, 1)
    nearest = a+t[:,:,None]*d
    edge = ((nearest-points[:,None])**2).sum(2).argmin(1)
    weights = np.zeros((len(points),3), np.float32)
    weights[np.arange(len(points)), edge] = 1-t[np.arange(len(points)),edge]
    weights[np.arange(len(points)), (edge+1)%3] = t[np.arange(len(points)),edge]
    if not len(points):
        raise ValueError('No visible contour samples')
    return torch.tensor(vertices), torch.tensor(weights)


def silhouettes(q, tri, faces):
    _, owner = rasterize(q, tri[faces['head']])
    global_owner = np.full_like(owner,-1)
    valid = owner>=0; global_owner[valid] = faces['head'][owner[valid]]
    result = {'head': attach_contour(boundary(valid)[::3],q,tri,global_owner)}
    _, fo = rasterize(q, tri[faces['face']])
    fg = np.full_like(fo,-1); fv = fo>=0; fg[fv] = faces['face'][fo[fv]]
    result['skin'] = attach_contour(lower_boundary(fv)[::2],q,tri,fg)
    for key in ('ear-left','ear-right'):
        # Isolated ear projection supplies the attraction even if initially
        # occluded; visible raster ownership remains the reported ear IoU.
        _, eo = rasterize(q, tri[faces[key]])
        eg = np.full_like(eo,-1); ev = eo>=0; eg[ev] = faces[key][eo[ev]]
        result[key] = attach_contour(boundary(ev)[::2],q,tri,eg)
    return result


def attached(q, attachment):
    indices, weights = attachment
    return (q[indices,:2]*weights[:,:,None]).sum(1)


def sample_dt(dt, points):
    grid = (points/(SIZE-1)*2-1)[None,None]
    sampled = F.grid_sample(dt[None,None],grid,align_corners=True,padding_mode='border').flatten()
    outside = (points-points.clamp(0,SIZE-1)).norm(dim=1)
    return sampled+outside


def symmetric_distance(a, b):
    distance = torch.cdist(a[None], b[None])[0]
    return .5*(distance.min(1).values.mean()+distance.min(0).values.mean())


def loss_frame(q, contours, target, thickness):
    index = int(torch.floor(thickness.detach()).clamp(3,13))-3
    fraction = thickness-(index+3)
    hp = target['head_points'][index]*(1-fraction)+target['head_points'][index+1]*fraction
    hdt = target['head_dt'][index]*(1-fraction)+target['head_dt'][index+1]*fraction
    def term(points, wanted, dt):
        # Target -> projected silhouette; projected -> exact target contour DT.
        return .5*(sample_dt(dt,points).mean()+torch.cdist(wanted[None],points[None])[0].min(1).values.mean())
    values = {'head': term(attached(q,contours['head']), hp, hdt),
              'skin': term(attached(q,contours['skin']),target['skin_points'],target['skin_dt'])}
    ears = [term(attached(q,contours[k]),target['ear_points'][k],target['ear_dt'][k]) for k in target['ears']]
    values['ear'] = torch.stack(ears).mean() if ears else q.sum()*0
    return values['head']+.5*values['skin']+.5*values['ear'], values
