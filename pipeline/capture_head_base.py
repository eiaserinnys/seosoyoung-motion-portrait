"""Capture the head, face, neck and ears against the Seedance references."""
import base64
import functools
import http.server
import io
import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / 'pipeline/headmodel/fit'
sys.path.insert(0, str(FIT))
from fit2d_sheets import image_for
from fit2f_core import load_targets_f

KEYS = [
    ('up-left', 192, -45, 30), ('up', 206, 0, 30), ('up-right', 219, 45, 30),
    ('left', 42, -45, 0), ('front', 3, 0, 0), ('right', 240, 45, 0),
    ('down-left', 303, -45, -30), ('down', 270, 0, -30), ('down-right', 264, 45, -30),
]
PROFILES = [(-90, 0), (-67.5, 0), (67.5, 0), (90, 0)]
INTERMEDIATE = [(yaw, pitch) for pitch in (-15, 15) for yaw in (-22.5, 22.5)]
STRIP = [(yaw, pitch) for pitch in (-20, 0, 20) for yaw in range(-90, 91, 15)]
BG = (30, 36, 48)
CANVAS_WIDTH = 1500
X_OFFSET = (CANVAS_WIDTH - 1024) // 2
DEST = ROOT / 'docs/captures'
WIP2 = DEST / 'wip2-head-base'
NECK_EDGE_BASELINE = 'cc3593a87b7b62432a88b5134f019d52393914c8'
FIX3 = [(-15, 0), (15, 20), (-30, 20), (45, 20), (15, -20), (0, -20), (-90, 0), (90, 0)]
FIX3_BOXES = {
    (-15, 0): (250, 300, 750, 650), (15, 20): (250, 300, 750, 650),
    (-30, 20): (275, 400, 725, 730), (45, 20): (275, 400, 725, 730),
    (15, -20): (250, 280, 500, 530), (0, -20): (250, 280, 500, 530),
    (-90, 0): (250, 370, 750, 750), (90, 0): (250, 370, 750, 750),
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def decode(data):
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGBA')
    return image.crop((0, 32, CANVAS_WIDTH, 1056))


def alpha(image):
    return np.asarray(image)[:, :, 3] > 127


def identified(image):
    pixels = np.asarray(image)
    return (pixels[:, :, 0] > 127) & (pixels[:, :, 3] > 127)


def contour(mask):
    lines, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not lines:
        raise RuntimeError('Empty rendered silhouette')
    samples = []
    for line in lines:
        points = line[:, 0, :].astype(float)
        following = np.roll(points, -1, axis=0)
        length = np.linalg.norm(following - points, axis=1)
        cumulative = np.r_[0, np.cumsum(length)]
        if cumulative[-1] == 0:
            samples.append(points[:1])
            continue
        positions = np.arange(0, cumulative[-1], 1.)
        indices = np.minimum(np.searchsorted(cumulative, positions, side='right') - 1, len(points) - 1)
        fraction = (positions - cumulative[indices]) / length[indices]
        samples.append(points[indices] + fraction[:, None] * (following[indices] - points[indices]))
    return np.vstack(samples)


def residual(target, rendered):
    values = np.r_[cKDTree(rendered).query(target)[0], cKDTree(target).query(rendered)[0]]
    return {'median_px': float(np.median(values)), 'max_px': float(values.max()),
            'p95_px': float(np.percentile(values, 95)), 'samples': len(values)}


def n3_metrics(mask, skin, hair):
    rendered = contour(mask)
    nearest = cKDTree(np.vstack([skin, hair])).query(rendered)[1]
    skin_rendered, hair_rendered = rendered[nearest < len(skin)], rendered[nearest >= len(skin)]
    if not len(skin_rendered) or not len(hair_rendered):
        raise RuntimeError('Missing a rendered silhouette category')
    result = {'skin': residual(skin, skin_rendered), 'hair': residual(hair, hair_rendered)}
    result['passed'] = (result['skin']['median_px'] <= 2 and result['skin']['p95_px'] <= 5
                        and result['hair']['median_px'] <= 2 and result['hair']['p95_px'] <= 6)
    return result


def jacobians(rest, current, triangles):
    chosen = triangles[np.all((rest[triangles, 1] >= 218) & (rest[triangles, 1] <= 347), axis=1)]
    source = np.stack([rest[chosen[:, 1]]-rest[chosen[:, 0]], rest[chosen[:, 2]]-rest[chosen[:, 0]]], axis=-1)
    deformed = np.stack([current[chosen[:, 1]]-current[chosen[:, 0]], current[chosen[:, 2]]-current[chosen[:, 0]]], axis=-1)
    matrices = deformed @ np.linalg.inv(source)
    determinant = np.linalg.det(matrices)
    singular = np.linalg.svd(matrices, compute_uv=False)
    return {'min_det_j': float(determinant.min()), 'max_anisotropy': float((singular[:, 0]/singular[:, 1]).max()),
            'triangles': len(chosen)}


def closed_holes(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=8)
    outside = set(np.unique(np.r_[labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    return int(sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, count) if i not in outside))


def closed_hole_locations(mask):
    count, labels, _, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=8)
    outside = set(np.unique(np.r_[labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    holes = (labels > 0) & ~np.isin(labels, list(outside))
    ys, xs = np.where(holes)
    return [[int(x-X_OFFSET), int(y)] for x, y in zip(xs, ys)]


def hole_mesh_samples(point, mesh, positions, atlas):
    """Record the triangles and atlas alpha underneath a screen-space hole."""
    p = np.asarray(point, dtype=float) + .5
    positions = np.asarray(positions, dtype=float)
    triangles = np.asarray(mesh['triangles'], dtype=int)
    tri = positions[triangles]
    a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    determinant = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    valid = np.abs(determinant) > 1e-8
    d = p-tri[:, 0]
    u = np.divide(d[:, 0]*b[:, 1]-d[:, 1]*b[:, 0], determinant, out=np.zeros_like(determinant), where=valid)
    v = np.divide(a[:, 0]*d[:, 1]-a[:, 1]*d[:, 0], determinant, out=np.zeros_like(determinant), where=valid)
    indices = np.flatnonzero(valid & (u >= 0) & (v >= 0) & (u+v <= 1))
    result = []
    for i in indices[:4]:
        weights = np.array([1-u[i]-v[i], u[i], v[i]])
        texel = weights @ np.asarray(mesh['uvs'])[triangles[i]] * atlas.shape[0] - .5
        x, y = np.rint(texel).astype(int)
        if 0 <= x < atlas.shape[1] and 0 <= y < atlas.shape[0]:
            ax, ay, _, _ = mesh['atlas']
            bx, by, _, _ = mesh['bbox']
            result.append({'triangle': int(i), 'canvas_texel': [int(x-ax+bx), int(y-ay+by)],
                           'atlas_texel': [int(x), int(y)], 'alpha': int(atlas[y, x, 3])})
    return result


def neck_gap(head, neck, combined, bounds):
    left = max(0, math.floor(bounds[0] + X_OFFSET))
    right = min(head.shape[1], math.ceil(bounds[1] + X_OFFSET) + 1)
    gap, columns = 0, 0
    pixels = np.zeros_like(head, dtype=bool)
    for x in range(left, right):
        h, n = np.flatnonzero(head[:, x]), np.flatnonzero(neck[:, x])
        if len(h) and len(n):
            columns += 1
            if n[0] > h[-1] + 1:
                pixels[h[-1]+1:n[0], x] = ~combined[h[-1]+1:n[0], x]
                gap += int(np.count_nonzero(pixels[h[-1]+1:n[0], x]))
    return gap, columns, pixels


def neck_opaque_band(image):
    pixels = np.asarray(image)
    mask = (pixels[:, :, 0] == 255) & (pixels[:, :, 3] == 255)
    rows = np.flatnonzero(mask.any(axis=1))
    for value in rows:
        row = int(value)
        xs = np.flatnonzero(mask[row])
        if len(xs) >= 3 and xs[-1] - xs[0] >= 2:
            return {'row': row, 'left': int(xs[0])+1-X_OFFSET, 'right': int(xs[-1])-1-X_OFFSET}
    return None


def ear_iou(rendered, frame, poses, translation):
    path = ROOT / f'pipeline/headmodel/data/frame_{frame:04d}/raw/ear.png'
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(path)
    scale, shift = transform(frame, poses, translation)
    matrix = np.array([[scale, 0, shift[0]], [0, scale, shift[1]]])
    target = cv2.warpAffine(target, matrix, (CANVAS_WIDTH, 1024), flags=cv2.INTER_NEAREST) > 127
    union = np.count_nonzero(rendered | target)
    return {'iou': float(np.count_nonzero(rendered & target) / union) if union else None,
            'rendered_px': int(rendered.sum()), 'target_px': int(target.sum())}


def transform(frame, poses, translation):
    front, pose = poses[3], poses[frame]
    scale = math.exp(front[3] - pose[3])
    shift = np.asarray(translation) - scale * np.asarray(pose[4:6]) + [X_OFFSET, 0]
    return scale, shift


def reference(frame, poses, translation):
    scale, shift = transform(frame, poses, translation)
    matrix = np.array([[scale, 0, shift[0]], [0, scale, shift[1]]])
    rgb = cv2.cvtColor(image_for(frame), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.warpAffine(rgb, matrix, (CANVAS_WIDTH, 1024), borderValue=BG))


def composite(image):
    background = Image.new('RGBA', image.size, BG + (255,))
    background.alpha_composite(image)
    return background.convert('RGB')


def still_neck_difference(rendered, still, mask):
    reference = np.asarray(composite(still), dtype=np.uint8)
    actual = np.asarray(composite(rendered), dtype=np.uint8)
    delta = np.abs(reference.astype(np.int16)-actual.astype(np.int16))
    per_pixel = delta.max(axis=2)
    mask = np.asarray(mask, dtype=bool)
    values = per_pixel[mask]
    changed = mask & (per_pixel > 0)
    yy, xx = np.where(changed)
    result = {'pixels': int(mask.sum()), 'changed_pixels': int(changed.sum()),
              'max_channel_difference': int(values.max()),
              'mean_max_channel_difference': float(values.mean()),
              'over_2_pixels': int(np.count_nonzero(mask & (per_pixel > 2))),
              'bbox': [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())] if len(xx) else None}
    return result, reference, actual, per_pixel


def make_crop(renders):
    bounds = np.asarray([item['bounds'] for item in renders.values()])
    left, top = bounds[:, :2].min(axis=0) + [X_OFFSET, 0]
    right, bottom = bounds[:, 2:].max(axis=0) + [X_OFFSET, 0]
    if left < 0 or right >= CANVAS_WIDTH or top < 0 or bottom >= 1024:
        raise RuntimeError(f'Head or face extends outside capture: {[left, top, right, bottom]}')
    size = math.ceil(max(right - left + 80, bottom - top + 160))
    center = (left + right) / 2
    return (math.floor(center - size / 2), math.floor(top - 40),
            math.floor(center - size / 2) + size, math.floor(top - 40) + size)


def tile(image, crop, size, label):
    if image.mode == 'RGBA':
        image = composite(image)
    result = image.crop(crop).resize((size, size), Image.Resampling.LANCZOS)
    ImageDraw.Draw(result).text((7, 7), label, fill='white', stroke_width=2, stroke_fill=BG)
    return result


def draw_targets(image, hair, skin):
    output = np.asarray(composite(image)).copy()
    for points, color in ((hair, (50, 235, 80)), (skin, (50, 120, 255))):
        for x, y in np.rint(points).astype(int):
            if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
                cv2.circle(output, (x, y), 2, color, -1)
    return Image.fromarray(output)


def save_sheets(renders, rows, poses, translation, crop):
    DEST.mkdir(parents=True, exist_ok=True)
    size = 360
    sheet = Image.new('RGB', (6 * size, 3 * size + 28), BG)
    ImageDraw.Draw(sheet).text((8, 7), 'Green: Seedance head target   Blue: Seedance skin target', fill='white')
    for i, (name, frame, yaw, pitch) in enumerate(KEYS):
        x, y = (i % 3) * 2 * size, (i // 3) * size + 28
        item, row = renders[(yaw, pitch)], rows[i]
        overlay = draw_targets(item['portrait'], row['hair_points'], row['skin_points'])
        sheet.paste(tile(overlay, crop, size, name + ' / model'), (x, y))
        sheet.paste(tile(reference(frame, poses, translation), crop, size, name + ' / Seedance'), (x + size, y))
    sheet.save(DEST / 'head_base_seedance_9.png')

    size = 280
    strip = Image.new('RGB', (13 * size, 3 * size + 28), BG)
    ImageDraw.Draw(strip).text((8, 7), 'Effective yaw -90 to +90; pitch -20 / 0 / +20', fill='white')
    for i, (yaw, pitch) in enumerate(STRIP):
        strip.paste(tile(renders[(yaw, pitch)]['portrait'], crop, size, f'{yaw:+g} / {pitch:+g}'),
                    ((i % 13) * size, (i // 13) * size + 28))
    strip.save(DEST / 'head_base_turn_strip.png')

    size = 500
    profiles = Image.new('RGB', (4 * size, size), BG)
    for i, (yaw, pitch) in enumerate(PROFILES):
        profiles.paste(tile(renders[(yaw, pitch)]['portrait'], crop, size, f'yaw {yaw:+g}'), (i * size, 0))
    profiles.save(DEST / 'head_base_profiles.png')

def previous_fix3_panel(sheet, pose):
    y = 0
    for key in FIX3:
        x0, y0, x1, y1 = FIX3_BOXES[key]
        width, height = (x1-x0)*2, (y1-y0)*2
        if key == pose:
            return sheet.crop((width, y+24, width*2, y+24+height))
        y += height+24
    raise KeyError(pose)


def save_edges(renders, rows, baseline):
    before = Image.open(baseline / 'head_base_edges_2x.png').convert('RGB')
    old_crop = json.loads((baseline / 'head_base_metrics.json').read_text())['common_crop_pixels']
    focus = (325, 400, 675, 630)
    size = ((focus[2]-focus[0])*2, (focus[3]-focus[1])*2)
    sheet = Image.new('RGB', (size[0]*2, (size[1]+24)*6), BG)
    for row, name in enumerate(('up', 'up-left', 'up-right', '-30/+20', '+45/+20', '+15/+20')):
        if row < 3:
            i = next(i for i, key in enumerate(KEYS) if key[0] == name)
            _, _, yaw, pitch = KEYS[i]
            current_tile = tile(draw_targets(renders[(yaw,pitch)]['portrait'],rows[i]['hair_points'],rows[i]['skin_points']),old_crop,360,'')
            box = tuple(round((v+X_OFFSET-old_crop[0])*360/(old_crop[2]-old_crop[0])) if j in (0,2)
                        else round((v-old_crop[1])*360/(old_crop[3]-old_crop[1])) for j,v in enumerate(focus))
            new_panel = current_tile.crop(box).resize(size, Image.Resampling.NEAREST)
        else:
            yaw,pitch = [(-30,20),(45,20),(15,20)][row-3]
            box=(focus[0]+X_OFFSET,focus[1],focus[2]+X_OFFSET,focus[3])
            new_panel = composite(renders[(yaw,pitch)]['portrait']).crop(box).resize(size,Image.Resampling.NEAREST)
        y = row*(size[1]+24)
        old_panel = before.crop((size[0],y+24,size[0]*2,y+24+size[1]))
        sheet.paste(old_panel,(0,y+24));sheet.paste(new_panel,(size[0],y+24))
        label=ImageDraw.Draw(sheet)
        label.text((6,y+4),name+' / '+NECK_EDGE_BASELINE[:7],fill='white')
        label.text((size[0]+6,y+4),name+' / current',fill='white')
    sheet.save(DEST / 'head_base_edges_2x.png')


def save_neck_edges(renders):
    rest = renders[(0, 0)]
    box = (330, 600, 680, 790)
    render_box = (box[0]+X_OFFSET, box[1], box[2]+X_OFFSET, box[3])
    width, height = (box[2]-box[0])*4, (box[3]-box[1])*4
    reference = composite(Image.open(ROOT / 'source/character/still.png').convert('RGBA')).crop(box)
    sheet = Image.new('RGB', (width*3, (height+26)*2), BG)
    for row, outfit in enumerate(('sweater', 'shirt')):
        images = (reference, composite(rest[outfit+'_before']).crop(render_box),
                  composite(rest[outfit+'_after']).crop(render_box))
        for col, (image, label) in enumerate(zip(images, ('original / sweater', outfit+' / '+NECK_EDGE_BASELINE[:7], outfit+' / current'))):
            sheet.paste(image.resize((width,height), Image.Resampling.NEAREST), (col*width, row*(height+26)+26))
            ImageDraw.Draw(sheet).text((col*width+6,row*(height+26)+5),label,fill='white')
    sheet.save(DEST / 'head_base_neck_edge.png')

    original = np.asarray(Image.open(ROOT / 'source/character/still.png').convert('RGBA'))
    x0, y0, x1, y1 = 339, 610, 662, 778
    reference_background = original[y0:y1,x0:x1,3] == 0
    result = {'bbox_canvas': [x0,y0,x1,y1], 'views': {}, 'portrait_changes': {}}
    for view, name in [('after', 'neck_background_test'), ('before', 'neck_background_before_view')]:
        rendered = np.asarray(rest[name])[y0:y1,x0+X_OFFSET:x1+X_OFFSET]
        opaque = (rendered[:,:,0] == 255) & (rendered[:,:,3] == 255)
        ys, xs = np.where(reference_background & opaque)
        bbox = [int(xs.min()+x0),int(ys.min()+y0),int(xs.max()+x0),int(ys.max()+y0)] if len(xs) else None
        result['views'][view] = {'opaque_on_original_background_px': int(len(xs)), 'affected_bbox': bbox}
    for outfit in ('sweater', 'shirt'):
        before, after = (np.asarray(rest[outfit+'_'+name]) for name in ('before', 'after'))
        result['portrait_changes'][outfit] = int(np.count_nonzero(np.any(before != after,axis=2)))
    return result


def save_fix3(renders, baseline):
    previous = Image.open(baseline / 'head_base_fix3.png').convert('RGB')
    rows = []
    for yaw, pitch in FIX3:
        x0, y0, x1, y1 = FIX3_BOXES[(yaw, pitch)]
        box = (x0 + X_OFFSET, y0, x1 + X_OFFSET, y1)
        before = previous_fix3_panel(previous,(yaw,pitch))
        after = composite(renders[(yaw, pitch)]['portrait'])
        size = ((x1-x0)*2, (y1-y0)*2)
        pair = Image.new('RGB', (size[0]*2, size[1]+24), BG)
        pair.paste(before, (0, 24))
        pair.paste(after.crop(box).resize(size, Image.Resampling.NEAREST), (size[0], 24))
        label = ImageDraw.Draw(pair)
        label.text((6, 5), f'{yaw:+g}/{pitch:+g}  {NECK_EDGE_BASELINE[:7]}', fill='white')
        label.text((size[0]+6, 5), 'current', fill='white')
        rows.append(pair)
    sheet = Image.new('RGB', (max(row.width for row in rows), sum(row.height for row in rows)), BG)
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(DEST / 'head_base_fix3.png')


def save_direction_color(baseline):
    before = Image.open(baseline / 'head_base_turn_strip.png').convert('RGB')
    after = Image.open(DEST / 'head_base_turn_strip.png').convert('RGB')
    poses = [(-60,0),(60,0),(-60,20),(60,20),(-45,-20),(45,-20)]
    size = 560
    sheet = Image.new('RGB', (size*2,(size+22)*len(poses)), BG)
    for row, (yaw,pitch) in enumerate(poses):
        x, y = (yaw+90)//15*280, {-20:0,0:1,20:2}[pitch]*280+28
        for col, image in enumerate((before,after)):
            panel = image.crop((x,y,x+280,y+280)).resize((size,size),Image.Resampling.NEAREST)
            sheet.paste(panel,(col*size,row*(size+22)+22))
            ImageDraw.Draw(sheet).text((col*size+5,row*(size+22)+5),
                                       f'{yaw:+d}/{pitch:+d} '+(NECK_EDGE_BASELINE[:7] if col==0 else 'current'),fill='white')
    sheet.save(DEST / 'head_base_direction_color.png')


def save_review_details(baseline):
    neck = Image.open(DEST / 'head_base_neck_edge.png').convert('RGB')
    shoulders = Image.new('RGB', (720, 728), BG)
    for row, (x0,y0,x1,y1) in enumerate(((160,220,520,560),(900,220,1260,560))):
        for col in range(2):
            shoulders.paste(neck.crop((x0+(col+1)*1400,y0+26,x1+(col+1)*1400,y1+26)),(col*360,row*364+24))
            ImageDraw.Draw(shoulders).text((col*360+5,row*364+5),
                ('left' if row==0 else 'right')+(' / '+NECK_EDGE_BASELINE[:7] if col==0 else ' / current'),fill='white')
    shoulders.save(DEST / 'head_base_shoulder_detail.png')

    turns = [Image.open(baseline / 'head_base_turn_strip.png').convert('RGB'),
             Image.open(DEST / 'head_base_turn_strip.png').convert('RGB')]
    poses = [(-60, 0), (60, 0), (-30, 0), (30, 0), (0, 20),
             (-60, -20), (-60, 20), (60, -20), (60, 20),
             (-30, -20), (-30, 20), (30, -20), (30, 20), (0, -20)]
    holes = Image.new('RGB', (1120, 574*len(poses)), BG)
    for row, (yaw, pitch) in enumerate(poses):
        x0, y0 = (yaw+90)//15*280, {-20:0,0:1,20:2}[pitch]*280+28
        for col, turn in enumerate(turns):
            panel=turn.crop((x0+80,y0+140,x0+220,y0+278)).resize((560,550),Image.Resampling.NEAREST)
            holes.paste(panel,(col*560,row*574+24))
            ImageDraw.Draw(holes).text((col*560+5,row*574+5),f'{yaw:+d}/{pitch:+d} '+
                                      ('cc3593a' if col==0 else 'current'),fill='white')
    holes.save(DEST / 'head_base_neck_holes.png')

    edges=Image.open(DEST / 'head_base_edges_2x.png').convert('RGB')
    jaw=Image.new('RGB',(1400,1452),BG)
    for row, source_row in enumerate((0,3,4)):
        for col in range(2):
            jaw.paste(edges.crop((col*700,source_row*484+24,(col+1)*700,source_row*484+484)),(col*700,row*484+24))
            ImageDraw.Draw(jaw).text((col*700+5,row*484+4),
                ('up','-30/+20','+45/+20')[row]+(' / '+NECK_EDGE_BASELINE[:7] if col==0 else ' / current'),fill='white')
    jaw.save(DEST / 'head_base_faint_jaw.png')


def save_still_comparison(assembled, baseline=None):
    rendered = assembled.crop((X_OFFSET, 0, X_OFFSET+1024, 1024))
    still = Image.open(ROOT / 'source/character/still.png').convert('RGBA').crop((0, 0, 1024, 1024))
    neck = np.asarray(Image.open(ROOT / 'sam3_masks/neck_mask.png').convert('L')) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = (cv2.dilate(neck.astype(np.uint8), kernel) > 0)[:1024]
    result, reference, actual, delta = still_neck_difference(rendered, still, mask)
    if baseline is not None:
        previous = np.asarray(composite(Image.open(baseline / 'head_base_assembled.png').convert('RGBA')))
        face = np.asarray(Image.open(ROOT / 'sam3_masks/face-skin_mask.png').convert('L')) > 0
        face = face[:1024].astype(np.uint8)
        expanded = cv2.dilate(face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
        eroded = cv2.erode(face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
        yy, xx = np.indices(mask.shape)
        band = mask & expanded & ~eroded & (xx >= 425) & (xx <= 575) & (yy >= 575) & (yy <= 630)
        previous_delta = np.abs(reference.astype(np.int16)-previous.astype(np.int16)).max(axis=2)
        result['jaw_band'] = {'pixels': int(band.sum()),
                              'previous_changed_pixels': int(np.count_nonzero(band & (previous_delta > 0))),
                              'changed_pixels': int(np.count_nonzero(band & (delta > 0))),
                              'previous_total_channel_error': int(previous_delta[band].sum()),
                              'total_channel_error': int(delta[band].sum())}
    rendered.save(DEST / 'head_base_assembled.png')
    yy, xx = np.where(mask)
    box = (max(0,int(xx.min())-20), max(0,int(yy.min())-20),
           min(1024,int(xx.max())+21), min(1024,int(yy.max())+21))
    heat = np.full_like(reference, BG)
    heat[mask] = np.stack([np.minimum(delta[mask]*4,255), np.zeros(mask.sum()),
                           np.zeros(mask.sum())], axis=1).astype(np.uint8)
    size = ((box[2]-box[0])*3, (box[3]-box[1])*3)
    sheet = Image.new('RGB', (size[0]*3, size[1]+24), BG)
    for column,(array,label) in enumerate(((reference,'original'),(actual,'assembled'),(heat,'RGB difference x4'))):
        sheet.paste(Image.fromarray(array).crop(box).resize(size,Image.Resampling.NEAREST),(column*size[0],24))
        ImageDraw.Draw(sheet).text((column*size[0]+5,5),label,fill='white')
    sheet.save(DEST / 'head_base_still_compare.png')
    return result


def save_temple_diagnosis(baseline):
    crop = (250 + X_OFFSET, 280, 500 + X_OFFSET, 530)
    sheet = Image.new('RGB', (1500, 1500), BG)
    for row, yaw in enumerate((0, 15, 30)):
        label = f'{yaw:+g}_-20'
        head = Image.open(baseline / f'{label}_head.png').convert('RGBA')
        face = Image.open(baseline / f'{label}_face.png').convert('RGBA')
        outside = alpha(face) & ~alpha(head)
        highlighted = np.asarray(composite(face)).copy()
        highlighted[outside] = [255, 40, 40]
        for column, image in enumerate((composite(head), composite(face), Image.fromarray(highlighted))):
            panel = image.crop(crop).resize((500, 500), Image.Resampling.NEAREST)
            ImageDraw.Draw(panel).text((6, 6), f'{yaw:+g}/-20 ' + ('head' if column == 0 else 'face + fill' if column == 1 else 'outside head'),
                                       fill='white', stroke_width=2, stroke_fill=BG)
            sheet.paste(panel, (column*500, row*500))
    sheet.save(DEST / 'head_base_temple_diagnosis.png')


def save_review2(renders, extra_ear, crop):
    baseline = DEST / 'review2-baseline'
    mask = np.asarray(Image.open(ROOT / 'sam3_masks/face-skin_mask.png').convert('L'))[:1024] > 127
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(mask, dtype='uint8')
    cv2.drawContours(solid, contours, -1, 1, cv2.FILLED)
    distance = cv2.distanceTransform(1-solid, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    old_head = Image.open(baseline / 'head_base_rest_head.png').convert('RGBA')
    new_head = renders[(0, 0)]['head_only']
    counts = {}
    box = (X_OFFSET+320, 490, X_OFFSET+690, 640)
    sheet = Image.new('RGB', ((box[2]-box[0])*4, (box[3]-box[1])*2+28), BG)
    for i, (label, image) in enumerate((('ee2a224', old_head), ('current', new_head))):
        head = np.asarray(image)[:, X_OFFSET:X_OFFSET+1024, 3] > 127
        outside = head & (distance > 1)
        outside[:515] = False
        outside[618:] = False
        counts[label] = int(outside.sum())
        canvas = np.asarray(composite(image)).copy()
        shifted = [line + np.array([[[X_OFFSET, 0]]], dtype='int32') for line in contours]
        cv2.drawContours(canvas, shifted, -1, (70, 150, 255), 1)
        yy, xx = np.where(outside)
        canvas[yy, xx+X_OFFSET] = [255, 60, 60]
        panel = Image.fromarray(canvas).crop(box).resize(((box[2]-box[0])*2, (box[3]-box[1])*2), Image.Resampling.NEAREST)
        sheet.paste(panel, (i*panel.width, 28))
        ImageDraw.Draw(sheet).text((i*panel.width+5, 6), f'{label}: outside >1px = {counts[label]}', fill='white')
    sheet.save(DEST / 'head_base_review2_cheek.png')

    strip = Image.new('RGB', (5*360, 2*360), BG)
    for row, sign in enumerate((-1, 1)):
        for col, magnitude in enumerate((50, 60, 70, 80, 90)):
            yaw = sign*magnitude
            item = renders[(yaw, 0)] if (yaw, 0) in renders else extra_ear[yaw]
            label = f'{yaw:+}: model {item["model_angle"][0]:+.1f} / tilt {item["ear_tilt"]:.1f}'
            strip.paste(tile(item['portrait'], crop, 360, label), (col*360, row*360))
    strip.save(DEST / 'head_base_review2_ear_tilt.png')

    ear = Image.new('RGB', (1200, 1200), BG)
    for row, yaw in enumerate((-60, 60)):
        box = (X_OFFSET+(450 if yaw < 0 else 250), 350, X_OFFSET+(750 if yaw < 0 else 550), 650)
        for col, key in enumerate(('legacy_portrait', 'portrait')):
            panel = tile(renders[(yaw, 0)][key], box, 600, f'{yaw:+} / '+('legacy' if col == 0 else '3D'))
            ear.paste(panel, (col*600, row*600))
    ear.save(DEST / 'head_base_review2_ear_edge.png')

    old_face = Image.open(baseline / 'head_base_m60_face.png').convert('RGBA')
    new_face = renders[(-60, 0)]['face_only']
    box = (X_OFFSET+480, 360, X_OFFSET+565, 445)
    sheet = Image.new('RGB', (680, 370), BG)
    for i, (label, image) in enumerate((('ee2a224', old_face), ('current', new_face))):
        panel = composite(image).crop(box).resize((340, 340), Image.Resampling.NEAREST)
        sheet.paste(panel, (i*340, 30))
        ImageDraw.Draw(sheet).text((i*340+5, 6), label+' / face only', fill='white')
    sheet.save(DEST / 'head_base_review2_fragments.png')
    return counts


def main():
    reuse_before = '--reuse-before' in sys.argv
    meta = json.loads((ROOT / 'bundle/assets/head-base.json').read_text())
    rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
    skin_mesh = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'face-skin')
    rest = np.asarray(skin_mesh['positions'], float)
    triangles = np.asarray(skin_mesh['triangles'], int)
    r24 = {item['name']: item['minDetJ'] for item in json.loads((ROOT / 'docs/face_hairless_metrics.json').read_text())['poses']}
    pose_rows = json.loads((FIT / 'results/fit2e/poses-refined.json').read_text())['frames']
    poses = {row['frame']: row['refined_pose'] for row in pose_rows}
    thickness = json.loads((FIT / 'results/fit2f/shared-n3.json').read_text())['thickness_px']
    targets = {target['frame']: target for target in load_targets_f([key[1] for key in KEYS], thickness)}
    for _, frame, _, _ in KEYS:
        scale, shift = transform(frame, poses, meta['rest_translation'])
        assert np.allclose(scale * np.asarray(poses[frame][4:6]) + shift,
                           np.asarray(meta['rest_translation']) + [X_OFFSET, 0], atol=1e-9)
    tasks = list(dict.fromkeys([(key[2], key[3]) for key in KEYS] + STRIP + PROFILES + INTERMEDIATE))
    key_positions = {(key[2], key[3]) for key in KEYS}
    jacobian_positions = key_positions | set(INTERMEDIATE) | {(-67.5, 0), (67.5, 0)}
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT / 'bundle')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, renders = [], {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--use-angle=swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            page.on('response', lambda r: errors.append(str(r.status) + ' ' + r.url) if r.status >= 400 else None)
            page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
            page.wait_for_function('window.studio?.ready', timeout=30000)
            page.evaluate('''async()=>{
              const rendererModule=await import('./renderer.mjs');
              const {Renderer}=rendererModule;window.rendererModule=rendererModule;
              const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
              Object.assign(canvas.style,{width:'1500px',height:'1600px',position:'fixed',left:'-2000px',top:'0'});
              document.body.append(canvas);window.headCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
              window.headModule=await import('./head-base.mjs');window.poseModule=await import('./pose-grid.mjs');
              window.rigModule=await import('./rig.mjs');window.solveData=await import('./face-solve-data.mjs');
            }''')
            # Compare each disconnected face component at rest before removing it.
            # The same renderer and exact pixel comparison cover both atlases and paths.
            islands = page.evaluate('''()=>{
              const meshes=studio.rig.meshes,skin=meshes.find(m=>m.id==='face-skin');
              const parent=skin.positions.map((_,i)=>i);
              const find=i=>parent[i]===i?i:(parent[i]=find(parent[i]));
              for(const t of skin.triangles)for(const i of t.slice(1))parent[find(i)]=find(t[0]);
              const groups=new Map();
              for(const t of skin.triangles){const id=find(t[0]);if(!groups.has(id))groups.set(id,[]);groups.get(id).push(t);}
              const candidates=[...groups.values()].filter(group=>group.length<10).sort((a,b)=>a.length-b.length);
              const hairless=meshes.filter(m=>m.id!=='face-skin'&&m.id!=='neck'&&!m.id.startsWith('ear-')&&m.deformerId!=='WarpBody').map(m=>m.id);
              const scenes=[['new_full',1,false,[]],['new_hairless',1,false,hairless],['legacy_full',0,false,[]],
                            ['new_before',1,true,[]],['legacy_before',0,true,[]]];
              const pose={...studio.p,CorrOutline:1,CorrTurn:1,ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0};
              const pixels=([name,active,before,hidden])=>{
                headCapture.render({...pose,CorrHeadBase:active},{...studio.state,mode:'all',wire:false,grid:false,masks:false,before,hidden:new Set(hidden)});
                const gl=headCapture.gl,data=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
                gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,data);
                if(gl.getError())throw Error('WebGL error during face component check');
                return data;
              };
              let baseline=scenes.map(pixels),accepted=[];
              for(const group of candidates){
                const ids=[...new Set(group.flat())].sort((a,b)=>a-b),set=new Set(ids);
                const prior=skin.triangles;
                skin.triangles=prior.filter(t=>!t.every(i=>set.has(i)));
                headCapture.faceBoundary=rendererModule.faceBoundaryDistances(skin);
                const after=scenes.map(pixels),difference={};
                for(let k=0;k<scenes.length;k++){
                  let count=0;for(let i=0;i<after[k].length;i+=4)
                    if(after[k][i]!==baseline[k][i]||after[k][i+1]!==baseline[k][i+1]||after[k][i+2]!==baseline[k][i+2]||after[k][i+3]!==baseline[k][i+3])count++;
                  difference[scenes[k][0]]=count;
                }
                const eligible=Object.values(difference).every(n=>n===0);
                skin.triangles=prior;headCapture.faceBoundary=rendererModule.faceBoundaryDistances(skin);
                const points=ids.map(i=>skin.positions[i]);
                accepted.push({vertices:ids,triangles:group.length,bbox:[Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))],difference,eligible,removed:false});
              }
              return accepted;
            }''')
            (ROOT / 'docs/head_base_islands.json').write_text(json.dumps(islands, ensure_ascii=False, indent=2) + '\n')
            print('Face components: '+json.dumps(islands, ensure_ascii=False), flush=True)
            prior_atlas = [base64.b64encode(subprocess.check_output(
                ['git', 'show', NECK_EDGE_BASELINE+':bundle/assets/'+name], cwd=ROOT)).decode('ascii')
                for name in ('texture_00.png', 'neck-keep.png')]
            page.evaluate('''async data=>{
              const images=await Promise.all(data.map(async value=>{const image=new Image();image.src='data:image/png;base64,'+value;await image.decode();return image;}));
              const gl=headCapture.gl;
              headCapture.baselineTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,headCapture.baselineTexture);
              headCapture.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,images[0]);
              headCapture.baselineKeep=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,headCapture.baselineKeep);
              headCapture.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,images[1]);
              const [x,y,w,h]=studio.rig.meshes.find(m=>m.id==='face-skin').atlas;
              headCapture.originalFace=studio.renderer.images.slice(0,2).map(image=>{
                const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;
                canvas.getContext('2d').drawImage(image,x,y,w,h,0,0,w,h);
                const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);headCapture.textureParams();
                gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,canvas);return tex;
              });
            }''', prior_atlas)
            core = page.evaluate('''()=>{
              const make=(yaw,pitch)=>({...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,[poseModule.FRAME_CACHE]:new Map()});
              const skin=studio.rig.meshes.find(m=>m.id==='face-skin'),p=make(0,0);let rest=0;
              for(const point of [...skin.positions,...solveData.FACE_SOLVE.points]){const q=headModule.headSolveWorld(point,p);rest=Math.max(rest,Math.hypot(q[0]-point[0],q[1]-point[1]));}
              const edges=[];for(const sign of [-1,1])for(const pitch of [-30,0,30]){
                const a=make(sign*44.99,pitch),b=make(sign*45.01,pitch),va=headModule.headBaseVertices(a).surface,vb=headModule.headBaseVertices(b).surface;
                let head=0,face=0;for(let i=0;i<va.length;i+=3)head=Math.max(head,Math.hypot(va[i]-vb[i],va[i+1]-vb[i+1]));
                for(const point of skin.positions){const x=headModule.headSolveWorld(point,a),y=headModule.headSolveWorld(point,b);face=Math.max(face,Math.hypot(x[0]-y[0],x[1]-y[1]));}
                edges.push({sign,pitch,head_px:head,face_px:face});
              }
              return {rest_max_px:rest,edges};
            }''')
            for i, (yaw, pitch) in enumerate(tasks):
                data = page.evaluate('''([yaw,pitch,key,measure,captureBefore])=>{
                  const p={...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0};
                  const meshes=studio.rig.meshes,skin=meshes.find(m=>m.id==='face-skin');
                  const hidden=meshes.filter(m=>m.id!=='face-skin'&&m.id!=='neck'&&!m.id.startsWith('ear-')&&m.deformerId!=='WarpBody').map(m=>m.id);
                  const state={...studio.state,mode:'all',wire:false,grid:false,before:false,surfaceOnly:false,masks:false,identify:null};
                  const capture=(ids,pose=p,identify=null,extra={})=>{headCapture.render(pose,{...state,...extra,identify,hidden:new Set(ids)});const error=headCapture.gl.getError();if(error)throw Error('WebGL error '+error);return headCapture.canvas.toDataURL('image/png').split(',')[1];};
                  const portrait=capture(hidden);
                  const legacyPortrait=Math.abs(yaw)===60&&pitch===0?capture(hidden,{...p,CorrHeadBase:0}):null;
                  const faceOnly=yaw===-60&&pitch===0?capture([...meshes.filter(m=>m.id!=='face-skin').map(m=>m.id),'head-base']):null;
                  const unionIds=meshes.filter(m=>m.id!=='face-skin').map(m=>m.id);
                  const union=key?capture(unionIds,p,null,{hideHeadCap:true}):null;
                  const unionWithCap=key?capture(unionIds):null;
                  const ears=key?capture(meshes.filter(m=>!m.id.startsWith('ear-')).map(m=>m.id),p,['ear-left','ear-right']):null;
                  const headNeck=capture(meshes.filter(m=>m.id!=='face-skin'&&m.id!=='neck').map(m=>m.id));
                  const headOnly=capture(meshes.map(m=>m.id));
                  const neckOnly=capture(meshes.filter(m=>m.id!=='neck').map(m=>m.id),p,'neck');
                  const neck=meshes.find(m=>m.id==='neck'),extension=headCapture.neckExtension;
                  headCapture.neckExtension=neck;
                  const neckArt=capture(meshes.filter(m=>m.id!=='neck').map(m=>m.id),p,'neck',{neckArtOnly:true});
                  headCapture.neckExtension=extension;
                  const isolated=[...meshes.filter(m=>m.id!=='face-skin').map(m=>m.id),'head-base'];
                  const restOld=yaw===0&&pitch===0?capture(isolated,{...p,CorrHeadBase:0}):null;
                  const restNew=yaw===0&&pitch===0?capture(isolated):null;
                  const restEars=yaw===0&&pitch===0?Object.fromEntries(['left','right'].map(side=>[side,capture(meshes.filter(m=>m.id!=='ear-'+side).map(m=>m.id),p,'ear-'+side)])):null;
                  const restBefore=captureBefore&&yaw===0&&pitch===0?capture(hidden,p,null,{before:true}):null;
                  const q={...p,[poseModule.FRAME_CACHE]:new Map()},points=headModule.headBaseVertices(q).surface;
                  let left=Infinity,top=Infinity,right=-Infinity,bottom=-Infinity;
                  const include=(x,y)=>{left=Math.min(left,x);top=Math.min(top,y);right=Math.max(right,x);bottom=Math.max(bottom,y);};
                  for(let j=0;j<points.length;j+=3)include(points[j],points[j+1]);
                  const skinPositions=skin.positions.map(point=>rigModule.deform3D(skin,point,q).slice(0,2));
                  const base=headModule.getHeadBase(),ring=base.open_boundary,ref=base.reference;
                  const axis=[0,2].map(c=>ring.reduce((sum,i)=>sum+ref[i*3+c],0)/ring.length),rows=new Map();
                  for(const point of neck.positions){const x=rigModule.deform3D(neck,point,q)[0],row=rows.get(point[1])||[Infinity,-Infinity];row[0]=Math.min(row[0],x);row[1]=Math.max(row[1],x);rows.set(point[1],row);}
                  const differences=[...rows.values()].map(([a,b])=>Math.abs((a+b)/2-axis[0]));
                  const neckAxis={center_x:axis[0],depth:axis[1],max_row_difference:Math.max(...differences),fallback_rows:differences.filter(d=>d>5).length};
                  const neckPositions=neck.positions.map(point=>rigModule.deform3D(neck,point,q).slice(0,2));
                  for(const point of skinPositions)include(point[0],point[1]);
                  for(const mesh of meshes.filter(m=>m.id.startsWith('ear-')))for(const point of mesh.positions){const v=rigModule.deform3D(mesh,point,q);include(v[0],v[1]);}
                  let restOriginalNeck=null,restOriginalNeckOnly=null,necklaceHidden=null,assembled=null;
                  let neckEdges=null;
                  if(yaw===0&&pitch===0){
                    assembled=capture([],p,null,{outfit:'sweater',bangs:false});
                    necklaceHidden=capture(['necklace'],p,null,{outfit:'sweater',bangs:false});
                    headCapture.neckExtension=neck;
                    restOriginalNeck=capture(hidden,p,null,{neckArtOnly:true});
                    restOriginalNeckOnly=neckArt;
                    headCapture.neckExtension=extension;
                    const hair=meshes.filter(m=>m.id==='hair-top'||m.id==='hairpin'||/^(bangs-|side-hair-|back-hair-|side-cranium-|tail-)/.test(m.id)).map(m=>m.id);
                    const current=headCapture.textures[0],currentKeep=headCapture.textures[9];neckEdges={};
                    for(const outfit of ['sweater','shirt']){
                      headCapture.textures[0]=headCapture.baselineTexture;
                      headCapture.textures[9]=headCapture.baselineKeep;
                      neckEdges[outfit+'_before']=capture(hair,p,null,{outfit});
                      headCapture.textures[0]=current;
                      headCapture.textures[9]=currentKeep;
                      neckEdges[outfit+'_after']=capture(hair,p,null,{outfit});
                    }
                    const onlyNeck=meshes.filter(m=>m.id!=='neck').map(m=>m.id);
                    neckEdges['neck_background_test']=capture(onlyNeck,p,'neck');
                    neckEdges['neck_background_before_view']=capture(onlyNeck,p,'neck',{before:true});
                    const onlyFace=meshes.filter(m=>m.id!=='face-skin').map(m=>m.id);
                    for(const before of [false,true])for(const legacy of [false,true]){
                      const index=before?8:7,original=headCapture.textures[index];
                      const pose=legacy?{...p,CorrHeadBase:0,ParamAngleX:30,ParamAngleY:20}:p;
                      const name=(before?'before':'after')+'_'+(legacy?'legacy':'rest');
                      headCapture.textures[index]=headCapture.originalFace[before?1:0];
                      neckEdges[name+'_original']=capture(onlyFace,pose,null,{before});
                      headCapture.textures[index]=original;
                      neckEdges[name+'_clean']=capture(onlyFace,pose,null,{before});
                    }
                  }
                  return {portrait,legacyPortrait,faceOnly,union,unionWithCap,ears,headNeck,headOnly,neckOnly,neckArt,neckAxis,restOld,restNew,restEars,restBefore,restOriginalNeck,restOriginalNeckOnly,necklaceHidden,assembled,neckEdges,skinPositions,neckPositions,bounds:[left,top,right,bottom],modelAngle:headModule.modelAngles(q),earTilt:headModule.earTiltDegrees(q),hidden};
                }''', [yaw, pitch, (yaw, pitch) in key_positions, (yaw, pitch) in jacobian_positions, not reuse_before])
                renders[(yaw, pitch)] = {'portrait': decode(data['portrait']),
                                         'bounds': data['bounds'], 'neck_axis': data['neckAxis'],
                                         'model_angle': data['modelAngle'], 'hidden': data['hidden'],
                                         'ear_tilt': data['earTilt'],
                                         'neck_positions': data['neckPositions']}
                for source, target in [('legacyPortrait', 'legacy_portrait'), ('faceOnly', 'face_only'), ('union', 'union'), ('unionWithCap', 'union_with_cap'), ('ears', 'ears'), ('headNeck', 'head_neck'),
                                       ('headOnly', 'head_only'), ('neckOnly', 'neck_only'), ('neckArt', 'neck_art'),
                                       ('restOld', 'rest_old'), ('restNew', 'rest_new'),
                                       ('restBefore', 'rest_before'),
                                       ('restOriginalNeck', 'rest_original_neck'), ('restOriginalNeckOnly', 'rest_original_neck_only'),
                                       ('necklaceHidden', 'necklace_hidden'), ('assembled', 'assembled')]:
                    if data[source] is not None:
                        renders[(yaw, pitch)][target] = decode(data[source])
                if data['restEars']:
                    renders[(yaw, pitch)]['rest_ears'] = {side: decode(value) for side, value in data['restEars'].items()}
                if data['neckEdges']:
                    renders[(yaw, pitch)].update({name: decode(value) for name, value in data['neckEdges'].items()})
                if data['skinPositions'] is not None:
                    renders[(yaw, pitch)]['skin_positions'] = data['skinPositions']
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    print(f'Captured {i + 1}/{len(tasks)} poses', flush=True)
            extra_ear = {}
            for yaw in (-80, -70, -50, 50, 70, 80):
                data = page.evaluate('''yaw=>{
                  const meshes=studio.rig.meshes;
                  const hidden=meshes.filter(m=>m.id!=='face-skin'&&m.id!=='neck'&&!m.id.startsWith('ear-')&&m.deformerId!=='WarpBody').map(m=>m.id);
                  const p={...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:0,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0};
                  headCapture.render(p,{...studio.state,mode:'all',wire:false,grid:false,before:false,surfaceOnly:false,masks:false,hidden:new Set(hidden)});
                  if(headCapture.gl.getError())throw Error('WebGL error during extra ear poses');
                  return {image:headCapture.canvas.toDataURL('image/png').split(',')[1],angle:headModule.modelAngles(p),tilt:headModule.earTiltDegrees(p)};
                }''', yaw)
                extra_ear[yaw] = {'portrait': decode(data['image']), 'model_angle': data['angle'], 'ear_tilt': data['tilt']}
            browser.close()
            if errors:
                raise RuntimeError('Browser errors: ' + str(errors))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    gaps = {}
    neck_mesh = next(mesh for mesh in rig['meshes'] if mesh['id'] == 'neck')
    atlas = np.asarray(Image.open(ROOT / 'bundle/assets/texture_00.png').convert('RGBA'))
    maximum_gap = None
    for yaw, pitch in tasks:
        item = renders[(yaw, pitch)]
        combined = alpha(item['head_neck'])
        hole_locations = closed_hole_locations(combined)
        hole_pixels = len(hole_locations)
        band = neck_opaque_band(item['neck_art'])
        if band:
            gap_pixels, columns, gap_mask = neck_gap(alpha(item['head_only']), identified(item['neck_only']), combined, (band['left'], band['right']))
        else:
            gap_pixels, columns, gap_mask = None, 0, np.zeros_like(combined, dtype=bool)
        evidence = []
        for x, y in hole_locations:
            evidence.append({'point': [x, y],
                             'head_alpha': int(np.asarray(item['head_only'])[y, x+X_OFFSET, 3]),
                             'combined_alpha': int(np.asarray(item['head_neck'])[y, x+X_OFFSET, 3]),
                             'neck_samples': hole_mesh_samples((x, y), neck_mesh, item['neck_positions'], atlas),
                             'face_samples': hole_mesh_samples((x, y), skin_mesh, item['skin_positions'], atlas)})
        gaps[(yaw, pitch)] = {'hole_pixels': hole_pixels, 'hole_locations_canvas': hole_locations, 'hole_evidence': evidence,
                              'neck_gap_pixels': gap_pixels,
                              'measured_columns': columns, 'opaque_band': band,
                              'passed': columns > 0 and hole_pixels == 0 and gap_pixels == 0}
        if gap_pixels is not None and (maximum_gap is None or gap_pixels > maximum_gap['pixels']):
            maximum_gap = {'yaw': yaw, 'pitch': pitch, 'pixels': gap_pixels, 'mask': gap_mask}

    if maximum_gap:
        image = np.asarray(composite(renders[(maximum_gap['yaw'], maximum_gap['pitch'])]['portrait'])).copy()
        image[maximum_gap['mask']] = [255, 50, 50]
        yy, xx = np.where(maximum_gap['mask'])
        box = (max(0, int(xx.min())-80), max(0, int(yy.min())-80),
               min(CANVAS_WIDTH, int(xx.max())+81), min(1024, int(yy.max())+81)) if len(xx) else (X_OFFSET+350, 430, X_OFFSET+650, 780)
        Image.fromarray(image).crop(box).save(DEST / 'head_base_max_gap.png')

    rows = []
    for name, frame, yaw, pitch in KEYS:
        item, target = renders[(yaw, pitch)], targets[frame]
        scale, shift = transform(frame, poses, meta['rest_translation'])
        hair = target['points'][target['labels'] == 1] * scale + shift
        skin_target = target['skin_all'][target['skin_keep']] * scale + shift
        n3 = n3_metrics(alpha(item['union']), skin_target, hair)
        n3_with_cap = n3_metrics(alpha(item['union_with_cap']), skin_target, hair)
        n4 = jacobians(rest, np.asarray(item['skin_positions']), triangles)
        n4['passed'] = n4['min_det_j'] >= .35 and n4['max_anisotropy'] <= 2
        rows.append({'name': name, 'frame': frame, 'yaw': yaw, 'pitch': pitch, 'model_angle': item['model_angle'],
                     'N3': n3, 'N3_with_cap': n3_with_cap, 'N4': n4, 'N10': gaps[(yaw, pitch)],
                     'N11': ear_iou(identified(item['ears']), frame, poses, meta['rest_translation']),
                     'r24_min_det_j': r24.get(name), 'hair_points': hair, 'skin_points': skin_target})
    extra = []
    for yaw, pitch in INTERMEDIATE + PROFILES:
        item = renders[(yaw, pitch)]
        row = {'name': f'pose-{yaw:+g}-{pitch:+g}', 'yaw': yaw, 'pitch': pitch}
        if (yaw, pitch) in jacobian_positions:
            row['N4'] = jacobians(rest, np.asarray(item['skin_positions']), triangles)
            row['N4']['passed'] = row['N4']['min_det_j'] >= .35 and row['N4']['max_anisotropy'] <= 2
        row['N10'] = gaps[(yaw, pitch)]
        extra.append(row)
    x0 = max(0, math.floor(rest[:, 0].min()) + X_OFFSET)
    x1 = min(CANVAS_WIDTH, math.ceil(rest[:, 0].max()) + X_OFFSET + 1)
    y0 = max(0, math.floor(rest[:, 1].min()))
    y1 = min(1024, math.ceil(rest[:, 1].max()) + 1)
    old = np.asarray(renders[(0, 0)]['rest_old'])[y0:y1, x0:x1]
    new = np.asarray(renders[(0, 0)]['rest_new'])[y0:y1, x0:x1]
    changed = int(np.count_nonzero(np.any(old != new, axis=2)))
    baseline = DEST / 'fix4-baseline'
    before = np.asarray(Image.open(baseline / '+0_+0_portrait.png').convert('RGBA'))
    after = np.asarray(renders[(0, 0)]['portrait'])
    difference = np.any(before != after, axis=2)
    previous_portrait = np.asarray(Image.open(DEST / 'review2-baseline/head_base_rest_portrait.png').convert('RGBA'))
    yy, xx = np.where(difference)
    rest_ear = np.zeros_like(difference)
    ear_occlusion, ear_pixel_difference = {}, {}
    for side in ('left', 'right'):
        old_ear = Image.open(baseline / f'rest_ear_{side}.png').convert('RGBA')
        mask = identified(old_ear)
        rest_ear |= mask
        current_ear = np.asarray(renders[(0, 0)]['rest_ears'][side])
        previous_ear = np.asarray(old_ear)
        ear_occlusion[side] = int(np.count_nonzero(mask & ~identified(renders[(0, 0)]['rest_ears'][side])))
        ear_region = (previous_ear[:, :, 0] > 0) | (current_ear[:, :, 0] > 0)
        ear_pixel_difference[side] = int(np.count_nonzero(ear_region & np.any(previous_portrait != after, axis=2)))
    rest_ear = cv2.dilate(rest_ear.astype('uint8'), np.ones((3, 3), dtype='uint8')) > 0
    older = np.asarray(Image.open(DEST / 'fix3-baseline/+0_+0_portrait.png').convert('RGBA'))
    older_difference = np.any(older != after, axis=2)
    original_neck = np.asarray(renders[(0, 0)]['rest_original_neck_only'])
    opaque_neck = (original_neck[:, :, 0] == 255) & (original_neck[:, :, 3] == 255)
    rest_diff = {'changed_pixels': int(difference.sum()), 'ear_changed_pixels': int(np.count_nonzero(difference & rest_ear)),
                 'neck_opaque_changed_pixels': int(np.count_nonzero(difference & ~rest_ear & opaque_neck)),
                 'other_changed_pixels': int(np.count_nonzero(difference & ~rest_ear & ~opaque_neck)),
                 'ear_changed_from_8330_pixels': int(np.count_nonzero(older_difference & rest_ear)), 'bbox': None}
    without_extension = np.asarray(renders[(0, 0)]['rest_original_neck'])
    extension_diff = np.any(without_extension != after, axis=2)
    dy, dx = np.where(extension_diff)
    rest_extension_diff = {'changed_pixels': int(extension_diff.sum()),
                           'on_opaque_original_neck_pixels': int(np.count_nonzero(extension_diff & opaque_neck)),
                           'bbox': [int(dx.min())-X_OFFSET, int(dy.min()), int(dx.max())-X_OFFSET, int(dy.max())] if len(dx) else None}
    if reuse_before:
        renders[(0, 0)]['rest_before'] = Image.open(DEST / 'head_base_rest_before.png').convert('RGBA')
    rest_before = np.asarray(renders[(0, 0)]['rest_before'])
    before_after_difference = np.any(rest_before != after, axis=2)
    rest_views = {'neck_opaque_changed_pixels': int(np.count_nonzero(before_after_difference & opaque_neck))}
    face_clean_checks = {}
    for view in ('before','after'):
        for pose_name in ('rest','legacy'):
            name = view+'_'+pose_name
            a = np.asarray(renders[(0,0)][name+'_original'])
            b = np.asarray(renders[(0,0)][name+'_clean'])
            face_clean_checks[name] = int(np.count_nonzero(np.any(a!=b,axis=2)))
    baseline_capture = DEST / 'fix11-baseline'
    previous_rest = np.asarray(Image.open(baseline_capture / 'head_base_rest_after.png').convert('RGBA'))
    rest_vs_previous = int(np.count_nonzero(np.any(previous_rest!=after,axis=2)))
    renders[(0, 0)]['rest_before'].save(DEST / 'head_base_rest_before.png')
    renders[(0, 0)]['portrait'].save(DEST / 'head_base_rest_after.png')
    composite(renders[(0, 0)]['necklace_hidden']).save(DEST / 'head_base_necklace_hidden.png')
    still_neck = save_still_comparison(renders[(0, 0)]['assembled'], baseline_capture)
    if len(xx):
        box = (max(0, int(xx.min())-8), max(0, int(yy.min())-8),
               min(CANVAS_WIDTH, int(xx.max())+9), min(1024, int(yy.max())+9))
        rest_diff['bbox'] = [int(xx.min())-X_OFFSET, int(yy.min()), int(xx.max())-X_OFFSET, int(yy.max())]
        before_crop, after_crop = composite(Image.fromarray(before)).crop(box), composite(Image.fromarray(after)).crop(box)
        comparison = Image.new('RGB', (before_crop.width*4, before_crop.height*2), BG)
        for col, image in enumerate((before_crop, after_crop)):
            comparison.paste(image.resize((image.width*2, image.height*2), Image.Resampling.NEAREST), (col*image.width*2, 0))
        comparison.save(DEST / 'head_base_rest_diff.png')
    crop = make_crop(renders)
    n6max = max(max(edge['head_px'], edge['face_px']) for edge in core['edges'])
    previous_metrics = json.loads(subprocess.check_output(
        ['git', 'show', '030dea1:docs/head_base_metrics.json'], cwd=ROOT))
    previous_keys = {item['name']: item for item in previous_metrics['poses']}
    for row in rows:
        prior = previous_keys[row['name']]
        row['N3_skin_median_change_px'] = row['N3']['skin']['median_px'] - prior['N3']['skin']['median_px']
        row['N11_previous'] = prior['N11']['iou']
        row['N11_change'] = row['N11']['iou'] - prior['N11']['iou']
    cheek_pixels = save_review2(renders, extra_ear, crop)
    metrics = {'N1': {'max_error_px': core['rest_max_px'], 'passed': core['rest_max_px'] <= 1e-3},
               'N2': {'passed': True, 'evidence': 'pipeline/test_head_base.mjs: legacy and feature hashes'},
               'N3c': {'changed_pixels': changed, 'passed': changed == 0, 'face_bbox': [x0, y0, x1, y1]},
               'N6': {'max_difference_px': n6max, 'passed': n6max <= .5, 'edges': core['edges']},
               'N7': {'passed': True, 'evidence': 'pipeline/test_head_base.mjs: projection fixture'},
               'N10': {'poses': [{'yaw': yaw, 'pitch': pitch, **gaps[(yaw, pitch)]} for yaw, pitch in tasks]},
               'cheek_outside_pixels': cheek_pixels, 'face_islands': islands,
               'rest_full_difference': rest_diff,
               'rest_ear_occluded_pixels': ear_occlusion, 'rest_ear_rgba_difference': ear_pixel_difference,
               'rest_fill_outside_head': json.loads((baseline / 'rest_analysis.json').read_text())['fill_outside_pixels'],
               'ear_depth_bias_px': meta['ear_depth_bias_px'],
               'neck_extension': meta['neck_extension'], 'rest_extension_difference': rest_extension_diff, 'rest_views': rest_views,
               'face_clean_checks': face_clean_checks, 'rest_changed_pixels_vs_cc3593a': rest_vs_previous,
               'rest_still_neck': still_neck,
               'neck_axis': {'center_x': renders[(0, 0)]['neck_axis']['center_x'],
                             'rest_depth': renders[(0, 0)]['neck_axis']['depth'],
                             'max_row_difference': max(item['neck_axis']['max_row_difference'] for item in renders.values()),
                             'max_fallback_rows': max(item['neck_axis']['fallback_rows'] for item in renders.values())},
               'temple_diagnosis': {'status': 'clipped', 'layer': 'exposed-skin',
                                    'inventory': json.loads((DEST / 'fix3-baseline/fill-inventory.json').read_text())},
               'poses': [{k: v for k, v in row.items() if k not in ('hair_points', 'skin_points')} for row in rows] + extra,
               'rest_translation': meta['rest_translation'], 'common_crop_pixels': crop,
               'hidden_meshes': renders[(0, 0)]['hidden'],
               'reference_transform': '(q - camera_translation) * exp(logscale_front - logscale_key) + rest_translation',
               'note': 'Angle-dependent displacement and outside-offset decay remain active.'}
    save_sheets(renders, rows, poses, meta['rest_translation'], crop)
    save_edges(renders, rows, baseline_capture)
    save_fix3(renders, baseline_capture)
    save_direction_color(baseline_capture)
    save_temple_diagnosis(DEST / 'fix3-baseline')
    metrics['neck_edge'] = save_neck_edges(renders)
    save_review_details(baseline_capture)
    (ROOT / 'docs/head_base_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'poses': len(tasks), 'N1': metrics['N1'], 'N6': metrics['N6']['max_difference_px'],
                      'N3c': metrics['N3c'], 'crop': crop, 'N3_passed': sum(row['N3']['passed'] for row in rows)}, ensure_ascii=False), flush=True)
    if changed:
        raise SystemExit('STOP: face pixels changed at rest while the head base was hidden')
    if any(face_clean_checks.values()):
        raise SystemExit('STOP: clean face changed rest or legacy path: '+str(face_clean_checks))
    worst = max(rows, key=lambda row: row['N3_skin_median_change_px'])
    if worst['N3_skin_median_change_px'] > .5:
        raise SystemExit('STOP: N3 skin median worsened by more than 0.5px: '+
                         worst['name']+' '+str(worst['N3_skin_median_change_px']))


if __name__ == '__main__':
    main()
