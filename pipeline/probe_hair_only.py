"""Hair-only WebGL/reference measurements in the frozen r15 registration frame.

Run --baseline to capture the three r15 neutral poses without changing checkout.
The normal run retains real draw order: non-target meshes are black occluders,
not hidden geometry. Reference segmentation is a diagnostic color/face mask,
not a replacement for the original SAM crown masks.
"""
import argparse
import base64
import functools
import hashlib
import http.server
import io
import json
import math
import subprocess
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '89fc408e7b8efb06acadb958324c31a30d0c0b84'
HAIR_CROWN = ['hair-top', 'bangs-left', 'bangs-right', 'side-hair-left',
              'side-hair-right', 'hairpin', 'back-hair-left', 'back-hair-right',
              'back-hair-center', 'side-cranium-left', 'side-cranium-right']
TAILS = ['tail-left-hair', 'tail-right-hair']
TIES = ['tail-left-tie', 'tail-right-tie']
GROUPS = {'crown': HAIR_CROWN, 'tails': TAILS, 'ties': TIES, 'face': ['face-skin'], 'attachment-hair': HAIR_CROWN}


def outline_rows(mask, stop=400):
    rows = []
    for y in range(80, stop + 1, 20):
        xs = np.flatnonzero(mask[y])
        rows.append({'y': y, 'span': [int(xs[0]), int(xs[-1])] if len(xs) else None})
    return rows


def top_gate(pitch, rendered, target):
    if pitch < 0:
        return {'kind': 'cropped-bound', 'maximum': 4, 'pass': rendered <= 4}
    return {'kind': 'target', 'target': target, 'error': rendered - target,
            'pass': abs(rendered - target) <= 8}


def approved_top_gate(yaw, pitch, rendered):
    """Approved completion bounds; reference residuals remain diagnostic."""
    if pitch > 0 and not yaw:
        return {'target': 92, 'tolerance': 8, 'pass': abs(rendered-92) <= 8}
    if yaw and pitch:
        maximum = 65 if pitch > 0 else 13
        return {'maximum': maximum, 'pass': rendered <= maximum}
    return None


def stem_gap(hair, tie):
    """Fable's percentile 20..80 columns, sampled every 2px above the tie."""
    yy, xx = np.where(tie)
    if not len(xx):
        return None
    lo, hi = [int(np.percentile(xx, p)) for p in [20, 80]]
    gaps = []
    for x in range(lo, hi+1, 2):
        ys = np.flatnonzero(tie[:, x])
        if not len(ys):
            continue
        top = int(ys[0])
        start = max(0, top-80)
        above = np.flatnonzero(hair[start:top, x])
        gaps.append(top-(start+int(above[-1]))-1 if len(above) else 80)
    return max(gaps, default=81)


def cheek_cover_rows(face, hair, side, rows):
    """Background pixels between visible skin edge and adjacent crown alpha."""
    out = []
    for y in rows:
        xs = np.flatnonzero(face[y])
        if not len(xs):
            continue
        edge = int(xs[-1] if side == 'right' else xs[0])
        line = hair[y, edge+1:] if side == 'right' else hair[y, :edge][::-1]
        hits = np.flatnonzero(line)
        out.append({'y': y, 'gap': int(hits[0]) if len(hits) else 1024})
    return out


def neutral_difference(a, b):
    if a.shape != b.shape:
        raise ValueError('Neutral capture dimensions differ')
    return int(np.any(a != b, axis=2).sum())


def span(mask, y):
    xs = np.flatnonzero(mask[y])
    return [int(xs[0]), int(xs[-1])] if len(xs) else None


def hairline_metrics(reference_face, rendered_face, eye_y):
    f, g = span(reference_face, 330), span(rendered_face, 330)
    samples, exposure = [], None
    if f and g:
        left, right = max(f[0], g[0]) + 6, min(f[1], g[1]) - 6
        for x in range(left, right, 12):
            a, b = np.flatnonzero(reference_face[:700, x]), np.flatnonzero(rendered_face[:700, x])
            if len(a) and len(b):
                samples.append({'x': x, 'reference': int(a[0]), 'render': int(b[0]), 'delta': int(b[0] - a[0])})
        reference_area = int(reference_face[150:eye_y, left:right].sum())
        render_area = int(rendered_face[150:eye_y, left:right].sum())
        exposure = {'referenceArea': reference_area, 'renderArea': render_area,
                    'ratioError': render_area / reference_area - 1 if reference_area else None,
                    'region': [left, 150, right, eye_y]}
    return {'hairline': samples, 'hairlineMedianError': float(np.median([h['delta'] for h in samples])) if samples else None,
            'foreheadExposure': exposure}


def bead_centers(rgb, delta, face):
    transform = np.float32([[1, 0, 152 + delta[0]], [0, 1, 20 + delta[1]]])
    rgb = cv2.warpAffine(rgb, transform, (1024, 740))
    face = cv2.warpAffine(face.astype(np.uint8), transform, (1024, 740), flags=cv2.INTER_NEAREST) > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 130)
            & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 65))
    blue &= ~cv2.dilate(face.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    yy, xx = np.indices(blue.shape)
    result = {}
    for side, (left, right) in {'left': (200, 375), 'right': (640, 800)}.items():
        roi = blue & (xx >= left) & (xx < right) & (yy >= 440) & (yy < 680)
        y, x = np.where(roi)
        if len(x) < 20:
            raise ValueError(f'Missing reference {side} bead pair')
        canvas = [(float(np.percentile(v, 2)) + float(np.percentile(v, 98))) / 2 for v in [x, y]]
        result[side] = {'refFrame': [canvas[i] - delta[i] for i in [0, 1]], 'refCanvas': canvas,
                        'pixels': len(x)}
    return result


def fill_holes(mask):
    # A cropped top/bottom can split the visible background in two. Connect
    # both sides outside the canvas before classifying enclosed holes.
    image = np.pad(mask.astype(np.uint8) * 255, 1)
    flooded = image.copy()
    cv2.floodFill(flooded, np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8), (0, 0), 255)
    return mask | (flooded[1:-1, 1:-1] == 0)


def reference(pose):
    name = pose['name']
    filename = {'left': 'left_45_ref_720.png', 'front': 'head_front_ref.png',
                'right': 'right_45_ref_720.png'}.get(name, name.replace('-', '_') + '_ref_720.png')
    rgb = np.array(Image.open(ROOT / 'source/references' / filename).convert('RGB'))
    delta = pose.get('registration', {}).get('delta', [0, 0])
    if name == 'front':
        full = np.array(Image.open(ROOT / 'sam3_masks/face-skin_mask.png').convert('L'))
        face = full[20:740, 152:872] > 127
    else:
        face = np.array(Image.open(ROOT / 'sam3_masks/references' / (name + '-face.png')).convert('L')) > 127
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = [hsv[:, :, i].astype(int) for i in range(3)]
    foreground = cv2.morphologyEx((rgb.max(2) > 22).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    foreground = np.isin(labels, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 400])
    skin = (value >= 150) & (saturation >= 20) & (saturation <= 125) & (hue <= 25)
    white = (saturation < 35) & (value >= 190)
    blue = (hue > 85) & (hue < 130) & (saturation > 110) & (value > 65)
    hair = foreground & ~cv2.dilate(face.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~skin & ~white & ~blue
    hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hair)
    hair = np.isin(labels, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] > 300])
    transform = np.float32([[1, 0, 152 + delta[0]], [0, 1, 20 + delta[1]]])
    mask = cv2.warpAffine(hair.astype(np.uint8), transform, (1024, 740), flags=cv2.INTER_NEAREST) > 0
    face_canvas = cv2.warpAffine(face.astype(np.uint8), transform, (1024, 740), flags=cv2.INTER_NEAREST) > 0
    canvas = cv2.warpAffine(rgb, transform, (1024, 740))
    if name == 'front':
        mask = np.zeros((740, 1024), bool)
        for key in ['hair-crown-all', 'side-hair-left', 'side-hair-right', 'tail-left-hair', 'tail-right-hair', 'tail-left-tie', 'tail-right-tie']:
            mask |= np.array(Image.open(ROOT / 'sam3_masks' / (key + '_mask.png')).convert('L'))[:740] > 127
    elif name in ['left', 'right']:
        mask = cv2.warpAffine(np.array(Image.open(ROOT / 'sam3_masks/references' / (name + '-crown.png')).convert('L')), transform, (1024, 740)) > 127
    head = fill_holes(mask | face_canvas)
    return {'rgb': canvas, 'mask': head & ~face_canvas, 'head': head, 'face': face_canvas,
            'beads': bead_centers(rgb, delta, face), 'delta': delta}


class Handler(http.server.SimpleHTTPRequestHandler):
    baseline = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        if not self.baseline:
            return super().do_GET()
        path = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        try:
            data = subprocess.check_output(['git', 'show', f'{BASELINE}:bundle/{path}'], cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def capture(poses, baseline):
    from playwright.sync_api import sync_playwright
    handler = type('SelectedHandler', (Handler,), {'baseline': baseline})
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(handler, directory=str(ROOT / 'bundle')))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    output = ROOT / 'debug/hair_h0' / ('r15' if baseline else 'current')
    output.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--use-angle=swiftshader'])
            page = browser.new_page(viewport={'width': 1100, 'height': 1600}, device_scale_factor=1)
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
            page.wait_for_function('window.studio?.ready', timeout=30000)
            page.add_style_tag(content='#stage{position:fixed!important;inset:0!important;width:1100px!important;height:1600px!important;max-width:none!important;max-height:none!important;}')
            geometry = {}
            for pose in poses:
                page.evaluate('pose=>{studio.setPose(pose.yaw,pose.pitch,0,"3D 구면");Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});Object.assign(studio.state,{auto:false,track:false});}', pose)
                groups = {'full': None, **GROUPS}
                if not baseline:
                    groups['stage-a-head'] = HAIR_CROWN + TAILS + TIES + ['face-skin']
                for name, ids in groups.items():
                    data = page.evaluate('''async({ids,attachment,tailIds,stageA})=>{const {renderer,p,state}=studio;
                     const {DOME}=await import('./dome-data.mjs');
                     const rear=DOME.rearPitch?.poses;
                     if(stageA)DOME.rearPitch.poses=[];
                     renderer.render(p,{...state,grid:false,wire:false,before:false,identify:ids||undefined,hidden:attachment?new Set([...state.hidden,...tailIds]):state.hidden});
                     const result={url:renderer.canvas.toDataURL(),view:renderer.view};
                     if(stageA)DOME.rearPitch.poses=rear;
                     return result;}''',
                     dict(ids=ids, attachment=name == 'attachment-hair', tailIds=TAILS + TIES, stageA=name == 'stage-a-head'))
                    assert data['view']['scale'] == 1, data['view']
                    rgba = Image.open(io.BytesIO(base64.b64decode(data['url'].split(',')[1]))).convert('RGBA')
                    ox, oy = data['view']['ox'], data['view']['oy']
                    assert ox == int(ox) and oy == int(oy)
                    rgba.crop((ox, oy, ox + 1024, oy + 740)).save(output / f'{pose["name"]}-{name}.png')
                if not pose['pitch']:
                    geometry[pose['name']] = page.evaluate('''async()=>{
                      const {deform3D}=await import('./rig.mjs'),{craniumFacing,craniumPosition,domeSection}=await import('./hair-surface.mjs');
                      const {rig,p}=studio;
                      const values=rig.meshes.map(m=>[m.id,m.positions.map(pt=>deform3D(m,pt,p)),m.id.startsWith('side-cranium')?m.positions.map(pt=>[craniumPosition(pt,p),craniumFacing(pt,p)]):null]);
                      values.push(Array.from({length:64},(_,i)=>domeSection(40+i*8)));
                      const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(values)));
                      return Array.from(new Uint8Array(bytes),v=>v.toString(16).padStart(2,'0')).join('');
                    }''')
            (output / 'geometry-sha256.json').write_text(json.dumps(geometry, indent=2) + '\n')
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    return output


def read_mask(path, threshold=200, alpha_threshold=40):
    a = np.array(Image.open(path).convert('RGBA'))
    return (a[:, :, 0] > threshold) & (a[:, :, 3] > alpha_threshold)


def sheets(rows, out):
    comparison = Image.new('RGB', (1080, len(rows) * 360), '#182330')
    overlay_sheet = Image.new('RGB', (1080, 1080), '#182330')
    portrait_sheet = Image.new('RGB', (1440, 1440), '#24303c')
    for i, row in enumerate(rows):
        ref, render, mask = row['ref'], row['render'], row['mask']
        portrait = Image.new('RGBA', (1024, 740), '#24303c')
        portrait.alpha_composite(Image.fromarray(render))
        tile = portrait.crop((152, 20, 872, 740)).convert('RGB').resize((480, 480))
        ImageDraw.Draw(tile).text((8, 8), row['name'], fill='white')
        portrait_sheet.paste(tile, ((i % 3) * 480, (i // 3) * 480))
        reference_hair = ref['rgb'].copy()
        reference_hair[~ref['mask']] = [24, 35, 48]
        actual = render[:, :, :3].copy()
        actual[~mask] = [24, 35, 48]
        over = np.zeros_like(actual) + [24, 35, 48]
        over[ref['mask']] = [40, 170, 220]
        over[mask] = [240, 110, 80]
        over[mask & ref['mask']] = [210, 210, 210]
        for j, (image, title) in enumerate([(reference_hair, 'reference'), (actual, 'render'), (over, 'overlay')]):
            tile = Image.fromarray(image.astype(np.uint8)).crop((152, 0, 872, 720)).resize((360, 360))
            ImageDraw.Draw(tile).text((6, 6), row['name'] + ' ' + title, fill='white')
            comparison.paste(tile, (j * 360, i * 360))
            if j == 2:
                overlay_sheet.paste(tile, ((i % 3) * 360, (i // 3) * 360))
    comparison.crop((0, 0, 1080, len(rows) * 360)).save(out / 'faceless_hair_9dir_comparison.png')
    overlay_sheet.save(out / 'hair_only_9dir_overlay.png')
    portrait_sheet.save(out / 'viewer_9dir_grid.png')


def measure(poses, current):
    rows, report = [], []
    for pose in poses:
        name = pose['name']
        ref = reference(pose)
        render = np.array(Image.open(current / f'{name}-full.png').convert('RGBA'))
        crown = read_mask(current / f'{name}-crown.png')
        mask = crown | read_mask(current / f'{name}-tails.png') | read_mask(current / f'{name}-ties.png')
        face = read_mask(current / f'{name}-face.png')
        head = fill_holes(mask | face)
        stage_a = fill_holes(read_mask(current / f'{name}-stage-a-head.png'))
        shifts = []
        for y in range(80, 521):
            a, b = span(head, y), span(stage_a, y)
            shifts.append({'y': y, 'shift': [a[i]-b[i] for i in [0, 1]] if a and b else None})
        top = int(np.where(head)[0].min()) if head.any() else 740
        exterior = []
        for a, b in zip(outline_rows(head, 520), outline_rows(ref['head'], 520)):
            exterior.append({'y': a['y'], 'render': a['span'], 'reference': b['span'],
                             'delta': [a['span'][k] - b['span'][k] for k in [0, 1]] if a['span'] and b['span'] else None})
        ties = {}
        tie_mask = read_mask(current / f'{name}-ties.png')
        stem_tie_mask = read_mask(current / f'{name}-ties.png', alpha_threshold=100)
        attachment_hair = read_mask(current / f'{name}-crown.png', 40) | read_mask(current / f'{name}-tails.png', 40)
        for side, (left, right) in {'left': (0, 502), 'right': (502, 1024)}.items():
            yy, xx = np.where(tie_mask[:, left:right])
            center = [float((np.percentile(v, 2) + np.percentile(v, 98)) / 2) + off
                      for v, off in [(xx, left), (yy, 0)]] if len(xx) else None
            target = ref['beads'][side]
            ties[side] = {**target, 'render': center,
                          'dBodyFrame': [center[k] - target['refFrame'][k] for k in [0, 1]] if center else None}
            if center:
                left_tie, right_tie = int(xx.min() + left), int(xx.max() + left)
                side_tie = stem_tie_mask.copy()
                side_tie[:, :left] = False
                side_tie[:, right:] = False
                ties[side]['hairOnlyStemGap'] = stem_gap(attachment_hair, side_tie)
                ties[side]['stemGap'] = stem_gap(attachment_hair | face, side_tie)
        neutral = None
        vertex_match = None
        if not pose['pitch']:
            neutral = neutral_difference(render, np.array(Image.open(ROOT / 'debug/hair_h0/r15' / f'{name}-full.png').convert('RGBA')))
            baseline_hash = json.loads((ROOT / 'debug/hair_h0/r15/geometry-sha256.json').read_text())
            current_hash = json.loads((current / 'geometry-sha256.json').read_text())
            vertex_match = baseline_hash[name] == current_hash[name]
        eye_y = int((pose['centers']['left'][1] + pose['centers']['right'][1]) / 2 - 20)
        hairline = hairline_metrics(ref['face'], face, eye_y)
        cover = None
        if pose['yaw']:
            neutral_name = 'left' if pose['yaw'] < 0 else 'right'
            near_side = 'right' if pose['yaw'] < 0 else 'left'
            baseline_dir = ROOT / 'debug/hair_h0/r15'
            baseline_rows = cheek_cover_rows(read_mask(baseline_dir / f'{neutral_name}-face.png'),
                                            read_mask(baseline_dir / f'{neutral_name}-crown.png', 40),
                                            near_side, range(500, 561))
            covered = [r['y'] for r in baseline_rows if r['gap'] == 0]
            cover = cheek_cover_rows(face, read_mask(current / f'{name}-crown.png', 40), near_side, covered)
        report.append({'name': name, 'yaw': pose['yaw'], 'pitch': pose['pitch'], 'registrationDelta': ref['delta'],
                       'topY': top, 'topGate': top_gate(pose['pitch'], top, int(np.where(ref['head'])[0].min()) if pose['yaw'] else 92) if pose['pitch'] else None,
                       'exterior': exterior, 'beads': ties, 'neutralChangedPixels': neutral,
                       'neutralVertexHashMatches': vertex_match, 'nearCheekCover': cover,
                       'posteriorProjectionShift': shifts, **hairline})
        rows.append({'name': name, 'ref': ref, 'render': render, 'mask': mask})
    return rows, report


def assess(poses):
    checks = []
    expected = {(yaw, pitch) for yaw in [-45, 0, 45] for pitch in [-30, 0, 30]}
    checks.append({'gate': 'nine poses', 'pass': {(p['yaw'], p['pitch']) for p in poses} == expected})
    for p in poses:
        name, yaw, pitch = p['name'], p['yaw'], p['pitch']
        def check(gate, passed, value):
            missing = isinstance(value, float) and not math.isfinite(value)
            item = {'pose': name, 'gate': gate, 'pass': bool(passed), 'value': None if missing else value}
            if missing:
                item['reason'] = 'Missing rendered or reference span; see exterior rows'
            checks.append(item)
        if not pitch:
            check('neutral pixels', p['neutralChangedPixels'] == 0, p['neutralChangedPixels'])
            check('neutral vertices and sections', p['neutralVertexHashMatches'] is True, p['neutralVertexHashMatches'])
        if p['topGate']:
            check('crown top', p['topGate']['pass'], p['topY'])
        approved_top = approved_top_gate(yaw, pitch, p['topY'])
        if approved_top is not None:
            check('approved crown top', approved_top['pass'], p['topY'])
        if pitch and not yaw:
            selected = [r for r in p['exterior'] if 100 <= r['y'] <= 300]
            error = max((max(abs(v) for v in r['delta']) if r['delta'] else float('inf') for r in selected), default=float('inf'))
            check('upper exterior', error <= (12 if pitch > 0 else 15), error)
        if pitch:
            median = p['hairlineMedianError']
            check('hairline median', median is not None and abs(median) <= 12, median)
            area = p['foreheadExposure']
            ratio = area['ratioError'] if area else None
            check('forehead exposure', ratio is not None and abs(ratio) <= .1, ratio)
        if yaw and pitch:
            rear = [r for r in p['exterior'] if (80 <= r['y'] <= 320 if pitch < 0 else 340 <= r['y'] <= 520)]
            side = 1 if yaw < 0 else 0
            error = max((abs(r['delta'][side]) if r['delta'] else float('inf') for r in rear), default=float('inf'))
            check('posterior exterior', error <= 12, error)
            if pitch < 0:
                cover = p['nearCheekCover']
                check('near cheek background gap', bool(cover) and all(r['gap'] == 0 for r in cover), cover)
        for side, bead in p['beads'].items():
            delta = bead['dBodyFrame']
            inherited = side == 'right' and name in ['up-right', 'down-left', 'down-right']
            check(side + ' bead body frame', delta is not None and max(abs(v) for v in delta) <= (20 if inherited else 8), delta)
            if pitch:
                gap = bead.get('stemGap')
                check(side + ' foreground stem gap', gap is not None and gap <= 15, gap)
    return checks


def summarize_metrics(data):
    """Keep release evidence small; full per-row samples belong in debug/."""
    result = json.loads(json.dumps(data))
    for p in result['poses']:
        exterior = p['exterior']
        p['exterior'] = {'sampledRows': len(exterior), 'maxAbsoluteError': max(
            (abs(v) for r in exterior if r['delta'] for v in r['delta']), default=None)}
        shifts = p['posteriorProjectionShift']
        worst = max((r for r in shifts if r['shift']), key=lambda r: max(abs(v) for v in r['shift']), default=None)
        p['posteriorProjectionShift'] = {'sampledRows': len(shifts), 'worstRow': worst,
                                       'maximum': max(abs(v) for v in worst['shift']) if worst else None}
        cover = p['nearCheekCover']
        p['nearCheekCover'] = {'sampledRows': len(cover), 'maximumGap': max((r['gap'] for r in cover), default=None)} if cover is not None else None
        p['hairline'] = {'sampleCount': len(p['hairline']), 'medianError': p['hairlineMedianError']}
    for c in result['checks'] + result.get('rawResidualFailures', []):
        if c['gate'] == 'near cheek background gap':
            rows = c['value'] or []
            c['value'] = {'sampledRows': len(rows), 'maximumGap': max((r['gap'] for r in rows), default=None)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output', default='docs', help='Directory for this measurement snapshot')
    args = parser.parse_args()
    features = json.loads((ROOT / 'source/viewer/face-feature-fit.json').read_text())
    poses = sorted(features['poses'], key=lambda p: (-p['pitch'], p['yaw']))
    current = capture([p for p in poses if not p['pitch']] if args.baseline else poses, args.baseline)
    if args.baseline:
        print('PASS r15 neutral baseline captures (3 poses, scale=1)')
        return
    rows, report = measure(poses, current)
    out = ROOT / args.output / 'captures'
    out.mkdir(parents=True, exist_ok=True)
    sheets(rows, out)
    checks = assess(report)
    bounds = json.loads(subprocess.check_output(['node', 'pipeline/hair_bounds.mjs'], cwd=ROOT, text=True))
    projection_shift = max(abs(v) for p in report for r in p['posteriorProjectionShift'] if r['shift'] for v in r['shift'])
    bounds['checks'].append({'gate': 'H2 raster exterior shift from zero-supplement stage a <=45px',
                             'pass': projection_shift <= 45, 'value': projection_shift})
    bounds['pass'] = all(c['pass'] for c in bounds['checks'])
    protected = [line.split('  ', 1)[1] for line in (ROOT / 'docs/hair-hspec-protected-sha256.txt').read_text().splitlines()]
    protected_pass = subprocess.run(['git', 'diff', '--quiet', BASELINE, '--', *protected], cwd=ROOT).returncode == 0
    invariant_gates = {'nine poses', 'neutral pixels', 'neutral vertices and sections',
                       'left foreground stem gap', 'right foreground stem gap', 'approved crown top'}
    invariants = [c for c in checks if c['gate'] in invariant_gates]
    invariant_pass = all(c['pass'] for c in invariants)
    strict_pass = all(c['pass'] for c in checks)
    accepted = bounds['pass'] and invariant_pass and protected_pass
    data = {'baseline': BASELINE, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'registrationSha256': hashlib.sha256((ROOT / 'source/viewer/face-feature-fit.json').read_bytes()).hexdigest(),
            'referenceMaskMethod': 'Fable probe/analyze2: frozen SAM face, foreground-minus-skin/white/blue, filled head silhouette; diagnostic segmentation',
            'canvas': [1100, 1600], 'scale': 1, 'hairMeshCount': 15, 'poses': report, 'checks': checks,
            'productSha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT / 'bundle' / name for name in ['hair-shell.mjs', 'hair-surface.mjs', 'dome-data.mjs', 'stereo-attachments.mjs']]},
            'strictRasterPass': strict_pass, 'bounds': bounds, 'invariantPass': invariant_pass,
            'invariants': invariants, 'rawResidualFailures': [c for c in checks if not c['pass'] and c['gate'] not in invariant_gates],
            'stemMaskMethod': 'visible crown | tails | face-skin; tie R>200/alpha>100, x percentile20..80 step2, empty pixels from row before tie',
            'protectedFilesPass': protected_pass, 'protectedFileCount': len(protected),
            'acceptance': 'strict pass' if strict_pass and accepted else 'bounded pass' if accepted else 'fail',
            'pass': accepted}
    (current / 'hair-r15-metrics-raw.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (ROOT / args.output / 'hair-r15-metrics.json').write_text(json.dumps(summarize_metrics(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{'name': p['name'], 'topY': p['topY'], 'neutralChangedPixels': p['neutralChangedPixels'],
                       'beadY': {s: b['dBodyFrame'][1] if b['dBodyFrame'] else None for s, b in p['beads'].items()}}
                      for p in report]))
    failed = [c for c in checks if not c['pass']]
    print(json.dumps({'pass': accepted, 'acceptance': data['acceptance'], 'rawResidualFailures': failed,
                      'bounds': bounds}, ensure_ascii=False))
    assert accepted, 'Hair-only invariant or bounded-model gate failed; inspect docs/hair-r15-metrics.json'


if __name__ == '__main__':
    main()
