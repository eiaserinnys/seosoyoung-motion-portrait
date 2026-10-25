"""Hair-only WebGL/reference measurements in the frozen r15 registration frame.
Run --baseline to capture nine r15 poses without changing checkout.
Non-target meshes remain black occluders. Reference segmentation is diagnostic.
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
from hair_solve_quality import outline_rows,top_gate,stem_gap,cheek_cover_rows,neutral_difference,span,hairline_metrics
from PIL import Image
if __package__:
    from .hair_report import sheets, summarize_metrics, write_release_report
    from .hair_revision_metrics import neutral_silhouette, rimlight_metrics, posterior_metrics, posterior_error_gate, posterior_shift_gate, protected_files_match, approved_top_gate
else:
    from hair_report import sheets, summarize_metrics, write_release_report
    from hair_revision_metrics import neutral_silhouette, rimlight_metrics, posterior_metrics, posterior_error_gate, posterior_shift_gate, protected_files_match, approved_top_gate

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '0b75a0211691965ca234ad9114073ba934568394'
HAIR_CROWN = ['hair-top', 'bangs-left', 'bangs-right', 'side-hair-left',
              'side-hair-right', 'hairpin', 'back-hair-left', 'back-hair-right',
              'back-hair-center', 'side-cranium-left', 'side-cranium-right']
TAILS = ['tail-left-hair', 'tail-right-hair']
TIES = ['tail-left-tie', 'tail-right-tie']
GROUPS = {'crown': HAIR_CROWN, 'tails': TAILS, 'ties': TIES, 'face': ['face-skin'], 'attachment-hair': HAIR_CROWN}


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
    dense = False
    solve = True

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        if not self.solve and path=='hair-solve-data.mjs':
            value=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())
            for pose in value['poses']:pose['du']=[[0,0]]*192
            value.pop('lower',None)
            data=('export const HAIR_SOLVE='+json.dumps(value,ensure_ascii=False)+';').encode()
            self.send_response(200);self.send_header('Content-Type','text/javascript')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            return
        if not self.baseline and not (self.dense and path == 'assets/seosoyoung-rig.json'):
            return super().do_GET()
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


def capture(poses, baseline, dense=False,solve=True):
    from playwright.sync_api import sync_playwright
    handler = type('SelectedHandler', (Handler,), {'baseline': baseline, 'dense': dense,'solve':solve})
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(handler, directory=str(ROOT / 'bundle')))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    output = ROOT / 'debug/hair_h0' / ('r15' if baseline else 'dense' if dense else 'current' if solve else 'solve-prior')
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
            geometry, mixes = {}, {}
            for pose in poses:
                page.evaluate('pose=>{studio.setPose(pose.yaw,pose.pitch,0,"3D 구면");Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});Object.assign(studio.state,{auto:false,track:false});}', pose)
                groups = {'full': None, **GROUPS}
                if not baseline:
                    groups['stage-a-head'] = HAIR_CROWN + TAILS + TIES + ['face-skin']
                for name, ids in groups.items():
                    data = page.evaluate('''async({ids,attachment,tailIds,stageA})=>{const {renderer,p,state}=studio;
                     const {DOME}=await import('./dome-data.mjs');
                     const rear=DOME.rearPitch?.poses,lower=DOME.rearLower?.poses;
                     if(stageA){DOME.rearPitch.poses=[];if(lower)DOME.rearLower.poses=[];}
                     renderer.render(p,{...state,grid:false,wire:false,before:false,identify:ids||undefined,hidden:attachment?new Set([...state.hidden,...tailIds]):state.hidden});
                     const result={url:renderer.canvas.toDataURL(),view:renderer.view};
                     if(stageA){DOME.rearPitch.poses=rear;if(lower)DOME.rearLower.poses=lower;}
                     return result;}''',
                     dict(ids=ids, attachment=name == 'attachment-hair', tailIds=TAILS + TIES, stageA=name == 'stage-a-head'))
                    assert data['view']['scale'] == 1, data['view']
                    rgba = Image.open(io.BytesIO(base64.b64decode(data['url'].split(',')[1]))).convert('RGBA')
                    ox, oy = data['view']['ox'], data['view']['oy']
                    assert ox == int(ox) and oy == int(oy)
                    rgba.crop((ox, oy, ox + 1024, oy + 740)).save(output / f'{pose["name"]}-{name}.png')
                if not pose['pitch']:
                    mixes[pose['name']] = page.evaluate('''async()=>{
                      const {referenceHairSample}=await import('./hair-surface.mjs');
                      const weights=studio.rig.meshes.flatMap(m=>m.positions.map(pt=>referenceHairSample(pt,studio.p).weight));
                      return Math.max(...weights);
                    }''')
                    geometry[pose['name']] = page.evaluate('''async()=>{
                      const {deform3D}=await import('./rig.mjs'),{craniumFacing,craniumPosition,domeSection}=await import('./hair-surface.mjs');
                      const {rig,p}=studio;
                      const values=rig.meshes.filter(m=>/^(hair-top|bangs|side-hair|hairpin|back-hair|side-cranium|tail-)/.test(m.id)).map(m=>[m.id,m.positions.map(pt=>deform3D(m,pt,p)),m.id.startsWith('side-cranium')?m.positions.map(pt=>[craniumPosition(pt,p),craniumFacing(pt,p)]):null]);
                      values.push(Array.from({length:64},(_,i)=>domeSection(40+i*8)));
                      const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(values)));
                      return Array.from(new Uint8Array(bytes),v=>v.toString(16).padStart(2,'0')).join('');
                    }''')
            (output / 'geometry-sha256.json').write_text(json.dumps(geometry, indent=2) + '\n')
            (output / 'reference-mix.json').write_text(json.dumps(mixes) + '\n')
            if baseline:
                (output / 'baseline.txt').write_text(BASELINE)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    return output


def read_mask(path, threshold=200, alpha_threshold=40):
    a = np.array(Image.open(path).convert('RGBA'))
    return (a[:, :, 0] > threshold) & (a[:, :, 3] > alpha_threshold)


def measure(poses, current):
    rows, report = [], []
    for pose in poses:
        name = pose['name']
        ref = reference(pose)
        render = np.array(Image.open(current / f'{name}-full.png').convert('RGBA'))
        crown = read_mask(current / f'{name}-crown.png')
        mask = crown | read_mask(current / f'{name}-tails.png') | read_mask(current / f'{name}-ties.png')
        dense_dir = ROOT / 'debug/hair_h0/dense'
        dense_mask = read_mask(dense_dir / f'{name}-crown.png') | read_mask(dense_dir / f'{name}-tails.png') | read_mask(dense_dir / f'{name}-ties.png')
        reduction_spans = {'all': neutral_silhouette(dense_mask, mask), **{group: neutral_silhouette(read_mask(dense_dir / f'{name}-{group}.png'), read_mask(current / f'{name}-{group}.png')) for group in ['crown', 'tails', 'ties']}}
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
        neutral, neutral_spans, rimlight, posterior = None, None, None, None
        baseline_dir = ROOT / 'debug/hair_h0/r15'
        vertex_match = None
        if not pose['pitch']:
            neutral = neutral_difference(render, np.array(Image.open(ROOT / 'debug/hair_solve_prior' / f'{name}-full.png').convert('RGBA')))
            baseline_hash = json.loads((ROOT / 'debug/hair_h0/r15/geometry-sha256.json').read_text())
            current_hash = json.loads((current / 'geometry-sha256.json').read_text())
            vertex_match = baseline_hash[name] == current_hash[name]
            neutral_spans = {group: neutral_silhouette(read_mask(baseline_dir / f'{name}-{group}.png'), read_mask(current / f'{name}-{group}.png')) for group in ['crown', 'tails', 'ties']}
            old_mask = read_mask(baseline_dir / f'{name}-crown.png') | read_mask(baseline_dir / f'{name}-tails.png') | read_mask(baseline_dir / f'{name}-ties.png')
            neutral_spans['all'] = neutral_silhouette(old_mask, mask)
        if pose['yaw'] and pose['pitch'] >= 0:
            old_render = np.array(Image.open(baseline_dir / f'{name}-full.png').convert('RGBA'))
            old_crown = read_mask(baseline_dir / f'{name}-crown.png')
            rimlight = rimlight_metrics(old_render, render, old_crown, crown, pose['yaw'])
            if pose['pitch']:
                old_head = fill_holes(old_crown | read_mask(baseline_dir / f'{name}-tails.png') | read_mask(baseline_dir / f'{name}-ties.png') | read_mask(baseline_dir / f'{name}-face.png'))
                posterior = posterior_metrics(ref['head'], head, old_head, pose['yaw'])
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
            cover = cheek_cover_rows(face, read_mask(current / f'{name}-crown.png', 40), near_side, range(480, 546) if pose['pitch'] > 0 else [y for y in covered if y<=545])
        report.append({'name': name, 'yaw': pose['yaw'], 'pitch': pose['pitch'], 'registrationDelta': ref['delta'],
                       'topY': top, 'topGate': top_gate(pose['pitch'], top, int(np.where(ref['head'])[0].min()) if pose['yaw'] else 92) if pose['pitch'] else None,
                       'exterior': exterior, 'beads': ties, 'neutralChangedPixels': neutral,
                       'neutralVertexHashMatches': vertex_match, 'nearCheekCover': cover,
                       'neutralSilhouette': neutral_spans, 'templeRimlight': rimlight, 'posteriorVolume': posterior,
                       'retriangulationSilhouette': reduction_spans,
                       'referenceMixMaximum': json.loads((current / 'reference-mix.json').read_text()).get(name),
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
        reduction = p['retriangulationSilhouette']
        check('P3 silhouette span <=2px', bool(reduction) and reduction['all']['pass'], reduction)
        if not pitch:
            silhouette = p['neutralSilhouette']
            check('neutral silhouette span <=2px', bool(silhouette) and silhouette['all']['pass'], silhouette)
            check('neutral pixel equality',p['neutralChangedPixels']==0,p['neutralChangedPixels'])
            if not yaw:
                check('front reference mix ==0', p['referenceMixMaximum'] == 0, p['referenceMixMaximum'])
        if yaw and pitch >= 0:
            rim = p['templeRimlight']
            check('temple rimlight reduction >=50%', rim is not None and rim['pass'], rim)
            if pitch:
                volume = p['posteriorVolume']
                check('upper diagonal posterior volume', volume is not None and volume['pass'], volume)
                error = posterior_error_gate(yaw, volume.get('meanAbsoluteDelta') if volume else None)
                check('upper diagonal posterior mean error', error['pass'], error)
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
            exposure_limit = .15 if yaw and pitch > 0 else .10
            check('forehead exposure', ratio is not None and abs(ratio) <= exposure_limit, ratio)
            if yaw and pitch > 0:
                check('upper diagonal forehead exposure', ratio is not None and abs(ratio) <= exposure_limit, ratio)
        if yaw and pitch:
            rear = [r for r in p['exterior'] if (80 <= r['y'] <= 320 if pitch < 0 else 340 <= r['y'] <= 520)]
            side = 1 if yaw < 0 else 0
            error = max((abs(r['delta'][side]) if r['delta'] else float('inf') for r in rear), default=float('inf'))
            check('posterior exterior', error <= 12, error)
            if pitch < 0:
                cover = p['nearCheekCover']
                check('near cheek background gap', bool(cover) and all(r['gap'] <= 2 for r in cover), cover)
        for side, bead in p['beads'].items():
            delta = bead['dBodyFrame']
            inherited = side == 'right' and name in ['up-right', 'down-left', 'down-right']
            check(side + ' bead body frame', delta is not None and max(abs(v) for v in delta) <= (20 if inherited else 8), delta)
            if pitch:
                gap = bead.get('stemGap')
                check(side + ' foreground stem gap', gap is not None and gap <= 17, gap)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--measure-only', action='store_true', help='Reassess saved frames without opening a browser')
    parser.add_argument('--output', default='docs', help='Directory for this measurement snapshot')
    parser.add_argument('--capture-unobserved',action='store_true')
    parser.add_argument('--capture-only',choices=['current','dense'])
    parser.add_argument('--comparison-base',default='4dba92d65c74d1ac13dfa31ac791bf2f568c2140')
    args = parser.parse_args()
    if args.capture_unobserved:
        import shutil
        import capture_vdd
        # Keep the existing capture harness's other outputs inside debug/.
        scratch=ROOT/'debug/vdd-split';(scratch/'docs/captures').mkdir(parents=True,exist_ok=True)
        if not (scratch/'bundle').exists():(scratch/'bundle').symlink_to(ROOT/'bundle',target_is_directory=True)
        capture_vdd.ROOT=scratch;capture_vdd.BASE=args.comparison_base
        capture_vdd.main()
        shutil.copyfile(scratch/'docs/captures/vdd_unobserved_angles_sheet.png',ROOT/args.output/'captures/vdd_unobserved_angles_sheet.png')
        return
    features = json.loads((ROOT / 'source/viewer/face-feature-fit.json').read_text())
    poses = sorted(features['poses'], key=lambda p: (-p['pitch'], p['yaw']))
    if args.capture_only:
        capture(poses,False,dense=args.capture_only=='dense')
        print('PASS captured '+args.capture_only);return
    baseline_dir = ROOT / 'debug/hair_h0/r15'
    stamp = baseline_dir / 'baseline.txt'
    if not args.measure_only and (args.baseline or not stamp.exists() or stamp.read_text() != BASELINE):
        capture(poses, True)
    if args.baseline:
        print('PASS frozen baseline captures (9 poses, scale=1)')
        return
    if not args.measure_only:
        capture(poses, False, dense=True)
        capture(poses, False)
    current = ROOT / 'debug/hair_h0/current'
    rows, report = measure(poses, current)
    out = ROOT / args.output / 'captures'
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    generated=current/'sheets';generated.mkdir(exist_ok=True)
    sheets(rows, generated)
    for name in ['faceless_hair_9dir_comparison.png','hair_only_9dir_overlay.png']:
        shutil.copyfile(generated/name,out/name)
    checks = assess(report)
    legacy_checks=checks
    target_gates={'crown top','upper exterior','hairline median','forehead exposure','posterior exterior',
                  'near cheek background gap','left foreground stem gap','right foreground stem gap'}
    neutral_gates={'nine poses','P3 silhouette span <=2px','neutral silhouette span <=2px','neutral pixel equality','front reference mix ==0'}
    checks=[c for c in checks if c['gate'] in target_gates|neutral_gates]
    bounds = json.loads(subprocess.check_output(['node', 'pipeline/hair_bounds.mjs'], cwd=ROOT, text=True))
    projection_shift = max(abs(v) for p in report for r in p['posteriorProjectionShift'] if r['shift'] for v in r['shift'])
    bounds['checks'].append(posterior_shift_gate(projection_shift))
    bounds['pass'] = all(c['pass'] for c in bounds['checks'])
    protected = [line.split('  ', 1) for line in (ROOT / 'docs/hair-hspec-protected-sha256.txt').read_text().splitlines()]
    protected_pass = protected_files_match(ROOT, protected)
    invariant_gates = neutral_gates|{'left foreground stem gap','right foreground stem gap'}
    invariants = [c for c in checks if c['gate'] in invariant_gates]
    invariant_pass = all(c['pass'] for c in invariants)
    strict_pass = all(c['pass'] for c in checks)
    chart=json.loads(subprocess.check_output(['node','pipeline/hair_solve_metrics.mjs'],cwd=ROOT,text=True,timeout=60))
    solution=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())
    budget=all(p['stats']['duRms']<=45 and p['stats']['duMax']<=115 for p in solution['poses'])
    neutral_pass=all(p['neutralVertexHashMatches'] for p in report if not p['pitch'])
    from hair_solve_quality import compare_captures,compare_split_captures
    comparison=compare_captures(poses,ROOT/'debug/hair_solve_prior',current)
    split=compare_split_captures(poses,ROOT/'debug/hair_solve_prior',current)
    split_pass=all(p['pass'] for domain in split.values() for p in domain['poses'] if p['name'] in {s['name'] for s in solution['poses']})
    accepted = chart['pass'] and budget and neutral_pass and invariant_pass and comparison['pass'] and split_pass and protected_pass
    data = {'baseline': BASELINE, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'registrationSha256': hashlib.sha256((ROOT / 'source/viewer/face-feature-fit.json').read_bytes()).hexdigest(),
            'referenceMaskMethod': 'Fable probe/analyze2: frozen SAM face, foreground-minus-skin/white/blue, filled head silhouette; diagnostic segmentation',
            'canvas': [1100, 1600], 'scale': 1, 'hairMeshCount': 15, 'poses': report, 'checks': checks,'legacyChecks':legacy_checks,
            'productSha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT / 'bundle' / name for name in ['hair-shell.mjs', 'hair-surface.mjs', 'dome-data.mjs', 'hair-solve.mjs','hair-solve-data.mjs','stereo-attachments.mjs', 'rig.mjs', 'renderer.mjs', 'assets/seosoyoung-rig.json']]},
            'hairSolveChart':chart,'hairSolveBudgetPass':budget,'neutralGeometryPass':neutral_pass,'priorComparison':comparison,'splitComparison':split,
            'neutralPixelBaseline':'same unchanged face calibration and r15 shell with both upper/lower residuals disabled (debug/hair_solve_prior)',
            'measurementMode': 'saved frames' if args.measure_only else 'single browser capture per variant',
            'measurementApproval': 'unchanged r15 prior; upper 12x16 chart restY515..545 fade, duRms<=45px/duMax<=115px; independent bead-local lower span chart; both minDet>=.25; neutral exact; stem<=17px',
            'probeSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'strictRasterPass': strict_pass, 'allChecksPass': strict_pass, 'bounds': bounds, 'invariantPass': invariant_pass,
            'invariants': invariants, 'rawResidualFailures': [c for c in checks if not c['pass'] and c['gate'] in target_gates],
            'stemMaskMethod': 'visible crown | tails | face-skin; tie R>200/alpha>100, x percentile20..80 step2, empty pixels from row before tie',
            'protectedFilesPass': protected_pass, 'protectedFileCount': len(protected),
            'acceptance': 'strict pass' if strict_pass and accepted else 'monotone prior pass' if accepted else 'fail',
            'pass': accepted}
    (current / 'hair-r15-metrics-raw.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (ROOT / args.output / 'hair-r15-metrics.json').write_text(json.dumps(summarize_metrics(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    write_release_report(ROOT, args.output, summarize_metrics(data))
    print(json.dumps([{'name': p['name'], 'topY': p['topY'], 'neutralChangedPixels': p['neutralChangedPixels'],
                       'beadY': {s: b['dBodyFrame'][1] if b['dBodyFrame'] else None for s, b in p['beads'].items()}}
                      for p in report]))
    failed = [c for c in checks if not c['pass']]
    print(json.dumps({'pass': accepted, 'acceptance': data['acceptance'], 'rawResidualFailures': failed,
                      'bounds': bounds}, ensure_ascii=False))
    assert accepted, 'Hair-only invariant or bounded-model gate failed; inspect docs/hair-r15-metrics.json'


if __name__ == '__main__':
    main()
