#!/usr/bin/env python3
"""Run headless WebGL checks and create review captures for the face viewer."""
import base64
import gzip
import json
import os
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / 'docs/captures'
STILL = ROOT / 'source/character/still.png'
KEYS = {
    'front': (0, 0), 'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1),
    'up-left': (-1, 1), 'up-right': (1, 1), 'down-left': (-1, -1), 'down-right': (1, -1),
}
OPENS = [1, .7, .4, .27, .25, .23, .21, 0]

sys_path = str(ROOT / 'pipeline')
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)
from keyshape_rig_runtime import Rig, holes, pose


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def to_rgba8(float_image):
    color = (float_image * 255).astype(np.float32)
    alpha = color[..., 3:4] / 255
    alpha = np.where(alpha >= 254.5 / 255, 1.0, alpha).astype(np.float32)
    rgb = np.where(alpha > 0, color[..., :3] / np.maximum(alpha, 1e-6), 0)
    result = np.clip(np.concatenate([rgb, np.minimum(color[..., 3:4], 255)], -1) + .5, 0, 255).astype(np.uint8)
    result[result[..., 3] == 0] = 0
    return result


def save_rgba(path, array):
    Image.fromarray(array, 'RGBA').save(path)


def capture_float(page, state, hair, hidden=None):
    payload = page.evaluate("""async ({pose, hair, hidden}) => {
      window.viewer.render(pose, {hair, hidden: hidden || []});
      const result = window.viewer.readPixels();
      const raw = new Uint8Array(result.data.buffer);
      const compressed = await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();
      const bytes = new Uint8Array(compressed); let encoded = '';
      for (let i = 0; i < bytes.length; i += 32768) encoded += String.fromCharCode(...bytes.subarray(i, i + 32768));
      return {float32: result.float32, data: btoa(encoded), glError: window.viewer.renderer.gl.getError()};
    }""", {'pose': state, 'hair': hair, 'hidden': hidden})
    if not payload['float32']:
        raise RuntimeError('검증 환경에서 RGBA32F 대상을 만들지 못했습니다')
    raw = gzip.decompress(base64.b64decode(payload['data']))
    image = np.frombuffer(raw, np.float32).reshape(1536, 1024, 4)[::-1].copy()
    if payload['glError']:
        raise RuntimeError(f"WebGL 오류: {payload['glError']}")
    return to_rgba8(image)


def capture_png(page, state, hair=False, hidden=None):
    encoded = page.evaluate("""({pose, hair, hidden}) => {
      window.viewer.render(pose, {hair, hidden: hidden || []});
      return document.querySelector('#stage').toDataURL('image/png').split(',')[1];
    }""", {'pose': state, 'hair': hair, 'hidden': hidden})
    return np.asarray(Image.open(BytesIO(base64.b64decode(encoded))).convert('RGBA'))


def full_state(u, v, open_value=1):
    return {'u': u, 'v': v, 'roll': 0, 'openL': open_value, 'openR': open_value,
            'gazeX': 0, 'gazeY': 0, 'breath': 0, 'bodyAngleX': 0}


def difference(actual, reference, outside_band=None):
    delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    visible = (actual[..., 3] > 0) | (reference[..., 3] > 0)
    alpha = delta[..., 3]
    value = delta[..., :3].max(-1)
    total_visible = max(1, int(visible.sum()))
    result = {
        'different_pixels': int((delta.max(-1) > 0).sum()),
        'max_channel_delta': int(delta.max()),
        'value_over_2': int((value > 2).sum()),
        'value_over_2_percent_visible': round(float((value[visible] > 2).mean() * 100), 4),
        'alpha_over_8': int((alpha > 8).sum()),
        'visible_pixels': total_visible,
    }
    if outside_band is not None:
        result['outside_edge_band_value_over_2'] = int(((value > 2) & ~outside_band).sum())
    return result


def silhouette_edge_band(reference, state):
    alpha = reference.silhouette_layer(state)[..., 3]
    return ndi.binary_dilation((alpha > 0) & (alpha < 255), structure=np.ones((3, 3), bool))


def nonfront_passes(metrics):
    return (metrics['value_over_2_percent_visible'] <= .6
            and metrics['alpha_over_8'] <= 110
            and metrics['outside_edge_band_value_over_2'] <= 4000)


def holes_within_limits(pixels, pieces, limits):
    return pixels <= limits[0] and (len(limits) == 1 or pieces <= limits[1])


def geometry_report(page, reference):
    payload = page.evaluate("""() => {
      const rig = window.viewer.rig, result = [], keys = rig.data.keys;
      const area = (p, t) => (p[t[1]][0]-p[t[0]][0])*(p[t[2]][1]-p[t[0]][1])-(p[t[1]][1]-p[t[0]][1])*(p[t[2]][0]-p[t[0]][0]);
      const base = new Map(rig.meshes.map(m => [m.id, m.triangles.map(t => Math.sign(area(m.positions, t)))]));
      for (const [key, [u,v]] of Object.entries(keys)) {
        const row = {key, visible: {}, clipped: {}};
        const p = {u,v,roll:0,openL:1,openR:1,gazeX:0,gazeY:0,breath:0,bodyAngleX:0,hair:true};
        for (const mesh of rig.meshes) {
          if (mesh.id === 'skull-plate' || rig.opacity(mesh,p) <= 0) continue;
          const positions = rig.positions(mesh,p), signs = base.get(mesh.id), clipped = rig.data.clipParts.includes(mesh.id);
          let count = 0; const centers = [];
          for (let i=0; i<mesh.triangles.length; i++) {
            const tri = mesh.triangles[i], value = area(positions,tri);
            if (Math.sign(value) === signs[i] && Math.abs(value) >= 1e-6) continue;
            count++;
            if (clipped) centers.push([(positions[tri[0]][0]+positions[tri[1]][0]+positions[tri[2]][0])/3,
                                       (positions[tri[0]][1]+positions[tri[1]][1]+positions[tri[2]][1])/3]);
          }
          if (!count) continue;
          if (clipped) row.clipped[mesh.id] = centers;
          else row.visible[mesh.id] = count;
        }
        result.push(row);
      }
      const eyes = [];
      for (const [key, [u,v]] of Object.entries(keys)) for (const open of [1,.7,.4,.27,.25,.23,.21,0]) {
        const p = {u,v,roll:0,openL:open,openR:open,gazeX:0,gazeY:0,breath:0,bodyAngleX:0,hair:true};
        let count=0;
        for (const mesh of rig.meshes) if (mesh.group === 'eye' && rig.opacity(mesh,p) > 0) {
          const positions=rig.positions(mesh,p),signs=base.get(mesh.id);
          for (let i=0;i<mesh.triangles.length;i++) { const value=area(positions,mesh.triangles[i]); if (Math.sign(value)!==signs[i]||Math.abs(value)<1e-6) count++; }
        }
        eyes.push({key,open,count});
      }
      return {rows:result,eyes};
    }""")
    old_checks = json.loads((ROOT / 'source/keyshapes/v6-checks.json').read_text())
    hair = set(reference.rig['hair'])
    all_counts, eye_counts, failures = {}, payload['eyes'], []
    for row in payload['rows']:
        mask = reference.clip_mask(pose(*KEYS[row['key']], hair=False))
        clipped_counts = {}
        for mesh_id, centers in row['clipped'].items():
            if not centers:
                continue
            xy = np.asarray(centers)
            x = np.clip(np.round(xy[:, 0]).astype(int), 0, 1023)
            y = np.clip(np.round(xy[:, 1]).astype(int), 0, 1535)
            count = int((mask[y, x] >= .5).sum())
            if count:
                clipped_counts[mesh_id] = count
        actual = {**row['visible'], **clipped_counts}
        key = row['key']
        expected = old_checks['flips_visible'].get(key, {})
        filtered = {name: count for name, count in actual.items()
                    if name not in hair and not name.startswith('still-fill-hair-')}
        all_counts[key] = filtered
        for name, count in filtered.items():
            if count > expected.get(name, 0):
                failures.append(f'{key}/{name}: {count}>{expected.get(name, 0)}')
    for row in eye_counts:
        if row['count']:
            failures.append(f"{row['key']}/{row['open']}: eye flips {row['count']}")
    return {'visible_flips': all_counts, 'eye_flips': eye_counts, 'failures': failures}


def skin_spill(page, reference, targets):
    ids = {mesh['id'] for mesh in reference.rig['meshes']}
    keep = {'face-skin', 'face-scalp'}
    hidden = sorted(ids - keep)
    result = {}
    for key, (u, v) in KEYS.items():
        state = full_state(u, v)
        image = capture_png(page, state, hair=False, hidden=hidden)
        silhouette_alpha = reference.silhouette_layer(pose(u, v, hair=False))[..., 3]
        target = targets[key]
        brow_y = min(target['brow_l_arch'][1], target['brow_r_arch'][1])
        outside = (image[..., 3] >= 128) & (silhouette_alpha < 128)
        outside[int(brow_y):, :] = False
        labels, count = ndi.label(outside)
        pieces = [int((labels == i + 1).sum()) for i in range(count)]
        result[key] = {'pixels': int(sum(size for size in pieces if size >= 20)),
                       'pieces': sorted((size for size in pieces if size >= 20), reverse=True)[:3]}
    return result


def verify_ui(page, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=output_dir / 'ui-wide.png', full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    page.screenshot(path=output_dir / 'ui-narrow.png', full_page=True)
    narrow_width = page.evaluate('document.documentElement.scrollWidth')
    auto = page.locator('#auto')
    auto.click()
    auto_active = auto.get_attribute('aria-pressed') == 'true'
    auto.click()
    track = page.locator('#track')
    track.click()
    box = page.locator('#viewport').bounding_box()
    page.mouse.move(box['x'] + box['width'] * .75, box['y'] + box['height'] * .25)
    tracked = page.evaluate('window.viewer.pose.u')
    track.click()
    page.locator('#reset').click()
    return {'narrow_scroll_width': narrow_width, 'viewport_width': 390,
            'auto_toggle': auto_active, 'pointer_tracking_u': round(tracked, 4),
            'reset_u': page.evaluate('window.viewer.pose.u')}


def verify_atlas(reference):
    data = reference.rig
    atlas = Image.open(ROOT / 'bundle/assets/texture_00.png').convert('RGBA')
    atlas_array = np.asarray(atlas)
    pixel_mismatches, uv_errors = 0, 0
    for mesh in data['meshes']:
        ax, ay, width, height = mesh['atlas']
        source = np.asarray(Image.open(ROOT / 'bundle/assets' / mesh['file']).convert('RGBA'))
        packed = atlas_array[ay:ay + height, ax:ax + width]
        pixel_mismatches += int(np.any(source != packed, axis=-1).sum())
        bx, by, bw, bh = mesh['bbox']
        for (x, y), (u, v) in zip(mesh['positions'], mesh['uvs']):
            expected = ((ax + max(0, min(width, x - bx))) / data['atlasSize'],
                        (ay + max(0, min(height, y - by))) / data['atlasSize'])
            if abs(u - expected[0]) > 1e-12 or abs(v - expected[1]) > 1e-12:
                uv_errors += 1
    front_fraction = max(abs(value - round(value)) for mesh in data['meshes'] for x, y in mesh['positions'] for value in (x, y))
    return {'meshes': len(data['meshes']), 'atlas_pixel_mismatches': pixel_mismatches,
            'uv_offset_errors': uv_errors, 'front_vertex_fractional_max': front_fraction,
            'mask_raster_margin_px': data['maskRasterMarginPx']}


def main():
    CAPTURES.mkdir(parents=True, exist_ok=True)
    reference = Rig()
    baseline = json.loads((ROOT / 'source/keyshapes/v6-checks.json').read_text())
    keyshapes = json.loads((ROOT / 'source/keyshapes/points_art_keyshapes.json').read_text())
    report = {'renderer': {}, 'atlas': {}, 'front': {}, 'nonfront_raw_comparison': {}, 'flips': {}, 'forehead': {}, 'holes': {}}
    with tempfile.TemporaryDirectory(prefix='615-viewer-') as temporary:
        states = Path(temporary)
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(ROOT)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True,
                    args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
                loading_page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
                loading_page.add_init_script("""(() => { const original = window.fetch.bind(window); window.fetch = (...args) => new Promise((resolve, reject) => setTimeout(() => original(...args).then(resolve, reject), 800)); })();""")
                loading_page.goto(f'http://127.0.0.1:{server.server_port}/bundle/index.html', wait_until='domcontentloaded')
                loading_page.wait_for_selector('#loading:not([hidden])', state='visible')
                loading_page.screenshot(path=CAPTURES / 'ui-loading.png', full_page=True)
                loading_page.close()
                page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
                page.goto(f'http://127.0.0.1:{server.server_port}/bundle/index.html', wait_until='networkidle')
                page.wait_for_function("window.viewer && document.querySelector('#status').textContent === '준비됨'", timeout=30000)
                report['renderer'] = page.evaluate("({float32:window.viewer.renderer.float32, canvas:[document.querySelector('#stage').width,document.querySelector('#stage').height], unpackPremultiply:window.viewer.renderer.gl.getParameter(window.viewer.renderer.gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL)})")
                report['atlas'] = verify_atlas(reference)
                report['ui'] = verify_ui(page, CAPTURES)
                page.set_viewport_size({'width': 1440, 'height': 1000})

                front_hair = capture_float(page, full_state(0, 0), True)
                front_nohair = capture_float(page, full_state(0, 0), False)
                save_rgba(states / 'front_hair.png', front_hair)
                save_rgba(states / 'front_nohair.png', front_nohair)
                still = np.asarray(Image.open(STILL).convert('RGBA'))
                report['front']['hair_on'] = difference(front_hair, still)
                front_pose = pose(0, 0, hair=False)
                edge = silhouette_edge_band(reference, front_pose)
                front_cpu = reference.render(front_pose)
                report['front']['hair_off'] = difference(front_nohair, front_cpu, edge)
                report['front']['hair_off']['outside_backing_edge_band'] = report['front']['hair_off']['outside_edge_band_value_over_2']
                save_rgba(states / 'dir_front.png', front_nohair)

                for key, (u, v) in KEYS.items():
                    if key == 'front':
                        image = front_nohair
                    else:
                        image = capture_float(page, full_state(u, v), False)
                    save_rgba(states / f'dir_{key}.png', image)
                    state = pose(u, v, hair=False)
                    cpu = reference.render(state)
                    result = difference(image, cpu, silhouette_edge_band(reference, state))
                    report['nonfront_raw_comparison'][key] = result
                    rgba = capture_png(page, full_state(u, v), hair=False)
                    save_rgba(states / f'holes_{key}.png', rgba)
                    hole_px, hole_count = holes(rgba)
                    allowed_px, allowed_pieces = baseline['holes'][key]
                    report['holes'][key] = {'pixels': hole_px, 'pieces': hole_count,
                                            'v6_limits': [allowed_px, allowed_pieces],
                                            'pass': holes_within_limits(hole_px, hole_count, [allowed_px, allowed_pieces])}
                    if key != 'front':
                        continue

                report['flips'] = geometry_report(page, reference)
                report['forehead'] = skin_spill(page, reference, keyshapes['targets'])

                for key, (u, v) in KEYS.items():
                    for open_value in [.7, .4, .27, .25, .23, .21, 0]:
                        state = full_state(u, v, open_value)
                        name = f'{open_value:.2f}'.replace('.', '_')
                        save_rgba(states / f'blink_{key}_{name}.png', capture_png(page, state, hair=False))
                for axis, (du, dv) in {'left': (-1, 0), 'up': (0, 1), 'upleft': (-1, 1)}.items():
                    hole_key = f'band {"왼쪽" if axis == "left" else "위" if axis == "up" else "왼쪽 위"}'
                    allowed_values = baseline['holes'][hole_key]
                    endpoint = 'left' if axis == 'left' else 'up' if axis == 'up' else 'up-left'
                    band_values = [0, .2, .4, .6, .8, 1]
                    for step in [.2, .4, .6, .8]:
                        value = f'{step:.1f}'.replace('.', '_')
                        image = capture_png(page, full_state(du * step, dv * step), hair=False)
                        save_rgba(states / f'band_{axis}_{value}.png', image)
                    for index, step in enumerate(band_values):
                        if step == 0:
                            image = front_nohair
                        elif step == 1:
                            image = np.asarray(Image.open(states / f'dir_{endpoint}.png').convert('RGBA'))
                        else:
                            value = f'{step:.1f}'.replace('.', '_')
                            image = np.asarray(Image.open(states / f'band_{axis}_{value}.png').convert('RGBA'))
                        hole_px, hole_count = holes(image)
                        allowed = allowed_values[index]
                        report['holes'][f'{hole_key} t={step:.1f}'] = {
                            'pixels': hole_px, 'pieces': hole_count, 'v6_limit_pixels': allowed,
                            'pass': holes_within_limits(hole_px, hole_count, [allowed])}

                from figures import build_figures
                build_figures(states, CAPTURES, STILL)
                browser.close()
        finally:
            server.shutdown()
            server.server_close()

    forehead_failures = {key: value for key, value in report['forehead'].items() if value['pixels']}
    hole_failures = {key: value for key, value in report['holes'].items() if not value['pass']}
    nonfront_failures = {key: value for key, value in report['nonfront_raw_comparison'].items()
                         if key != 'front' and not nonfront_passes(value)}
    required_failures = []
    if report['front']['hair_on']['different_pixels']:
        required_failures.append('front hair-on is not pixel-identical')
    if report['front']['hair_off']['outside_edge_band_value_over_2']:
        required_failures.append('front hair-off differs by more than 2 outside the backing edge band')
    required_failures.extend(report['flips']['failures'])
    if forehead_failures:
        required_failures.append('skin appears outside the scalp above the brows')
    if hole_failures:
        required_failures.append('head/neck holes exceed v6 checks values')
    if nonfront_failures:
        required_failures.append('non-front comparison exceeds a v6 threshold')
    if report['atlas']['atlas_pixel_mismatches']:
        required_failures.append('atlas samples differ byte-for-byte from the source parts')
    if report['atlas']['uv_offset_errors'] or report['atlas']['front_vertex_fractional_max']:
        required_failures.append('atlas UVs or front vertex coordinates have an offset')
    if report['renderer']['unpackPremultiply']:
        required_failures.append('browser premultiplied atlas pixels on upload')
    report['status'] = {
        'required_failures': required_failures,
        'nonfront_thresholds': {'value_over_2_percent_visible_max': .6, 'alpha_over_8_max': 110,
                                'outside_edge_band_value_over_2_max': 4000},
        'nonfront_failures': nonfront_failures,
        'blink_visual_review': 'awaiting screenshot inspection',
    }
    (CAPTURES / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Viewer verification', '',
             f"- Renderer: RGBA32F={report['renderer']['float32']}, target={report['renderer']['canvas']}",
             f"- Front with hair: {report['front']['hair_on']['different_pixels']} different pixels",
             f"- Front without hair: {report['front']['hair_off']['outside_edge_band_value_over_2']} pixels differ by >2 outside the silhouette edge band",
             '- Non-front limits: value >2 ≤0.6% visible, alpha >8 ≤110, and value >2 outside the edge band ≤4,000.', '']
    for key, value in report['nonfront_raw_comparison'].items():
        lines.append(f"- {key}: {value['different_pixels']} different, max {value['max_channel_delta']}, >2 {value['value_over_2']} ({value['value_over_2_percent_visible']}% visible), alpha >8 {value['alpha_over_8']}, outside band >2 {value.get('outside_edge_band_value_over_2', 0)}")
    lines.extend(['', '- Eye triangle flips: ' + str(sum(row['count'] for row in report['flips']['eye_flips'])),
                  '- Skin spill above brows: ' + str({key: value['pixels'] for key, value in report['forehead'].items()}),
                  '- Holes: ' + str({key: value['pixels'] for key, value in report['holes'].items()}),
                  '- Blink figure review: `blink.png`, `blink_transition.png` (visual review pending).',
                  '- Captures: `front_diff.png`, `nine_nohair.png`, `band_left.png`, `band_up.png`, `band_upleft.png`, `ui-loading.png`, `ui-wide.png`, `ui-narrow.png`.',
                  '- Required numeric failures: ' + (', '.join(required_failures) if required_failures else 'none'), ''])
    (CAPTURES / 'verification.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if required_failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
