"""Check protected assets against the main-branch merge base."""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
base = subprocess.check_output(['git', 'merge-base', 'HEAD', 'origin/main'], cwd=ROOT, text=True).strip()


def previous(name):
    return subprocess.check_output(['git', 'show', f'{base}:bundle/{name}'], cwd=ROOT)


def verify():
    report = {'merge_base': base, 'atlas_outside_neck': {}}
    rig = json.loads((ROOT / 'bundle/assets/seosoyoung-rig.json').read_text())
    original = json.loads(previous('assets/seosoyoung-rig.json'))
    del rig['corrections'], original['corrections']
    assert rig == original, 'rig differs outside the corrections metadata'
    for name in ('texture_00.png', 'texture_before_00.png'):
        old = np.asarray(Image.open(io.BytesIO(previous('assets/' + name))).convert('RGBA'))
        new = np.asarray(Image.open(ROOT / 'bundle/assets' / name).convert('RGBA'))
        assert old.shape == new.shape
        diff = np.any(old != new, axis=2)
        x, y, w, h = next(m['atlas'] for m in rig['meshes'] if m['id'] == 'neck')
        diff[y:y+h, x:x+w] = False
        report['atlas_outside_neck'][name] = int(diff.sum())
        assert not diff.any(), f'{name} changed outside the neck atlas'
    current = (ROOT / 'bundle/hair-shell.mjs').read_text()
    current = current.replace("import {headEarWorld} from './head-base.mjs';\n", '')
    current = current.replace('HEAD_CENTER,clamp,correction,artWorld,headBaseActive}',
                              'HEAD_CENTER,clamp,correction,artWorld}')
    current = current.replace(
        'export function sideWorld(point,side,p,ear=false){return ear?(headBaseActive(p)?headEarWorld(point,side,p):earWorld(point,side,p)):canopyWorld(point,p);}',
        'export function sideWorld(point,side,p,ear=false){return ear?earWorld(point,side,p):canopyWorld(point,p);}')
    assert current == previous('hair-shell.mjs').decode(), 'hair-shell changed outside ear routing'
    report['rig_except_corrections'] = 'identical'
    report['hair_shell_except_ear_routing'] = 'identical'
    for name in ('hair-surface.mjs', 'dome-data.mjs'):
        assert (ROOT / 'bundle' / name).read_bytes() == previous(name), name
    report['other_assets'] = 'identical'
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False))
