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
    assert current == previous('hair-shell.mjs').decode(), 'hair-shell differs from main'
    report['rig_except_corrections'] = 'identical'
    report['hair_shell'] = 'identical'
    for name in ('hair-surface.mjs', 'dome-data.mjs'):
        assert (ROOT / 'bundle' / name).read_bytes() == previous(name), name
    report['other_assets'] = 'identical'
    fit = ROOT / 'pipeline/headmodel/fit/results'
    with np.load(fit/'fit2f/fit-state-n3.npz') as original, np.load(fit/'dome/fit-state-dome.npz') as dome:
        assert all(np.array_equal(original[key], dome[key]) for key in original.files if key != 'vertices')
        weights = dome['w'].copy()
        assert np.array_equal(original['vertices'][original['vertices'][:, 1] < 2],
                              dome['vertices'][original['vertices'][:, 1] < 2])
    metadata = json.loads((ROOT/'bundle/assets/head-base.json').read_text())
    binary = (ROOT/'bundle/assets/head-base.bin').read_bytes()
    count = metadata['vertexCount']
    def array(name):
        region = metadata['offsets'][name]
        return np.frombuffer(binary, dtype='<f4', count=region['length']//4, offset=region['offset'])
    with np.load(fit/'fit2e/final-state.npz') as original:
        expected = original['D_k'] * (1-weights)[None, :, None]
    fields = array('fields').reshape(9, count, 3)[:, :-1]
    selected = weights > 0
    field_error = float(np.max(np.abs(fields[:, selected] - expected[:, selected])))
    assert field_error < 1e-5, field_error
    with np.load(ROOT/'pipeline/headmodel/cheek-field.npz') as original:
        expected = original['displacement'] * (1-weights)[:, None]
    cheek = array('cheek_field').reshape(count, 3)[:-1]
    cheek_error = float(np.max(np.abs(cheek - expected)))
    assert cheek_error < 1e-6, cheek_error
    report['dome_weighted_fields'] = {'vertices': int(selected.sum()), 'max_error': field_error}
    report['dome_weighted_cheek'] = {'max_error': cheek_error}
    report['checks_passed'] = 10
    return report


if __name__ == '__main__':
    result = verify()
    print(json.dumps(result, ensure_ascii=False))
    print(f"{result['checks_passed']} passed")
