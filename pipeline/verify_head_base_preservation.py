"""Verify protected assets and the fixed-outline fields against the merge base."""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
base = subprocess.check_output(['git', 'merge-base', 'HEAD', 'origin/main'], cwd=ROOT, text=True).strip()


def previous(name):
    return subprocess.check_output(['git', 'show', f'{base}:bundle/{name}'], cwd=ROOT)


def verify():
    checks = []
    report = {'merge_base': base}

    def check(condition, name):
        assert condition, name
        checks.append(name)

    rig = json.loads((ROOT/'bundle/assets/seosoyoung-rig.json').read_text())
    check(rig == json.loads(previous('assets/seosoyoung-rig.json')), 'rig')
    for name in ('texture_00.png', 'texture_before_00.png'):
        old = np.asarray(Image.open(io.BytesIO(previous('assets/'+name))).convert('RGBA'))
        new = np.asarray(Image.open(ROOT/'bundle/assets'/name).convert('RGBA'))
        check(np.array_equal(old, new), name)
    for name in ('hair-shell.mjs', 'hair-surface.mjs', 'dome-data.mjs', 'renderer.mjs', 'app.mjs',
                 'face-outline.mjs', 'face-solve.mjs', 'feature-data.mjs', 'skin-fill.mjs'):
        check((ROOT/'bundle'/name).read_bytes() == previous(name), name)

    fit = ROOT/'pipeline/headmodel/fit/results'
    with np.load(fit/'fit2f/fit-state-n3.npz') as original, np.load(fit/'fixed-outline/fit-state-fixed.npz') as fixed_data:
        check(all(np.array_equal(original[key], fixed_data[key]) for key in original.files if key != 'vertices'),
              'source topology and metadata')
        original_vertices, fixed, w = original['vertices'].copy(), fixed_data['vertices'].copy(), fixed_data['w'].copy()
        triangles = original['triangles'].copy()
    with np.load(ROOT/'pipeline/headmodel/cheek-field.npz') as source:
        original_cheek = source['displacement'].copy()
    check(np.allclose(fixed[:, 1], original_vertices[:, 1]+w*original_cheek[:, 1], rtol=0, atol=1e-12),
          'vertical coordinate changes only by the transferred cheek field')
    with np.load(fit/'fit2f/surface-a.npz') as source:
        ear = source['ear_excluded'].copy()
    a, b, c = triangles.T
    edges = np.vstack((np.c_[a, b], np.c_[b, c], np.c_[c, a]))
    adjacency = sparse.coo_matrix((np.ones(len(edges)*2),
                                   (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                                  shape=(len(fixed), len(fixed))).tocsr()
    adjacency.data[:] = 1
    for _ in range(3):
        ear |= (adjacency @ ear.astype(float)) > 0
    metadata = json.loads((ROOT/'bundle/assets/head-base.json').read_text())
    binary = (ROOT/'bundle/assets/head-base.bin').read_bytes()
    count = metadata['vertexCount']

    def array(name):
        region = metadata['offsets'][name]
        return np.frombuffer(binary, dtype='<f4', count=region['length']//4, offset=region['offset'])

    with np.load(fit/'fit2e/final-state.npz') as source:
        raw, nodes, sigma = source['D_k'], source['theta_k'], float(source['sigma'])
        theta = next(item['model'] for item in metadata['keys'] if item['frame'] == 3)
        distance = (nodes-theta)*[1., metadata['pitch_weight']]
        logits = -np.sum(distance*distance, axis=1)/(2*sigma*sigma)
        weights = np.exp(logits-logits.max()); weights /= weights.sum()
        rest = np.einsum('k,kij->ij', weights, raw)
        expected = raw*(1-w)[None, :, None] + rest[None, :, :]*w[None, :, None]
    fields = array('fields').reshape(9, count, 3)[:, :-1]
    selected = ~ear
    selected[metadata['open_boundary']] = False
    field_error = float(np.max(np.abs(fields[:, selected]-expected[:, selected])))
    check(field_error < 1e-5, 'rest-blended angle fields')
    expected_cheek = (original_cheek*(1-w)[:, None]).astype('<f4')
    cheek = array('cheek_field').reshape(count, 3)[:-1]
    check(np.array_equal(cheek, expected_cheek), 'fixed-outline cheek field')
    report['field_max_error'] = field_error
    report['field_checked_vertices'] = int(selected.sum())
    report['checks'] = checks
    report['checks_passed'] = len(checks)
    return report


if __name__ == '__main__':
    result = verify()
    print(json.dumps(result, ensure_ascii=False))
    print(f"{result['checks_passed']} passed")
