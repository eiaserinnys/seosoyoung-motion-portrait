#!/usr/bin/env python3
"""Measure the exported posed head outline against yaw zero."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    meta = json.loads((ROOT/'bundle/assets/head-base.json').read_text())
    binary = (ROOT/'bundle/assets/head-base.bin').read_bytes()
    count, scale = meta['vertexCount'], meta['s_ref']

    def array(name):
        data = meta['offsets'][name]
        return np.frombuffer(binary, dtype='<f4', count=data['length']//4, offset=data['offset'])

    vertices = array('vertices').reshape(count, 3)[:-1].astype(float)
    fields = array('fields').reshape(9, count, 3)[:, :-1].astype(float)
    cheek = array('cheek_field').reshape(count, 3)[:-1].astype(float)
    with np.load(ROOT/'pipeline/headmodel/fit/results/fit2f/surface-a.npz') as source:
        ear = source['ear_excluded'].copy()
    nodes = np.asarray(meta['theta_k'])
    rest = np.asarray(next(key['model'] for key in meta['keys'] if key['frame'] == 3))

    def project(yaw, pitch):
        theta = np.array([yaw, pitch])
        distance = (nodes-theta)*[1., meta['pitch_weight']]
        logits = -np.sum(distance*distance, axis=1)/(2*meta['sigma']**2)
        weights = np.exp(logits-logits.max()); weights /= weights.sum()
        d = (theta-rest)*[1., meta['pitch_weight']]
        cheek_weight = np.exp(-np.dot(d, d)/(2*meta['cheek_sigma']**2))
        v = vertices+np.einsum('k,kij->ij', weights, fields)/scale+cheek_weight*cheek
        a, b = np.radians([yaw, pitch])
        # fit_core.project: y is inverted, then pitch, then yaw.
        y = np.cos(b)*v[:, 1]+np.sin(b)*v[:, 2]
        z = -np.sin(b)*v[:, 1]+np.cos(b)*v[:, 2]
        x = np.cos(a)*v[:, 0]+np.sin(a)*z
        return x, y

    heights = np.round(np.arange(-2.3, 11.7001, 1.), 1)
    rows = []
    for pitch in (0, 12):
        x0, y0 = project(0, pitch)
        for yaw in (-35, -25, -16, 16, 25, 35):
            x, y = project(yaw, pitch)
            direction = -np.sign(yaw)
            for height in heights:
                reference = (np.abs(y0-height) <= .15) & ~ear
                posed = (np.abs(y-height) <= .15) & ~ear
                near = ((direction*x[posed]).max()-(direction*x0[reference]).max())*scale
                far = ((-direction*x[posed]).max()-(-direction*x0[reference]).max())*scale
                rows.append({'pitch': pitch, 'yaw': yaw, 'height': float(height),
                             'near_px': float(near), 'far_px': float(far),
                             'near_pass': bool(abs(near) <= 8) if pitch == 0 else None})
    report = {'reference': 'same pitch, yaw 0', 'projection': 'pitch then yaw, original model units',
              'height_band': .15, 'scale_px_per_ict': scale, 'rows': rows}
    path = ROOT/'docs/head_base_fixed_outline_metrics.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    checked = [row for row in rows if row['pitch'] == 0]
    fails = [row for row in checked if row['near_pass'] is False]
    print(json.dumps({'rows': len(rows), 'pitch0_near_max_abs': max(abs(row['near_px']) for row in checked),
                      'pitch0_far_y_ge_5.7_max_abs': max(abs(row['far_px']) for row in checked if row['height'] >= 5.7),
                      'pitch0_failure_count': len(fails), 'pitch0_failures': fails}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
