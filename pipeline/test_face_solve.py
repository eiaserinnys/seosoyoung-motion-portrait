"""Independent gates for the generated joint least-squares face solution."""
import copy
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def assert_solution(data):
    assert data['family'] == 'face-0'
    n = len(data['points'])
    assert len(data['Z']) == n
    assert np.isfinite(data['Z']).all()
    assert len(data['poses']) == 9
    for p in data['poses']:
        assert np.shape(p['du']) == (n, 2)
        assert np.isfinite(p['du']).all()
        assert p['s'] == 1
        s = p['stats']
        assert s['rms'] <= 1
        assert s['reversedTriangles'] == 0 and s['minDetJ'] > 0
        if p['pitch']:
            assert s['minDetJ'] > .05
        if p['yaw'] or p['pitch']:
            threshold = .5 if p['yaw'] and p['pitch'] else .9 if p['yaw'] else .6
            limit = 25 if p['yaw'] and p['pitch'] else 16 if p['yaw'] else 12
            assert s['explained3D'] >= threshold, (p['name'], s)
            assert s['duRms'] <= limit, (p['name'], s)
        else:
            assert p['t'] == [0, 0] and np.count_nonzero(p['du']) == 0
    cv = data['crossValidation']
    assert {p['heldOut'] for p in cv} == {p['name'] for p in data['poses'] if p['yaw'] or p['pitch']}
    for p in cv:
        assert all(np.isfinite(p[k]) and p[k] > 0 for k in ['silhouetteRms', 'featureRms'])


def assert_family45(data):
    family = data['family45']
    n = len(data['points'])
    for side in ['Left', 'Right']:
        assert np.shape(family['Z'+side]) == (n,)
        assert np.isfinite(family['Z'+side]).all()
    assert [p['name'] for p in family['poses']] == ['left67', 'left90', 'right67', 'right90']
    for p in family['poses']:
        assert np.shape(p['du']) == (n, 2)
        assert np.isfinite(p['du']).all()
        s = p['stats']
        threshold = .35 if abs(p['yaw']) == 67.5 else .22
        assert s['reversedTriangles'] == 0 and s['minDetJ'] >= threshold
        assert s['explained3D'] >= .90
        assert s['rms'] < 1e-6 and s['observationRms'] < 1e-6


class FaceSolveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT/'source/viewer/face-solve.json').read_text())

    def test_solution_gates(self):
        assert_solution(self.data)

    def test_family45_gates(self):
        assert_family45(self.data)

    def test_family45_gate_rejects_violations(self):
        for key, value in [('rms', 1e-5), ('observationRms', 1e-5),
                           ('explained3D', .89), ('minDetJ', .349), ('reversedTriangles', 1)]:
            bad = copy.deepcopy(self.data)
            bad['family45']['poses'][0]['stats'][key] = value
            with self.assertRaises(AssertionError, msg=key):
                assert_family45(bad)
        bad = copy.deepcopy(self.data)
        bad['family45']['poses'][1]['stats']['minDetJ'] = .219
        with self.assertRaises(AssertionError):
            assert_family45(bad)

    def test_pitch_refit_preserves_neutral_xy_and_the_approved_side_family(self):
        previous = json.loads(subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', '0b75a02:source/viewer/face-solve.json']))
        self.assertEqual(self.data['family45'],previous['family45'])
        for key in ['points','triangles','samples','anchors','meshCount']:
            self.assertEqual(self.data[key],previous[key])
        from optimize_3d_2d import projection
        for old,new in zip(previous['poses'],self.data['poses']):
            if new['pitch']:continue
            a,b=projection(np.array(self.data['points']),new['yaw'],0)
            before=a+np.array(previous['Z'])[:,None]*b+old['t']+old['du']
            after=a+np.array(self.data['Z'])[:,None]*b+new['t']+new['du']
            np.testing.assert_allclose(after,before,atol=1e-9)

    def test_gates_reject_injected_violations(self):
        for key, value in [('rms', 1.01), ('explained3D', 0), ('duRms', 100),
                           ('minDetJ', -1), ('reversedTriangles', 1)]:
            bad = copy.deepcopy(self.data)
            p = next(p for p in bad['poses'] if p['yaw'])
            p['stats'][key] = value
            with self.assertRaises(AssertionError, msg=key):
                assert_solution(bad)
        bad = copy.deepcopy(self.data)
        bad['crossValidation'][0]['silhouetteRms'] = float('nan')
        with self.assertRaises(AssertionError):
            assert_solution(bad)

    def test_source_hashes(self):
        for key, name in [('outlineTraces', 'outline-traces'), ('featureFit', 'face-feature-fit'), ('depth', 'face-depth')]:
            self.assertEqual(self.data['sources'][key], hashlib.sha256(
                (ROOT/f'source/viewer/{name}.json').read_bytes()).hexdigest())

    def test_schur_matches_direct_joint_system(self):
        from optimize_3d_2d import schur_pose
        w = np.array([4., 0., 8.])
        lap = np.array([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
        r = .25*lap + np.diag(.02*np.where(w > 0, 1., 1e-4))
        effective, inverse_w = schur_pose(w, r)
        d = np.array([8., -2., 3.])
        du = inverse_w@d
        np.testing.assert_allclose((np.diag(w)+r)@du, w*d, atol=1e-10)
        np.testing.assert_allclose(d@effective@d, np.sum(w*(du-d)**2)+du@r@du, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
