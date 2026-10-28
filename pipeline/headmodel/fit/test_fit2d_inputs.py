"""Pose-space contract against all nine registered key normalizations."""
import unittest
import numpy as np
from fit_core import ROOT, read_json, normalize_pose
from fit2d_core import pose_affine


class InputContract(unittest.TestCase):
    def test_nine_key_affines(self):
        registration = read_json(ROOT/'source/viewer/ponytail-seedance-registration.json')
        self.assertEqual(len(registration['keys']), 9)
        for key in registration['keys']:
            frame = int(key['sourceFrame'])
            manifest = read_json(ROOT/f'pipeline/headmodel/data/frame_{frame:04d}/manifest.json')
            actual = pose_affine(manifest['to720'])
            raw = registration['video'][frame-1]['pose']
            np.testing.assert_allclose(actual, key['normalization'], atol=1e-6, rtol=0)
            np.testing.assert_allclose(normalize_pose(raw, actual),
                                       normalize_pose(raw, key['normalization']), atol=1e-6, rtol=0)
            # Deliberately applying the image transform must fail this contract.
            self.assertGreater(np.max(np.abs(normalize_pose(raw, manifest['to720'][:2])-
                                            normalize_pose(raw, actual))), 30)
