"""Stage 4 gates recompute predictions; fabricated summary metrics cannot pass."""
import copy
import json
import unittest
import hashlib
from pathlib import Path
import numpy as np
from ponytail_model import basis, predict, assert_registration, block_holdout, shape_system,fit_continuous_shapes
from ponytail_observations import match_patch
from seedance_temporal import project_frames,fit_trajectory

ROOT = Path(__file__).resolve().parents[1]

class PonytailRegistrationTest(unittest.TestCase):
    def test_block_holdout_has_three_contiguous_untrained_blocks(self):
        mask = block_holdout(361)
        self.assertEqual(int(mask.sum()), 54)
        self.assertEqual(int(np.count_nonzero(np.diff(mask.astype(int)) == 1)), 3)

    def test_ncc_tracks_translated_texture(self):
        rng = np.random.default_rng(5)
        im = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        shifted = np.roll(np.roll(im, 7, axis=1), -4, axis=0)
        p, score = match_patch(im[40:61,40:61], shifted, [50,50], 15)
        np.testing.assert_allclose(p, [57,46], atol=.2)
        self.assertGreater(score,.99)

    def test_linear_shape_block_recovers_multiple_landmarks_and_both_axes(self):
        rng=np.random.default_rng(12)
        X=rng.normal(0,40,(9,3));C=rng.normal(0,2,(10,9,2));C[0]=0
        model=dict(points3D=X.tolist(),residual=C.tolist())
        poses=np.array([[y,p,2,.1,340,420] for y in [-45,-30,0,20,45] for p in [-30,-10,0,15,30]])
        obs=np.array([predict(model,p) for p in poses])
        a,b=shape_system(poses,X,obs)
        fitted=np.linalg.lstsq(a,b,rcond=1e-10)[0]
        np.testing.assert_allclose(a@fitted,b,atol=1e-8)

    def test_vectorized_camera_projection_matches_independent_frame_evaluation(self):
        rng=np.random.default_rng(19)
        model=dict(points3D=rng.normal(0,40,(9,3)).tolist(),residual=rng.normal(0,2,(10,9,2)).tolist())
        poses=np.array([[0,0,0,0,340,420],[35,-20,4,.1,350,400],[-42,25,-3,-.1,330,430]])
        np.testing.assert_allclose(project_frames(model,poses),[predict(model,p) for p in poses],atol=1e-10)

    def test_cubic_fit_uses_image_space_rms_cones_and_excludes_heldout_shapes(self):
        rng=np.random.default_rng(21)
        X=rng.normal(0,40,(9,3));C=rng.normal(0,1,(10,9,2));C[0]=0
        model=dict(points3D=X.tolist(),residual=C.tolist())
        angles=[(0,0),(0,0),(-45,0),(45,0),(0,30),(0,-30),(-45,30),(45,30),(-45,-30),(45,-30)]
        poses=np.array([[y,p,0,0,340,420] for y,p in angles]+[[y,p,1,.1,350,410] for y,p in [(-20,10),(10,-20),(25,15)]])
        obs=project_frames(model,poses);keys=[dict(name='canonical-rest')]+[dict(name='front' if i==1 else str(i)) for i in range(1,10)]
        a,_=fit_continuous_shapes(keys,obs,X,poses,np.array([True,False,True]))
        changed=obs.copy();changed[-2]+=1000
        b,_=fit_continuous_shapes(keys,changed,X,poses,np.array([True,False,True]))
        np.testing.assert_allclose(a['residual'],b['residual'],atol=1e-8)
        np.testing.assert_allclose(predict(a,poses[0]),obs[0],atol=1e-9)
        for i in range(1,10):
            error=np.sqrt(np.mean(np.sum((predict(a,poses[i])-obs[i])**2,axis=1)))
            self.assertLessEqual(error,.5 if i==1 else 3.5)

    def test_temporal_camera_training_does_not_read_held_out_image_points(self):
        rng=np.random.default_rng(20)
        model=dict(points3D=rng.normal(0,40,(9,3)).tolist(),residual=np.zeros((10,9,2)).tolist())
        true=np.array([[y,3,0,0,340,420] for y in np.linspace(-12,12,12)])
        obs=project_frames(model,true);initial=true.copy();initial[6,0]+=30
        weights=np.ones(12);weights[4:8]=0
        altered=obs.copy();altered[4:8]+=100
        a=fit_trajectory(model,obs,initial,weights,max_nfev=8)
        b=fit_trajectory(model,altered,initial,weights,max_nfev=8)
        np.testing.assert_allclose(a,b,atol=1e-8)
        self.assertLess(float(np.max(np.abs(np.diff(a[:,:3],axis=0)))),10)

    def test_published_model_predictions_and_all_gates(self):
        data = json.loads((ROOT/'source/viewer/ponytail-seedance-registration.json').read_text())
        assert_registration(data)
        self.assertEqual(len(data['video']),361)
        self.assertEqual(len(data['keys']),9)
        self.assertEqual(data['weights'],{'keyframe':10.0,'video':1.0})
        self.assertEqual(data['holdout']['indices'],np.flatnonzero(block_holdout(361)).tolist())
        for row in data['keys']:
            actual = predict(data['model'], row['pose'])
            np.testing.assert_allclose(actual, row['prediction'], atol=1e-7)
            self.assertEqual(row['source'],'seedance')
            self.assertNotIn(row['sourceFrame']-1,data['holdout']['indices'])
            self.assertEqual(row['normalizedSha256'],hashlib.sha256((ROOT/row['image']).read_bytes()).hexdigest())
        self.assertEqual(data['holdout']['trainingIndices'], np.flatnonzero(~block_holdout(361)).tolist())

    def test_gate_rejects_corrupt_model_observations_and_jacobian(self):
        data = json.loads((ROOT/'source/viewer/ponytail-seedance-registration.json').read_text())
        for field in ['landmarks','radii']:
            bad = copy.deepcopy(data)
            bad['keys'][0][field] = (np.array(bad['keys'][0][field])+50).tolist()
            with self.assertRaises(AssertionError): assert_registration(bad)
        bad=copy.deepcopy(data)
        bad['model']['residual'][0][0][0]+=50
        with self.assertRaises(AssertionError): assert_registration(bad)
        bad=copy.deepcopy(data)
        bad['field']['minDetJ']=.24
        with self.assertRaises(AssertionError): assert_registration(bad)
        bad=copy.deepcopy(data);bad['video'][20]['pose'][0]+=50
        with self.assertRaises(AssertionError):assert_registration(bad)

if __name__=='__main__':unittest.main()
