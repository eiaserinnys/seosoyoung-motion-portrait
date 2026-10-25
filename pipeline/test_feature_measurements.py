"""Image evidence for opening envelopes and the front nose detector bias."""
import hashlib
import io
import json
import subprocess
import unittest
import numpy as np
from PIL import Image
from fit_features import ROOT,measure,measure_opening,SEEDS

class FeatureMeasurementTest(unittest.TestCase):
    def test_each_pitch_pose_preserves_its_own_registered_measurements(self):
        data=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
        for pose in data['poses']:
            if not pose['pitch']:continue
            with self.subTest(pose=pose['name']):
                self.assertEqual(pose['centers'],pose['registeredCenters'])
                scale=pose['registration']['scale']
                for side in ['left','right']:
                    np.testing.assert_allclose(pose['scales'][side],np.array(pose['measuredScales'][side])*scale,atol=1e-12)

    def test_face_verifier_uses_each_independent_measured_opening(self):
        from verify_hairless_face import expected_opening
        data=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
        poses=[dict(yaw=p['yaw'],pitch=p['pitch'],feature=p) for p in data['poses']]
        for p in poses:
            if not p['pitch']:continue
            for side in ['left','right']:
                self.assertEqual(expected_opening(poses,p,side),p['feature']['measuredScales'][side][1]*p['feature']['registration']['scale'])

    def test_down_diagonals_keep_their_own_artwork_asymmetry(self):
        data=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
        poses={p['name']:p for p in data['poses']}
        dr,dl=poses['down-right'],poses['down-left']
        self.assertEqual(dr['centers'],dr['registeredCenters'])
        self.assertEqual(dr['scales'],dr['measuredScales'])
        front=data['front']
        self.assertEqual(dl['centers'],dl['registeredCenters'])
        self.assertNotAlmostEqual(dl['centers']['mouth'][0]-front['mouth'][0],front['mouth'][0]-dr['centers']['mouth'][0])

    def test_cute_up_references_preserve_the_protected_upper_pixels(self):
        for name,stop,sha in [
            ('up-left',360,'74279089d6dac474ae26dba26cea5f4c7edf4998648230ab95da5ac306e00120'),
            ('up',315,'a8ee13b0fd05528cf26005f6544973289e3a97ba7979573648a60c61e7b425e6'),
            ('up-right',360,'26c16fdb0bb5b339677498df67b2be51b90f4dd04732ace9febfeefa64bd3411'),
        ]:
            path='source/references/'+name.replace('-','_')+'_ref_720.png'
            with self.subTest(pose=name):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha)
                before=subprocess.check_output(['git','show','f6112b6:'+path],cwd=ROOT,timeout=60)
                old=np.array(Image.open(io.BytesIO(before)).convert('RGB'))
                new=np.array(Image.open(ROOT/path).convert('RGB'))
                np.testing.assert_array_equal(new[:stop],old[:stop])

    def test_cute_up_mouth_measurements_follow_the_new_ink(self):
        data=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
        for name,seed,expected_y in [('up-left',(226,432),432.),('up',(358,376),382.),('up-right',(511,432),431.5)]:
            with self.subTest(pose=name):
                rgb=np.array(Image.open(ROOT/'source/references'/(name.replace('-','_')+'_ref_720.png')).convert('RGB'))
                measured=measure(rgb,seed,'mouth')
                self.assertAlmostEqual(measured['center'][1]-20,expected_y,delta=1)
                pose=next(p for p in data['poses'] if p['name']==name)
                self.assertEqual(pose['measurements']['mouth'],measured)
                self.assertEqual(SEEDS[name][5],seed)

    def test_neutral_feature_poses_and_mask_manifest_entries_are_unchanged(self):
        for path,key in [('source/viewer/face-feature-fit.json','poses'),('sam3_masks/references/manifest.json',None)]:
            baseline='4dba92d' if not key else 'f6112b6'
            old=json.loads(subprocess.check_output(['git','show',baseline+':'+path],cwd=ROOT,timeout=60))
            new=json.loads((ROOT/path).read_text())
            if key:
                old={p['name']:p for p in old[key]};new={p['name']:p for p in new[key]}
            for name,value in old.items():
                if not key or name in {'left','front','right'}:
                    self.assertEqual(json.dumps(new[name],ensure_ascii=False),json.dumps(value,ensure_ascii=False),name)

    def test_up_masks_end_at_the_new_chin(self):
        for name,chin in [('up',474),('up-left',503),('up-right',508)]:
            mask=np.array(Image.open(ROOT/'sam3_masks/references'/(name+'-face.png')))
            self.assertEqual(int(np.where(mask>127)[0].max()),chin,name)
            self.assertFalse(np.any(mask[chin+1:]>127),name+' neck below chin')

    def test_front_nose_bias_is_five_pixels(self):
        rgb=np.array(Image.open(ROOT/'source/references/head_front_ref.png').convert('RGB'))
        nose=measure(rgb,(508-152,497-20),'nose')
        self.assertEqual(497-nose['center'][1],5)

    def test_down_eye_uses_lash_to_liner_height_instead_of_iris_height(self):
        extents=[]
        for name,file,seed in [('front','head_front_ref.png',(264,412)),('down','down_ref_720.png',SEEDS['down'][0])]:
            rgb=np.array(Image.open(ROOT/'source/references'/file).convert('RGB'))
            iris=measure(rgb,seed,'left');opening=measure_opening(rgb,iris['center'])
            self.assertGreater(opening['openingExtent'][1],iris['extent'][1]+5,name)
            extents.append(opening['openingExtent'][1])
        self.assertAlmostEqual(extents[1]/extents[0],51/82)
        self.assertLess(extents[1]/extents[0],.7)

if __name__=='__main__':unittest.main()
