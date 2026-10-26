import unittest
import numpy as np
from PIL import Image
from verify_eye_clip_pixels import judge_pose, pixel_evidence

class EyeClipPixelsTest(unittest.TestCase):
    def test_full_clip_requires_zero_for_both_sides(self):
        self.assertTrue(judge_pose(15,{'left':0,'right':0},{'left':20,'right':20})['passed'])
        for side in ['left','right']:
            bad={'left':0,'right':0};bad[side]=1
            self.assertFalse(judge_pose(15,bad,{'left':20,'right':20})['passed'])

    def test_low_angle_uses_separate_side_regression_counts(self):
        self.assertTrue(judge_pose(1,{'left':1,'right':0},{'left':1,'right':0})['passed'])
        self.assertFalse(judge_pose(1,{'left':0,'right':1},{'left':1,'right':0})['passed'])

    def test_outside_pixel_is_detected_at_its_actual_coordinate(self):
        eye=np.zeros((2,2,4),np.uint8);eye[1,0]=[255,255,255,255]
        face=np.full_like(eye,255)
        inputs={'left':Image.fromarray(eye),'faceMask':Image.fromarray(face)}
        self.assertEqual(pixel_evidence(inputs,'left')['pixels'],0)
        face[1,0]=[0,0,0,255];inputs['faceMask']=Image.fromarray(face)
        self.assertEqual(pixel_evidence(inputs,'left'),{'pixels':1,'coordinates720':[[0,1]]})

if __name__=='__main__':unittest.main()
