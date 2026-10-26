import unittest
import numpy as np
from PIL import Image
from verify_eye_clip_pixels import judge_pose, pixel_evidence
import verify_eye_clip_pixels as clip

class EyeClipPixelsTest(unittest.TestCase):
    def test_one_pixel_edge_is_reported_separately_from_real_outside(self):
        eye=np.full((5,5,4),255,np.uint8)
        face=np.zeros_like(eye);face[:,:,3]=255;face[:,2:,:3]=255
        result=clip.edge_evidence({'left':Image.fromarray(eye),'faceMask':Image.fromarray(face)},'left')
        self.assertEqual(result['edgeBandPixels'],5)
        self.assertEqual(result['outsideBeyondBandPixels'],5)
        self.assertEqual(result['outsideBeyondBandCoordinates720'],[[0,y] for y in range(5)])

    def test_all_outside_values_are_kept_without_boundary_filtering(self):
        eye=np.array([[[144,144,144,220],[255,255,255,255],[255,255,255,255]]],np.uint8)
        face=np.array([[[125,125,125,255],[0,0,0,255],[200,200,200,255]]],np.uint8)
        rows=clip.outside_samples({'left':Image.fromarray(eye),'faceMask':Image.fromarray(face)},'left')
        self.assertEqual(rows,[dict(x=0,y=0,mask_value=125,eye_value=144,eye_alpha=220),dict(x=1,y=0,mask_value=0,eye_value=255,eye_alpha=255)])

    def test_full_clip_requires_zero_only_for_far_side(self):
        self.assertTrue(judge_pose(15,{'left':3,'right':0},{'left':1,'right':20})['passed'])
        self.assertFalse(judge_pose(15,{'left':0,'right':1},{'left':20,'right':20})['passed'])
        self.assertTrue(judge_pose(-15,{'left':0,'right':3},{'left':20,'right':1})['passed'])
        self.assertFalse(judge_pose(-15,{'left':1,'right':0},{'left':20,'right':20})['passed'])

    def test_low_angle_uses_separate_side_regression_counts(self):
        self.assertTrue(judge_pose(1,{'left':1,'right':0},{'left':1,'right':0})['passed'])
        result=judge_pose(1,{'left':0,'right':1},{'left':1,'right':0})
        self.assertTrue(result['passed'])
        self.assertEqual(result['increasedSides'],['right'])

    def test_outside_pixel_is_detected_at_its_actual_coordinate(self):
        eye=np.zeros((2,2,4),np.uint8);eye[1,0]=[255,255,255,255]
        face=np.full_like(eye,255)
        inputs={'left':Image.fromarray(eye),'faceMask':Image.fromarray(face)}
        self.assertEqual(pixel_evidence(inputs,'left')['pixels'],0)
        face[1,0]=[0,0,0,255];inputs['faceMask']=Image.fromarray(face)
        self.assertEqual(pixel_evidence(inputs,'left'),{'pixels':1,'coordinates720':[[0,1]]})

if __name__=='__main__':unittest.main()
