"""Boundary evidence, isolated temple gaps and the prior-only occluded case."""
import unittest
import numpy as np
from fit_outline import classify_boundary,qualify_background,blend_profile,monotone_profile

class OutlineOcclusionTest(unittest.TestCase):
    def test_four_pixel_neighborhood_classifies_background_skin_and_hair(self):
        for rgb,expected in [([8,9,10],'BG'),([250,202,170],'SKIN'),([90,63,49],'HAIR')]:
            img=np.full((2,20,3),rgb,np.uint8)
            self.assertEqual(classify_boundary(img,1,10,-4),expected)
            self.assertEqual(classify_boundary(img,1,10,4),expected)

    def test_repeated_temple_gap_is_not_multiple_silhouette_observations(self):
        rows=[dict(restY=y,**{s:dict(referenceY=286,**{'class':'BG'},validBG=True) for s in ['left','right']}) for y in [410,420,430,440]]
        qualify_background(rows)
        self.assertTrue(all(not row[s]['validBG'] for row in rows for s in ['left','right']))

    def test_fully_occluded_profile_uses_prior_plus_translation(self):
        target,weight=blend_profile([350,355,365,385],[400,420,400,405],[False]*4,-6)
        np.testing.assert_array_equal(target,[344,349,359,379]);np.testing.assert_array_equal(weight,0)

    def test_cosine_seam_reaches_measured_core_without_a_quartic_fit(self):
        target,weight=blend_profile([0]*10,[10]*10,[False,False,True,True,True,True,True,True,False,False],0)
        self.assertEqual(target[0],0);self.assertEqual(target[4],10)
        self.assertAlmostEqual(target[2],5);self.assertAlmostEqual(target[7],5)

    def test_lower_half_width_cannot_rebound_or_cross(self):
        rows=[[y,y,502-w/2,y,502+w/2] for y,w in [(440,310),(450,320),(465,312),(480,314),(490,318),(510,280),(616,-3)]]
        out=monotone_profile(rows);widths=[r[4]-r[2] for r in out]
        self.assertEqual(widths,[310,320,312,312,312,280,11])

if __name__=='__main__':unittest.main()
