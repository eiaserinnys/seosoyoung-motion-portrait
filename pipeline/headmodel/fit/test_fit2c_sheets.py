"""The jaw report must expose missed targets and preserve 720px distances."""
import unittest
import numpy as np
from fit2c_sheets import jaw_details


class JawReportContract(unittest.TestCase):
    def test_three_pixel_offset_fails_median_target_and_exposes_worst_side(self):
        wanted=np.array([[10.,13.],[20.,13.],[30.,13.]])
        model=wanted-[0.,3.]
        result=jaw_details(model,wanted)
        self.assertEqual(result['median_px'],3.)
        self.assertEqual(result['max_px'],3.)
        self.assertFalse(result['target_median_le_2'])
        self.assertTrue(result['target_max_le_5'])
        self.assertEqual(result['regions']['screen-left']['max_px'],3.)
        exact=jaw_details(wanted,wanted)
        self.assertEqual(exact['max_px'],0.)
        self.assertTrue(exact['target_median_le_2'])


if __name__=='__main__':unittest.main()
