import unittest
import numpy as np
from capture_eye_final import stage_measure


class StageReportTest(unittest.TestCase):
    def fixture(self):
        rgba=np.zeros((1024,1024,4),np.uint8)
        rgba[250:350,400:605]=[210,190,170,255]
        owned=np.zeros_like(rgba)
        geometry=dict(innerRow=[0,1,2],face=[[450,300,-200],[502,300,-200],[550,300,-200]],faceRest=[[450,223],[502,223],[550,223]])
        return rgba,owned,geometry

    def test_all_stage_failures_are_reported_without_stopping(self):
        rgba,owned,g=self.fixture()
        self.assertTrue(stage_measure('down',rgba,owned,g)['passed'])
        owned[310,600]=[255,255,255,255]
        rgba[250:270,605:610]=[210,190,170,255]
        rgba[300,502,:3]=20
        result=stage_measure('down',rgba,owned,g)
        self.assertFalse(result['passed'])
        self.assertEqual(result['exposedPixelsBelowEarLine'],1)
        self.assertGreater(result['downHalfWidthDifference'],2)
        self.assertGreater(result['maxForeheadLuminanceValley'],10)


if __name__=='__main__':unittest.main()
