import unittest
import capture_eye_sclera_identify as identify

class ScleraIdentifyTest(unittest.TestCase):
    def test_visible_right_temporal_and_two_times_bound_are_both_required(self):
        self.assertTrue(identify.judge_pair(10,20)['passed'])
        self.assertFalse(identify.judge_pair(10,0)['passed'])
        self.assertFalse(identify.judge_pair(10,21)['passed'])
        self.assertFalse(identify.judge_pair(0,1)['passed'])

if __name__=='__main__':unittest.main()
