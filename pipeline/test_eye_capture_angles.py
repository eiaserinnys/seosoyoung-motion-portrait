import json
from pathlib import Path
import unittest
from eye_capture_angles import inverse_achieved, achieved

ROOT=Path(__file__).resolve().parents[1]
BOOT=json.loads((ROOT/'source/references/seedance/eye-bootstrap.json').read_text())
ROWS=json.loads((ROOT/'sam3_masks/references/seedance/selection.json').read_text())['frames']

class EyeCaptureAnglesTest(unittest.TestCase):
    def test_neutral_inverse_and_known_requested_pose(self):
        self.assertEqual(inverse_achieved(BOOT['achieved']['c'],0,0),[0.,0.])
        c=BOOT['achieved']['c'];target=achieved(c,23,-18)
        for a,b in zip(inverse_achieved(c,*target),[23,-18]):self.assertAlmostEqual(a,b,places=9)

    def test_all_24_reference_angles_are_matched(self):
        c=BOOT['achieved']['c']
        for row in ROWS:
            with self.subTest(pose=row['name']):
                requested=inverse_achieved(c,row['yaw'],row['pitch'])
                for a,b in zip(achieved(c,*requested),[row['yaw'],row['pitch']]):
                    self.assertAlmostEqual(a,b,places=9)

if __name__=='__main__':unittest.main()
