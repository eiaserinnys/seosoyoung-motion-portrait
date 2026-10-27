"""A1 must preserve every neutral viewer pixel while replacing face and ear calibration."""
from pathlib import Path
import numpy as np
import pytest
from pipeline.face_outline_stage_a.neutral_render import render_pair

@pytest.mark.timeout(60)
def test_neutral_face_and_ear_update_changes_zero_rendered_pixels():
 before,after=render_pair(Path(__file__).resolve().parents[1])
 delta=np.abs(after.astype(np.int16)-before.astype(np.int16))
 assert int(np.count_nonzero(np.any(delta,axis=2)))==0
 assert int(delta.max())==0
