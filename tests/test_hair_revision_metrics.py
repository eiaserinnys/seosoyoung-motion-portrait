"""Approved overhang contract, with the same mask fixtures as the hair probe."""
import numpy as np
import pytest
from pipeline.hair_revision_metrics import posterior_metrics


def masks(upper_base, upper_current, yaw):
    reference = np.zeros((740, 1024), bool)
    reference[80:521, 200:700] = True
    baseline, current = reference.copy(), reference.copy()
    baseline[80:221] = False
    current[80:221] = False
    baseline[80:221, 200:upper_base] = True
    current[80:221, 200:upper_current] = True
    baseline[340:441, 700:730] = True
    current[340:441, 700:720] = True
    values = [reference, current, baseline]
    return values if yaw < 0 else [np.fliplr(mask) for mask in values]


@pytest.mark.parametrize('yaw', [-45, 45])
def test_widening_inside_reference_is_not_an_overhang_regression(yaw):
    result = posterior_metrics(*masks(600, 650, yaw), yaw)
    assert result['improvement'] == 10
    assert result['upperMaximumOverhang'] == -50
    assert result['upperMaximumIncrease'] == 0
    assert result['pass']


@pytest.mark.parametrize('yaw', [-45, 45])
def test_positive_overhang_increase_accepts_thirteen_and_rejects_fourteen_pixels(yaw):
    for current, increase in [(774, 12), (775, 13), (776, 14)]:
        result = posterior_metrics(*masks(762, current, yaw), yaw)
        assert result['upperMaximumOverhang'] == current-700
        assert result['upperMaximumIncrease'] == increase
        assert result['pass'] is (increase <= 13)


@pytest.mark.parametrize('yaw', [-45, 45])
def test_upper_guard_only_checks_inclusive_rows_120_through_200(yaw):
    reference, current, baseline = masks(762, 766, -45)
    for y in [80, 115, 116, 119, 201, 216, 219, 220]:
        current[y, 766:790] = True
    frames = [reference, current, baseline]
    if yaw > 0:
        frames = [np.fliplr(mask) for mask in frames]
    result = posterior_metrics(*frames, yaw)
    assert result['upperRows'] == [120, 200]
    assert result['upperMaximumIncrease'] == 4 and result['pass']
    assert result['upperWorstRow']['y'] == 120
    for y in [120, 200]:
        outside = current.copy()
        outside[y, 766:776] = True
        frames = [reference, outside, baseline]
        if yaw > 0:
            frames = [np.fliplr(mask) for mask in frames]
        result = posterior_metrics(*frames, yaw)
        assert result['upperMaximumIncrease'] == 14 and not result['pass']
        assert result['upperWorstRow']['y'] == y
