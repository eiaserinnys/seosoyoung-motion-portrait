"""Contracts for the hair-only measurements, independent of WebGL fixtures."""
import numpy as np
from pipeline.probe_hair_only import outline_rows, bead_centers, top_gate, neutral_difference, hairline_metrics, assess, stem_gap, cheek_cover_rows, fill_holes, summarize_metrics


def test_outline_samples_canvas_rows_and_keeps_missing_rows_explicit():
    mask = np.zeros((740, 1024), bool)
    mask[100:121, 200:700] = True
    rows = outline_rows(mask)
    assert [row['y'] for row in rows] == list(range(80, 401, 20))
    assert rows[0]['span'] is None
    assert rows[1]['span'] == [200, 699]


def test_down_crop_is_a_bound_and_up_is_a_measured_target():
    assert top_gate(-30, 4, 25)['pass']
    assert not top_gate(-30, 5, 25)['pass']
    assert top_gate(30, 100, 100)['pass']
    assert not top_gate(30, 109, 100)['pass']


def test_beads_use_body_frame_instead_of_face_registration():
    rgb = np.zeros((720, 720, 3), np.uint8)
    rgb[510:530, 140:180] = [20, 120, 220]
    rgb[510:530, 510:550] = [20, 120, 220]
    beads = bead_centers(rgb, [19, 72], np.zeros((720, 720), bool))
    assert beads['left']['refFrame'] == [311.5, 539.5]
    assert beads['left']['refCanvas'] == [330.5, 611.5]
    assert beads['right']['refFrame'] == [681.5, 539.5]


def test_neutral_pixel_comparison_cannot_accept_a_changed_pixel():
    a = np.zeros((10, 10, 4), np.uint8)
    b = a.copy()
    assert neutral_difference(a, b) == 0
    b[3, 4, 0] = 1
    assert neutral_difference(a, b) == 1


def test_hairline_measurement_detects_a_lowered_fringe_and_lost_forehead_area():
    ref = np.zeros((740, 1024), bool)
    render = ref.copy()
    ref[200:500, 400:600] = True
    render[210:500, 400:600] = True
    result = hairline_metrics(ref, render, 400)
    assert result['hairlineMedianError'] == 10
    assert result['foreheadExposure']['ratioError'] < 0


def test_acceptance_gate_rejects_the_observed_top_bead_and_neutral_regressions():
    poses = [dict(name=f'{yaw},{pitch}', yaw=yaw, pitch=pitch, topY=(60 if yaw else 97) if pitch > 0 else 0,
                  topGate={'pass': True} if pitch else None, neutralChangedPixels=0,
                  neutralVertexHashMatches=True, exterior=[{'y': y, 'delta': [0, 0]} for y in range(80, 521, 20)],
                  hairlineMedianError=0, foreheadExposure={'ratioError': 0},
                  nearCheekCover=[{'y': 520, 'gap': 0}],
                  beads={s: {'dBodyFrame': [0, 0], 'stemGap': 0} for s in ['left', 'right']})
             for yaw in [-45, 0, 45] for pitch in [-30, 0, 30]]
    assert all(c['pass'] for c in assess(poses))
    poses[0]['beads']['left']['dBodyFrame'][1] = -119
    poses[1]['neutralChangedPixels'] = 1
    poses[2]['topGate']['pass'] = False
    failures = [c for c in assess(poses) if not c['pass']]
    assert {c['gate'] for c in failures} == {'left bead body frame', 'neutral pixels', 'crown top'}
    poses[5]['exterior'][1]['delta'] = None
    import json
    serialized = json.dumps(assess(poses), allow_nan=False)
    assert 'Missing rendered or reference span' in serialized


def test_stem_gap_rejects_a_gap_in_the_central_columns_without_counting_side_overlap():
    hair = np.zeros((200, 100), bool)
    tie = hair.copy()
    tie[100:130, 30:70] = True
    hair[20:101, 30:70] = True
    assert stem_gap(hair, tie) == 0
    hair[91:101, 38:62] = False
    assert stem_gap(hair, tie) == 10
    hair[:, 38:62] = False
    assert stem_gap(hair, tie) == 81


def test_stem_gap_rejects_one_empty_column_even_when_an_adjacent_column_connects():
    hair = np.zeros((200, 100), bool)
    tie = hair.copy()
    tie[100:130, 30:70] = True
    hair[20:101, 30:70] = True
    hair[85:101, 50] = False
    assert stem_gap(hair, tie) == 16


def test_visible_face_closes_an_occluded_hair_column_but_leaves_real_background_gaps():
    hair = np.zeros((200, 100), bool)
    face = hair.copy()
    tie = hair.copy()
    tie[100:130, 30:70] = True
    face[20:101, 30:70] = True
    assert stem_gap(hair, tie) == 81
    assert stem_gap(hair | face, tie) == 0
    face[84:101, 50] = False
    assert stem_gap(hair | face, tie) == 17


def test_approved_top_gate_uses_pose_bounds_and_rejects_regressions():
    from pipeline import probe_hair_only as probe
    for yaw, pitch, boundary in [(0, 30, 100), (-45, 30, 65), (45, 30, 65), (-45, -30, 12), (45, -30, 12)]:
        assert probe.approved_top_gate(yaw, pitch, boundary)['pass']
        assert not probe.approved_top_gate(yaw, pitch, boundary + 1)['pass']
    assert probe.approved_top_gate(0, 30, 84)['pass']
    assert not probe.approved_top_gate(0, 30, 83)['pass']
    assert probe.approved_top_gate(0, -30, 7) is None


def test_foreground_stem_gate_accepts_15_and_rejects_16_pixels():
    poses = [dict(name='up', yaw=0, pitch=30, topY=97, topGate={'pass': True},
                  neutralChangedPixels=None, neutralVertexHashMatches=None,
                  exterior=[{'y': y, 'delta': [0, 0]} for y in range(80, 521, 20)],
                  hairlineMedianError=0, foreheadExposure={'ratioError': 0}, nearCheekCover=None,
                  beads={s: {'dBodyFrame': [0, 0], 'stemGap': 15} for s in ['left', 'right']})]
    checks = assess(poses)
    stems = [c for c in checks if c['gate'].endswith('foreground stem gap')]
    assert len(stems) == 2 and all(c['pass'] for c in stems)
    poses[0]['beads']['left']['stemGap'] = 16
    assert any(not c['pass'] for c in assess(poses) if c['gate'] == 'left foreground stem gap')


def test_summary_keeps_worst_residual_and_does_not_mutate_raw_samples():
    raw = {'poses': [{'exterior': [{'delta': [-9, 4]}],
                      'posteriorProjectionShift': [{'y': 100, 'shift': [0, -12]}, {'y': 101, 'shift': [3, 0]}],
                      'nearCheekCover': [{'y': 520, 'gap': 7}],
                      'hairline': [{'delta': 4}], 'hairlineMedianError': 4}],
           'checks': [{'gate': 'near cheek background gap', 'pass': False, 'value': [{'y': 520, 'gap': 7}]}]}
    summary = summarize_metrics(raw)
    p = summary['poses'][0]
    assert p['posteriorProjectionShift']['maximum'] == 12
    assert p['posteriorProjectionShift']['worstRow']['y'] == 100
    assert p['exterior']['maxAbsoluteError'] == 9
    assert p['nearCheekCover']['maximumGap'] == 7
    assert summary['checks'][0]['value']['maximumGap'] == 7
    assert isinstance(raw['poses'][0]['exterior'], list)


def test_cropped_hair_does_not_turn_the_outside_background_into_a_filled_hole():
    mask = np.zeros((20, 20), bool)
    mask[:, 9:11] = True
    np.testing.assert_array_equal(fill_holes(mask), mask)
    mask[5:15, 5:15] = True
    mask[7:13, 7:13] = False
    assert fill_holes(mask)[7:13, 7:13].all()


def test_near_cheek_scan_rejects_background_between_skin_and_crown():
    face = np.zeros((740, 1024), bool)
    hair = face.copy()
    face[500:561, 400:600] = True
    hair[500:561, 600:650] = True
    assert all(r['gap'] == 0 for r in cheek_cover_rows(face, hair, 'right', range(500, 561)))
    hair[520, 600:604] = False
    assert cheek_cover_rows(face, hair, 'right', [520]) == [{'y': 520, 'gap': 4}]
