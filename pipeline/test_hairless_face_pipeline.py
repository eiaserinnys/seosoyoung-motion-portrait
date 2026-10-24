"""Face-only jaw correction and registered alpha difference contracts."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import capture_hairless_face as capture
import fit_outline as fit
import verify_hairless_face as verify


class DiagonalJawTest(unittest.TestCase):
    def pose(self, kind, measured):
        rows=[[450,450,300,450,700],[480,480,310,480,690],
              [548,548,320,548,680],[616,616,330,616,670]]
        samples=[{s:dict(validBG=False,**{'class':kind},xCanvas=measured[j])
                  for j,s in enumerate(['left','right'])} for _ in rows]
        return dict(yaw=-45,pitch=30,rows=rows,boundarySamples=samples)

    def test_skin_bounds_only_inward_and_eases_from_480(self):
        p=self.pose('SKIN',[350,650]);fit.correct_up_diagonal(p)
        self.assertEqual(p['rows'][0][2::2],[300,700])
        self.assertEqual(p['rows'][1][2::2],[310,690])
        np.testing.assert_allclose(p['rows'][2][2::2],[335,665])
        self.assertEqual(p['rows'][-1][2::2],[350,650])
        p=self.pose('SKIN',[200,800]);before=copy.deepcopy(p)
        fit.correct_up_diagonal(p);self.assertEqual(p,before)

    def test_hair_accepts_only_outward_evidence_above_three_pixels(self):
        p=self.pose('HAIR',[326,674]);fit.correct_up_diagonal(p)
        self.assertEqual(p['rows'][-1][2::2],[326,674])
        for measured in ([328,672],[400,600]):
            p=self.pose('HAIR',measured);before=copy.deepcopy(p)
            fit.correct_up_diagonal(p);self.assertEqual(p,before)

    def test_valid_background_and_other_seven_poses_are_unchanged(self):
        for yaw,pitch in [(0,30),(-45,-30),(0,-30),(45,-30),(-45,0),(0,0),(45,0)]:
            p=self.pose('SKIN',[350,650]);p.update(yaw=yaw,pitch=pitch)
            before=copy.deepcopy(p);fit.correct_up_diagonal(p);self.assertEqual(p,before)
        p=self.pose('SKIN',[350,650])
        for sample in p['boundarySamples']:
            for side in ['left','right']:sample[side]['validBG']=True
        before=copy.deepcopy(p);fit.correct_up_diagonal(p);self.assertEqual(p,before)

    def test_each_side_is_monotone_without_moving_the_opposite_side(self):
        p=self.pose('HAIR',[200,800]);fit.correct_up_diagonal(p)
        rows=np.array(p['rows'])
        self.assertTrue(np.all(np.diff(rows[np.argmin(rows[:,2]):,2])>=0))
        self.assertTrue(np.all(np.diff(rows[np.argmax(rows[:,4]):,4])<=0))
        # Changing only the near side must leave the far side at its own bound.
        p=self.pose('SKIN',[200,650]);fit.correct_up_diagonal(p)
        self.assertEqual([r[2] for r in p['rows']],[300,310,320,330])

    def test_generated_rows_match_scope_chin_and_single_peak_contract(self):
        data=json.loads((fit.ROOT/'source/viewer/outline-traces.json').read_text())
        before=json.loads(subprocess.check_output(
            ['git','show','83d4a80:source/viewer/outline-traces.json'],text=True,timeout=60))
        old={p['name']:p for p in before['poses']}
        for p in data['poses']:
            if p['name'] not in ['up-left','up-right']:
                self.assertEqual(p['rows'],old[p['name']]['rows'],p['name'])
            else:
                mid=(p['rows'][-1][2]+p['rows'][-1][4])/2
                self.assertLessEqual(abs(mid-{'up-left':430.1,'up-right':573.9}[p['name']]),20)
                self.assertNotEqual(p['rows'],old[p['name']]['rows'])
            if p['pitch']:
                shape=verify.outline_shape({'name':p['name'],'outline':p})
                self.assertEqual((shape['peakRestY'],shape['peakCount']),(450,1))
                self.assertLessEqual(shape['maxLowerWidthIncrease'],.5)
                self.assertLessEqual(shape['maxLowerHalfWidthIncrease'],.5)
            self.assertEqual(p['yChin'],old[p['name']]['yChin'])


class SilhouetteDifferenceTest(unittest.TestCase):
    def difference(self, alpha, reference):
        rgba=np.zeros((1024,1024,4),np.uint8);rgba[:,:,3]=alpha
        p=dict(name='fixture',comparisonTop=[[0,347],[1024,347]])
        with patch.object(capture,'raw_mask',return_value=reference),patch.object(
                capture,'matrix',return_value=np.array([[1,0,0],[0,1,0]],float)):
            return capture.silhouette_difference(p,Image.fromarray(rgba))

    def test_alpha_threshold_region_counts_overlap_and_depth(self):
        alpha=np.zeros((1024,1024),np.uint8);ref=alpha.copy()
        alpha[400:405,400:405]=255;ref[400:405,500:505]=255
        alpha[500:505,400:405]=128;ref[500:505,400:405]=255
        alpha[200:205,400:405]=255;ref[200:205,500:505]=255
        alpha[600:605,400:405]=127
        red,cyan,gray,values=self.difference(alpha,ref)
        self.assertEqual(values,dict(redPixels=25,cyanPixels=25,redMaxDepth=3.,cyanMaxDepth=3.))
        self.assertEqual(np.count_nonzero(gray),25)
        self.assertFalse(red[:347].any() or cyan[:347].any())
        self.assertTrue(red[400,400] and cyan[400,500] and gray[500,400])
        self.assertFalse(red[600,400])

    def test_identical_empty_masks_have_zero_depth(self):
        empty=np.zeros((1024,1024),np.uint8)
        *_,values=self.difference(empty,empty)
        self.assertEqual(values,dict(redPixels=0,cyanPixels=0,redMaxDepth=0.,cyanMaxDepth=0.))

    def test_reference_is_registered_with_nearest_neighbor(self):
        raw=np.zeros((720,720),np.uint8);raw[400:405,400:405]=255
        p=dict(name='fixture',comparisonTop=[[0,347],[1024,347]])
        rgba=Image.new('RGBA',(1024,1024))
        with patch.object(capture,'raw_mask',return_value=raw),patch.object(
                capture,'matrix',return_value=np.array([[1,0,20],[0,1,30]],float)):
            _,cyan,_,values=capture.silhouette_difference(p,rgba)
        self.assertTrue(cyan[430:435,420:425].all())
        self.assertEqual(values['cyanPixels'],25)

    def test_diff_visual_uses_exact_colors_and_stays_inside_region(self):
        masks=[np.zeros((1024,1024),bool) for _ in range(3)]
        for mask,x in zip(masks,[400,410,420]):mask[400,x]=True
        p=dict(comparisonTop=[[0,347],[1024,347]])
        tile=np.array(capture.difference_image(p,Image.new('RGB',(1024,1024),'white'),masks))
        self.assertEqual(tuple(tile[400,400]),(255,64,64))
        self.assertEqual(tuple(tile[400,410]),(64,224,255))
        self.assertEqual(tuple(tile[200,400]),capture.BG)

    def test_metrics_write_keeps_existing_fields_and_refreshes_difference(self):
        values=dict(redPixels=25,cyanPixels=9,redMaxDepth=3.,cyanMaxDepth=2.)
        row=dict(name='fixture',existingMetric=42,targetRms=0.,rawVisibleRms=0.,rawChinError=0.,
                 lengthErrors={},eyes={'left':{'shear':0.}},cheekError=0.)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'docs').mkdir()
            with patch.object(verify,'ROOT',root),patch.object(verify,'metrics',
                    return_value=[copy.deepcopy(row)]),patch.object(
                    verify,'protected_changes',return_value=[]),patch.object(verify,'failures',return_value=[]),patch('builtins.print'):
                # Real gates are covered by verify.self_test; isolate JSON persistence.
                with patch.object(verify,'self_test'):
                    verify.write_metrics([],{ 'fixture':values })
                result=json.loads((root/'docs/face_hairless_metrics.json').read_text())
        self.assertEqual(result['poses'],[dict(row,**values)])
        self.assertEqual(result['protectedChanges'],[])


if __name__=='__main__':unittest.main()
