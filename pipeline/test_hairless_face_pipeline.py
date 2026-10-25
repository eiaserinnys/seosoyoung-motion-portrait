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


class AngleDomainGateTest(unittest.TestCase):
    def test_joint_wrapper_is_allowed_but_the_prior_stays_protected(self):
        paths='bundle/hair-shell.mjs\nbundle/rig.mjs\npipeline/hair_solve_quality.py\n'
        with patch.object(verify.subprocess,'check_output',return_value=paths):
            self.assertEqual(verify.protected_changes(),['bundle/hair-shell.mjs'])

    def test_only_approved_angle_bounds_are_exempt_from_rig_protection(self):
        old=dict(parameters=[dict(id='ParamAngleX',min=-50,max=50,default=0)],meshes=[dict(id='face-skin')],
                 deformers=[dict(id='WarpHead',keyforms=dict(ParamAngleX=[dict(value=v,output=v) for v in [-50,0,50]]))])
        current=copy.deepcopy(old);current['parameters'][0].update(min=-90,max=90)
        current['deformers'][0]['keyforms']['ParamAngleX']=[dict(value=v,output=v) for v in [-90,-50,0,50,90]]
        self.assertTrue(verify.angle_domain_only(old,current))
        for target,key,value in [(current['meshes'][0],'id','changed'),(current['parameters'][0],'max',91),
                                 (current['parameters'][0],'default',1)]:
            saved=target[key];target[key]=value
            self.assertFalse(verify.angle_domain_only(old,current))
            target[key]=saved
        current['deformers'][0]['keyforms']['ParamAngleX'][0]['output']=-89
        self.assertFalse(verify.angle_domain_only(old,current))


class DiagonalJawTest(unittest.TestCase):
    def pose(self, kind, measured):
        rows=[[450,450,300,450,700],[480,480,310,480,690],
              [548,548,320,548,680],[570,570,330,570,670]]
        samples=[{s:dict(validBG=False,**{'class':kind},xCanvas=measured[j])
                  for j,s in enumerate(['left','right'])} for _ in rows]
        return dict(yaw=-45,pitch=30,rows=rows,boundarySamples=samples)

    def test_skin_bounds_only_inward_and_eases_from_480(self):
        p=self.pose('SKIN',[350,650]);fit.correct_up_diagonal(p)
        self.assertEqual(p['rows'][0][2::2],[300,700])
        self.assertEqual(p['rows'][1][2::2],[310,690])
        np.testing.assert_allclose(p['rows'][2][2::2],[335,665])
        self.assertGreater(p['rows'][-1][2],330)
        self.assertLess(p['rows'][-1][4],670)
        p=self.pose('SKIN',[200,800]);before=copy.deepcopy(p)
        fit.correct_up_diagonal(p);self.assertEqual(p,before)

    def test_hair_accepts_only_outward_evidence_above_three_pixels(self):
        p=self.pose('HAIR',[326,674]);fit.correct_up_diagonal(p)
        self.assertLess(p['rows'][-1][2],330)
        self.assertGreater(p['rows'][-1][4],670)
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

    def test_chin_rows_keep_prior_plus_offset_for_every_class_after_monotone(self):
        for kind in ['SKIN','HAIR','BG']:
            p=self.pose(kind,[200,800])
            p['rows']+=[[y,y,400,y,600] for y in [590,600,610,616]]
            p['priorRows']=[[r[0],r[2]-2,r[4]+2] for r in p['rows']]
            p['offsets']={'left':2.,'right':-2.}
            p['boundarySamples']+=[copy.deepcopy(p['boundarySamples'][-1]) for _ in range(4)]
            fit.correct_up_diagonal(p)
            for i,row in enumerate(p['rows']):
                if row[0] in {600,610,616}:
                    self.assertEqual(row[2::2],[p['priorRows'][i][1]+2,p['priorRows'][i][2]-2])
                np.testing.assert_allclose(row[1::2],[row[0]]*2)

    def test_570_to_600_ramp_returns_to_base_at_three_quarters_then_one(self):
        p=self.pose('SKIN',[350,650])
        p['rows']+=[[590,590,340,590,660],[600,600,400,600,600],
                    [610,610,420,610,580],[616,616,440,616,560]]
        p['boundarySamples']+=[copy.deepcopy(p['boundarySamples'][-1]) for _ in range(4)]
        p['priorRows']=[[r[0],r[2],r[4]] for r in p['rows']];p['offsets']={'left':0.,'right':0.}
        base=copy.deepcopy(p['rows']);fit.correct_up_diagonal(p)
        # At590 the SKIN result contributes25%, the original contour75%.
        inward=340+float(fit.sphere_ease(110/136))*10
        self.assertAlmostEqual(p['rows'][4][2],.25*inward+.75*340)
        self.assertAlmostEqual(p['rows'][4][4],1000-p['rows'][4][2])
        for i in [5,6,7]:self.assertEqual(p['rows'][i][2::2],base[i][2::2])

    def test_generated_rows_match_scope_chin_and_single_peak_contract(self):
        data=json.loads((fit.ROOT/'source/viewer/outline-traces.json').read_text())
        before=json.loads(subprocess.check_output(
            ['git','show','f6112b6:source/viewer/outline-traces.json'],text=True,timeout=60))
        old={p['name']:p for p in before['poses']}
        for p in data['poses']:
            if not p['name'].startswith('up') and p['name'] not in {'down-left','down-right'}:
                self.assertEqual(p['rows'],old[p['name']]['rows'],p['name'])
            elif p['name'].startswith('up') and p['yaw']:
                mid=(p['rows'][-1][2]+p['rows'][-1][4])/2
                mirror=next(q for q in data['poses'] if q['pitch']==p['pitch'] and q['yaw']==-p['yaw'])
                own=p['priorRows'][-1][1:];other=mirror['priorRows'][-1][1:]
                own_mid=(sum(own)+p['offsets']['left']+p['offsets']['right'])/2
                other_mid=(sum(other)+mirror['offsets']['left']+mirror['offsets']['right'])/2
                self.assertAlmostEqual(mid,(own_mid+1004-other_mid)/2)
                self.assertLessEqual(abs(mid-{'up-left':430.1,'up-right':573.9}[p['name']]),10)
                jaw=verify.diagonal_jaw({'name':p['name'],'outline':p})
                self.assertEqual([r[0] for r in jaw['rows']],[570,590,600,610,616])
                self.assertEqual(jaw['leftMaxOutwardStep'],0)
                self.assertEqual(jaw['rightMaxOutwardStep'],0)
                self.assertLessEqual(jaw['maxSecondDifference'],15)
                self.assertLessEqual(jaw['maxMidpointReversal'],5)
                self.assertNotEqual(p['rows'],old[p['name']]['rows'])
                for i,row in enumerate(p['rows']):
                    if row[0] in {600,610,616}:
                        measured=p['measuredRows'][i]
                        np.testing.assert_allclose(measured[2::2],np.array(p['priorRows'][i][1:])+[p['offsets']['left'],p['offsets']['right']])
            if p['name'].startswith('up'):
                mirror=next(q for q in data['poses'] if q['pitch']==p['pitch'] and q['yaw']==-p['yaw'])
                def chin(q):
                    r=q['registration']
                    return r['eyeMid'][1]+(q['rawYChin']-r['referenceEyeMid'][1])*r['scale']
                self.assertAlmostEqual(p['yChin'],(chin(p)+chin(mirror))/2)
                self.assertLess(p['yChin'],old[p['name']]['yChin']-30)
            if p['pitch']:
                shape=verify.outline_shape({'name':p['name'],'outline':p})
                self.assertEqual((shape['peakRestY'],shape['peakCount']),(450,1))
                self.assertLessEqual(shape['maxLowerWidthIncrease'],.5)
                self.assertLessEqual(shape['maxLowerHalfWidthIncrease'],.5)
            if p['name'] in {'down-left','down-right'}:
                self.assertAlmostEqual(p['yChin'],608.000150279906)
                dr=next(q for q in data['poses'] if q['name']=='down-right')
                if p['name']=='down-left':
                    self.assertEqual(p['rows'],[[sy,yr,1004-xr,yl,1004-xl] for sy,yl,xl,yr,xr in dr['rows']])
                    for a,b in zip(p['boundarySamples'],dr['boundarySamples']):
                        for left,right in [('left','right'),('right','left')]:
                            self.assertEqual(a[left]['xCanvas'],1004-b[right]['xCanvas'])
                            self.assertEqual(a[left]['validBG'],b[right]['validBG'])
            elif not p['name'].startswith('up'):self.assertEqual(p['yChin'],old[p['name']]['yChin'])


class ApprovedDiagonalGateTest(unittest.TestCase):
    def valid(self):
        row=copy.deepcopy(next(p for p in json.loads((fit.ROOT/'docs/face_hairless_metrics.json').read_text())['poses'] if p['name']=='up-left'))
        row.update(redBgPixels=120,redBgMaxDepth=2.,cyanPixels=1500,cyanMaxDepth=8.,
                   redHairPixels=1000000,redHairMaxDepth=1000.,
                   diagonalJaw=dict(chinMidpointError=10.,leftMaxOutwardStep=0.,rightMaxOutwardStep=0.,
                                    maxSecondDifference=15.,maxMidpointReversal=5.))
        return row

    def test_exact_approved_limits_pass_and_hair_remains_reference_only(self):
        for name,yaw in [('up-left',-45),('up-right',45)]:
            row=self.valid();row.update(name=name,yaw=yaw)
            self.assertEqual(verify.failures([row],[]),[])

    def test_every_approved_gate_rejects_an_injected_violation(self):
        for key,value in [('redBgPixels',121),('redBgMaxDepth',2.001),('cyanPixels',1501),('cyanMaxDepth',8.001)]:
            with self.subTest(key=key):
                row=self.valid();row[key]=value
                self.assertTrue(verify.failures([row],[]),key)
        for key,value in [('chinMidpointError',10.001),('leftMaxOutwardStep',.001),('rightMaxOutwardStep',.001),
                          ('maxSecondDifference',15.001),('maxMidpointReversal',5.001)]:
            with self.subTest(key=key):
                row=self.valid();row['diagonalJaw'][key]=value
                self.assertTrue(verify.failures([row],[]),key)

    def test_jaw_summary_measures_injected_outward_step_and_reversal(self):
        rows=[[y,y,400+i*10,y,600-i*20] for i,y in enumerate([570,590,600,610,616])]
        p=dict(name='up-left',outline={'rows':rows})
        summary=verify.diagonal_jaw(p)
        self.assertEqual(summary['leftMaxOutwardStep'],0)
        self.assertEqual(summary['rightMaxOutwardStep'],0)
        self.assertEqual(summary['maxSecondDifference'],0)
        self.assertEqual(summary['maxMidpointReversal'],0)
        rows[1][4]=601;summary=verify.diagonal_jaw(p)
        self.assertEqual(summary['rightMaxOutwardStep'],1)
        self.assertEqual(summary['maxSecondDifference'],42)
        self.assertEqual(summary['maxMidpointReversal'],5.5)


class SilhouetteDifferenceTest(unittest.TestCase):
    def difference(self, alpha, reference):
        rgba=np.zeros((1024,1024,4),np.uint8);rgba[:,:,3]=alpha
        p=dict(name='fixture',comparisonTop=[[0,347],[1024,347]])
        with patch.object(capture,'raw_mask',return_value=reference),patch.object(
                capture,'matrix',return_value=np.array([[1,0,0],[0,1,0]],float)),patch.object(
                capture,'warped_reference',return_value=np.zeros((1024,1024,3),np.uint8)):
            return capture.silhouette_difference(p,Image.fromarray(rgba))

    def test_alpha_threshold_region_counts_overlap_and_depth(self):
        alpha=np.zeros((1024,1024),np.uint8);ref=alpha.copy()
        alpha[400:405,400:405]=255;ref[400:405,500:505]=255
        alpha[500:505,400:405]=128;ref[500:505,400:405]=255
        alpha[200:205,400:405]=255;ref[200:205,500:505]=255
        alpha[600:605,400:405]=127
        red,hair,cyan,gray,values=self.difference(alpha,ref)
        self.assertEqual(values,dict(redPixels=25,cyanPixels=25,redMaxDepth=3.,cyanMaxDepth=3.,
                                    redBgPixels=25,redHairPixels=0,redBgMaxDepth=3.,redHairMaxDepth=0.))
        self.assertFalse(hair.any())
        self.assertEqual(np.count_nonzero(gray),25)
        self.assertFalse(red[:347].any() or cyan[:347].any())
        self.assertTrue(red[400,400] and cyan[400,500] and gray[500,400])
        self.assertFalse(red[600,400])

    def test_identical_empty_masks_have_zero_depth(self):
        empty=np.zeros((1024,1024),np.uint8)
        *_,values=self.difference(empty,empty)
        self.assertEqual(values,dict(redPixels=0,cyanPixels=0,redMaxDepth=0.,cyanMaxDepth=0.,
                                    redBgPixels=0,redHairPixels=0,redBgMaxDepth=0.,redHairMaxDepth=0.))

    def test_reference_is_registered_with_nearest_neighbor(self):
        raw=np.zeros((720,720),np.uint8);raw[400:405,400:405]=255
        p=dict(name='fixture',comparisonTop=[[0,347],[1024,347]])
        rgba=Image.new('RGBA',(1024,1024))
        with patch.object(capture,'raw_mask',return_value=raw),patch.object(
                capture,'matrix',return_value=np.array([[1,0,20],[0,1,30]],float)),patch.object(
                capture,'warped_reference',return_value=np.zeros((1024,1024,3),np.uint8)):
            _,_,cyan,_,values=capture.silhouette_difference(p,rgba)
        self.assertTrue(cyan[430:435,420:425].all())
        self.assertEqual(values['cyanPixels'],25)

    def test_diff_visual_uses_exact_colors_and_stays_inside_region(self):
        masks=[np.zeros((1024,1024),bool) for _ in range(4)]
        for mask,x in zip(masks,[400,410,420,430]):mask[400,x]=True
        p=dict(comparisonTop=[[0,347],[1024,347]])
        tile=np.array(capture.difference_image(p,Image.new('RGB',(1024,1024),'white'),masks))
        self.assertEqual(tuple(tile[400,400]),(255,64,64))
        self.assertEqual(tuple(tile[400,410]),(191,95,95))
        self.assertEqual(tuple(tile[400,420]),(64,224,255))
        self.assertEqual(tuple(tile[400,430]),(128,128,128))
        self.assertEqual(tuple(tile[200,400]),capture.BG)

    def test_registered_rgb_39_is_background_40_is_occlusion(self):
        rgba=np.zeros((1024,1024,4),np.uint8)
        rgba[400:405,400:405,3]=255;rgba[400:405,410:415,3]=255
        warped=np.full((1024,1024,3),39,np.uint8);warped[400:405,410:415,2]=40
        p=dict(name='fixture',comparisonTop=[[0,347],[1024,347]])
        with patch.object(capture,'raw_mask',return_value=np.zeros((1024,1024),np.uint8)),patch.object(
                capture,'matrix',return_value=np.array([[1,0,0],[0,1,0]],float)),patch.object(
                capture,'warped_reference',return_value=warped):
            bg,hair,cyan,gray,values=capture.silhouette_difference(p,Image.fromarray(rgba))
        self.assertEqual(values,dict(redPixels=50,cyanPixels=0,redMaxDepth=3.,cyanMaxDepth=0.,
                                    redBgPixels=25,redHairPixels=25,redBgMaxDepth=3.,redHairMaxDepth=3.))
        self.assertFalse(np.any(bg&hair))
        self.assertEqual(np.count_nonzero(bg|hair),values['redPixels'])

    def test_geometry_overlay_draws_solid_bg_and_half_transparent_hair(self):
        masks=[np.zeros((1024,1024),bool) for _ in range(4)]
        for mask,x in zip(masks,[400,410,420,430]):mask[400,x]=True
        p=dict(meshes=[dict(id='face-skin',triangles=[])],boundary=[],feature={'centers':{}})
        with patch.object(capture,'reference_boundary'):
            out=np.array(capture.geometry_overlay(p,Image.new('RGB',(1024,1024),'white'),masks,grid=False))
        self.assertEqual(tuple(out[400,400]),(255,64,64))
        self.assertEqual(tuple(out[400,410]),(255,159,159))

    def test_difference_label_preserves_all_metrics_and_fits_the_tile(self):
        values=dict(redBgPixels=100000,redBgMaxDepth=100.,redHairPixels=100000,cyanPixels=100000,cyanMaxDepth=100.)
        label=capture.difference_label('down-right',values)
        self.assertEqual(label.replace('\n',' '),'down-right | R_bg:100000(100.0px) R_hair:100000 C:100000(100.0px)')
        draw=capture.ImageDraw.Draw(Image.new('RGB',(340,340)))
        self.assertLessEqual(draw.multiline_textbbox((6,6),label)[2],334)

    def test_metrics_write_keeps_existing_fields_and_refreshes_difference(self):
        values=dict(redPixels=25,cyanPixels=9,redMaxDepth=3.,cyanMaxDepth=2.,
                    redBgPixels=16,redHairPixels=9,redBgMaxDepth=2.,redHairMaxDepth=2.)
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
