"""Contracts for removing stationary eyes/hair while retaining profile detail."""
import unittest
import numpy as np
from bake_face90 import clean_skin, bake_uv, profile_canvas
from capture_side90 import changed_pixels


class ProfileBakeTest(unittest.TestCase):
    def test_pixel_invariance_gate_rejects_a_single_rgb_or_alpha_change(self):
        image=np.zeros((4,4,4),np.uint8)
        self.assertEqual(changed_pixels(image,image.copy()),0)
        for channel in [0,3]:
            changed=image.copy();changed[1,2,channel]=1
            self.assertEqual(changed_pixels(image,changed),1)

    def test_occluding_eyes_and_hair_are_removed_but_nose_and_lips_stay(self):
        rgb=np.full((720,720,3),[250,215,190],np.uint8)
        rgb[370:430,200:270]=[30,20,15]
        rgb[200:300,200:270]=[30,20,15]
        rgb[516:520,150:175]=[80,40,30]
        mask=np.full((720,720),255,np.uint8)
        clean=clean_skin(rgb,mask,'left90')
        self.assertGreater(int(clean[400,230,0]),200)
        self.assertGreater(int(clean[250,230,0]),200)
        self.assertLess(int(clean[518,160,0]),100)
        self.assertLess(np.max(np.abs(clean[350,180].astype(float)-clean[351,180])),5)

    def test_triangle_bake_samples_solved_canvas_positions(self):
        source=np.zeros((1024,1024,3),np.uint8)
        source[200:401,200:401]=[210,180,150]
        rest=np.array([[10,10],[100,10],[10,100]],float)
        solved=rest+200
        baked=bake_uv(source,rest,solved,[[0,1,2]])
        np.testing.assert_array_equal(baked[20,20],[210,180,150])
        wrong=bake_uv(source,rest,solved+500,[[0,1,2]])
        self.assertFalse(np.array_equal(baked[20,20],wrong[20,20]))

    def test_detail_cannot_copy_background_hair_outside_eroded_face(self):
        rgb=np.full((720,720,3),[250,215,190],np.uint8)
        mask=np.zeros((720,720),np.uint8);mask[200:600,165:350]=255
        rgb[470:490,140:164]=[40,25,15]
        rgb[520:540,142:164]=255
        clean=clean_skin(rgb,mask,'left90')
        self.assertTrue(np.all(clean[470:490,140:164,0]>220))
        self.assertTrue(np.all(clean[520:540,142:164,1]<230))

    def test_registered_profile_nose_and_lips_are_exactly_mirrored(self):
        for angle in ['67','90']:
            left=profile_canvas('left'+angle);right=profile_canvas('right'+angle)
            np.testing.assert_array_equal(left[440:590,260:440],right[440:590,565:745][:,::-1])

    def test_feathering_does_not_pull_white_background_into_the_detail(self):
        rgb=np.full((720,720,3),[250,215,190],np.uint8)
        mask=np.full((720,720),255,np.uint8)
        rgb[470:475,140:155]=[80,40,30];rgb[475:490,140:155]=255
        clean=clean_skin(rgb,mask,'left90')
        self.assertLess(int(clean[475,145,2]),195)


if __name__=='__main__':unittest.main()
