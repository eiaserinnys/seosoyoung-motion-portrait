"""Contracts for removing stationary eyes/hair while retaining profile detail."""
import unittest
import numpy as np
from bake_face90 import clean_skin, bake_uv
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


if __name__=='__main__':unittest.main()
