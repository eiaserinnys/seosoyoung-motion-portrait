"""Closed-hole and two-ring fairing contracts before the full template run."""
import unittest
import numpy as np
from egg_biharmonic import cap_loops, biharmonic_fields, two_ring_boundary


class BiharmonicContracts(unittest.TestCase):
    def test_cap_centroid_and_modes_share_the_same_average(self):
        v=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]])
        modes=np.array([v*.1,v*.2])
        out,tri,fields=cap_loops(v,np.empty((0,3),int),modes,[np.array([0,1,2])])
        np.testing.assert_allclose(out[-1],v.mean(0))
        np.testing.assert_allclose(fields[:,-1],modes.mean(1))
        self.assertEqual(len(tri),3)
        self.assertTrue(np.all(tri[:,2]==3))

    def test_two_rings_and_exact_biharmonic_linear_fields(self):
        edges=np.array([[0,1],[1,2],[2,3],[3,4]])
        ring=two_ring_boundary(np.array([True]*5),edges,np.array([0,4]))
        np.testing.assert_array_equal(ring,[0,1,3,4])
        base=np.column_stack([np.arange(5),np.zeros(5),np.zeros(5)])
        perturbed=base.copy(); perturbed[2,2]=100.
        result=biharmonic_fields(perturbed,edges,np.array([2]))
        np.testing.assert_allclose(result,base,atol=1e-12)
        field=base*.2
        np.testing.assert_allclose(biharmonic_fields(perturbed+3*field,edges,np.array([2])),
                                   result+3*biharmonic_fields(field,edges,np.array([2])),atol=1e-12)
        np.testing.assert_array_equal(result[ring],perturbed[ring])


if __name__=='__main__':unittest.main()
