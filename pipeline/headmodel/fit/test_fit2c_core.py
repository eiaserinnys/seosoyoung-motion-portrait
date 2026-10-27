"""Observable deformation contracts for the bounded three-frame 2c fit."""
import unittest
import numpy as np
import torch
from fit2c_core import ear_transform,normal_deform,delta_prior,constrain


class Fit2cContracts(unittest.TestCase):
    def test_ear_similarity_identity_and_feathered_translation(self):
        v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        weights=torch.tensor([[1.,.5,0.],[0.,0.,0.]])
        groups=[torch.tensor([0]),torch.tensor([2])]
        params=torch.zeros(2,7,requires_grad=True)
        np.testing.assert_allclose(ear_transform(v,params,weights,groups).detach(),v)
        with torch.no_grad():params[0,3]=2.
        out=ear_transform(v,params,weights,groups)
        np.testing.assert_allclose((out-v).detach()[:,0],[2.,1.,0.])
        out.sum().backward(); self.assertGreater(float(params.grad[0,3]),0.)

    def test_delta_is_normal_only_has_26px_limit_and_neck_is_fixed(self):
        vertices=torch.zeros(3,3); normals=torch.tensor([[0.,0.,1.]]*3)
        delta=torch.nn.Parameter(torch.tensor([40.,-40.,12.])); head=torch.tensor([True,True,False])
        beta=torch.nn.Parameter(torch.ones(100)*5); t=torch.nn.Parameter(torch.tensor(20.))
        constrain(beta,t,delta,head)
        np.testing.assert_allclose(delta.detach(),[26.,-26.,0.]); self.assertEqual(float(beta.detach().max()),3.)
        self.assertEqual(float(t.detach()),14.)
        result=normal_deform(vertices,normals,delta,head,torch.tensor(20.))
        np.testing.assert_allclose(result.detach()[:,:2],0.); np.testing.assert_allclose(result.detach()[:,2],[1.3,-1.3,0.],atol=1e-6)

    def test_delta_prior_increases_for_rough_displacement(self):
        edges=torch.tensor([[0,1],[1,2]])
        head=torch.ones(3,dtype=torch.bool); face=torch.ones(3,dtype=torch.bool)
        flat=torch.ones(3); rough=torch.tensor([0.,5.,0.])
        self.assertGreater(float(delta_prior(rough,edges,head,face)),float(delta_prior(flat,edges,head,face)))


if __name__=='__main__':unittest.main()
