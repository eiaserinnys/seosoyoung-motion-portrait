"""2b contracts: immutable rotations and bounded, stronger shape prior."""
import unittest
import torch
from run_falsify2 import fixed_cameras, constrain_parameters, shape_prior


class Falsify2Contracts(unittest.TestCase):
    def test_adam_moves_similarity_but_preserves_registered_rotation(self):
        base=torch.tensor([[-36.83,.70,0.,3.16,469.,397.],
                           [23.51,-20.15,0.,3.16,252.,317.]])
        delta=torch.nn.Parameter(torch.zeros(2,3))
        rate=torch.tensor([.004,.35,.35])
        optimizer=torch.optim.Adam([delta],lr=1.)
        for _ in range(3):
            optimizer.zero_grad()
            poses=fixed_cameras(base,delta,rate)
            ((poses[:,3:]-base[:,3:]-1)**2).sum().backward()
            optimizer.step()
        poses=fixed_cameras(base,delta,rate)
        torch.testing.assert_close(poses[:,:3],base[:,:3],rtol=0,atol=0)
        self.assertTrue(torch.all(poses[:,3:]!=base[:,3:]))

    def test_beta_and_thickness_clamp_after_update(self):
        beta=torch.nn.Parameter(torch.tensor([-4.,-2.,0.,2.,4.]))
        thickness=torch.nn.Parameter(torch.tensor(18.))
        constrain_parameters(beta,thickness)
        torch.testing.assert_close(beta,torch.tensor([-3.,-2.,0.,2.,3.]))
        self.assertEqual(float(thickness.detach()),14.)
        with torch.no_grad():thickness.fill_(-1.)
        constrain_parameters(beta,thickness)
        self.assertEqual(float(thickness.detach()),3.)

    def test_beta_l2_weight_is_point_one(self):
        value=shape_prior(torch.tensor([2.,-2.]),torch.zeros(3))
        self.assertAlmostEqual(float(value),.8,places=6)


if __name__=='__main__':unittest.main()
