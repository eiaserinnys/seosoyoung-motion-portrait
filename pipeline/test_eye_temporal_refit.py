import copy
import unittest
import numpy as np
import fit_eye_sam3 as fit

class TemporalRefitTest(unittest.TestCase):
    def test_censored_lower_bound_has_no_cost_when_model_is_wider(self):
        errors=np.array([-6.,-1.,0.,1.,6.])
        weights=fit.temporal_huber_weights(errors,np.ones(5,dtype=bool),3.)
        np.testing.assert_allclose(weights,[.5,1.,0.,0.,0.])
        np.testing.assert_allclose(fit.temporal_huber_weights(errors,np.zeros(5,dtype=bool),3.),[.5,1.,1.,1.,.5])

    def test_refit_changes_only_two_temporal_tables(self):
        previous=dict(aLaw='cos(max(theta,0))',noise={},status='old',warnings=[],left=dict(r_T=[1,2],r_N=[3,4],kappa=1,delta0=[0,.1]),right=dict(r_T=[5,6],r_N=[7,8],kappa=1))
        frozen=copy.deepcopy(previous)
        evaluation=dict(noise={},status='old',warnings=[],fit={s:dict(dTemp=dict(knotLogs=[0.,-.2])) for s in ['left','right']})
        result=fit.replace_temporal_tables(previous,evaluation)
        self.assertEqual(previous,frozen)
        for side in ['left','right']:
            self.assertEqual(result[side]['r_T'],[0.,-.2])
            result[side]['r_T']=frozen[side]['r_T']
        self.assertEqual(result,frozen)

    def test_quality_metadata_tracks_the_new_uncensored_fit(self):
        previous=dict(left=dict(r_T=[0]),right=dict(r_T=[0]),noise=dict(dTemp=dict(sigma=1.)))
        evaluation=dict(status='CONSENSUS PASS',warnings=[],noise=dict(dTemp=dict(sigma=2.)),fit={s:dict(dTemp=dict(knotLogs=[0.])) for s in ['left','right']})
        result=fit.replace_temporal_tables(previous,evaluation)
        self.assertEqual(result['noise']['dTemp']['sigma'],2.)

if __name__=='__main__':unittest.main()
