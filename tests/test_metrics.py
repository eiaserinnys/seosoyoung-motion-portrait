import unittest
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('metrics',Path(__file__).parents[1]/'tools/deformation_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MetricsTests(unittest.TestCase):
    def test_known_affine_and_injected_nonaffine_error(self):
        a=np.array([[x,y] for y in range(3) for x in range(3)],float)
        b=a@np.diag([.5,1.5])+[2,3];r=m.metrics(a,b)
        np.testing.assert_allclose(r['principal_stretches'],[1.5,.5])
        self.assertLess(r['nonaffine_rms'],1e-12)
        b[4,1]+=1
        self.assertGreater(m.metrics(a,b)['nonaffine_rms'],.1)
        self.assertGreater(m.grid_curvature_change(a,b,2,2),.1)
    def test_key_order_uses_first_axis_fastest(self):
        t={'parameters':[{'id':'x','default':0},{'id':'y','default':0}],
           'normal_binding_bands':[{'binding_indices':[0,1]}],
           'normal_parameter_bindings':[{'parameter_id':'x','keys':[-1,0,1]}, {'parameter_id':'y','keys':[-1,0,1]}],
           'warp_keyforms':[{'position_start':i*2} for i in range(9)],
           'position_floats':[v for y in [-1,0,1] for x in [-1,0,1] for v in [x,y]]}
        o={'binding_band_index':0,'keyform_start':0,'vertex_count':1}
        np.testing.assert_allclose(m.ordinary_points(t,'warp',o,{'x':.25,'y':-.5}),[[.25,-.5]])
    def test_identity_and_sphere_depth(self):
        a=np.array([[0,0],[.5,2.]])
        np.testing.assert_allclose(m.sphere_projection(a,[0,0],1,0,0),a)
        np.testing.assert_allclose(m.sphere_projection([[0,0]],[0,0],1,30,0),[[.5,0]],atol=1e-12)
if __name__=='__main__':unittest.main()
