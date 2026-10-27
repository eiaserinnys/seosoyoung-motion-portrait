"""Free-X projection must preserve hard observations and report infeasibility."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).parent/'face_outline_stage_a/refinement/triangle_projection.py'
spec=importlib.util.spec_from_file_location('triangle_projection',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
@pytest.mark.timeout(60)
def test_free_x_projection_closes_an_inversion_without_moving_fixed_nodes():
 rest=np.array([[0.,0.],[1.,0.],[0.,1.]])
 world=np.array([[0.,0.],[-1.,0.],[0.,1.]])
 q,report=module.project(rest,world,[[0,1,2]],{0,2})
 assert q[1,0]>0 and report['movedIds']==[1]
 assert np.array_equal(q[[0,2]],world[[0,2]])
 assert np.array_equal(q[:,1],world[:,1])
@pytest.mark.timeout(60)
def test_fixed_inversion_cannot_be_silently_accepted():
 rest=np.array([[0.,0.],[1.,0.],[0.,1.]])
 with pytest.raises(module.ProjectionConflict,match='fixed observations'):
  module.project(rest,rest[[0,2,1]],[[0,1,2]],{0,1,2})
