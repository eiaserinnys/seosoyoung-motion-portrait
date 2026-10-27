"""Six small contracts for the specified 2e field, each bounded by 30 seconds."""
import signal
import unittest
import numpy as np
from scipy import sparse
import fit2e_core as e


class Contracts(unittest.TestCase):
    def setUp(self):
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30 seconds')))
        signal.alarm(30)

    def tearDown(self):
        signal.alarm(0)

    def test_1_phi(self):
        nodes=np.array([[y,p] for p in [-27,0,27] for y in [-37,0,37]],float)
        weights=e.phi(np.vstack([nodes,[90,0]]),nodes)
        np.testing.assert_allclose(weights.sum(1),1,atol=1e-14)
        self.assertTrue(np.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all())
        # Real key angles become available only after the authorized pose fit.
        from fit_core import FIT,read_json,write_json
        path=FIT/'results/fit2e/poses-refined.json'
        if path.exists():
            keys=read_json(FIT/'results/fit2e/prepare.json')['split']['keys']
            poses={r['frame']:r['refined_pose'] for r in read_json(path)['frames']}
            actual=np.array([poses[f][:2] for f in keys])
            checked=e.phi(np.vstack([actual,[90,0]]),actual); values=np.diag(checked[:9])
            np.testing.assert_allclose(checked.sum(1),1,atol=1e-14)
            self.assertTrue(np.isfinite(checked).all() and (checked>=0).all() and (checked<=1).all())
            report={'key_frames':keys,'nodes_deg':actual.tolist(),'self_weights':values.tolist(),
                    'passed':True,'self_weight_is_measurement_only':True,'sigma_deg':18.,'pitch_weight':1.4,
                    'correction':'담당 채택: 자기 가중 하한 대신 실측값만 기록'}
            write_json(FIT/'results/fit2e/basis-check.json',report)
            print('ACTUAL KEY SELF PHI',report,flush=True)

    def test_2_loads(self):
        weights=np.array([[.8,.2],[.4,.6]])
        loads=e.node_loads(weights,np.array([100,200]),100,np.array([[0,0],[37,0]]))
        np.testing.assert_allclose(loads['s'],[.96,.76])
        np.testing.assert_allclose(loads['lambda'],1000*loads['s'])
        self.assertTrue((loads['s']>=.5).all())

    def test_3_clamp(self):
        import torch
        value=torch.tensor([[15.,-14.,3.,1.,2.,3.]],requires_grad=True)
        e.clamp_rotation(value)
        np.testing.assert_array_equal(value.detach().numpy(),[[12.,-12.,3.,1.,2.,3.]])
        value=torch.tensor([[75.,-70.,15.,1.,2.,3.]],requires_grad=True)
        e.clamp_rotation(value,.2)
        np.testing.assert_array_equal(value.detach().numpy(),[[60.,-60.,15.,1.,2.,3.]])
        # Three-ring exclusion is the requested additional preparation contract.
        tri=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4]])
        self.assertTrue(e.excluded_vertices_3ring(tri,6,np.zeros(6,bool)).all())
        grid=np.arange(17**2).reshape(17,17); a=grid[:-1,:-1].ravel()
        tri=np.vstack([np.column_stack([a,a+1,a+18]),np.column_stack([a,a+18,a+17])])
        selected=e.excluded_vertices_3ring(tri,17**2,np.zeros(17**2,bool))
        self.assertTrue(selected[grid[3,8]])
        self.assertFalse(selected[grid[4,8]])

    def test_4_matvec(self):
        rng=np.random.default_rng(11); n=200; k=3
        xx,yy=np.meshgrid(np.arange(20.),np.arange(10.)); vertices=np.column_stack([xx.ravel(),yy.ravel(),np.zeros(n)])
        a=np.arange(n).reshape(10,20)[:-1,:-1].ravel()
        triangles=np.vstack([np.column_stack([a,a+1,a+21]),np.column_stack([a,a+21,a+20])])
        lap,mass=e.geometric_operator(vertices,triangles)
        energy=lap.T@sparse.diags(1/mass)@lap; weights=rng.uniform(size=(3,k)); weights/=weights.sum(1)[:,None]
        constraints=[sparse.random(17,n*3,density=.02,random_state=rng,format='csr') for _ in range(3)]
        rows=[{'c':c,'w':rng.uniform(.1,3,size=17),'b':rng.normal(size=17)} for c in constraints]
        loads=e.node_loads(weights,np.full(3,17),17,np.array([[-37,0],[0,0],[37,0]]))
        system=e.FieldSystem(energy,rows,weights,loads,rho=1e4)
        data=sparse.vstack([sparse.hstack([weights[f,j]*r['c'] for j in range(k)]) for f,r in enumerate(rows)])
        explicit=data.T@sparse.diags(np.concatenate([r['w'] for r in rows]))@data
        average=np.ones((n,n))/n**2
        for j in range(k):
            block=loads['lambda'][j]*sparse.kron(energy,sparse.eye(3))+loads['mu'][j]*sparse.eye(3*n)
            block+=sparse.kron(sparse.csr_matrix(average),sparse.eye(3))*1e4
            select=sparse.csr_matrix(([1.],([j],[0])),shape=(k,1))
            explicit+=sparse.kron(select@select.T,block)
        coupling=sparse.diags(loads['nu'].sum(1))-sparse.csr_matrix(loads['nu'])
        explicit+=sparse.kron(coupling,sparse.eye(n*3))
        x=rng.normal(size=k*n*3)
        np.testing.assert_allclose(system.matvec(x),explicit@x,atol=1e-8,rtol=0)
        from scipy.sparse.linalg import splu
        preconditioner=system.block_preconditioner()
        expected=np.concatenate([splu(explicit[j*n*3:(j+1)*n*3,j*n*3:(j+1)*n*3].tocsc()).solve(x[j*n*3:(j+1)*n*3])
                                 for j in range(k)])
        np.testing.assert_allclose(preconditioner@x,expected,atol=1e-8,rtol=0)
        gs=system.gauss_seidel_preconditioner()
        y=rng.normal(size=len(x))
        self.assertAlmostEqual(float(x@(gs@y)),float(y@(gs@x)),places=8)
        self.assertGreater(float(x@(gs@x)),0.)

    def test_5_split(self):
        keys=[1000+i for i in range(9)]
        rows=[{'frame':i,'role':'sample'} for i in range(121)]+[{'frame':i,'role':'key'} for i in keys]
        split=e.split_frames(rows,{0,1,1000})
        self.assertEqual(len(split['heldout']),30)
        self.assertFalse(set(split['train'])&set(split['heldout']))
        self.assertTrue(set(keys)-{1000}<=set(split['train']))
        self.assertFalse({0,1,1000}&set(split['noise']))

    def test_6_noise(self):
        points=np.array([[1.,1.],[2.,2.],[3.,1.]])
        np.testing.assert_equal(e.symmetric_distance(points,points),{'median_px':0.,'max_px':0.})


if __name__=='__main__':
    unittest.main(verbosity=2)
