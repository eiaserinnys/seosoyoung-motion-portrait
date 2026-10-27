"""Pixel-unit Gaussian angle fields; fixed-shape pose refinement and linear solve."""
import time
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator, splu, cg
from fit_core import project
from fit2d_core import linear_camera
from fit2f_core import (boundary_edges, geometric_operator, surface_contour, pair_rows)


def phi(angles,nodes,sigma=18.):
    delta=(np.asarray(angles)[:,None,:2]-np.asarray(nodes)[None,:,:2])*[1.,1.4]
    logits=-(delta**2).sum(2)/(2*sigma**2)
    logits-=logits.max(1,keepdims=True)
    values=np.exp(logits)
    return values/values.sum(1,keepdims=True)


def node_loads(weights,pairs,p_ref,nodes,sigma=18.):
    s=(np.asarray(weights)**2*np.asarray(pairs)[:,None]).sum(0)/p_ref
    if np.any(s<=0): raise ValueError('Empty angle node load')
    delta=(np.asarray(nodes)[:,None,:2]-np.asarray(nodes)[None,:,:2])*[1.,1.4]
    g=np.exp(-(delta**2).sum(2)/(2*sigma**2))
    nu=.05*g*(s[:,None]+s[None,:])/2
    np.fill_diagonal(nu,0.)
    return {'s':s,'lambda':1000*s,'mu':.01*s,'nu':nu}


def split_frames(rows,excluded=()):
    excluded=set(excluded)
    keys=[r['frame'] for r in rows if r['role']=='key']
    samples=[r['frame'] for r in rows if r['role']=='sample']
    held=set(samples[::4]); all_frames=[r['frame'] for r in rows]
    return {'train':[f for f in all_frames if f not in held and f not in excluded],
            'heldout':[f for f in all_frames if f in held and f not in excluded],
            'noise':[f for f in all_frames if f not in excluded],
            'keys':keys,'excluded':[f for f in all_frames if f in excluded]}


def excluded_vertices_3ring(tri,count,ear):
    selected=np.zeros(count,bool); selected[np.unique(boundary_edges(tri))]=True
    for _ in range(3):
        selected[np.unique(tri[selected[tri].any(1)])]=True
    return selected|np.asarray(ear,bool)


def symmetric_distance(a,b):
    values=np.r_[cKDTree(a).query(b)[0],cKDTree(b).query(a)[0]]
    return {'median_px':float(np.median(values)),'max_px':float(values.max())}


def clamp_rotation(delta,rate=1.):
    with torch.no_grad(): delta[:,:3].clamp_(-12./rate,12./rate)


def refine_poses(vertices,tri,targets,initial):
    """150 Adam steps, 30 correspondence updates, shape and thickness fixed."""
    started=time.perf_counter(); dtype=torch.float64
    base=torch.tensor(np.asarray(initial),dtype=dtype)
    rate=torch.tensor([.2,.2,.2,.0015,.20,.20],dtype=dtype)
    delta=torch.nn.Parameter(torch.zeros_like(base))
    optimizer=torch.optim.Adam([delta],lr=1.)
    cut_edges=boundary_edges(tri); history=[]; clamp_max=0.
    for outer in range(30):
        tick=time.perf_counter(); gate=15.-9.*min(outer,7)/7
        poses=(base+delta*rate).detach().numpy(); rows=[]
        for index,(pose,target) in enumerate(zip(poses,targets)):
            model=surface_contour(vertices,tri,pose,boundary_cache=cut_edges)
            pairs=pair_rows(model,target,gate); mi,ti=pairs['model_indices'],pairs['target_indices']
            if not len(mi): continue
            anchor=(vertices[model['ids'][mi]]*model['weights'][mi,:,None]).sum(1)
            rows.append((index,torch.tensor(anchor,dtype=dtype),
                         torch.tensor(target['points'][ti],dtype=dtype),
                         torch.tensor(target['normals'][ti],dtype=dtype),
                         torch.tensor(np.where(target['labels'][ti]==0,10.,1.)/3.,dtype=dtype)))
        if not rows: raise ValueError('No pose correspondences')
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True); total=0.
            for index,anchor,wanted,normals,weight in rows:
                residual=((project(anchor,base[index]+delta[index]*rate)[:,:2]-wanted)*normals).sum(1)
                loss=(weight*residual.square()).sum()
                if not torch.isfinite(loss): raise FloatingPointError('Pose loss is nonfinite')
                loss.backward(); total+=float(loss.detach())
            optimizer.step()
            # delta stores rate-scaled parameters; enforce the bound in degrees.
            clamp_rotation(delta,.2)
            clamp_max=max(clamp_max,float((delta[:,:3]*.2).detach().abs().max()))
        record={'step':(outer+1)*5,'gate_px':gate,'pairs':sum(len(r[1]) for r in rows),
                'loss_sum':total,'seconds':time.perf_counter()-tick,
                'rotation_max_abs_delta_deg':clamp_max}
        history.append(record); print('2E POSE',record,flush=True)
    poses=(base+delta*rate).detach().numpy()
    return poses,{'iterations':150,'rate':rate.tolist(),'history':history,
                  'maximum_during_optimization_deg':clamp_max,
                  'elapsed_seconds':time.perf_counter()-started}


class FieldSystem:
    """Matrix-free full normal equations. Mean penalties use rank-three updates."""
    def __init__(self,energy,rows,weights,loads,rho=1e4):
        self.energy=energy; self.rows=rows; self.phi=weights; self.loads=loads; self.rho=rho
        self.k=weights.shape[1]; self.n=energy.shape[0]; self.width=3*self.n
        self.shape=(self.k*self.width,)*2

    def matvec(self,x):
        d=np.asarray(x).reshape(self.k,self.n,3); result=np.empty_like(d)
        for k in range(self.k):
            result[k]=self.loads['lambda'][k]*(self.energy@d[k])+self.loads['mu'][k]*d[k]
            result[k]+=self.rho*d[k].mean(0)/self.n
            for l in range(self.k):
                result[k]+=self.loads['nu'][k,l]*(d[k]-d[l])
        flat=result.reshape(self.k,-1); vectors=d.reshape(self.k,-1)
        for f,row in enumerate(self.rows):
            combined=self.phi[f]@vectors
            contribution=row['c'].T@(row['w']*(row['c']@combined))
            flat+=self.phi[f,:,None]*contribution[None,:]
        return flat.ravel()

    def rhs(self):
        result=np.zeros((self.k,self.width))
        for f,row in enumerate(self.rows):
            contribution=row['c'].T@(row['w']*row['b'])
            result+=self.phi[f,:,None]*contribution[None,:]
        return result.ravel()

    def block_preconditioner(self):
        solvers=[]; identity=sparse.eye(self.width,format='csc')
        average=np.tile(np.eye(3),(self.n,1))/self.n
        for k in range(self.k):
            block=self.loads['lambda'][k]*sparse.kron(self.energy,sparse.eye(3),format='csc')
            block+=(self.loads['mu'][k]+self.loads['nu'][k].sum())*identity
            for f,row in enumerate(self.rows):
                cw=row['c'].multiply(np.sqrt(row['w'])[:,None])
                block+=self.phi[f,k]**2*(cw.T@cw)
            factor=splu(block.tocsc()); update=factor.solve(average)
            small=np.eye(3)/self.rho+average.T@update
            solvers.append((factor,update,small))
        def block_solve(k,value):
            factor,update,small=solvers[k]; z=factor.solve(value)
            return z-update@np.linalg.solve(small,average.T@z)
        self.block_solve=block_solve
        def solve(x):
            x=np.asarray(x).reshape(self.k,-1); result=[]
            for k,value in enumerate(x):
                result.append(block_solve(k,value))
            return np.asarray(result).ravel()
        return LinearOperator(self.shape,matvec=solve,dtype=np.float64)

    def gauss_seidel_preconditioner(self):
        """Symmetric block Gauss-Seidel keeps the fallback suitable for CG."""
        def offdiag(k,l,value):
            result=-self.loads['nu'][k,l]*value
            for f,row in enumerate(self.rows):
                result+=self.phi[f,k]*self.phi[f,l]*(row['c'].T@(row['w']*(row['c']@value)))
            return result
        def solve(x):
            rhs=np.asarray(x).reshape(self.k,-1); y=np.zeros_like(rhs); diagonal_y=np.zeros_like(rhs)
            for k in range(self.k):
                adjusted=rhs[k].copy()
                for l in range(k): adjusted-=offdiag(k,l,y[l])
                y[k]=self.block_solve(k,adjusted); diagonal_y[k]=adjusted
            result=np.zeros_like(rhs)
            for k in range(self.k-1,-1,-1):
                adjusted=diagonal_y[k].copy()
                for l in range(k+1,self.k): adjusted-=offdiag(k,l,result[l])
                result[k]=self.block_solve(k,adjusted)
            return result.ravel()
        return LinearOperator(self.shape,matvec=solve,dtype=np.float64)


def constraint_rows(base,tri,displacement,pose,target,gate,cut_edges):
    model=surface_contour(base+displacement,tri,pose,boundary_cache=cut_edges)
    pairs=pair_rows(model,target,gate); mi,ti=pairs['model_indices'],pairs['target_indices']
    ids=model['ids'][mi]; bary=model['weights'][mi]; normals=target['normals'][ti]
    a,t=linear_camera(pose); coefficients=bary[:,:,None]*(normals@a)[:,None,:]
    c=sparse.csr_matrix((coefficients.ravel(),(np.repeat(np.arange(len(mi)),9),
                        (ids[:,:,None]*3+np.arange(3)).ravel())),shape=(len(mi),base.size))
    original=(base[ids]*bary[:,:,None]).sum(1)@a.T+t
    return {'c':c,'b':((target['points'][ti]-original)*normals).sum(1),
            'w':np.where(target['labels'][ti]==0,10.,1.)/3.,'frame':target['frame'],
            'skin_pairs':int((target['labels'][ti]==0).sum()),
            'hair_pairs':int((target['labels'][ti]==1).sum())}


def solve_field(vertices,tri,poses,targets,sref,nodes,p_ref,sigma=18.,failure_callback=None):
    started=time.perf_counter(); base=vertices*sref; poses=np.asarray(poses).copy()
    poses[:,3]-=np.log(sref); mixing=phi(poses[:,:2],nodes,sigma)
    lap,mass=geometric_operator(base,tri); energy=(lap.T@sparse.diags(1/mass)@lap).tocsr()
    d=np.zeros((len(nodes),len(base),3)); cut_edges=boundary_edges(tri); history=[]
    load=None; fallback_used=False
    for iteration in range(8):
        tick=time.perf_counter(); gate=15.-9.*iteration/7
        rows=[constraint_rows(base,tri,np.einsum('k,kic->ic',mix,d),pose,target,gate,cut_edges)
              for mix,pose,target in zip(mixing,poses,targets)]
        counts=np.array([len(row['b']) for row in rows])
        if load is None:
            load=node_loads(mixing,counts,p_ref,nodes,sigma)
        system=FieldSystem(energy,rows,mixing,load); operator=LinearOperator(system.shape,matvec=system.matvec,dtype=np.float64)
        preconditioner=system.block_preconditioner(); rhs=system.rhs(); iterations=0
        def callback(_):
            nonlocal iterations
            iterations+=1
        solution,info=cg(operator,rhs,x0=d.ravel(),M=preconditioner,rtol=1e-6,atol=0.,maxiter=600,callback=callback)
        attempts=[{'preconditioner':'block_diagonal','iterations':iterations,'info':int(info)}]
        if info:
            if fallback_used:
                if failure_callback: failure_callback(history,attempts)
                raise RuntimeError('CG failed after the single preconditioner fallback was used')
            fallback_used=True; iterations=0
            solution,info=cg(operator,rhs,x0=solution,M=system.gauss_seidel_preconditioner(),
                             rtol=1e-6,atol=0.,maxiter=600,callback=callback)
            attempts.append({'preconditioner':'symmetric_block_gauss_seidel','iterations':iterations,'info':int(info)})
            if info:
                if failure_callback: failure_callback(history,attempts)
                raise RuntimeError('CG failed with block Gauss-Seidel fallback')
        d=solution.reshape(d.shape)
        if not np.isfinite(d).all(): raise FloatingPointError('Nonfinite angle field')
        residual=np.concatenate([np.abs(row['c']@(mix@d.reshape(len(nodes),-1))-row['b'])
                                 for mix,row in zip(mixing,rows)])
        record={'iteration':iteration+1,'gate_px':gate,'pairs':int(counts.sum()),
                'counts':{str(r['frame']):int(len(r['b'])) for r in rows},'cg':attempts,
                'relative_normal_residual':float(np.linalg.norm(operator@solution-rhs)/np.linalg.norm(rhs)),
                'median_px':float(np.median(residual)),'max_px':float(residual.max()),
                'node_d_max_px':np.linalg.norm(d,axis=2).max(1).tolist(),'seconds':time.perf_counter()-tick}
        history.append(record)
        print('2E FIELD',{k:v for k,v in record.items() if k!='counts'},flush=True)
    return d,load,history,time.perf_counter()-started


def field_stats(lap,mass,displacement,excluded):
    values=np.linalg.norm((lap@displacement)/mass[:,None],axis=1); used=values[~excluded]
    stats={key:float(np.percentile(used,p)) for key,p in [('p50',50),('p95',95),('p99',99)]}
    stats.update(max=float(used.max()),passed=bool(np.percentile(used,99)<=.03 and used.max()<=.05))
    return stats,values


def similarity_scale(base,d):
    a=base-base.mean(0); b=base+d-(base+d).mean(0)
    u,s,vh=np.linalg.svd(a.T@b); signs=np.ones(3); signs[-1]=np.linalg.det(u@vh)
    return float(np.dot(s,signs)/(a*a).sum()-1.)
