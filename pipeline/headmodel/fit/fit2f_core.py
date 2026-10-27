"""Label-free silhouettes, tapered mask targets and pixel-unit surface editing."""
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu
from fit_core import (FIT, ROOT, read_json, write_json, project, rasterize, boundary,
                      clean_mask, attach_contour)
from fit2d_core import (pose_affine, contour_normals, classify_face, linear_camera,
                        sample_polyline, normalized_laplacian)

RESULTS=FIT/'results/fit2f'


def edges(tri):
    return np.sort(np.vstack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)


def boundary_edges(tri):
    unique,count=np.unique(edges(tri),axis=0,return_counts=True)
    return unique[count==1]


def compact_surface(v,tri):
    used=np.unique(tri); inverse=np.full(len(v),-1,int); inverse[used]=np.arange(len(used))
    return v[used].copy(),inverse[tri],used


def load_surface(name):
    with np.load(RESULTS/(name+'.npz')) as data: return {k:data[k].copy() for k in data.files}


def taper_amount(skin_or_ear,points,thickness):
    distance=cv2.distanceTransform((1-np.asarray(skin_or_ear,np.uint8)),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    xy=np.rint(points).astype(int)
    return thickness*np.minimum(1.,distance[xy[:,1],xy[:,0]]/20.)


def make_target(head,skin,ear,neck,thickness,frame=0):
    hp=boundary(head); hn=contour_normals(hp)
    inner=np.rint(hp-3*hn).astype(int)
    inner[:,0]=np.clip(inner[:,0],0,head.shape[1]-1)
    inner[:,1]=np.clip(inner[:,1],0,head.shape[0]-1)
    x,y=inner.T
    # The inward three-pixel probe owns the label. Skin precedes ear.
    labels=np.where(skin[y,x]>0,0,np.where(ear[y,x]>0,2,1))
    moved=hp-taper_amount(skin|ear,hp,thickness)[:,None]*hn
    sp,sn,keep=classify_face(skin,head,ear,neck)
    hair=labels==1; excluded=labels==2
    points=np.vstack([sp[keep],moved[hair],hp[excluded]])
    normals=np.vstack([sn[keep],hn[hair],hn[excluded]])
    point_labels=np.r_[np.zeros(keep.sum(),int),np.ones(hair.sum(),int),np.full(excluded.sum(),2,int)]
    return {'frame':int(frame),'head':head,'skin':skin,'ear':ear,'neck':neck,
            'points':points,'normals':normals,'labels':point_labels,
            'ear_points':hp[excluded],'head_points_all':moved,'head_boundary':hp,
            'head_labels':labels,'skin_all':sp,'skin_keep':keep,'thickness_px':thickness}


def load_targets_f(frames,thickness):
    result=[]
    for frame in frames:
        directory=ROOT/f'pipeline/headmodel/data/frame_{frame:04d}'
        def mask(name):
            value=cv2.imread(str(directory/(name+'.png')),0)
            if value is None or value.shape!=(720,720): raise ValueError(directory/name)
            return (value>0).astype(np.uint8)
        h,e,n=mask('head'),mask('raw/ear'),mask('raw/neck')
        s=clean_mask(mask('raw/face skin') & (1-n))
        result.append(make_target(h,s,e,n,thickness,frame))
    return result


def raster_silhouette(q,tri):
    """Reject faces which cannot cover a pixel before the unchanged z-buffer.

    The imported renderer otherwise expands every subpixel/back-facing GLB
    triangle's bounding box. Owners are mapped back to original face IDs.
    """
    p=q[tri]; a,b,c=p[:,0],p[:,1],p[:,2]
    det=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    xy=p[:,:,:2]; lo=xy.min(1); hi=xy.max(1)
    # Existing barycentric tolerance permits 1e-5 outside each edge. A point
    # can exceed either bbox end by at most 2e-5 times that axis's span.
    pad=2e-5*(hi-lo)+1e-6
    first=np.maximum(np.ceil(lo-.5-pad),0)
    last=np.minimum(np.floor(hi-.5+pad),719)
    selected=np.flatnonzero((det < -1e-8)&(first<=last).all(1))
    depth,owner=rasterize(q,tri[selected],cull_backfaces=True)
    visible=owner>=0; owner[visible]=selected[owner[visible]]
    return depth,owner


def surface_contour(v,tri,pose,exclude_boundary=True,boundary_cache=None):
    q=project(torch.tensor(v),torch.tensor(pose,dtype=torch.float64)).numpy()
    _,owner=raster_silhouette(q,tri)
    visible=owner>=0
    samples=sample_polyline(boundary(visible),1.,closed=True)
    ids,w=attach_contour(samples,q,tri,owner); ids,w=ids.numpy(),w.numpy().astype(float)
    xy=(q[ids,:2]*w[:,:,None]).sum(1)
    valid=np.ones(len(ids),bool)
    if exclude_boundary:
        open_edges=set(map(tuple,boundary_edges(tri) if boundary_cache is None else boundary_cache))
        attached_edges=np.sort(np.take_along_axis(ids,np.argsort(w,axis=1)[:,-2:],axis=1),axis=1)
        valid=np.array([tuple(edge) not in open_edges for edge in attached_edges])
        single=(w>1e-7).sum(1)==1
        open_vertices=np.unique(np.array(list(open_edges),int))
        valid &= ~(single & np.isin(ids[np.arange(len(ids)),w.argmax(1)],open_vertices))
    return {'xy':xy[valid],'ids':ids[valid],'weights':w[valid],
            'head_mask':visible,'projected':q,'boundary_rejected':int((~valid).sum())}


def pair_rows(model,target,gate):
    xy=model['xy']; p=target['points']; labels=target['labels']
    if not len(xy):
        return {'model_indices':np.empty(0,int),'target_indices':np.empty(0,int)}
    distance,nearest=cKDTree(p).query(xy)
    mi=np.arange(len(xy)); ti=nearest
    distance2,nearest2=cKDTree(xy).query(p)
    mi=np.r_[mi,nearest2]; ti=np.r_[ti,np.arange(len(p))]
    keep=(np.r_[distance,distance2]<=gate)&(labels[ti]!=2)
    ear=target['ear_points']
    if len(ear):
        tree=cKDTree(ear)
        keep &= (tree.query(xy[mi])[0]>10.) & (tree.query(p[ti])[0]>10.)
    return {'model_indices':mi[keep],'target_indices':ti[keep]}


def geometric_operator(v,tri):
    """Positive clamped cotangent stiffness and barycentric pixel-squared mass."""
    p=v[tri]; cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    twice_area=np.linalg.norm(cross,axis=1)
    if (twice_area<=0).any(): raise ValueError('Degenerate surface triangle')
    mass=np.bincount(tri.ravel(),weights=np.repeat(twice_area/6.,3),minlength=len(v))
    row=[]; col=[]; values=[]
    for k in range(3):
        a=(k+1)%3; b=(k+2)%3
        cot=((p[:,a]-p[:,k])*(p[:,b]-p[:,k])).sum(1)/twice_area*.5
        row.extend([tri[:,a],tri[:,b]]); col.extend([tri[:,b],tri[:,a]]); values.extend([cot,cot])
    adjacency=sparse.csr_matrix((np.concatenate(values),(np.concatenate(row),np.concatenate(col))),shape=(len(v),len(v)))
    # CSR sums the two incident-triangle cot contributions first. Only the
    # resulting undirected edge weight is clamped, preserving their cancellation.
    adjacency.data=np.maximum(adjacency.data,0.)
    adjacency.eliminate_zeros()
    lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    if (mass<=0).any(): raise ValueError('Isolated or massless vertex')
    return lap.tocsr(),mass


def excluded_vertices(tri,count,ear=None):
    selected=np.zeros(count,bool); selected[np.unique(boundary_edges(tri))]=True
    for _ in range(2):
        touching=selected[tri].any(1); selected[np.unique(tri[touching])]=True
    if ear is not None: selected|=ear
    return selected


def curvature_stats(base,tri,d,ear=None):
    lap,mass=geometric_operator(base,tri)
    curvature=np.linalg.norm((lap@d)/mass[:,None],axis=1)
    excluded=excluded_vertices(tri,len(base),ear)
    used=curvature[~excluded]
    stats={k:float(np.percentile(used,p)) for k,p in [('p50',50),('p95',95),('p99',99)]}
    stats.update(max=float(used.max()),passed=bool(np.percentile(used,99)<=.03 and used.max()<=.05),
                 excluded_vertices=int(excluded.sum()),unit='1/px',operator='cotangent: sum per edge, then clamp')
    return stats,curvature,excluded


def solve_surface(v,tri,poses,targets,sref,geometric=True,lam=None):
    base=np.asarray(v,float)*sref; d=np.zeros_like(base); started=time.perf_counter()
    poses=np.asarray(poses,float).copy(); poses[:,3]-=np.log(sref)
    if geometric:
        lap,mass=geometric_operator(base,tri)
        energy=lap.T@sparse.diags(1/mass)@lap
        weight=1000. if lam is None else lam
    else:
        lap=normalized_laplacian(tri,len(base)); energy=lap.T@lap
        weight=300. if lam is None else lam
    regularizer=weight*sparse.kron(energy,sparse.eye(3),format='csc')+.01*sparse.eye(base.size,format='csc')
    trace=[]
    for iteration in range(10):
        tick=time.perf_counter(); gate=15.-9.*min(iteration,7)/7
        rr=[]; cc=[]; vv=[]; rhs=[]; weights=[]; offset=0; counts={}
        for pose,target in zip(poses,targets):
            model=surface_contour((base+d),tri,pose)
            pairs=pair_rows(model,target,gate); mi=pairs['model_indices']; ti=pairs['target_indices']
            ids=model['ids'][mi]; bary=model['weights'][mi]; normals=target['normals'][ti]
            a,t=linear_camera(pose); coefficients=bary[:,:,None]*(normals@a)[:,None,:]
            count=len(mi)
            rr.append(np.repeat(np.arange(offset,offset+count),9)); cc.append((ids[:,:,None]*3+np.arange(3)).ravel())
            vv.append(coefficients.ravel())
            original=(base[ids]*bary[:,:,None]).sum(1)@a.T+t
            rhs.append(((target['points'][ti]-original)*normals).sum(1))
            weights.append(np.where(target['labels'][ti]==0,10.,1.)/3.)
            counts[str(target['frame'])]={'skin':int((target['labels'][ti]==0).sum()),
                'hair':int((target['labels'][ti]==1).sum()),'cut_rejected':model['boundary_rejected']}
            offset+=count
        if offset==0: raise ValueError('No accepted silhouette correspondences')
        c=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(offset,base.size))
        b=np.concatenate(rhs); w=np.concatenate(weights); cw=c.multiply(np.sqrt(w)[:,None]).tocsc()
        d=splu((regularizer+cw.T@cw).tocsc()).solve(np.asarray(c.T@(w*b)).ravel()).reshape(-1,3)
        if not np.isfinite(d).all(): raise FloatingPointError('Nonfinite displacement')
        residual=np.abs(c@d.ravel()-b)
        record={'iteration':iteration+1,'gate_px':gate,'pairs':offset,'counts':counts,
                'median_px':float(np.median(residual)),'max_px':float(residual.max()),
                'd_max_px':float(np.linalg.norm(d,axis=1).max()),'seconds':time.perf_counter()-tick}
        trace.append(record); print('2F ICP',record,flush=True)
    return (base+d)/sref,d,trace,time.perf_counter()-started


def measure_surface(v,tri,poses,targets):
    rows=[]
    for pose,target in zip(poses,targets):
        model=surface_contour(v,tri,pose); tree=cKDTree(model['xy'])
        row={'frame':target['frame']}
        for label,name in [(0,'skin'),(1,'hair')]:
            pts=target['points'][target['labels']==label]
            if len(target['ear_points']): pts=pts[cKDTree(target['ear_points']).query(pts)[0]>10.]
            dist=tree.query(pts)[0]
            row[name]={'median_px':float(np.median(dist)),'max_px':float(dist.max()),'count':len(pts)}
        # IoU uses the tapered closed head target. Skin remains on the original boundary.
        wanted=np.zeros((720,720),np.uint8)
        cv2.fillPoly(wanted,[np.rint(target['head_points_all']).astype(np.int32)],1)
        mask=model['head_mask']; row['head_iou']=float((mask&(wanted>0)).sum()/(mask|(wanted>0)).sum())
        rows.append(row)
    return rows


def fit_cameras_f(vertices,tri,targets,initial):
    """100 camera-only Adam steps, ten 2f correspondence rounds, no shape fit.

    Camera parameterization and rates are the existing 2d camera convention.
    The ten correspondence rounds retain 2f's fixed 15-to-6px gate schedule.
    """
    started=time.perf_counter(); dtype=torch.float64
    base=torch.tensor(np.asarray(initial),dtype=dtype)
    rate=torch.tensor([.0015,.20,.20],dtype=dtype)
    delta=torch.nn.Parameter(torch.zeros(len(targets),3,dtype=dtype))
    optimizer=torch.optim.Adam([delta],lr=1.)
    cut_edges=boundary_edges(tri); history=[]
    for outer in range(10):
        tick=time.perf_counter(); gate=15.-9.*min(outer,7)/7
        poses=torch.cat([base[:,:3],base[:,3:]+delta*rate],1).detach().numpy()
        rows=[]; counts={}
        for index,(pose,target) in enumerate(zip(poses,targets)):
            model=surface_contour(vertices,tri,pose,boundary_cache=cut_edges)
            pairs=pair_rows(model,target,gate)
            mi,ti=pairs['model_indices'],pairs['target_indices']
            counts[str(target['frame'])]=len(mi)
            if not len(mi): continue
            anchor=(vertices[model['ids'][mi]]*model['weights'][mi,:,None]).sum(1)
            rows.append((index,torch.tensor(anchor,dtype=dtype),
                         torch.tensor(target['points'][ti],dtype=dtype),
                         torch.tensor(target['normals'][ti],dtype=dtype),
                         torch.tensor(np.where(target['labels'][ti]==0,10.,1.)/3.,dtype=dtype)))
        if not rows: raise ValueError('No camera refinement correspondences')
        for inner in range(10):
            optimizer.zero_grad(set_to_none=True); total=0.
            for index,anchor,wanted,normals,weight in rows:
                pose=torch.cat([base[index,:3],base[index,3:]+delta[index]*rate])
                residual=((project(anchor,pose)[:,:2]-wanted)*normals).sum(1)
                loss=(weight*residual.square()).sum()
                if not torch.isfinite(loss): raise FloatingPointError('2f camera refinement diverged')
                loss.backward(); total+=float(loss.detach())
            optimizer.step()
        record={'iteration':(outer+1)*10,'gate_px':gate,'pairs':sum(counts.values()),
                'counts':counts,'loss_sum':total,'seconds':time.perf_counter()-tick}
        history.append(record); print('2F CAMERA',record,flush=True)
    poses=torch.cat([base[:,:3],base[:,3:]+delta*rate],1).detach().numpy()
    np.testing.assert_array_equal(poses[:,:3],np.asarray(initial)[:,:3])
    return poses,{'iterations':100,'history':history,'elapsed_seconds':time.perf_counter()-started,
                  'rotation_max_abs_delta_deg':0.,'rate':[.0015,.20,.20],
                  'shape_fixed':True,'thickness_fixed':True}


def remesh_surface(v,tri,sref):
    """Exactly the requested one remeshing pass; no topology/weight search."""
    tool='pymeshlab 2025.7'
    try:
        import pymeshlab
    except ImportError:
        import pyacvd
        import pyvista as pv
        mesh=pv.PolyData(v,np.column_stack([np.full(len(tri),3),tri]).ravel())
        cluster=pyacvd.Clustering(mesh)
        area=np.linalg.norm(np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]),axis=1).sum()/2
        cluster.subdivide(2); cluster.cluster(max(4,int(area/(np.sqrt(3)/2*(8/sref)**2))))
        out=cluster.create_mesh(); result=(np.array(out.points),out.faces.reshape(-1,4)[:,1:].copy())
        tool='pyacvd fallback'
    else:
        ms=pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(v,tri))
        ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(8/sref),iterations=10,adaptive=False)
        m=ms.current_mesh(); result=(m.vertex_matrix().copy(),m.face_matrix().copy())
    return *result,tool


def surface_distances(v,tri,w,other_tri,sref):
    import trimesh
    # Closest triangle distances, not nearest vertex spacing error.
    def closest(points,vertices,faces):
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        batches=[trimesh.proximity.closest_point(mesh,p)[1] for p in np.array_split(points,max(1,int(np.ceil(len(points)/1000))))]
        return np.concatenate(batches)*sref
    a=closest(v,w,other_tri); b=closest(w,v,tri)
    both=np.r_[a,b]
    return {'max':float(both.max()),'p99':float(np.percentile(both,99)),
            'original_to_remesh_max':float(a.max()),'remesh_to_original_max':float(b.max()),
            'method':'bidirectional vertex-to-triangle closest distances'}
