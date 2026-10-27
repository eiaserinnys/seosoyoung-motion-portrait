"""Fixed-camera global fit and pixel-unit linear Laplacian surface editing."""
from __future__ import annotations
import numpy as np
from fit_core import FIT, ROOT, read_json

RESULTS = FIT/'results/fit2d'


def pose_affine(to720):
    """Video poses already use 720px; image transforms start in source 640px."""
    return np.asarray(to720, dtype=float)[:2] @ np.diag([640/720, 640/720, 1.])


def frame_sets():
    rows = read_json(ROOT/'pipeline/headmodel/data/frames.json')['frames']
    keys = [r['frame'] for r in rows if r['role'] == 'key']
    samples = [r['frame'] for r in rows if r['role'] == 'sample']
    assert len(keys) == 9 and len(samples) == 121
    return keys + samples[:120:5], [r['frame'] for r in rows], [samples[i] for i in (0,20,40,60,80,100)]


def template():
    with np.load(FIT/'results/fit2c/egg-template-v5.npz') as b:
        v, tri, modes = (b[k].copy() for k in ('neutral_vertices','triangles','identity_modes'))
        masks = {k:b[n].copy() for k,n in [('head','head_mask'),('face','face_mask'),
                    ('ear-left','ear_left_mask'),('ear-right','ear_right_mask')]}
    # A retains 2b's lower-Face loss definition; B uses the entire Face mask.
    faces = {'head':np.flatnonzero(masks['head'][tri].all(1)),
             'face':np.flatnonzero((masks['face'] & (v[:,1]<=1.5))[tri].all(1))}
    for k in ('ear-left','ear-right'):
        faces[k] = np.flatnonzero((masks[k][tri].sum(1)>=2) & masks['head'][tri].all(1))
    return v,tri,modes,masks,faces


def global_fit(v, tri, modes, faces, targets, initial):
    """Prescribed A33: similarity 300 then joint 500, fixed rotations."""
    import time
    import torch
    from fit_core import project, silhouettes, loss_frame
    from run_falsify2 import fixed_cameras, constrain_parameters, shape_prior
    neutral=torch.tensor(v); basis=torch.tensor(modes.reshape(100,-1))
    base=torch.tensor(initial); original=base.clone()
    beta=torch.nn.Parameter(torch.zeros(100))
    logaxes=torch.nn.Parameter(torch.zeros(3))
    thickness=torch.nn.Parameter(torch.tensor(7.))
    history=[]; times=[]; started=time.perf_counter()
    for stage,steps,rate_values in [('similarity',300,[.004,.35,.35]),
                                    ('joint',500,[.001,.15,.15])]:
        rate=torch.tensor(rate_values); delta=torch.nn.Parameter(torch.zeros(len(targets),3))
        joint=stage=='joint'
        for p in (beta,logaxes,thickness): p.requires_grad_(joint)
        groups=[{'params':[delta],'lr':1.}]
        if joint:
            groups += [{'params':[beta],'lr':.012},{'params':[logaxes],'lr':.002},
                       {'params':[thickness],'lr':.025}]
        optimizer=torch.optim.Adam(groups)
        for iteration in range(steps):
            tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            vertices=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
            poses=fixed_cameras(base,delta,rate)
            projected=[project(vertices,p) for p in poses]
            if iteration%5==0:
                contours=[silhouettes(q.detach().numpy(),tri,faces) for q in projected]
            terms=[loss_frame(q,c,t,thickness) for q,c,t in zip(projected,contours,targets)]
            loss=torch.stack([term[0] for term in terms]).mean()+shape_prior(beta,logaxes)
            if not torch.isfinite(loss): raise FloatingPointError('A global fit diverged')
            loss.backward(); optimizer.step(); constrain_parameters(beta,thickness)
            times.append(time.perf_counter()-tick)
            if iteration%50==0 or iteration==steps-1:
                row={'stage':stage,'iteration':iteration+1,'loss':float(loss.detach()),
                     'beta_at_limit':int((beta.detach().abs()>=3.-1e-6).sum()),
                     'scale_xyz':logaxes.detach().exp().tolist(),'thickness_px':float(thickness.detach()),
                     'seconds':time.perf_counter()-started}
                history.append(row); print('A',row,flush=True)
        base=fixed_cameras(base,delta,rate).detach()
    vertices=((neutral+(beta@basis).reshape(-1,3))*logaxes.exp()).detach().numpy()
    torch.testing.assert_close(base[:,:3],original[:,:3],rtol=0,atol=0)
    shared={'beta':beta.detach().tolist(),'scale_xyz':logaxes.detach().exp().tolist(),
            'thickness_px':float(thickness.detach()),
            'beta_at_limit':int((beta.detach().abs()>=3.-1e-6).sum()),
            'iterations':{'similarity':300,'joint':500},'history':history,
            'elapsed_seconds':time.perf_counter()-started,
            'iteration_mean_seconds':float(np.mean(times)),
            'rotation_max_abs_delta_deg':0.,
            'weights':{'head':1.,'skin':.5,'ear':.5,'beta_L2_sum':.1,'logaxes_L2_sum':.1}}
    return vertices,base.numpy(),shared


def fit_cameras(vertices,tri,faces,targets,initial,thickness,steps=150):
    """One independent-camera batch, without retaining 130 autograd graphs."""
    import time
    import torch
    from fit_core import project, silhouettes, loss_frame
    from run_falsify2 import fixed_cameras
    started=time.perf_counter(); times=[]; history=[]
    v=torch.tensor(vertices); base=torch.tensor(initial); rate=torch.tensor([.0015,.20,.20])
    delta=torch.nn.Parameter(torch.zeros(len(targets),3)); optimizer=torch.optim.Adam([delta],lr=1.)
    fixed_t=torch.tensor(thickness)
    for iteration in range(steps):
        tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        if iteration%5==0:
            contours=[silhouettes(project(v,p).detach().numpy(),tri,faces)
                      for p in fixed_cameras(base,delta,rate)]
        total=0.
        for index,(target,contour) in enumerate(zip(targets,contours)):
            # Reconstruct each small graph over the same batched parameter.
            p=torch.cat([base[index,:3],base[index,3:]+delta[index]*rate])
            loss,_=loss_frame(project(v,p),contour,target,fixed_t)
            if not torch.isfinite(loss): raise FloatingPointError('A camera fit diverged')
            (loss/len(targets)).backward(); total+=float(loss.detach())/len(targets)
        optimizer.step(); times.append(time.perf_counter()-tick)
        if iteration%25==0 or iteration==steps-1:
            row={'iteration':iteration+1,'loss':total,'seconds':time.perf_counter()-started}
            history.append(row); print('A CAMERA',row,flush=True)
    poses=fixed_cameras(base,delta,rate).detach().numpy()
    np.testing.assert_array_equal(poses[:,:3],initial[:,:3])
    return poses,{'iterations':steps,'elapsed_seconds':time.perf_counter()-started,
                  'iteration_mean_seconds':float(np.mean(times)),'history':history}


def linear_camera(pose):
    """Numerically recover the exact orthographic 2x3 map in the given units."""
    import torch
    from fit_core import project
    q=project(torch.tensor(np.vstack([np.zeros(3),np.eye(3)])),
              torch.tensor(pose,dtype=torch.float64)).numpy()[:,:2]
    return (q[1:]-q[0]).T,q[0]


def normalized_laplacian(tri,count):
    from scipy import sparse
    from egg_pca import mesh_edges
    edges=mesh_edges(tri)
    a,b=np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]]
    adjacency=sparse.csr_matrix((np.ones(len(a)),(a,b)),shape=(count,count))
    degree=np.asarray(adjacency.sum(1)).ravel()
    return sparse.eye(count,format='csr')-sparse.diags(1/degree)@adjacency


def contour_normals(points):
    """Outward normals of a closed image-space polyline, independent of winding."""
    import cv2
    p=np.asarray(points,dtype=float)
    tangent=np.roll(p,-2,axis=0)-np.roll(p,2,axis=0)
    normal=np.column_stack([tangent[:,1],-tangent[:,0]])
    if cv2.contourArea(p.astype(np.float32),oriented=True)<0: normal=-normal
    return normal/np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)


def classify_face(skin,head,ear,neck):
    """Keep only the skin's exposed/background or neck-facing outer contour."""
    from fit_core import boundary
    points=boundary(skin); normals=contour_normals(points)
    xy=np.rint(points+4*normals).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<head.shape[1])&(xy[:,1]>=0)&(xy[:,1]<head.shape[0])
    indices=np.flatnonzero(inside); x,y=xy[inside].T
    keep=~inside
    # Neck owns this classification before the residual H & ~S & ~E hair set.
    keep[indices]=((head[y,x]==0)|(neck[y,x]>0)) & (ear[y,x]==0)
    contour_xy=np.rint(points).astype(int)
    keep &= ear[contour_xy[:,1],contour_xy[:,0]]==0
    return points,normals,keep


def prepare_b_targets(targets,thickness):
    import cv2
    from fit_core import boundary, eroded_target
    for target in targets:
        directory=ROOT/f'pipeline/headmodel/data/frame_{target["frame"]:04d}'
        def mask(name):
            value=cv2.imread(str(directory/(name+'.png')),cv2.IMREAD_GRAYSCALE)
            if value is None: raise FileNotFoundError(directory/name)
            return (value>0).astype(np.uint8)
        target['neck']=mask('raw/neck')
        points,normals,keep=classify_face(target['skin'],target['head'],mask('raw/ear'),target['neck'])
        eroded=eroded_target(target['head'],thickness)
        target.update(face_points=points[keep],face_normals=normals[keep],head_eroded=eroded,
                      face_all_points=points,face_keep=keep,
                      head_boundary=boundary(eroded))
    return targets


def sample_polyline(points,spacing=1.,closed=False):
    points=np.asarray(points)
    if closed: points=np.r_[points,points[:1]]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    samples=np.arange(0,arc[-1],spacing)
    return np.column_stack([np.interp(samples,arc,points[:,k]) for k in range(2)])


def face_interface(tri,masks,faces):
    """Edges shared by full-Face and non-Face head triangles are not silhouettes."""
    from egg_pca import mesh_edges
    full=masks['face'][tri].all(1)
    face_edges=mesh_edges(tri[full])
    other_edges=mesh_edges(tri[np.isin(np.arange(len(tri)),faces['head']) & ~full])
    encoded=set(map(tuple,other_edges))
    return set(tuple(edge) for edge in face_edges if tuple(edge) in encoded)


def model_contours(vertices,tri,masks,faces,pose,interface=None):
    import torch
    from fit_core import project, rasterize, boundary, lower_boundary, attach_contour
    q=project(torch.tensor(vertices),torch.tensor(pose,dtype=torch.float64)).numpy()
    def attachment(points,owner):
        ids,weights=attach_contour(points,q,tri,owner)
        ids,weights=ids.numpy(),weights.numpy().astype(float)
        xy=(q[ids,:2]*weights[:,:,None]).sum(1)
        return {'ids':ids,'weights':weights,'xy':xy}
    def raster(ids):
        _,owner=rasterize(q,tri[ids],cull_backfaces=True)
        global_owner=np.full_like(owner,-1); valid=owner>=0
        global_owner[valid]=ids[owner[valid]]
        return valid,global_owner
    head,head_owner=raster(faces['head'])
    raw=sample_polyline(boundary(head),closed=True)
    result={'head':attachment(raw,head_owner)}
    pixels=np.rint(raw).astype(int); owned=head_owner[pixels[:,1],pixels[:,0]]
    full_face=masks['face'][tri].all(1)
    keep=(owned>=0)&full_face[np.maximum(owned,0)]
    full_face_ids=np.flatnonzero(full_face & masks['head'][tri].all(1))
    face,face_owner=raster(full_face_ids)
    lower=sample_polyline(lower_boundary(face))
    parts=[]
    if keep.any(): parts.append(attachment(raw[keep],head_owner))
    parts.append(attachment(lower,face_owner))
    combined={key:np.concatenate([p[key] for p in parts]) for key in ('ids','weights','xy')}
    if interface is None: interface=face_interface(tri,masks,faces)
    allowed=[]
    for ids,weights in zip(combined['ids'],combined['weights']):
        edge=tuple(sorted(ids[np.argsort(weights)[-2:]]))
        allowed.append(edge not in interface)
    result['face']={k:value[np.asarray(allowed)] for k,value in combined.items()}
    for key in ('ear-left','ear-right'):
        visible,owner=raster(faces[key])
        result[key]=attachment(sample_polyline(boundary(visible),closed=True),owner)
    result['head_mask']=head; result['head_owner']=head_owner; result['projected']=q
    return result


def contour_residual(model,target):
    from scipy.spatial import cKDTree
    a=cKDTree(target).query(model)[0]; b=cKDTree(model).query(target)[0]
    distance=np.r_[a,b]
    return {'median_px':float(np.median(distance)),'mean_px':float(distance.mean()),
            'max_px':float(distance.max()),'model_count':len(model),'target_count':len(target)}


def correspondence_rows(vertices,tri,masks,faces,poses,targets,thickness,gate,interface):
    """Bidirectional nearest neighbors, gated by Euclidean pixel distance."""
    from scipy.spatial import cKDTree
    from fit_core import boundary
    rows=[]; counts={}
    for pose,target in zip(poses,targets):
        contours=model_contours(vertices,tri,masks,faces,pose,interface)
        specifications=[('head',target['head_boundary'],None,1.),
                        ('face',target['face_points'],target['face_normals'],10.)]
        specifications += [(key,boundary(mask),None,1.) for key,mask in target['ears'].items()]
        a,t=linear_camera(pose)
        frame_counts={}
        for key,points,normals,weight in specifications:
            if normals is None:
                normals=contour_normals(points)
                # Every target boundary pixel participates (one-pixel sampling).
            model=contours[key]; xy=model['xy']
            distance,nearest=cKDTree(points).query(xy)
            accepted=distance<=gate
            model_ids=np.flatnonzero(accepted); target_ids=nearest[accepted]
            distance,nearest=cKDTree(xy).query(points)
            accepted=distance<=gate
            model_ids=np.r_[model_ids,nearest[accepted]]
            target_ids=np.r_[target_ids,np.flatnonzero(accepted)]
            frame_counts[key]=len(model_ids)
            rows.append({'ids':model['ids'][model_ids],'weights':model['weights'][model_ids],
                         'normal':normals[target_ids],'target':points[target_ids],
                         'a':a,'t':t,'weight':weight,'frame':target['frame'],'kind':key})
        counts[str(target['frame'])]=frame_counts
    return rows,counts


def solve_icp(vertices,tri,masks,faces,poses,targets,sref,thickness,lam=300.,iterations=25,
              *,sample_spacing,reference_spacing):
    """Absolute 3N displacement, sum energies; refactor the normal matrix per ICP."""
    import time
    from scipy import sparse
    from scipy.sparse.linalg import splu
    started=time.perf_counter(); base=vertices.astype(float)*sref; d=np.zeros_like(base)
    pxposes=np.array(poses,dtype=float); pxposes[:,3]-=np.log(sref)
    lap=normalized_laplacian(tri,len(base)); interface=face_interface(tri,masks,faces)
    regularizer=lam*sparse.kron(lap.T@lap,sparse.eye(3),format='csc')+.01*sparse.eye(base.size,format='csc')
    weight_multiplier=sample_spacing/reference_spacing
    trace=[]; stable=0; tight_start=None
    for iteration in range(iterations):
        tick=time.perf_counter()
        # Convergence at a loose gate triggers the final five 5px iterations;
        # this prevents early stopping from skipping the prescribed tight gate.
        if iterations>5 and tight_start is None and (stable>=2 or iteration>=iterations-5):
            tight_start=iteration; stable=0
        gate=(5. if tight_start is not None else
              40.-35.*iteration/max(iterations-5 if iterations>5 else iterations-1,1))
        rows,counts=correspondence_rows(base+d,tri,masks,faces,pxposes,targets,thickness,gate,interface)
        rr=[]; cc=[]; vv=[]; rhs=[]; offset=0; total_weights=[]
        for row in rows:
            count=len(row['ids']); normal_map=row['normal']@row['a']
            coefficients=row['weights'][:,:,None]*normal_map[:,None,:]
            indices=row['ids'][:,:,None]*3+np.arange(3)
            rr.append(np.repeat(np.arange(offset,offset+count),9)); cc.append(indices.ravel()); vv.append(coefficients.ravel())
            original=(base[row['ids']]*row['weights'][:,:,None]).sum(1)@row['a'].T+row['t']
            rhs.append(((row['target']-original)*row['normal']).sum(1))
            total_weights.extend([row['weight']*weight_multiplier]*count); offset+=count
        c=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(offset,base.size))
        rhs=np.concatenate(rhs); weights=np.asarray(total_weights)
        cw=c.multiply(np.sqrt(weights)[:,None]).tocsc()
        matrix=regularizer+cw.T@cw
        d=splu(matrix.tocsc()).solve(np.asarray(c.T@(weights*rhs)).ravel()).reshape(-1,3)
        if not np.isfinite(d).all(): raise FloatingPointError('Nonfinite B displacement')
        residual=np.abs(c@d.ravel()-rhs)
        record={'iteration':iteration+1,'gate_px':gate,'pairs':offset,'pairs_by_frame':counts,
                'residual_median_px':float(np.median(residual)),'residual_max_px':float(residual.max()),
                'displacement_max_px':float(np.linalg.norm(d,axis=1).max()),
                'seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-started}
        if trace and abs(record['residual_median_px']-trace[-1]['residual_median_px'])<.05:
            stable+=1
        else: stable=0
        record['stable_changes']=stable
        trace.append(record); print('B ICP',record,flush=True)
        if stable>=2 and tight_start is not None and iteration-tight_start>=4: break
    return (base+d)/sref,trace


def measure_b(vertices,tri,masks,faces,poses,targets):
    """Dense classified-target contour distances; actual head-occluded ear IoU."""
    from fit_core import boundary
    from fit_sheets import iou
    interface=face_interface(tri,masks,faces); report=[]
    landmarks={r['frame']:r for r in read_json(ROOT/'pipeline/headmodel/data/landmarks.json')['frames']}
    nose_id=read_json(FIT/'results/fit2c/egg-template-v5.json')['nose_vertex']
    for pose,target in zip(poses,targets):
        contours=model_contours(vertices,tri,masks,faces,pose,interface)
        head=contour_residual(contours['head']['xy'],target['head_boundary'])
        face=contour_residual(contours['face']['xy'],target['face_points'])
        from scipy.spatial import cKDTree
        distance=cKDTree(contours['face']['xy']).query(target['face_points'])[0]
        worst=np.argsort(distance)[-10:][::-1]
        face['worst_target_points']=[{'xy':target['face_points'][i].tolist(),'distance_px':float(distance[i])}
                                     for i in worst]
        # Pixel staircase variation after subtracting a five-sample local trend;
        # a spatial mask-boundary diagnostic, not temporal segmentation jitter.
        p=target['face_all_points']; keep=target['face_keep']
        smooth=sum(np.roll(p,j,axis=0) for j in (-2,-1,0,1,2))/5
        jitter=np.abs(((p-smooth)*contour_normals(p)).sum(1))[keep]
        face['mask_spatial_jitter_estimate']={'median_px':float(np.median(jitter)),
            'p95_px':float(np.percentile(jitter,95)),
            'method':'normal residual to five-pixel local boundary average; includes real corners, not temporal jitter'}
        ears={}
        for key,wanted in target['ears'].items():
            detail=contour_residual(contours[key]['xy'],boundary(wanted))
            ear_owned=np.isin(contours['head_owner'],faces[key])
            from fit_core import rasterize
            _,isolated=rasterize(contours['projected'],tri[faces[key]],cull_backfaces=True)
            detail.update(visible_iou=iou(ear_owned,wanted),isolated_iou=iou(isolated>=0,wanted))
            ears[key]=detail
        x,y,score=landmarks[target['frame']]['keypoints'][23]
        projected_nose=contours['projected'][nose_id,:2]
        report.append({'frame':target['frame'],'head':head,'face':face,'ears':ears,
                       'head_iou':iou(contours['head_mask'],target['head_eroded']),
                       'nose_record_only':{'score':score,'detected_xy':[x,y],
                           'projected_xy':projected_nose.tolist(),
                           'error_px':float(np.linalg.norm(projected_nose-[x,y])) if score>=.5 else None}})
    return report
