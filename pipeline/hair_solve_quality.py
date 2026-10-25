"""Compare each actual rendered pose on the prior's fixed observation domain."""
import numpy as np
from functools import lru_cache


def outline_rows(mask, stop=400):
    rows = []
    for y in range(80, stop + 1, 20):
        xs = np.flatnonzero(mask[y])
        rows.append({'y': y, 'span': [int(xs[0]), int(xs[-1])] if len(xs) else None})
    return rows


def top_gate(pitch, rendered, target):
    if pitch < 0:
        return {'kind': 'cropped-bound', 'maximum': 4, 'pass': rendered <= 4}
    return {'kind': 'target', 'target': target, 'error': rendered - target,
            'pass': abs(rendered - target) <= 8}


def stem_gap(hair, tie):
    """Fable's percentile 20..80 columns, sampled every 2px above the tie."""
    yy, xx = np.where(tie)
    if not len(xx):
        return None
    lo, hi = [int(np.percentile(xx, p)) for p in [20, 80]]
    gaps = []
    for x in range(lo, hi+1, 2):
        ys = np.flatnonzero(tie[:, x])
        if not len(ys):
            continue
        top = int(ys[0])
        start = max(0, top-80)
        above = np.flatnonzero(hair[start:top, x])
        gaps.append(top-(start+int(above[-1]))-1 if len(above) else 80)
    return max(gaps, default=81)


def cheek_cover_rows(face, hair, side, rows):
    """Background pixels between visible skin edge and adjacent crown alpha."""
    out = []
    for y in rows:
        xs = np.flatnonzero(face[y])
        if not len(xs):
            continue
        edge = int(xs[-1] if side == 'right' else xs[0])
        line = hair[y, edge+1:] if side == 'right' else hair[y, :edge][::-1]
        hits = np.flatnonzero(line)
        out.append({'y': y, 'gap': int(hits[0]) if len(hits) else 1024})
    return out


def neutral_difference(a, b):
    if a.shape != b.shape:
        raise ValueError('Neutral capture dimensions differ')
    return int(np.any(a != b, axis=2).sum())


def span(mask, y):
    xs = np.flatnonzero(mask[y])
    return [int(xs[0]), int(xs[-1])] if len(xs) else None


def hairline_metrics(reference_face, rendered_face, eye_y):
    f, g = span(reference_face, 330), span(rendered_face, 330)
    samples, exposure = [], None
    if f and g:
        left, right = max(f[0], g[0]) + 6, min(f[1], g[1]) - 6
        for x in range(left, right, 12):
            a, b = np.flatnonzero(reference_face[:700, x]), np.flatnonzero(rendered_face[:700, x])
            if len(a) and len(b):
                samples.append({'x': x, 'reference': int(a[0]), 'render': int(b[0]), 'delta': int(b[0] - a[0])})
        reference_area = int(reference_face[150:eye_y, left:right].sum())
        render_area = int(rendered_face[150:eye_y, left:right].sum())
        exposure = {'referenceArea': reference_area, 'renderArea': render_area,
                    'ratioError': render_area / reference_area - 1 if reference_area else None,
                    'region': [left, 150, right, eye_y]}
    return {'hairline': samples, 'hairlineMedianError': float(np.median([h['delta'] for h in samples])) if samples else None,
            'foreheadExposure': exposure}



@lru_cache(maxsize=1)
def atlas_alpha(path):
    from PIL import Image
    return np.array(Image.open(path).convert('RGBA'))[:,:,3]/255.


def raster_stencil_sampler(pose,xy,grid,texture):
    """Associate an alpha boundary with its painted triangle, not nearby UVs."""
    ids=np.array(pose['triangles']);tri=xy[ids];a,b,c=tri[:,0],tri[:,1],tri[:,2]
    ab=b-a;ac=c-a;det=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    lo=tri.min(axis=1);hi=tri.max(axis=1);vertices=pose['vertices']
    uv=np.array([v['uv'] for v in vertices]);facing=np.array([v['facing'] for v in vertices])
    front=np.array([v['front'] for v in vertices]);order=np.array([v['order'] for v in vertices])
    culled=np.array([v.get('cull',v['front']) for v in vertices])[ids[:,0]]
    raw=np.array([[v.get('x',0),v.get('y',0)] for v in vertices])
    clip=np.array([v.get('clipFromY') or 1e9 for v in vertices])[ids[:,0]]
    alpha=atlas_alpha(str(texture));height,width=alpha.shape
    def sample(pixel,front_only):
        point=np.array(pixel)+.5
        candidates=np.flatnonzero((np.abs(det)>1e-8)&(~culled|(det>0))&np.all(point>=lo-1,axis=1)&np.all(point<=hi+1,axis=1))
        if front_only:candidates=candidates[front[ids[candidates,0]]]
        if not len(candidates):return None
        ap=point-a[candidates];d=det[candidates]
        u=(ap[:,0]*ac[candidates,1]-ap[:,1]*ac[candidates,0])/d
        v=(ab[candidates,0]*ap[:,1]-ab[candidates,1]*ap[:,0])/d
        weights=np.c_[1-u-v,u,v];inside=np.all(weights>=-.02,axis=1)&np.all(weights<=1.02,axis=1)
        weights=weights[inside];candidates=candidates[inside]
        if not len(candidates):return None
        tex=np.einsum('ij,ijk->ik',weights,uv[ids[candidates]])
        x=np.clip(tex[:,0]*width-.5,0,width-1);y=np.clip(tex[:,1]*height-.5,0,height-1)
        ix=x.astype(int);iy=y.astype(int);tx=x-ix;ty=y-iy
        ix1=np.minimum(ix+1,width-1);iy1=np.minimum(iy+1,height-1)
        opacity=(1-ty)*((1-tx)*alpha[iy,ix]+tx*alpha[iy,ix1])+ty*((1-tx)*alpha[iy1,ix]+tx*alpha[iy1,ix1])
        rest=np.einsum('ij,ijk->ik',weights,raw[ids[candidates]])
        # A far back ribbon is clipped to the face/side-hair mask. At the
        # movable exterior that mask is either absent or owned by a front
        # layer; the clipped backing cannot own the silhouette observation.
        clipped=(np.sign(pose.get('yaw',0))*(rest[:,0]-502)>0)&(rest[:,1]>clip[candidates])
        valid=(opacity>.15)&(np.sum(weights*facing[ids[candidates]],axis=1)>1e-6)&~clipped
        if not valid.any():return None
        candidates=candidates[valid];weights=weights[valid]
        pick=np.argmax(order[ids[candidates,0]])
        return weights[pick]@grid[ids[candidates[pick]]]
    return sample


def compare_arrays(reference_head,reference_face,prior_head,prior_face,current_head,current_face,stop=400):
    def span(mask,y):
        xs=np.flatnonzero(mask[y]);return [xs[0],xs[-1]] if len(xs) else None
    before=[];after=[]
    for y in range(80,stop+1,20):
        target,prior,current=span(reference_head,y),span(prior_head,y),span(current_head,y)
        if target is None or prior is None:continue
        before.extend(np.array(prior)-target)
        after.extend(np.array(current)-target if current else [current_head.shape[1]]*2)
    ref_top=int(np.where(reference_head)[0].min())
    before.append(int(np.where(prior_head)[0].min())-ref_top)
    after.append(int(np.where(current_head)[0].min())-ref_top)
    a,b=span(reference_face,330),span(prior_face,330)
    if a and b:
        for x in range(max(a[0],b[0])+6,min(a[1],b[1])-6,12):
            target,prior,current=[np.flatnonzero(m[:700,x]) for m in [reference_face,prior_face,current_face]]
            if len(target) and len(prior):
                before.append(int(prior[0])-int(target[0]))
                after.append(int(current[0])-int(target[0]) if len(current) else current_face.shape[0])
    first=float(np.sqrt(np.mean(np.square(np.array(before,float)))))
    last=float(np.sqrt(np.mean(np.square(np.array(after,float)))))
    return {'priorRms':first,'correctedRms':last,'sampleCount':len(before),'pass':last<=first}


def compare_captures(poses,prior_dir,current_dir):
    from probe_hair_only import reference,read_mask,fill_holes
    rows=[]
    for pose in poses:
        name=pose['name'];ref=reference(pose)
        def masks(directory):
            face=read_mask(directory/(name+'-face.png'))
            hair=read_mask(directory/(name+'-crown.png'))|read_mask(directory/(name+'-tails.png'))|read_mask(directory/(name+'-ties.png'))
            return fill_holes(hair|face),face
        prior,prior_face=masks(prior_dir);current,current_face=masks(current_dir)
        rows.append(dict(name=name,**compare_arrays(ref['head'],ref['face'],prior,prior_face,current,current_face)))
    return {'pass':all(r['pass'] for r in rows),'definition':'actual raster RMS in pixels: head L/R spans at y80..400 step20, topY, and hairline columns step12; fixed reference/prior-visible domain','poses':rows}


def lower_span_comparison(reference_mask,prior,current):
    """Fixed reference/prior-visible rows; missing current spans are failures."""
    before=[];after=[]
    for lo,hi in [(0,502),(502,1024)]:
        for y in range(550,721,10):
            r,a,b=[np.flatnonzero(mask[y,lo:hi]) for mask in [reference_mask,prior,current]]
            if len(r) and len(a):
                target=r[[0,-1]];before.extend(a[[0,-1]]-target)
                after.extend(b[[0,-1]]-target if len(b) else [1024,1024])
    first=float(np.sqrt(np.mean(np.square(before)))) if before else None
    last=float(np.sqrt(np.mean(np.square(after)))) if after else None
    return dict(priorRms=first,correctedRms=last,sampleCount=len(before),**{'pass':last is not None and last<=first})


def compare_split_captures(poses,prior_dir,current_dir):
    from probe_hair_only import reference,read_mask,fill_holes
    upper=[];lower=[]
    for pose in poses:
        name=pose['name'];ref=reference(pose)
        def masks(directory):
            face=read_mask(directory/(name+'-face.png'))
            crown=read_mask(directory/(name+'-crown.png'))
            tail=read_mask(directory/(name+'-tails.png'))|read_mask(directory/(name+'-ties.png'))
            return fill_holes(crown|face),face,tail
        a,af,at=masks(prior_dir);b,bf,bt=masks(current_dir)
        upper.append(dict(name=name,**compare_arrays(ref['head'],ref['face'],a,af,b,bf,520)))
        lower.append(dict(name=name,**lower_span_comparison(ref['mask'],at,bt)))
    return dict(upper=dict(definition='crown | face only, row spans80..520 step20 + topY + hairline',poses=upper),
                lower=dict(definition='separate left/right tails | ties, fixed rows550..720 step10',poses=lower))


def bare_face_limits(root,features):
    """Record reference hairlines above the immutable skin's alpha support."""
    import json
    import subprocess
    from probe_hair_only import reference,read_mask
    inputs=json.loads(subprocess.check_output(['node',str(root/'pipeline/hair_solve_inputs.mjs'),'--bare-face'],timeout=60))
    rows=[]
    for pose in inputs:
        name=pose['name'];ref=reference(features[name]);prior=read_mask(root/'debug/hair_solve_prior'/(name+'-face.png'))
        a,b=span(ref['face'],330),span(prior,330)
        sample=raster_stencil_sampler(pose,np.array([v['xy'] for v in pose['vertices']]),np.ones((len(pose['vertices']),1)),root/'bundle/assets/texture_00.png')
        points=[]
        if a and b:
            for x in range(max(a[0],b[0])+6,min(a[1],b[1])-6,12):
                target=np.flatnonzero(ref['face'][:,x])
                if not len(target):continue
                bare=next((y for y in range(150,600) if sample([x,y],False) is not None),None)
                if bare is not None and bare>target[0]:points.append(dict(x=x,referenceTop=int(target[0]),bareFaceTop=bare,minimumGap=bare-int(target[0])))
        rows.append(dict(name=name,columns=points))
    return dict(definition='alpha-supported bare skin with every hair mesh absent; same frozen face, UV atlas, triangle culling and canvas frame; positive gap cannot be exposed by moving hair alone',poses=rows)
