"""Compare each actual rendered pose on the prior's fixed observation domain."""
import numpy as np


def compare_arrays(reference_head,reference_face,prior_head,prior_face,current_head,current_face):
    def span(mask,y):
        xs=np.flatnonzero(mask[y]);return [xs[0],xs[-1]] if len(xs) else None
    before=[];after=[]
    for y in range(80,401,20):
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
