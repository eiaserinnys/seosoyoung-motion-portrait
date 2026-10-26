"""Export continuous, local scales constrained by the actual 2D triangles.

At each registered pose minimize ||1-s||², 0<=s<=1, subject to the signed
output area being >= .2 times each positive input area. Only vertices in a
violating triangle enter the active set. Shared artwork vertices are welded.
The determinant of the affine triangle Jacobian equals this area ratio, so
pitch Y and spatial changes of the vertex scales are included exactly.
"""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]

def area(q):
    a,b,c=q[:,0],q[:,1],q[:,2]
    return (b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])

def limit_stage(points,delta,triangles,kappa=.2):
    initial=area(points[triangles]);valid=initial>0
    triangles=triangles[valid];initial=initial[valid]
    raw=area((points+delta)[triangles])/initial
    bad=raw<kappa
    active=set(triangles[bad].ravel().tolist())
    if not active:return np.ones(len(points)),dict(triggered=0,vertices=0,minimum=float(raw.min()))
    # Expand only when zeroing the current active vertices exposes an actual
    # neighbouring constraint. This is an active-set solve, not a global fit.
    while True:
        ids=np.array(sorted(active));scales=np.ones(len(points));scales[ids]=0
        bad=area((points+delta*scales[:,None])[triangles])/initial<kappa-1e-10
        extra=set(triangles[bad].ravel().tolist())-active
        if not extra:break
        active.update(extra)
    touched=np.any(np.isin(triangles,ids),axis=1)
    tri=triangles[touched];den=initial[touched]
    columns=np.full(len(points),-1,int);columns[ids]=np.arange(len(ids))
    def locations(s):
        scale=np.ones(len(points));scale[ids]=s
        return (points+delta*scale[:,None])[tri]
    def constraint(s):
        return area(locations(s))/den-kappa
    def jacobian(s):
        q=locations(s);a,b,c=q[:,0],q[:,1],q[:,2]
        gradients=np.stack((np.column_stack((b[:,1]-c[:,1],c[:,0]-b[:,0])),
                            np.column_stack((c[:,1]-a[:,1],a[:,0]-c[:,0])),
                            np.column_stack((a[:,1]-b[:,1],b[:,0]-a[:,0]))),axis=1)
        derivatives=np.sum(gradients*delta[tri],axis=2)/den[:,None]
        j=np.zeros((len(tri),len(ids)))
        for vertex in range(3):
            cols=columns[tri[:,vertex]];mask=cols>=0
            j[np.flatnonzero(mask),cols[mask]]+=derivatives[mask,vertex]
        return j
    result=minimize(lambda s:np.square(s-1).sum(),np.zeros(len(ids)),jac=lambda s:2*(s-1),
                    bounds=[(0,1)]*len(ids),constraints=dict(type='ineq',fun=constraint,jac=jacobian),
                    method='SLSQP',options=dict(maxiter=150,ftol=1e-11))
    if not result.success or constraint(result.x).min()<-1e-7:
        raise ValueError(('local area projection did not converge',result.message,float(constraint(result.x).min())))
    scales=np.ones(len(points));scales[ids]=result.x
    return scales,dict(triggered=int((raw<kappa).sum()),vertices=int((scales<1-1e-8).sum()),
                       minimum=float((area((points+delta*scales[:,None])[triangles])/initial).min()),
                       maxDisplacementChange=float(np.linalg.norm(delta*(1-scales[:,None]),axis=1).max()))

def main():
    source=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/hair_attachment_geometry.mjs')],timeout=60))
    atlas=np.array(Image.open(ROOT/'bundle/assets/texture_00.png'))
    coordinates=sorted({tuple(q) for mesh in source[0]['meshes'] for q in mesh['rest']})
    lookup={q:i for i,q in enumerate(coordinates)}
    x=sorted({q[0] for q in coordinates}|{0,1024});y=sorted({q[1] for q in coordinates}|{0,1024})
    xi={v:i for i,v in enumerate(x)};yi={v:i for i,v in enumerate(y)}
    output=dict(neutral=1,x=x,y=y,poses=[]);report=[]
    for pose in sorted(source,key=lambda p:(p['pitch'],p['yaw'])):
        arrays={key:np.zeros((len(coordinates),2)) for key in ['fringe','rawCanopy','flare','residual']}
        triangles=[]
        for mesh in pose['meshes']:
            ids=np.array([lookup[tuple(q)] for q in mesh['rest']])
            for key in arrays:arrays[key][ids]=np.array(mesh['points'][key])[:,:2]
            for t in mesh['indices']:
                uv=np.array(mesh['uvs'])[t].mean(axis=0);u,v=np.clip(uv*atlas.shape[0],0,atlas.shape[0]-1).astype(int)
                if atlas[v,u,3]>127:
                    tri=ids[t]
                    if area(arrays['fringe'][tri][None])[0]>1e-8:triangles.append(tri)
        triangles=np.array(triangles);points=arrays['fringe'];row=dict(yaw=pose['yaw'],pitch=pose['pitch']);stats=dict(name=pose['name'])
        for stage,delta in [('attachment',arrays['rawCanopy']-points),('flare',arrays['flare']),('residual',arrays['residual'])]:
            scales,metrics=limit_stage(points,delta,triangles)
            row[stage]={str(yi[q[1]]*len(x)+xi[q[0]]):float(s) for q,s in zip(coordinates,scales) if s<1-1e-10}
            points=points+delta*scales[:,None];stats[stage]=metrics
        output['poses'].append(row);report.append(stats)
    text=json.dumps(output,ensure_ascii=False,separators=(',',':'))
    (ROOT/'bundle/hair-limits-data.mjs').write_text('// Generated by pipeline/fit_hair_attachment.py.\nexport const HAIR_LIMITS='+text+';\n')
    (ROOT/'.local/forehead-temple').mkdir(parents=True,exist_ok=True)
    (ROOT/'.local/forehead-temple/local-projection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
