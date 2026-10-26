"""Audit opaque forward triangles, all bangs vertices, and side-hair movement."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from fit_hair_attachment import area

ROOT=Path(__file__).resolve().parents[1]

def audit(rows):
    atlas=np.array(Image.open(ROOT/'bundle/assets/texture_00.png'))
    output=[]
    for row in rows:
        maximum=0.;location=None;count=0;normal_count=0;stage_counts={k:0 for k in ['rawAttachment','attachment','rawFlare','flare','rawResidual','residual']}
        bangs_max=0.;bangs_location=None;side_max=0.;side_location=None
        for mesh in row['meshes']:
            q={key:np.array(value)[:,:2] for key,value in mesh['points'].items()}
            triangles=np.array(mesh['indices']);reference=area(q['fringe'][triangles])
            uv=np.array(mesh['uvs'])[triangles].mean(axis=1);tex=np.clip(uv*atlas.shape[0],0,atlas.shape[0]-1).astype(int)
            selected=(reference>1e-8)&(atlas[tex[:,1],tex[:,0],3]>127)
            legacy=q['rawCanopy']+q['flare']+q['residual']
            shifts=np.linalg.norm(q['final']-legacy,axis=1)
            if mesh['id'].startswith('bangs-'):
                i=int(np.argmax(shifts))
                if shifts[i]>bangs_max:
                    bangs_max=float(shifts[i]);bangs_location=dict(mesh=mesh['id'],vertex=i,rest=mesh['rest'][i])
            if mesh['id'].startswith('side-hair-'):
                i=int(np.argmax(shifts))
                if shifts[i]>side_max:
                    side_max=float(shifts[i]);side_location=dict(mesh=mesh['id'],vertex=i,rest=mesh['rest'][i])
            normal=selected&(area(legacy[triangles])>0)
            normal_count+=int(normal.sum())
            ids=np.unique(triangles[normal])
            if len(ids):
                delta=np.linalg.norm(q['final'][ids]-legacy[ids],axis=1);i=int(np.argmax(delta))
                count+=int((delta>2+1e-8).sum())
                if delta[i]>maximum:
                    maximum=float(delta[i]);v=int(ids[i]);location=dict(mesh=mesh['id'],vertex=v,rest=mesh['rest'][v],
                        before=legacy[v].tolist(),after=q['final'][v].tolist())
            stages=dict(rawAttachment=q['rawCanopy'],attachment=q['canopy'],rawFlare=q['canopy']+q['flare'],
                        flare=q['limitedFlare'],rawResidual=q['limitedFlare']+q['residual'],residual=q['final'])
            for key,positions in stages.items():stage_counts[key]+=int((area(positions[triangles])[selected]<=0).sum())
        output.append(dict(name=row['name'],normalTriangles=normal_count,normalMaxDisplacement=maximum,
                           normalVerticesOver2Px=count,normalWorst=location,bangsMaxDisplacement=bangs_max,bangsWorst=bangs_location,
                           sideMaxDisplacement=side_max,sideWorst=side_location,reversed=stage_counts))
    return output

if __name__=='__main__':
    result=audit(json.loads(Path(sys.argv[1]).read_text()))
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
