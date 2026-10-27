"""Read-only topology evidence for v4's remaining local facial defects."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from fit_core import FIT,load_model,write_json
from egg_pca import mesh_edges,adjacency,vertex_normals
from egg_biharmonic import boundary_loops


def inspect():
    original,tri,_=load_model()
    data=np.load(FIT/'results/fit2c/egg-template-v4.npz')
    v=data['neutral_vertices']; faces=data['triangles']; selected=data['feature_mask']
    edges=mesh_edges(faces); adj=adjacency(edges,len(v)); normals=vertex_normals(data['original_augmented'],faces)
    original_area=np.cross(original[tri[:,1]]-original[tri[:,0]],original[tri[:,2]]-original[tri[:,0]])
    new_area=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
    keep=~selected[faces].any(1); retained=faces[keep]
    loops=boundary_loops(retained)
    near=[loop for loop in loops if selected[edges[np.any(np.isin(edges,loop),axis=1)].ravel()].any()]
    report={'feature_original_back_facing_vertices':int((selected[:len(original)]&(normals[:len(original),2]<=0)).sum()),
            'original_feature_triangles_with_negative_xy_area':int(((original_area[:,2]<0)&selected[tri].all(1)).sum()),
            'v4_feature_triangles_with_negative_xy_area':int(((new_area[:,2]<0)&selected[faces].all(1)).sum()),
            'v4_feature_triangles_with_positive_xy_area':int(((new_area[:,2]>0)&selected[faces].all(1)).sum()),
            'remove_incident_faces':int((~keep).sum()),'remaining_boundary_loops':[
                {'count':len(loop),'xyz_min':v[loop].min(0).tolist(),'xyz_max':v[loop].max(0).tolist()} for loop in loops]}
    samples={}
    for label,mask in [('nose',(np.abs(v[:,0])<1.8)&(v[:,1]>-2)&(v[:,1]<2)),
                       ('eye',(np.abs(v[:,0])>1.5)&(np.abs(v[:,0])<4)&(v[:,1]>2)&(v[:,1]<5)),
                       ('mouth',(np.abs(v[:,0])<3)&(v[:,1]>-4.5)&(v[:,1]<-2))]:
        ids=np.flatnonzero(mask&selected)
        degree=np.asarray(adj.sum(1)).ravel()
        residual=np.linalg.norm(v-adj@v/degree[:,None],axis=1)
        worst=ids[np.argsort(residual[ids])[-8:][::-1]]
        samples[label]=[{'id':int(i),'original':data['original_augmented'][i].tolist(),'v4':v[i].tolist(),
                         'original_normal_z':float(normals[i,2]),'degree':int(degree[i]),
                         'laplacian_residual':float(residual[i])} for i in worst]
    report['local_samples']=samples
    write_json(FIT/'results/fit2c/v5-diagnosis.json',report)
    print(report,flush=True)


if __name__=='__main__':inspect()
