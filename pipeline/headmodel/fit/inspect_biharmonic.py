"""Read-only diagnostic of fixed components after the first fairing sheet."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from egg_pca import adjacency,mesh_edges,boundary_vertices
from fit_core import FIT,load_model,define_regions
v,t,_=load_model(); regions,_=define_regions(v,t)
face=np.zeros(len(v),bool); face[regions['face']]=True
edges=mesh_edges(t); outer=boundary_vertices(face,edges)
graph=adjacency(edges,len(v))[outer][:,outer]
count,labels=connected_components(graph,directed=False)
with np.load(FIT/'results/fit2c/egg-template.npz') as d:
    fixed=d['fixed_rings']; egg=d['neutral_vertices']
    print('FIXED COMPONENTS')
    for i in range(count):
        ids=outer[labels==i]
        print(len(ids),'center',v[ids].mean(0),'range',v[ids].min(0),v[ids].max(0))
    for name,point in [('nose',np.array([0.,-1.7,11.7])),('left brow',np.array([-6.,4.,7.]))]:
        ids=fixed[np.argsort(np.linalg.norm(v[fixed]-point,axis=1))[:5]]
        print(name,'nearest fixed',ids.tolist(),v[ids].tolist(),'moves',np.linalg.norm(egg[ids]-v[ids],axis=1).tolist())
