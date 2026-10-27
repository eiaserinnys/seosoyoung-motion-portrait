"""Tripo generation/alignment runner and the single T1-B3 runner."""
import argparse
import json
import sys
from pathlib import Path
import time
import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
FIT=HERE.parent/'fit'
sys.path.insert(0,str(FIT))
from fit_core import read_json, write_json, ROOT
from fit2d_core import frame_sets, template
from fit2f_core import (boundary_edges, load_targets_f, surface_contour, remesh_surface,
                       solve_surface, curvature_stats, measure_surface)
from fit_sheets import shade, label
from run_fit2f import context
from tripo_api import OUT, generate_all, TripoError
from tripo_align import align, rotation


def topology(mesh):
    import pymeshlab
    ms=pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)))
    measures=ms.get_topological_measures()
    return {'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),
            'manifold':bool(measures['non_two_manifold_edges']==0 and measures['non_two_manifold_vertices']==0),
            'components':int(measures['connected_components_number']),
            'nonmanifold_edges':int(measures['non_two_manifold_edges']),
            'nonmanifold_vertices':int(measures['non_two_manifold_vertices'])}


def prepare_mesh(path,name):
    import trimesh
    import pymeshlab
    scene=trimesh.load(path,force='scene',process=False)
    mesh=scene.to_geometry()
    raw=topology(mesh)
    # UV/normal seams duplicate geometric vertices in GLB. Connectivity must
    # be measured in position space before selecting the physical component.
    mesh.merge_vertices(merge_tex=True,merge_norm=True)
    parts=mesh.split(only_watertight=False); original=topology(mesh)
    largest=max(parts,key=lambda m:m.area)
    ms=pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(np.asarray(largest.vertices),np.asarray(largest.faces)))
    ms.meshing_close_holes(maxholesize=len(largest.vertices),refinehole=False)
    value=ms.current_mesh()
    clean=trimesh.Trimesh(value.vertex_matrix().copy(),value.face_matrix().copy(),process=False)
    stats={'raw':raw,'original_after_position_weld':original,'original_components':len(parts),
           'prepared':topology(clean),'largest_component_selection':'surface area',
           'hole_fill':'one pymeshlab meshing_close_holes, all boundary loop sizes'}
    write_json(OUT/(name+'-mesh.json'),stats)
    return np.asarray(clean.vertices).copy(),np.asarray(clean.faces).copy()


def orientation_preview(v,tri,name):
    center=(v.min(0)+v.max(0))/2; radius=np.linalg.norm(v-center,axis=1).max()
    images=[]
    for yaw,pitch in [(0,0),(90,0),(180,0),(-90,0),(0,90),(0,-90)]:
        image,_,_=shade(v-center,tri,[yaw,pitch,0,np.log(160/radius),200,200],400,cull_backfaces=True)
        label(image,f'{name} raw yaw {yaw} pitch {pitch}'); images.append(image)
    path=HERE/'sheets'/(name+'-axes.png'); cv2.imwrite(str(path),np.vstack([np.hstack(images[:3]),np.hstack(images[3:])]))
    print('AXIS_REVIEW',name,str(path),flush=True)
    answer=json.loads(input('Enter proper 3x3 raw-to-ICT rotation matrix as JSON: '))
    matrix=np.asarray(answer,float)
    if matrix.shape!=(3,3) or not np.allclose(matrix@matrix.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(matrix),1):
        raise ValueError('Expected proper orthogonal axis conversion')
    return matrix


def verify_visibility(v,tri,name):
    from fit_core import project, rasterize
    from fit2f_core import raster_silhouette
    center=(v.min(0)+v.max(0))/2; radius=np.linalg.norm(v-center,axis=1).max()
    q=project(torch.tensor(v-center),torch.tensor([0.,0.,0.,np.log(300/radius),360.,360.])).numpy()
    tick=time.perf_counter(); before,owners=rasterize(q,tri,cull_backfaces=True); full_seconds=time.perf_counter()-tick
    tick=time.perf_counter(); after,fast_owners=raster_silhouette(q,tri); fast_seconds=time.perf_counter()-tick
    np.testing.assert_array_equal(owners,fast_owners); np.testing.assert_array_equal(before,after)
    write_json(OUT/(name+'-visibility-equivalence.json'),{'pixels_and_owners_identical':True,
               'triangles':len(tri),'full_seconds':full_seconds,'prefilter_seconds':fast_seconds})
    print('VISIBILITY_EQUIVALENCE',name,full_seconds,fast_seconds,flush=True)


def slice_surface(v,tri,origin,normal):
    import trimesh
    mesh=trimesh.Trimesh(v,tri,process=False)
    sliced=trimesh.intersections.slice_mesh_plane(mesh,np.array(normal),np.array(origin),cap=False)
    sliced.remove_unreferenced_vertices()
    return np.array(sliced.vertices),np.array(sliced.faces)


def generate_align():
    started=time.perf_counter()
    try: paths=generate_all()
    except TripoError as error:
        write_json(OUT/'status.json',{'status':'part2_stopped','error_code':error.code,'message':str(error),
                   'elapsed_seconds':time.perf_counter()-started})
        print('PART2_STOP',error,flush=True); return
    cameras,shared,sref=context(); frames=frame_sets()[0]
    initial=np.array([cameras[str(f)] for f in frames])
    a=np.load(FIT/'results/fit2d/a-state.npz')['vertices']
    _,_,_,mask,_=template(); head=a[mask['head']]
    center=(head.min(0)+head.max(0))/2; height=np.ptp(head[:,1])
    alignment={}
    for name,path in paths.items():
        if (OUT/(name+'-a-state.npz')).exists():
            alignment[name]=read_json(OUT/(name+'-alignment.json'))
            print('REUSE ALIGNED',name,flush=True)
            continue
        v,tri=prepare_mesh(path,name)
        matrix=orientation_preview(v,tri,name)
        verify_visibility(v,tri,name)
        converted=v@matrix.T
        raw_height=np.ptp(converted[:,1]); raw_center=(converted.min(0)+converted.max(0))/2
        factor=height/raw_height
        # Initial global uniform datum conversion, followed by the specified
        # nine shared affine unknowns. No vertex-level deformation here.
        normalized=(converted-raw_center)*factor+center
        low=normalized[:,1].min()+.15*np.ptp(normalized[:,1])
        v,tri=slice_surface(normalized,tri,[0,low,0],[0,1,0])
        hair=name=='T2'; thickness=0. if hair else shared['thickness_px']
        first,first_poses,pass1=align(v,tri,frames,initial,thickness,hair)
        # Locate the chin by the bottommost *used skin* target, not image
        # landmarks or semantic labels on the generated surface.
        target=load_targets_f([3],pass1['thickness_px'])[0]
        skin=target['points'][target['labels']==0]
        chin_target=skin[skin[:,1].argmax()]
        model=surface_contour(first,tri,first_poses[frames.index(3)])
        nearest=np.linalg.norm(model['xy']-chin_target,axis=1).argmin()
        chin=(first[model['ids'][nearest]]*model['weights'][nearest,:,None]).sum(0)
        normal=np.array([0.,np.cos(np.radians(20)),np.sin(np.radians(20))])
        origin=chin-np.array([0.,10/sref,0.])
        cut,cut_tri=slice_surface(first,tri,origin,normal)
        # Undo pass 1 before refining the *same* rotation+diagonal-scale model;
        # composing two affine fits would introduce an unauthorized shear.
        rot=rotation(torch.tensor(np.radians(pass1['rotation_offset_deg']),dtype=torch.float64)).numpy()
        source=((cut-np.array(pass1['translation_ict']))@rot)/np.array(pass1['scale_xyz'])
        final,poses,pass2=align(source,cut_tri,frames,initial,pass1['thickness_px'],hair,pass1,first_poses)
        np.savez_compressed(OUT/(name+'-a-state.npz'),vertices=final,triangles=cut_tri,source_vertices=source,
                            open_edges=boundary_edges(cut_tri))
        frame_poses={str(f):p.tolist() for f,p in zip(frames,poses)}
        write_json(OUT/(name+'-poses.json'),{'frames':frame_poses})
        info={'raw_to_ict':matrix.tolist(),'normalization_scale':factor,'normalization_center':raw_center.tolist(),
              'world_initial_center':center.tolist(),'pass1':pass1,'pass2':pass2,
              'neck_cut':{'chin_ict':chin.tolist(),'origin':origin.tolist(),'normal':normal.tolist(),
                          'below_chin_px':10,'backward_degrees':20},
              'scale_xyz_raw_units':(factor*np.array(pass2['scale_xyz'])).tolist()}
        write_json(OUT/(name+'-alignment.json'),info); alignment[name]=info
    write_json(OUT/'status.json',{'status':'aligned','elapsed_seconds':time.perf_counter()-started,
                                'models':list(alignment)})


def t1_b3():
    if not (OUT/'T1-a-state.npz').exists():
        print('PART2 T1-B3 skipped: no aligned T1',flush=True); return
    cameras,shared,sref=context()
    source=np.load(OUT/'T1-a-state.npz'); v,tri=source['vertices'],source['triangles']
    new,new_tri,tool=remesh_surface(v,tri,sref)
    from fit2f_core import surface_distances, edges
    unique=np.unique(edges(new_tri),axis=0)
    lengths=np.linalg.norm(new[unique[:,0]]-new[unique[:,1]],axis=1)*sref
    remesh_info={'tool':tool,'vertices':len(new),'faces':len(new_tri),
        'edge_length_px':{'p10':float(np.percentile(lengths,10)),'median':float(np.median(lengths)),
                          'p90':float(np.percentile(lengths,90))},
        'distance_px':surface_distances(v,tri,new,new_tri,sref)}
    write_json(OUT/'T1-remesh.json',remesh_info)
    info=read_json(OUT/'T1-alignment.json')['pass2']; poses=read_json(OUT/'T1-poses.json')['frames']
    frames=[3,42,264]; targets=load_targets_f(frames,info['thickness_px'])
    result,d,trace,elapsed=solve_surface(new,new_tri,[poses[str(f)] for f in frames],targets,sref)
    stats,curvature,excluded=curvature_stats(new*sref,new_tri,d)
    np.savez_compressed(OUT/'T1-b3-state.npz',vertices=result,triangles=new_tri,base_vertices=new,
                        displacement_px=d,curvature=curvature,excluded=excluded,open_edges=boundary_edges(new_tri))
    keys=frame_sets()[0][:9]
    residual=measure_surface(result,new_tri,[poses[str(f)] for f in keys],load_targets_f(keys,info['thickness_px']))
    write_json(OUT/'T1-b3-residuals.json',{'tool':tool,'vertices':len(new),'faces':len(new_tri),
        'lambda':1000.,'mu':.01,'curvature':stats,'icp':trace,'elapsed_seconds':elapsed,'frames':residual,
        'd_px':{'max':float(np.linalg.norm(d,axis=1).max()),'mean':float(np.linalg.norm(d,axis=1).mean()),
                 'p95':float(np.percentile(np.linalg.norm(d,axis=1),95))}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['generate-align','t1-b3'])
    phase=parser.parse_args().phase
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    {'generate-align':generate_align,'t1-b3':t1_b3}[phase]()
