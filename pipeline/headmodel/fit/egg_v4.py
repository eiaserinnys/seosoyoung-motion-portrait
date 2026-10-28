"""Feature-hull biharmonic template without a normal-direction condition."""
from egg_v3 import inside_polygon,prepare_template


def feature_mask(vertices,_normals,face_mask,polygon):
    return face_mask & inside_polygon(vertices[:,:2],polygon)


if __name__=='__main__':
    import torch,cv2
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    prepare_template(version='v4',selector=feature_mask,
                     selection_text='Face excluding ears and neck, all vertices inside expanded neutral XY feature polygon; no normal filter')
