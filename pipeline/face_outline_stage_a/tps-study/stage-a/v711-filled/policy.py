"""Coordinator correction 1: measured reference boundary, not merely band-free."""
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.spatial import cKDTree

def reference_geometry(mask):
 # Same signed-distance pixel convention as the approved v2 TPS pipeline.
 sd=distance_transform_edt(~mask)-distance_transform_edt(mask)
 contours,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 edge=np.concatenate([c[:,0,:] for c in contours]).astype(int)
 return sd,edge,cKDTree(edge)

def observed_targets(targets,band,near_boxes,geometry,threshold):
 q=np.asarray(targets,float);h,w=band.shape;inframe=(q[:,0]>=0)&(q[:,0]<=w-1)&(q[:,1]>=0)&(q[:,1]<=h-1)
 sd,edge,tree=geometry;values=map_coordinates(sd,[q[:,1],q[:,0]],order=1,mode='constant',cval=np.inf)
 distance,nearest=tree.query(q);pixel=edge[nearest];boundary_band=band[pixel[:,1],pixel[:,0]]
 inear=np.zeros(len(q),bool)
 for x0,y0,x1,y1 in near_boxes:inear|=(q[:,0]>=x0)&(q[:,0]<x1)&(q[:,1]>=y0)&(q[:,1]<y1)
 close=np.abs(values)<=threshold;measured=inframe&close&~boundary_band&~inear
 return measured,{'inFrame':inframe,'signedDistance':values,'nearBoundary':close,'nearestBoundaryPixel':pixel,'nearestBoundaryDistance':distance,'hairband':boundary_band,'nearEarBox':inear}
