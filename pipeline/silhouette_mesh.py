"""Constrained alpha-contour triangulation with holes and attachment vertices."""
import cv2
import numpy as np
import triangle
from shapely.geometry import Polygon,Point
from shapely import make_valid

def polygons(shape):
    if shape.geom_type=='Polygon':yield shape
    elif hasattr(shape,'geoms'):
        for part in shape.geoms:yield from polygons(part)

def silhouette_mesh(im,bbox,atlas,size,pins=(),seams=()):
    x,y,w,h=bbox;ax,ay,_,_=atlas
    domain=cv2.dilate((np.array(im.getchannel('A'))>8).astype(np.uint8),np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(domain,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    positions=[];triangles=[]
    if hierarchy is None:return dict(positions=[],uvs=[],triangles=[],topology='alpha-silhouette')
    for outer,relation in enumerate(hierarchy[0]):
        if relation[3]>=0 or len(contours[outer])<3:continue
        rings=[];child=relation[2]
        while child>=0:
            if len(contours[child])>=3:rings.append(contours[child].reshape(-1,2))
            child=hierarchy[0][child][0]
        shape=make_valid(Polygon(contours[outer].reshape(-1,2),rings))
        for poly in polygons(shape):
            poly=poly.simplify(.65,preserve_topology=True)
            if poly.area<1:continue
            vertices=[];segments=[];holes=[]
            for number,ring in enumerate([poly.exterior,*poly.interiors]):
                coords=list(ring.coords)[:-1];start=len(vertices);vertices.extend(coords)
                segments.extend([[start+i,start+(i+1)%len(coords)] for i in range(len(coords))])
                if number:
                    point=Polygon(ring).representative_point();holes.append([point.x,point.y])
            for px,py in pins:
                q=(float(px-x),float(py-y))
                if poly.contains(Point(q)) and q not in vertices:vertices.append(q)
            # Constrain the curved horizon itself. A triangle spanning the
            # asin limb otherwise interpolates a transparent outer wisp into
            # the opaque inner surface even when both boundaries are correct.
            for seam in seams:
                previous=None
                for px,py in seam:
                    q=(float(px-x),float(py-y))
                    if not poly.contains(Point(q)):
                        previous=None;continue
                    if q in vertices:index=vertices.index(q)
                    else:index=len(vertices);vertices.append(q)
                    if previous is not None and previous!=index:segments.append([previous,index])
                    previous=index
            data={'vertices':vertices,'segments':segments}
            if holes:data['holes']=holes
            result=triangle.triangulate(data,'pq20a150')
            offset=len(positions);positions.extend(result['vertices'].tolist())
            triangles.extend((result['triangles']+offset).tolist())
    return dict(positions=[[float(x+px),float(y+py)] for px,py in positions],
                uvs=[[(ax+px)/size,(ay+py)/size] for px,py in positions],triangles=triangles,topology='alpha-silhouette')
