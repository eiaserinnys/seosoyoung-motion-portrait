"""Pixel gates for the visible ribbon, its axis, and the original face seam."""
import numpy as np


def validate_pixels(name,rgba,identified,geometry):
    A=np.array(geometry['face'])[geometry['innerRow'],:2]
    rest=np.array(geometry['faceRest'])[geometry['innerRow'],:2]
    yy,xx=np.indices(rgba.shape[:2]);EL,ER=A[0],A[-1]
    below=(ER[0]-EL[0])*(yy+.5-EL[1])-(ER[1]-EL[1])*(xx+.5-EL[0])>=0
    # Identify pass paints only exposed backing red; face/ears overwrite black.
    owned=(identified[:,:,0]>127)&(identified[:,:,1]<20)&(identified[:,:,2]<20)&(identified[:,:,3]>127)
    protruding=int(np.count_nonzero(owned&below))
    assert protruding==0,(name,'Exposed ribbon beyond the ear attachment line',protruding)
    halfwidth=0
    if name=='down':
        for y in range(360):
            xs=np.flatnonzero(rgba[y,:,3]>127)
            if len(xs):halfwidth=max(halfwidth,abs(1004-int(xs[0])-int(xs[-1])))
        assert halfwidth<=2,(name,'Upper dome half-width asymmetry',halfwidth)
    luminance=rgba[:,:,:3]@np.array([.2126,.7152,.0722])
    valleys=[]
    # Vertical profiles through the forehead join; restrict to the crown where
    # brow ink and the side boundary cannot be mistaken for an alpha/RGB seam.
    for (x,y),(rx,ry) in zip(A,rest):
        if not (442<=rx<=580 and ry<260):continue
        ix,iy=round(x),round(y)
        values=luminance[iy-4:iy+5,ix]
        alpha=rgba[iy-4:iy+5,ix,3]
        if len(values)!=9 or np.min(alpha)<250:continue
        valleys.append(max(0.,float(min(values[0],values[-1])-np.min(values[2:7]))))
    valley=max(valleys,default=0.)
    assert valley<=10,(name,'Forehead boundary luminance valley',valley)
    return dict(name=name,exposedPixelsBelowEarLine=protruding,downHalfWidthDifference=halfwidth,
                maxForeheadLuminanceValley=valley,foreheadProfileCount=len(valleys))
