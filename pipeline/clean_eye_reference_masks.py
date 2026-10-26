"""Approved v2.9.1 cleanup and centroid geometry, copied without numerical changes."""
import numpy as np
import cv2
YY, XX = np.indices((720, 720))

def pts(mask):
 y,x=np.where(mask);return np.column_stack([x,y]).astype(float)

def fill(mask):
 contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 res=np.zeros(mask.shape,'uint8');cv2.drawContours(res,contours,-1,1,cv2.FILLED);return res>0

def median_extreme(p,axis,high):
 u=p@axis;m=u.max() if high else u.min();return np.median(p[(u>=m-3) if high else (u<=m+3)],axis=0)

def prepare(rgb,raw,lsam,p):
 E=np.array(p['E']);ci=np.array(p['cI']);local=np.stack([XX-ci[0],YY-ci[1]],axis=-1)@E;tau,nu=local[:,:,0],local[:,:,1]
 y,x=np.where(raw);xlo,xhi,ylo,yhi=x.min(),x.max(),y.min(),y.max();width=xhi-xlo+1
 roi=(XX>=xlo-6)&(XX<=xhi+6)&(YY>=ylo-12)&(YY<=yhi);dark=rgb.max(2)<95;pupil=(tau/(.35*p['a']))**2+(nu/(.45*p['b']))**2<1
 n,lab,stats,_=cv2.connectedComponentsWithStats((roi&dark&~pupil).astype('uint8'))
 ids=[i for i in range(1,n) if stats[i,cv2.CC_STAT_WIDTH]>=.45*width and nu[lab==i].mean()<0];limg=np.isin(lab,ids)
 samkeep=np.zeros_like(raw)
 if limg.any():
  near=cv2.dilate(limg.astype('uint8'),np.ones((3,3),'uint8'))>0;n,lab=cv2.connectedComponents(lsam.astype('uint8'));mean=nu[limg].mean()
  for i in range(1,n):
   cc=lab==i
   if (cc&near).any() or abs(nu[cc].mean()-mean)<=10:samkeep|=cc
 else:samkeep=lsam.copy()
 lash=cv2.morphologyEx((samkeep|limg).astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),'uint8'))>0
 central=abs(tau)<=.3*p['w'];arcp=pts(limg&central);top=float(arcp[:,1].min()) if len(arcp) else float(p['apex'][1]);base=central&~dark&((rgb.max(2)-rgb.min(2))>20)
 patch=base&(YY>=top-14)&(YY<=top-4);source='upper'
 if patch.sum()<5:patch=base&(YY>=yhi+2)&(YY<=yhi+12);source='cheek'
 mus=(float(np.median(rgb.max(2)[patch])),float(np.median((rgb[:,:,0]-rgb[:,:,2])[patch]))) if patch.sum()>=5 else None
 return dict(lash=lash,limg=limg,mus=mus,source=source,patchCount=int(patch.sum()),coverage=float(lash[:,raw.any(0)].any(0).mean()))

def centroid_model(c,b,top,cut):
 lo=max(-1.,(top-c)/b);hi=min(1.,(cut-c)/b) if np.isfinite(cut) else 1.
 if lo>=hi:return float('nan')
 def A(u):return u*np.sqrt(max(0,1-u*u))+np.arcsin(u)
 area=A(hi)-A(lo)
 if area<1e-10:return float('nan')
 moment=(-2/3)*((max(0,1-hi*hi))**1.5-(max(0,1-lo*lo))**1.5)
 return c+b*moment/area

def root_centroid(observed,b,top,cut):
 xs=np.linspace(observed-3*b,observed+3*b,301);valid=[]
 for x in xs:
  v=centroid_model(x,b,top,cut)
  if np.isfinite(v):valid.append((x,v-observed))
 for (lo,fl),(hi,fh) in zip(valid,valid[1:]):
  if fl<=0<=fh:
   for _ in range(60):
    mid=(lo+hi)/2;fm=centroid_model(mid,b,top,cut)-observed
    if fm<0:lo=mid
    else:hi=mid
   c=(lo+hi)/2;h=.01;der=(centroid_model(c+h,b,top,cut)-centroid_model(c-h,b,top,cut))/(2*h)
   return c,float(der)
 return None,None

def cleanup(rgb, raw, iraw, lsam, p, prep, other, side='left', is_still=False):
    E = np.array(p['E'])
    ci = np.array(p['cI'])
    lash = prep['lash']
    mus = prep['mus']
    source = prep['source']
    if mus is None:
        mus = other['mus']
        source = 'other-eye'
    info = dict(coverage=prep['coverage'], imageArchArea=int(prep['limg'].sum()), patchSource=source, patchCount=prep['patchCount'])
    if mus is None:
        return None, dict(**info, error='all skin patches unavailable')
    muL, muC = mus
    L = rgb.max(2)
    RB = rgb[:, :, 0] - rgb[:, :, 2]
    RG = rgb[:, :, 0] - rgb[:, :, 1]
    GB = rgb[:, :, 1] - rgb[:, :, 2]
    dark = L < 95
    limg = prep['limg'] if prep['limg'].any() else lash
    present_limg = limg.any(0)
    valid_limg = np.flatnonzero(present_limg)
    top_limg = np.interp(np.arange(720), valid_limg, np.where(limg, YY, 720).min(0).astype(float)[valid_limg])
    tau0 = (np.stack([XX - ci[0], YY - ci[1]], axis=-1) @ E)[:, :, 0]
    lash_ir = lash & (YY <= top_limg[None, :] + np.where(abs(tau0) <= 0.65 * p['a'], 11, 15))
    present = lash.any(0)
    if not present.any():
        return None, dict(**info, error='lash empty')
    valid = np.flatnonzero(present)
    last = np.interp(np.arange(720), valid, np.where(lash_ir, YY, -1).max(0).astype(float)[valid])
    below_ir = YY > last[None, :]
    below_om = below_ir
    skin = (abs(L - muL) < 14) & (abs(RB - muC) < 14)
    peach = (RB >= 45) & (RG >= 38) & (GB <= 22) & (L >= 185)
    cand = iraw & below_ir & ~skin & ~peach
    warm_sclera = (rgb.min(2) > 165) & ((L - rgb.min(2)) < 40) & (abs(tau0) > 0.55 * p['a'])
    over_inclusive = bool(cand.sum() > 0.92 * p['area0'])
    if over_inclusive:
        cand = cv2.morphologyEx((cand & ~warm_sclera).astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), 'uint8')) > 0
    else:
        cand = cv2.morphologyEx(cand.astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), 'uint8')) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype('uint8'))
    eligible = [i for i in range(1, n) if 200 <= stats[i, cv2.CC_STAT_AREA] <= 1.5 * p['area0']]
    used_fallback = False
    if not eligible:
        used_fallback = True
        sclera_w = (rgb.min(2) > 165) & ((L - rgb.min(2)) < 40)
        dark_canthus = (RG > 45) & (L < 160)
        cfb = cv2.morphologyEx((raw & below_ir & ~skin & ~peach & ~sclera_w & ~dark_canthus).astype('uint8'), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        cand = cv2.morphologyEx((cfb & below_ir & ~skin & ~peach).astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), 'uint8')) > 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype('uint8'))
        eligible = [i for i in range(1, n) if 200 <= stats[i, cv2.CC_STAT_AREA] <= 1.5 * p['area0']]
    if not eligible:
        return None, dict(**info, error='no eligible iris component', candidateAreas=stats[1:, cv2.CC_STAT_AREA].tolist())
    sx, sy = np.rint(ci).astype(int)
    idx = int(lab[sy, sx]) if 0 <= sx < 720 and 0 <= sy < 720 else 0
    seedhit = idx in eligible
    if not seedhit:
        pp = pts(np.isin(lab, eligible))
        k = int(np.argmin(np.sum((pp - ci) ** 2, axis=1)))
        xx, yy = pp[k].astype(int)
        idx = int(lab[yy, xx])
    iris = fill(lab == idx)
    icore = cv2.morphologyEx((iris & ~warm_sclera).astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), 'uint8')) > 0
    icore = iris
    lp = pts(lash)
    lt = (lp - ci) @ E
    alluv = np.stack([XX - ci[0], YY - ci[1]], axis=-1) @ E
    alltau = alluv[:, :, 0]
    allnu = alluv[:, :, 1]
    iuv = (pts(iris) - ci) @ E
    itau = iuv[:, 0]
    inu = iuv[:, 1]
    if side == 'left':
        tau_lo = lt[:, 0].min() - 3
        tau_hi = min(lt[:, 0].max() + 3, np.median(itau[itau >= itau.max() - 3]) + 3.2)
    else:
        tau_lo = max(lt[:, 0].min() - 3, np.median(itau[itau <= itau.min() + 3]) - 3.2)
        tau_hi = lt[:, 0].max() + 3
    lateral = (alltau >= tau_lo) & (alltau <= tau_hi) & (allnu <= np.median(inu[inu >= inu.max() - 3]) + 4.2)
    bridge = cv2.dilate(iris.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    dil = cv2.dilate(raw.astype('uint8'), np.ones((3, 3), 'uint8')) > 0
    sclera_cand = ((~skin & ~peach & ~dark & (RB < 42) & (RG < 34)) & dil) & below_om & lateral
    candO = (bridge | sclera_cand) & below_om & lateral & ~skin & ~peach
    n, lab = cv2.connectedComponents(candO.astype('uint8'))
    ids = np.unique(lab[iris & candO])
    ids = ids[ids > 0]
    om_raw = fill(iris | (np.isin(lab, ids) & sclera_cand))
    om = cv2.morphologyEx(om_raw.astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), 'uint8')) > 0
    if om.sum() < 10:
        return None, dict(**info, error='opening empty')
    env_cand = ((raw & ~skin & ~peach) | om | limg) & (YY >= top_limg[None, :] - 1)
    ne, labe = cv2.connectedComponents(env_cand.astype('uint8'))
    idse = np.unique(labe[(bridge | iris) & env_cand])
    idse = idse[idse > 0]
    om_env = fill(np.isin(labe, idse))
    om_cols = np.flatnonzero(om.any(0))
    top_om = np.where(om, YY, 720).min(0)
    rim = om & (YY <= top_om[None, :] + 12) & (rgb.min(2) < 105) & (L < 130)
    lash = lash | rim
    info.update(
        coverage=float(lash[:, om.any(0)].any(0).mean()),
        lashCoverageRefined=float(lash[:, om.any(0)].any(0).mean()),
        lashCoverageRaw=prep['coverage'],
        rawCoverage=prep['coverage'],
        muL=muL,
        muC=muC,
        irisArea=int(iris.sum()),
        irisCoreArea=int(icore.sum()),
        openingArea=int(om.sum()),
        seedContained=seedhit,
        usedFallback=used_fallback,
        overInclusiveIris=over_inclusive,
        irisRecovered=used_fallback,
        seedDistance=float(np.min(np.linalg.norm(pts(iris) - ci, axis=1))),
    )
    return (om, om_raw, om_env, iris, icore, lash, skin, last), info
