import sys, json, ast, hashlib, os
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SHIFT = 0.0
cv2.setNumThreads(1)
OUT = ROOT / 'debug/sam3-eye-v291'
if SHIFT: OUT = OUT / ('plus2' if SHIFT > 0 else 'minus2')
CACHE = ROOT / 'sam3_masks/references/seedance'
sys.path.insert(0, str(ROOT / 'pipeline'))
from segment_eye_references import normalized_image

from clean_eye_reference_masks import pts, fill, median_extreme, prepare, centroid_model, root_centroid, cleanup

OUT.mkdir(exist_ok=True, parents=True)
rows = json.loads((CACHE / 'selection.json').read_text())['frames']
old_rest = json.loads((ROOT / 'source/references/seedance/eye-bootstrap.json').read_text())['front']
YY, XX = np.indices((720, 720))
hashes = {}
area0 = {s: int((np.asarray(Image.open(ROOT / f'sam3_masks/eye-{s}-iris_mask.png').convert('L')) > 127).sum()) for s in ['left', 'right']}

def load(path):
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return np.array(Image.open(path).convert('L')) > 127

def load_cropped(path, box=(152, 20, 152 + 720, 20 + 720)):
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return np.array(Image.open(path).convert('L').crop(box)) > 127

def make_pred(seed, E_binoc, base_prior, yaw, pitch, side):
    sy = yaw * (1 if side == 'left' else -1)
    ratio = float(np.cos(np.deg2rad(min(sy, 0.0))))
    return dict(
        cI=seed.tolist(),
        cO=seed.tolist(),
        E=E_binoc.tolist(),
        w=float(base_prior['w'] * ratio),
        a=float(base_prior['a'] * ratio),
        b=float(base_prior['bI'] * np.cos(np.deg2rad(pitch))),
        apex=(seed - base_prior['aUp'] * ratio * E_binoc[:, 1]).tolist(),
        area0=area0[side], sy=sy,
    )

def opening26(om_env, om_inner, im, side, sy, pitch, t_lash=None):
    phi = float(old_rest[side]['phi'])
    t = np.array([np.cos(phi), np.sin(phi)])
    n = np.array([-t[1], t[0]])
    E = np.column_stack([t, n])
    uv_in = pts(om_inner) @ E
    uv_env = pts(om_env) @ E
    uv_im = pts(im) @ E
    umin = float(np.median(uv_env[uv_env[:, 0] <= uv_env[:, 0].min() + 3, 0]))
    umax = float(np.median(uv_env[uv_env[:, 0] >= uv_env[:, 0].max() - 3, 0]))
    tlo_im = float(np.median(uv_im[uv_im[:, 0] <= uv_im[:, 0].min() + 3, 0]))
    thi_im = float(np.median(uv_im[uv_im[:, 0] >= uv_im[:, 0].max() - 3, 0]))
    imu = 0.5 * (tlo_im + thi_im)
    aobs_im = 0.5 * (thi_im - tlo_im)
    is_left = side == 'left'
    dTemp_obs = (imu - umin) if is_left else (umax - imu)
    dNas_obs = (umax - imu) if is_left else (imu - umin)
    dTemp = dTemp_obs
    dNas = dNas_obs
    c_in = uv_in[abs(uv_in[:, 0] - imu) <= 15] if (abs(uv_in[:, 0] - imu) <= 15).sum() >= 10 else uv_in
    vmin_in = float(np.median(c_in[c_in[:, 1] <= c_in[:, 1].min() + 3, 1]))
    vmin = float(vmin_in - t_lash)
    vmax = float(np.median(uv_in[uv_in[:, 1] >= uv_in[:, 1].max() - 3, 1]))
    u0 = imu
    f_up = float(old_rest[side]['aUp'] / (old_rest[side]['aUp'] + old_rest[side]['bLow']))
    h = float(vmax - vmin)
    v0 = float(vmin + f_up * h)
    cO = E @ np.array([u0, v0])
    w = float(umax - umin)
    aUp = float(v0 - vmin)
    bLow = float(vmax - v0)
    return dict(
        cO=cO,
        w=w,
        h=h,
        E=E,
        Pmin=E @ np.array([umin, v0]),
        Pmax=E @ np.array([umax, v0]),
        Papex=E @ np.array([u0, vmin]),
        Pbot=E @ np.array([u0, vmax]),
        aUp=aUp,
        bLow=bLow,
        phi=phi,
        dTemp=float(dTemp),
        dNas=float(dNas),
        dTempObs=float(dTemp_obs),
        dNasObs=float(dNas_obs),
        tLash=float(t_lash), hInner=float(vmax-vmin_in),
    )

def measure26(om, om_raw, im, icore, lm, last, q, p, side, sy, pitch, rest_ref=None, iris_recovered=False):
    E = q['E']
    is_left = side == 'left'
    uv_in = pts(om) @ E
    uv_im = pts(im) @ E
    uv_ic = pts(icore) @ E
    olo_in = float(np.median(uv_in[uv_in[:, 0] <= uv_in[:, 0].min() + 3, 0]))
    ohi_in = float(np.median(uv_in[uv_in[:, 0] >= uv_in[:, 0].max() - 3, 0]))
    tlo_im = float(np.median(uv_im[uv_im[:, 0] <= uv_im[:, 0].min() + 3, 0]))
    thi_im = float(np.median(uv_im[uv_im[:, 0] >= uv_im[:, 0].max() - 3, 0]))
    tlo_ic = float(np.median(uv_ic[uv_ic[:, 0] <= uv_ic[:, 0].min() + 3, 0]))
    thi_ic = float(np.median(uv_ic[uv_ic[:, 0] >= uv_ic[:, 0].max() - 3, 0]))
    aobs_im = 0.5 * (thi_im - tlo_im)
    aobs = 0.5 * (thi_ic - tlo_ic)
    b0_ref = rest_ref['bI'] if rest_ref is not None else old_rest[side]['bI']
    bp = float(b0_ref * np.cos(np.deg2rad(pitch)))
    touchL = tlo_im - olo_in <= 2
    touchR = ohi_in - thi_im <= 2
    touch_nasal = touchR if is_left else touchL
    a = aobs_im
    wx = 0.0 if touchL and touchR else 0.5 if touchL or touchR or iris_recovered else 1.0
    oc = {'bottom': False}
    local = (pts(im) - q['cO']) @ E
    mean = local.mean(0)
    tau, nu = local.T
    op = (pts(om_raw) - q['cO']) @ E
    obot = float(np.median(op[op[:, 1] >= op[:, 1].max() - 3, 1]))
    ibot = float(np.median(nu[nu >= nu.max() - 3]))
    tlo = float(np.median(tau[tau <= tau.min() + 3]))
    thi = float(np.median(tau[tau >= tau.max() - 3]))
    nasal_overrun = bool((thi_im > ohi_in + 1e-9) if is_left else (tlo_im < olo_in - 1e-9))
    ct = (tlo + thi) / 2
    xs = np.arange(max(0, int(q['Pmin'][0]) - 10), min(720, int(q['Pmax'][0]) + 11))
    lp = (np.column_stack([xs, last[xs]]) - q['cO']) @ E
    order = np.argsort(lp[:, 0])
    top = float(np.interp(mean[0], lp[order, 0], lp[order, 1]))
    cn_cen, der = root_centroid(float(mean[1]), bp, top, float('inf'))
    provisional = cn_cen
    oc['bottom'] = bool(obot - ibot <= 2 and provisional is not None and ibot - provisional < 0.65 * bp)
    cut = obot if oc['bottom'] else float('inf')
    if oc['bottom']:
        cn_cen, der = root_centroid(float(mean[1]), bp, top, cut)
    if cn_cen is not None:
        cn = cn_cen
        wy = 1.0 if (der is not None and der >= 0.35) else 0.5
    else:
        cn = float(mean[1])
        wy = 0.0
    if iris_recovered:
        wx = min(wx, 0.5)
        wy = min(wy, 0.5)
    bot = float(np.median(nu[nu >= nu.max() - 3]))
    b_prior = bp
    b_obs = bool(not oc['bottom'] and (0.65 * b_prior <= bot - cn <= 1.06 * b_prior))
    bi = float(bot-cn) if b_obs else None
    return dict(
        nasalIrisOverrun=nasal_overrun,
        nasalTouch=bool(touch_nasal),
        nasalCutDepth=float(max(0., (thi_im-ohi_in) if is_left else (olo_in-tlo_im))),
        touchL=bool(touchL),
        touchR=bool(touchR),
        bObserved=bool(b_obs),
        cI=q['cO'] + E @ np.array([ct, cn]),
        a=float(a),
        aObs=float(aobs),
        bI=bi,
        bPrior=float(b_prior),
        psi=0.0,
        delta=np.array([ct, cn]) / (rest_ref['w'] if rest_ref is not None else q['w']),
        occluded=oc,
        weightX=float(wx),
        weightY=float(wy),
        cArc=None,
        cCentroid=cn_cen,
        centroidDerivative=der,
        centroidTop=top,
        centroidCut=cut if np.isfinite(cut) else None,
        centroidObserved=float(mean[1]),
        irisVerticalDisagree=False,
        equatorFromCentroidModel=cn_cen is not None,
        centroidNoRoot=cn_cen is None,
        arcs=[],
        arcRejected=[],
    )

def lash_thickness(om, im, limg):
    # Screen-space columns, central 30% of observed opening width about iris mid-x.
    iy, ix = np.where(im); oy, ox = np.where(om)
    center = .5*(ix.min()+ix.max()); width=ox.max()-ox.min()+1
    cols=np.flatnonzero(om.any(0)&limg.any(0)&(abs(np.arange(720)-center)<=.15*width))
    if not len(cols): raise ValueError('No valid columns for tLash')
    top_o=np.where(om,YY,720).min(0);top_l=np.where(limg,YY,720).min(0)
    return float(np.median(top_o[cols]-top_l[cols])), int(len(cols))

def finish_flags(q, om_env, om, side):
    q['weightDistance']=.5 if q.get('irisRecovered',False) else 1.
    inner=pts(om)@q['E'];outer=pts(om_env)@q['E']
    lo=lambda v:float(np.median(v[v[:,0]<=v[:,0].min()+3,0]))
    hi=lambda v:float(np.median(v[v[:,0]>=v[:,0].max()-3,0]))
    gaps=[lo(inner)-lo(outer),hi(outer)-hi(inner)]
    q['gapTemporal']=gaps[0 if side=='left' else 1]
    q['gapNasal']=gaps[1 if side=='left' else 0]

def main():
    global rest0, old_rest
    # 1. Measure still.png (q_s(0)) with the exact same v2.6 pipeline
    still_path = ROOT / 'source/character/still.png'
    hashes[str(still_path)] = hashlib.sha256(still_path.read_bytes()).hexdigest()
    still_rgb = np.array(Image.open(still_path).convert('RGB').crop((152, 20, 152 + 720, 20 + 720))).astype('int16')
    front_fit = json.loads((ROOT / 'source/viewer/face-feature-fit.json').read_text())['front']
    centers0 = np.array([front_fit['left'], front_fit['right']], dtype=float) - np.array([152.0, 20.0])
    t0_b = centers0[1] - centers0[0]
    t0_b /= np.linalg.norm(t0_b)
    E0_b = np.column_stack([t0_b, [-t0_b[1], t0_b[0]]])
    still_raws = {}
    still_preds = {}
    still_prep = {}
    for si, s in enumerate(['left', 'right']):
        raw, ir, lm = [load_cropped(ROOT / f'sam3_masks/eye-{s}-{k}_mask.png') for k in ['opening', 'iris', 'upper-lashes']]
        still_raws[s] = (raw, ir, lm)
        still_preds[s] = make_pred(centers0[si], E0_b, old_rest[s], 0.0, 0.0, s)
        still_prep[s] = prepare(still_rgb, raw, lm, still_preds[s])

    rest = {}
    still_comparison = {}
    for s in ['left', 'right']:
        raw, ir, lm = still_raws[s]
        clean, stats = cleanup(still_rgb, raw, ir, lm, still_preds[s], still_prep[s], still_prep['right' if s == 'left' else 'left'], s, is_still=False)
        om, om_raw, om_env, im, icore, lash, skin, last = clean
        t0, tcols = lash_thickness(om, im, still_prep[s]['limg'])
        q0 = opening26(om_env, om, im, s, 0.0, 0.0, t_lash=t0+SHIFT)
        q0.update(measure26(om, om_raw, im, icore, lash, last, q0, still_preds[s], s, 0.0, 0.0, rest_ref=None, iris_recovered=False))
        rest[s] = dict(
            cO=q0['cO'].tolist(),
            cI=q0['cI'].tolist(),
            w=float(q0['w']),
            h=float(q0['h']),
            aUp=float(q0['aUp']),
            bLow=float(q0['bLow']),
            a=float(q0['a']),
            bI=float(q0['bI']),
            phi=float(q0['phi']),
            psi=0.0,
            delta=q0['delta'].tolist(),
            irisArea=int(stats['irisArea']),
            tLash=float(t0+SHIFT), tLashMeasured=float(t0), tLashColumns=tcols,
            dTemp=float(q0['dTemp']),
            dNas=float(q0['dNas']),
        )
        finish_flags(q0,om_env,om,s)
        rest[s].update(gapTemporal=q0['gapTemporal'],gapNasal=q0['gapNasal'])
        old_s = old_rest[s]
        old_h = float(old_s['aUp'] + old_s['bLow'])
        still_comparison[s] = dict(
            old=dict(w=old_s['w'], h=old_h, aUp=old_s['aUp'], bLow=old_s['bLow'], a=old_s['a'], bI=old_s['bI'], deltaX_px=old_s['delta'][0] * old_s['w'], deltaY_px=old_s['delta'][1] * old_s['w']),
            v26=dict(w=rest[s]['w'], h=rest[s]['h'], aUp=rest[s]['aUp'], bLow=rest[s]['bLow'], a=rest[s]['a'], bI=rest[s]['bI'], deltaX_px=rest[s]['delta'][0] * rest[s]['w'], deltaY_px=rest[s]['delta'][1] * rest[s]['w']),
            diffPx=dict(
                cO_norm=float(np.linalg.norm(np.array(rest[s]['cO']) - (np.array(old_s['cO']) - np.array([152.0, 20.0])))),
                cI_norm=float(np.linalg.norm(np.array(rest[s]['cI']) - (np.array(old_s['cI']) - np.array([152.0, 20.0])))),
                w=float(rest[s]['w'] - old_s['w']),
                h=float(rest[s]['h'] - old_h),
                a=float(rest[s]['a'] - old_s['a']),
                bI=float(rest[s]['bI'] - old_s['bI']),
                deltaX_px=float(rest[s]['delta'][0] * rest[s]['w'] - old_s['delta'][0] * old_s['w']),
                deltaY_px=float(rest[s]['delta'][1] * rest[s]['w'] - old_s['delta'][1] * old_s['w']),
            ),
        )

    rest0=json.loads(json.dumps(rest));self_consistency={};pass2_preds={};pass2_prep={}
    for si,ss in enumerate(['left','right']):
        raw,ir,lm=still_raws[ss]
        pass2_preds[ss]=make_pred(centers0[si],E0_b,rest0[ss],0.,0.,ss)
        pass2_prep[ss]=prepare(still_rgb,raw,lm,pass2_preds[ss])
    for ss in ['left','right']:
        raw,ir,lm=still_raws[ss]
        clean,stats=cleanup(still_rgb,raw,ir,lm,pass2_preds[ss],pass2_prep[ss],pass2_prep['right' if ss=='left' else 'left'],ss,is_still=False)
        if clean is None:raise ValueError('Still pass2 cleanup failed: '+ss)
        om,om_raw,om_env,im,icore,lash,skin,last=clean
        t2,cols=lash_thickness(om,im,pass2_prep[ss]['limg'])
        qq=opening26(om_env,om,im,ss,0.,0.,t_lash=rest0[ss]['tLash'])
        qq.update(measure26(om,om_raw,im,icore,lash,last,qq,pass2_preds[ss],ss,0.,0.,rest_ref=rest0[ss],iris_recovered=stats['irisRecovered']))
        r2={k:(qq[k].tolist() if isinstance(qq[k],np.ndarray) else qq[k]) for k in ['cO','cI','w','h','aUp','bLow','a','bI','phi','psi','delta','dTemp','dNas']}
        finish_flags(qq,om_env,om,ss)
        r2.update(gapTemporal=qq['gapTemporal'],gapNasal=qq['gapNasal'])
        r2.update(tLash=rest0[ss]['tLash'],tLashMeasured=t2,tLashColumns=cols,irisArea=int(stats['irisArea']))
        diffs={k:float(np.linalg.norm(np.array(r2[k])-np.array(rest0[ss][k]))) for k in ['cO','cI']}
        diffs.update({k:float(abs(r2[k]-rest0[ss][k])) for k in ['w','h']})
        rel={k:float(abs(r2[k]/rest0[ss][k]-1)) for k in ['a','bI','irisArea']}
        self_consistency[ss]=dict(differences=diffs,relativeDifferences=rel)
        rest[ss]=r2

    registration_path=ROOT/'source/viewer/ponytail-seedance-registration.json'
    hashes[str(registration_path)]=hashlib.sha256(registration_path.read_bytes()).hexdigest()
    registration=json.loads(registration_path.read_text());sil_by_name={}
    (OUT/'silhouettes').mkdir(exist_ok=True)
    skin_source=ROOT/'source/references/seedance/silhouettes'
    manifest_path=skin_source/'manifest.json';hashes[str(manifest_path)]=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for row in rows:
        source_path=skin_source/f"{row['name']}.png";hashes[str(source_path)]=hashlib.sha256(source_path.read_bytes()).hexdigest()
        mask=np.array(Image.open(source_path).convert('L'))>127
        edge=mask&~(cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0)
        dist=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        sil_by_name[row['name']]=(mask,dist,'Stage4 skin_contour reconstructed pixel mask; per-frame RNG42; nearest affine warp')
        Image.fromarray(mask.astype(np.uint8)*255).save(OUT/'silhouettes'/f"{row['name']}.png")

    def censor_test(om_env, om, q, side, sy, row):
        sil,dist,source=sil_by_name[row['name']]
        outward=-1 if side=='left' else 1
        distance=float('inf')
        for step in range(301):
            x,y=np.rint(q['cO']+step*outward*q['E'][:,0]).astype(int)
            if not (0<=x<720 and 0<=y<720 and sil[y,x]):
                distance=float(step);break
        env=q['dTemp'];theta=-sy
        q.update(dTempEnv=env,dTempSil=distance,silhouetteSource=source,
                 temporalAtFaceEdge=bool(theta>=15 and distance-env<=5),
                 silhouetteSuspect=bool(theta<15 and distance-env<=5))
        if q['temporalAtFaceEdge']:q['dTemp']=min(env,distance)

    # 2. Measure 24 Seedance frames (48 eyes)
    preds = {}
    for row in rows:
        centers = np.array(row['eyeCenters'])
        t = centers[1] - centers[0]
        t /= np.linalg.norm(t)
        E = np.column_stack([t, [-t[1], t[0]]])
        preds[row['name']] = {}
        for si, s in enumerate(['left', 'right']):
            preds[row['name']][s] = make_pred(centers[si], E, rest0[s], row['yaw'], row['pitch'], s)

    records = []
    visual = []
    pixelchecks = []
    for row in rows:
        rgb = np.array(normalized_image(row).convert('RGB')).astype('int16')
        pairs = {}
        prepared = {}
        raws = {}
        for side in ['left', 'right']:
            raw, ir, lm = [load(CACHE / f"{row['name']}-eye-{side}-{k}.png") for k in ['opening-raw', 'iris', 'upper-lashes']]
            raws[side] = (raw, ir, lm)
            prepared[side] = prepare(rgb, raw, lm, preds[row['name']][side])
        for side in ['left', 'right']:
            raw, ir, lm = raws[side]
            p = preds[row['name']][side]
            clean, stats = cleanup(rgb, raw, ir, lm, p, prepared[side], prepared['right' if side == 'left' else 'left'], side, is_still=False)
            pairs[side] = (clean, stats, raw, ir, lm)
        if any(x[0] is None for x in pairs.values()):
            for side, (clean, stats, *_) in pairs.items():
                records.append(dict(name=row['name'], side=side, yaw=row['yaw'], pitch=row['pitch'], role=row['role'], frame=row['frame'], clean=stats, error='pair clean failed'))
            continue
        for side, (clean, stats, raw, ir, lraw) in pairs.items():
            om, om_raw, om_env, im, icore, lm, skin, last = clean
            p = preds[row['name']][side]
            sy = row['yaw'] * (1 if side == 'left' else -1)
            q = opening26(om_env, om, im, side, sy, row['pitch'], t_lash=rest0[side]['tLash'])
            q['tLashObs'],q['tLashObsColumns']=lash_thickness(om,im,prepared[side]['limg'])
            q.update(measure26(om, om_raw, im, icore, lm, last, q, p, side, sy, row['pitch'], rest_ref=rest0[side], iris_recovered=stats['irisRecovered']))
            q['irisRecovered']=stats['irisRecovered']
            finish_flags(q,om_env,om,side)
            censor_test(om_env,om,q,side,sy,row)
            q['measurementFailed'] = bool(q['aUp'] < 2 or q['bLow'] < 4)
            records.append(dict(name=row['name'], side=side, yaw=row['yaw'], pitch=row['pitch'], role=row['role'], frame=row['frame'], q=q, clean=stats, prediction=p))
            if row['name'] == 'front' and side == 'left':
                for x, y in [(220, 415), (225, 420), (230, 425), (235, 430), (230, 410), (240, 440), (251, 371)]:
                    pixelchecks.append(dict(x=x, y=y, rgb=rgb[y, x].tolist(), skin=bool(skin[y, x]), opening=bool(om[y, x]), iris=bool(im[y, x])))
            img = Image.fromarray(rgb.astype('uint8'))
            dr = ImageDraw.Draw(img)
            for mask, col in [(sil_by_name[row['name']][0], '#ff8800'), (lm, '#ffff00'), (om_env, '#00ee77'), (im, '#0088ff')]:
                cs, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cs:
                    line = [tuple(z) for z in c.reshape(-1, 2)]
                    if len(line) > 1:
                        dr.line(line + [line[0]], fill=col, width=1)
            ep = [tuple(q['cI'] + q['E'] @ np.array([q['a'] * np.cos(t), (q['bI'] if q['bI'] is not None else 0.) * np.sin(t)])) for t in np.linspace(0, 2 * np.pi, 100)]
            dr.line(ep, fill='#ff00ff', width=1)
            x, y = np.where(raw)[1], np.where(raw)[0]
            box = (int(x.min()) - 6, int(y.min()) - 18, int(x.max()) + 7, int(y.max()) + 7)
            visual.append((row['name'] + ' ' + side, img.crop(box), stats['coverage'], q['weightX'], q['weightY'], q['temporalAtFaceEdge']))

    def serial(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    valid = [r for r in records if 'q' in r]
    diagnostics = dict(
        irisRecoveredEyes=[r['name']+':'+r['side'] for r in valid if r['clean']['irisRecovered']],
        bObservedCount=sum(r['q']['bObserved'] for r in valid),
        measurementFailed=sum(r['q']['measurementFailed'] for r in valid),
    )
    result = dict(K=rest0,rawStillIrisArea=area0,rest=rest,restMinusK=self_consistency,
        silhouetteSource=json.loads(manifest_path.read_text()),diagnostics=diagnostics,
        records=records,pixelChecks=pixelchecks,hashes=hashes)
    (ROOT / 'source/viewer/sam3-eye-measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=serial) + '\n')
    for page in range(2):
        sheet = Image.new('RGB', (1000, 6 * 180), '#eeeeee')
        dr = ImageDraw.Draw(sheet)
        for j, (name, im, cov, wx, wy, outlier) in enumerate(visual[page * 24:(page + 1) * 24]):
            im.thumbnail((245, 145))
            x = j % 4 * 250
            y = j // 4 * 180
            sheet.paste(im, (x, y))
            dr.text((x + 3, y + 145), name, fill='black')
            dr.text((x + 3, y + 160), f'cov={cov:.2f} wA/y={wx}/{wy} censor={outlier}', fill='black')
        sheet.save(OUT / f'overlay-{page}.png')

    summary = dict(
        eyes=len(records),
        measured=len(valid),
        coveragePass=sum(r['clean'].get('coverage', 0) >= 0.9 for r in records),
        imageArchMissing=sum(r['clean'].get('imageArchArea', 0) == 0 for r in records),
        failures=[dict(name=r['name'], side=r['side'], clean=r['clean']) for r in records if 'q' not in r],
        irisMin=min(r['clean']['irisArea'] for r in valid),
        measurementFailed=sum(r['q']['measurementFailed'] for r in valid),
        arcsValid=0,
        centroidNoRoot=sum(r['q']['centroidNoRoot'] for r in valid),
        disagree=sum(r['q']['irisVerticalDisagree'] for r in valid),
        diagnostics=diagnostics,
        stillComparison=still_comparison,
        pixelChecks=pixelchecks,
    )
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('INPUT_HASH_CHANGES', [p for p, h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h])

if __name__ == "__main__":
    main()
