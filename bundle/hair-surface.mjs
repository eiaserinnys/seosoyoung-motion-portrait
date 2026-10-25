import {DOME} from './dome-data.mjs';
import {HEAD_CENTER,correctedLift,clamp,radians,artWorld,correction,rotateXYZ} from './geometry.mjs';
const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
export const sphereEase=t=>(1-Math.cos(clamp(t,0,1)*Math.PI))*.5;
export const pitchEase=(t,p)=>{
 const weight=sphereEase(Math.abs(p.ParamAngleY*correction(p,'CorrTurn'))/30);
 return smooth(t)+(sphereEase(t)-smooth(t))*weight;
};
// Piecewise spherical weights stay in the convex hull of the approved keys.
// This prevents trigonometric cardinal overshoot from exceeding the 45px cap.
export function hairPoseValue(poses,p,value){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
 return poses.reduce((sum,pose)=>{
  if(Math.sign(pitch)!==Math.sign(pose.pitch))return sum;
  const yawWeight=pose.yaw===0?1-sphereEase(Math.abs(yaw)/45):Math.sign(yaw)===Math.sign(pose.yaw)?sphereEase(Math.abs(yaw)/45):0;
  return sum+yawWeight*sphereEase(Math.abs(pitch)/30)*value(pose);
 },0);
}
const smax=(a,b,k=8)=>(a+b+Math.sqrt((a-b)**2+k*k))/2;
// The front dome meets the rear shell at longitude ±90°, where Z is cz.
export function hairLift(point,p,relief=0) {
 const [x,y]=point,{cx,rx,cz}=domeSection(y);
 const s=clamp((x-HEAD_CENTER[0]-cx)/Math.max(1,rx),-1,1);
 const u=clamp((250-y)/192,0,1),cap=Math.sqrt(Math.max(.08,1-u*u));
 const lowerMix=smooth((y-230)/120),skinY=clamp(y,220,350);
 const zSkinMid=correctedLift([502,skinY],0,p)[2];
 const zCenter=(195*cap*(1-lowerMix)+(zSkinMid+12)*lowerMix)*correction(p,'CorrHairDepth');
 const zDome=cz+(zCenter-cz)*Math.sqrt(Math.max(0,1-s*s));
 const zFace=correctedLift([x,skinY],0,p)[2];
 const foreheadFade=(1-smooth((Math.abs(s)-.72)/.28))*smooth((y-210)/35);
 return [x-HEAD_CENTER[0],y-HEAD_CENTER[1],zDome+(smax(zDome,zFace+6)-zDome)*foreheadFade+relief];
}
// One upper latitude arc. Its negative top depth only drives pitch Y;
// yaw X/Z always belong to hairLift, including at the rear seam.
export function pitchHairDepth(point,p,relief=0) {
 const y=point[1],base=correctedLift(point,0,p)[2];
 if(y>=310)return base+relief;
 const latitude=sphereEase((310-y)/302);
 const pitch=p.ParamAngleY*correction(p,'CorrTurn');
 const yawWeight=sphereEase(Math.abs(p.ParamAngleX*correction(p,'CorrTurn'))/45)*sphereEase(Math.max(0,pitch)/30);
 const topZ=(DOME.pitchArc.topZ+(DOME.pitchArc.diagonalTopZ-DOME.pitchArc.topZ)*yawWeight)*correction(p,'CorrHairDepth');
 return base+(topZ-base)*latitude+relief;
}
const capRadius=y=>DOME.rx*Math.sqrt(Math.max(0,1-((y-DOME.cy)/DOME.ry)**2));
function frontSection(y){
 const rows=DOME.frontSeam,cy=clamp(y,60,350),i=Math.min(rows.length-2,Math.floor((cy-60)/2));
 const a=rows[i],b=rows[i+1],t=(cy-a[0])/(b[0]-a[0]);
 const l=a[1]+(b[1]-a[1])*t,r=a[2]+(b[2]-a[2])*t;
 const scale=y<60?capRadius(y)/capRadius(60):1;
 return {rx:(r-l)/2*scale,cx:((r+l)/2-DOME.cx)*scale};
}
export const domeRadius=y=>domeSection(y).rx;
export const domeSection=y=>{
 const f=frontSection(y),t=clamp((y-350)/115,0,1),w=t*t*(3-2*t);
 const rx=f.rx+(DOME.lower.rx-f.rx)*w,cx=f.cx*(1-w);
 const upperDepth=Math.sqrt(Math.max(1,2*capRadius(y)**2-f.rx**2));
 const rz=upperDepth+(DOME.lower.rz-upperDepth)*w;
 const extent=capRadius(y)-(capRadius(465)-DOME.lower.extent45)*w;
 return {rx,rz,cx,cz:Math.hypot(rx,rz)-Math.SQRT2*extent};
};

export const projectedLongitude=(x,y)=>{const {cx,rx}=domeSection(y);return Math.asin(clamp((x-DOME.cx-cx)/Math.max(.001,rx),-1,1));};
// Orthographic inverse of an elliptical section, shared by all hair surfaces.
export function hairSurface(point,p,reference=0,relief=0){
 const [x,y]=point,{rx,rz:depth,cz,cx}=domeSection(y),rz=depth*correction(p,'CorrHairDepth'),ref=radians(reference);
 const projected=Math.hypot(rx*Math.cos(ref),rz*Math.sin(ref));
 const phase=Math.atan2(rz*Math.sin(ref),rx*Math.cos(ref));
 const lambda=Math.asin(clamp((x-DOME.cx-cx)/Math.max(.001,projected),-1,1))-phase;
 const residual=x-DOME.cx-cx-projected*Math.sin(lambda+phase);
 const yaw=radians(p.ParamAngleX*correction(p,'CorrTurn'));
 const restX=cx+rx*Math.sin(lambda)+residual,restZ=cz+rz*Math.cos(lambda);
 const xx=restX*Math.cos(yaw)+restZ*Math.sin(yaw);
 const zz=-restX*Math.sin(yaw)+restZ*Math.cos(yaw);
 return artWorld([xx,y-HEAD_CENTER[1],zz+relief],{...p,ParamAngleX:0});
}
// A finite tangent at the photographed limb prevents the thin alpha fringe
// from folding around an infinite sqrt slope. XY remains exact at rest.
export function canopySurface(point,p,back=false){
 const [x,y]=point,{rx,rz,cz,cx}=domeSection(y);
 const nx=(x-HEAD_CENTER[0]-cx)/Math.max(1,rx),v=1-nx*nx;
 const root=v>=.16?Math.sqrt(v):.4*Math.exp((v-.16)/.32);
 const t=clamp((y-60)/120,0,1),polar=t*t*(3-2*t);
 return artWorld([x-HEAD_CENTER[0],y-HEAD_CENTER[1],(cz+(back?-1:1)*rz*correction(p,'CorrHairDepth')*root)*polar],p);
}

// The narrow crack-prevention collar is folded onto the horizon at rest,
// then unfolds geometrically over the first ten degrees; alpha stays one.
function seamLongitude(lambda,p){
 const t=clamp(Math.abs(p.ParamAngleX*correction(p,'CorrTurn'))/10,0,1),w=pitchEase(t,p);
 return lambda+Math.sign(lambda)*Math.max(0,Math.PI/2-Math.abs(lambda))*(1-w);
}
export function posteriorLocal(y,lambda,p,includePosterior=true){
 const {rx,rz:depth,cz,cx}=domeSection(y),rz=depth*correction(p,'CorrHairDepth');
 const local=[cx+rx*Math.sin(lambda),y-HEAD_CENTER[1],cz+rz*Math.cos(lambda)];
 if(!includePosterior||!p.ParamAngleY||!p.ParamAngleX)return local;
 const rear=Math.sin(Math.max(0,Math.abs(lambda)-Math.PI/2));
 const latitude=Math.cos(Math.asin(clamp((y-DOME.cy)/DOME.ry,-1,1)))*sphereEase((y-200)/120);
 const delta=[hairPoseValue(DOME.rearPitch.poses,p,q=>q.rx)*Math.sin(lambda),
              hairPoseValue(DOME.rearPitch.poses,p,q=>q.y),
              hairPoseValue(DOME.rearPitch.poses,p,q=>q.rz)*Math.cos(lambda)].map(v=>v*rear*latitude);
 const zero=artWorld([0,0,0],p),world=artWorld(delta,p).map((v,i)=>v-zero[i]);
 const weight=Math.min(1,45/Math.max(1,Math.hypot(...delta),Math.hypot(...world)));
 const base=local.map((v,i)=>v+delta[i]*weight);
 const L=DOME.rearLower;if(!L)return base;
 const lw=sphereEase((y-L.yStart)/(L.yFull-L.yStart))*(1-sphereEase((y-L.yEnd)/L.fade));
 const amp=Math.min(L.cap,hairPoseValue(L.poses,p,q=>q.amp))*rear*lw;
 return [base[0]+amp*Math.sin(lambda),base[1],base[2]+amp*Math.cos(lambda)];
}
export function craniumPosition([x,y,lambda],p,includePosterior=true){
 lambda=seamLongitude(lambda,p);
 return artWorld(posteriorLocal(y,lambda,p,includePosterior),p);
}
export function craniumFacing([x,y,lambda],p){
 lambda=seamLongitude(lambda,p);
 const section=domeSection(y),a=domeSection(y-.5),b=domeSection(y+.5),gain=correction(p,'CorrHairDepth');
 if(section.rx<.001)return rotateXYZ([0,-1,0],p.ParamAngleX*correction(p,'CorrTurn'),p.ParamAngleY*correction(p,'CorrTurn'),0)[2];
 const nx=Math.sin(lambda)/Math.max(.001,section.rx),nz=Math.cos(lambda)/(section.rz*gain);
 const ny=-nx*((b.cx-a.cx)+(b.rx-a.rx)*Math.sin(lambda))-nz*((b.cz-a.cz)+(b.rz-a.rz)*gain*Math.cos(lambda));
 const n=[nx,ny,nz];
 const length=Math.hypot(...n);
 return rotateXYZ(n.map(v=>v/length),p.ParamAngleX*correction(p,'CorrTurn'),p.ParamAngleY*correction(p,'CorrTurn'),0)[2];
}
export function craniumSeam(){return 1;}
// The front photograph has no texture resolution at its projected horizon.
// Supply the missing near-side RGB from its independent reference photograph;
// this never changes coverage, alpha, or the common geometric surface.
export function referenceHairSample(point,p){
 const [x,y]=point,{cx,rx}=domeSection(y),theta=p.ParamAngleX*correction(p,'CorrTurn');
 const normalized=(x-DOME.cx-cx)/Math.max(.001,rx),side=theta<0?'right':'left';
 const near=theta*normalized<0;
 const lambda=Math.asin(clamp(normalized,-1,1));
 const weight=near?sphereEase((Math.abs(lambda)*180/Math.PI-30)/25)*sphereEase(Math.abs(theta)/30)*(1-sphereEase((y-430)/60)):0;
 if(!weight)return {side,point:[0,0,0],weight:0};
 const q=craniumPosition([0,y,lambda],{...p,CorrTurn:1,ParamAngleX:side==='left'?45:-45,ParamAngleY:0,ParamAngleZ:0});
 return {side,point:q,weight};
}
