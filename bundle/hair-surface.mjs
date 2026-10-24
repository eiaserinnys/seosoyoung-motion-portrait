import {DOME} from './dome-data.mjs';
import {HEAD_CENTER,HEAD_RADIUS,correctedLift,clamp,radians,artWorld,correction,rotateXYZ} from './geometry.mjs';
import {gridValue,frameValue} from './pose-grid.mjs';
const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
const rollShift=([x,y],p)=>rotateXYZ([x,y,0],0,0,p.ParamAngleZ);
export function hairDepthReliefAt([x,y]){
 const prior=DOME.depthPrior;
 const cell=(values,v)=>{
  const t=clamp((v-values[0])/(values[1]-values[0]),0,values.length-1),i=Math.min(values.length-2,Math.floor(t));
  return [i,t-i];
 };
 const [i,u]=cell(prior.x,x),[j,v]=cell(prior.y,y),rows=prior.relief;
 const a=rows[j][i]+(rows[j][i+1]-rows[j][i])*u,b=rows[j+1][i]+(rows[j+1][i+1]-rows[j+1][i])*u;
 return a+(b-a)*v;
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
// Keep the yaw dome depth; pitch uses a pole-convergent upper arc so its
// center does not sink below the rim when the head looks downward.
export function pitchHairDepth(point,p,relief=0) {
 const q=correctedLift(point,0,p),y=point[1];
 if(y<310){
  const t=clamp((y-(HEAD_CENTER[1]-HEAD_RADIUS))/246,0,1),dx=(point[0]-HEAD_CENTER[0])/DOME.rx;
  const v=1-dx*dx-((y-DOME.cy)/DOME.ry)**2,upper=v>=.16?Math.sqrt(v):Math.max(0,.4+(v-.16)/.8);
  q[2]=q[2]*smooth(t)+30*correction(p,'CorrHairDepth')*upper*Math.sin(Math.PI*t)**2;
 }
 return q[2]+relief;
}
export function fringeRotation(point,p,relief=0) {
 const local=hairLift(point,p,relief);
 if(!p.ParamAngleY)return artWorld(local,p);
 const unrolled={...p,ParamAngleZ:0};
 const pitchWeight=smooth(Math.abs(p.ParamAngleY*correction(p,'CorrTurn'))/30);
 const {cx,rx}=domeSection(point[1]),u=clamp((point[0]-502-cx)/Math.max(1,rx),-1,1);
 const detail=hairDepthReliefAt(point)*(1-u*u)*smooth((point[1]-70)/80);
 const depth=local[2]+(pitchHairDepth(point,p,relief)+detail-local[2])*pitchWeight;
 const lifted=[local[0],local[1],depth];
 const yawOnly={...unrolled,ParamAngleY:0};
 const dx0=artWorld(local,yawOnly)[0]-artWorld(lifted,yawOnly)[0];
 const q=artWorld(lifted,unrolled);q[0]+=dx0;
 if(!p.ParamAngleZ)return q;
 return artWorld(q.map((v,i)=>v-(i<2?HEAD_CENTER[i]:0)),{...p,ParamAngleX:0,ParamAngleY:0});
}
// All front layers and their rear seam consume this same reference residual.
export function hairPitchShift([x,y],p,world){
 if(!p.ParamAngleY)return [0,0,0];
 const rows=frameValue(p,'hair-pitch-rows',()=>{
  const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
  const grid=value=>gridValue(DOME.pitchPoses,yaw,pitch,value);
  return {top:grid(pose=>pose.crown.deltaY),spans:DOME.pitchRows.map((y,i)=>[y,grid(pose=>pose.crown.rows[i][1]),grid(pose=>pose.crown.rows[i][2])])};
 });
 let i=0;while(i<rows.spans.length-2&&y>rows.spans[i+1][0])i++;
 const a=rows.spans[i],b=rows.spans[i+1],t=smooth((y-a[0])/(b[0]-a[0]));
 const [offset,scale]=[1,2].map(i=>a[i]+(b[i]-a[i])*t);
 const local=rotateXYZ([world[0]-502,world[1]-568,0],0,0,-p.ParamAngleZ);
 return rollShift([offset+(local[0]+502)*scale,rows.top*(1-smooth((y-240)/140))],p);
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
 const t=clamp(Math.abs(p.ParamAngleX*correction(p,'CorrTurn'))/10,0,1),w=t*t*(3-2*t);
 return lambda+Math.sign(lambda)*Math.max(0,Math.PI/2-Math.abs(lambda))*(1-w);
}
export function craniumPosition([x,y,lambda],p){
 lambda=seamLongitude(lambda,p);
 const {rx,rz:depth,cz,cx}=domeSection(y),rz=depth*correction(p,'CorrHairDepth');
 return artWorld([cx+rx*Math.sin(lambda),y-HEAD_CENTER[1],cz+rz*Math.cos(lambda)],p);
}
export function craniumFacing([x,y,lambda],p){
 lambda=seamLongitude(lambda,p);
 const section=domeSection(y),a=domeSection(y-.5),b=domeSection(y+.5),gain=correction(p,'CorrHairDepth');
 if(section.rx<.001)return rotateXYZ([0,-1,0],p.ParamAngleX*correction(p,'CorrTurn'),0,0)[2];
 const nx=Math.sin(lambda)/Math.max(.001,section.rx),nz=Math.cos(lambda)/(section.rz*gain);
 const ny=-nx*((b.cx-a.cx)+(b.rx-a.rx)*Math.sin(lambda))-nz*((b.cz-a.cz)+(b.rz-a.rz)*gain*Math.cos(lambda));
 const n=[nx,ny,nz];
 const length=Math.hypot(...n);
 return rotateXYZ(n.map(v=>v/length),p.ParamAngleX*correction(p,'CorrTurn'),0,0)[2];
}
export function craniumSeam(){return 1;}
// The front photograph has no texture resolution at its projected horizon.
// Supply the missing near-side RGB from its independent reference photograph;
// this never changes coverage, alpha, or the common geometric surface.
export function referenceHairSample(point,p){
 const [x,y]=point,{cx,rx}=domeSection(y),theta=p.ParamAngleX*correction(p,'CorrTurn');
 const normalized=(x-DOME.cx-cx)/Math.max(.001,rx),side=theta<0?'right':'left';
 const near=theta*normalized<0;
 const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
 const lambda=Math.asin(clamp(normalized,-1,1));
 const weight=near?smooth((Math.abs(lambda)*180/Math.PI-45)/25)*smooth(Math.abs(theta)/35)*(1-smooth((y-320)/60)):0;
 const q=craniumPosition([0,y,lambda],{...p,CorrTurn:1,ParamAngleX:side==='left'?45:-45,ParamAngleY:0,ParamAngleZ:0});
 return {side,point:q,weight};
}
