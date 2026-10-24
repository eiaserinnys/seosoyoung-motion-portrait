import {HEAD_CENTER,artWorld,correctedLift,correction,clamp,radians} from './geometry.mjs';
import {OUTLINE_ROWS,OUTLINE_POSES} from './outline-data.mjs';
import {gridValue,frameValue} from './pose-grid.mjs';
export {OUTLINE_ROWS} from './outline-data.mjs';
// Cubic Hermite interpolation keeps the sampled mask residual smooth in height.
export function outlineRow(y){
 if(y<=OUTLINE_ROWS[0][0])return OUTLINE_ROWS[0].slice(1);
 for(let i=1;i<OUTLINE_ROWS.length;i++)if(y<=OUTLINE_ROWS[i][0]){
  const a=OUTLINE_ROWS[i-1],b=OUTLINE_ROWS[i],prev=OUTLINE_ROWS[Math.max(0,i-2)],next=OUTLINE_ROWS[Math.min(OUTLINE_ROWS.length-1,i+1)];
  const h=b[0]-a[0],t=(y-a[0])/h;
  return a.slice(1).map((v,j)=>{
   const k=j+1,m0=(b[k]-prev[k])/(b[0]-prev[0]),m1=(next[k]-a[k])/(next[0]-a[0]);
   return (2*t**3-3*t*t+1)*v+(t**3-2*t*t+t)*h*m0+(-2*t**3+3*t*t)*b[k]+(t**3-t*t)*h*m1;
  });
 }
 return OUTLINE_ROWS.at(-1).slice(1);
}
function legacyOutline(point,world,p){
 const angle=p.ParamAngleX*correction(p,'CorrTurn'),theta=radians(clamp(angle,-45,45));
 const weight=correction(p,'CorrOutline')*clamp((point[1]-250)/80,0,1);
 if(!theta||!weight)return world;
 const sampleY=Math.max(465,point[1]);
 const [l,r,ll,lr,rl,rr,ty]=outlineRow(sampleY);
 const [frontL,frontR]=outlineRow(point[1]),u=clamp((point[0]-frontL)/(frontR-frontL),0,1);
 const half=(r-l)/2,offset=(l+r)/2-HEAD_CENTER[0];
 const residual=(yaw,targets)=>{
  const pose={...p,ParamAngleX:yaw,CorrTurn:1,ParamAngleY:0,ParamAngleZ:0};
  return [targets[0]-offset-artWorld(correctedLift([HEAD_CENTER[0]-half,sampleY],0,p),pose)[0],
          targets[1]-offset-artWorld(correctedLift([HEAD_CENTER[0]+half,sampleY],0,p),pose)[0]];
 };
 const neg=residual(-45,[ll,lr]),pos=residual(45,[rl,rr]);
 const near=(pos[1]-neg[0])/2,far=(pos[0]-neg[1])/2;
 const minus=-near*(1-u)-far*u,plus=far*(1-u)+near*u;
 const odd=Math.sin(theta)/Math.sin(Math.PI/4),even=(1-Math.cos(theta))/(1-Math.cos(Math.PI/4));
 const shift=((plus-minus)*odd+(plus+minus)*even)*.5*weight;
 const dy=(ty-Math.min(sampleY,616))*even*weight,roll=radians(p.ParamAngleZ);
 return [world[0]+shift*Math.cos(roll)-dy*Math.sin(roll),world[1]+shift*Math.sin(roll)+dy*Math.cos(roll),world[2]];
}
const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
const SOURCE_ROWS=OUTLINE_ROWS.filter(r=>r[0]>=465);
const POSE_BY_KEY=new Map(OUTLINE_POSES.map(pose=>[`${pose.yaw}:${pose.pitch}`,pose]));
function targetRow(pose,y){
 const rows=pose.rows;
 if(y<=rows[0][0]&&pose.pitch===0)return rows[0];
 let i=0;while(i<rows.length-2&&y>SOURCE_ROWS[i+1][0])i++;
 const src=SOURCE_ROWS,t=clamp((y-src[i][0])/(src[i+1][0]-src[i][0]),0,1);
 return rows[i].map((v,j)=>v+(rows[i+1][j]-v)*t);
}
// Stage 1: True 3D face mask surface replacing the vertical jaw ribbon for pitch,
// while preserving pitch=0 projection identically.
export function faceLift([x,y],relief,p){
 const old=correctedLift([x,y],relief,p);
 if(y<=400)return old;
 const v=clamp((x-HEAD_CENTER[0])/145,-1,1),s=clamp((y-465)/(616-465),0,1);
 const z0=correctedLift([502,465],0,p)[2],zEdge0=correctedLift([345,465],0,p)[2];
 const zChin=68*correction(p,'CorrDepth')/.78;
 const zMid=z0-(z0-zChin)*Math.pow(s,1.25),arch=(z0-zEdge0)*(1-.75*s);
 const zJaw=zMid-arch+arch*(1-v*v)+relief,t=smooth((y-400)/65);
 return [old[0],old[1],old[2]+(zJaw-old[2])*t];
}
export function faceWorld(point,relief,p){
 const old=correctedLift(point,relief,p),neu=faceLift(point,relief,p);
 const q=artWorld(neu,p),p0={...p,ParamAngleY:0,ParamAngleZ:0};
 const dx0=artWorld(old,p0)[0]-artWorld(neu,p0)[0],roll=radians(p.ParamAngleZ);
 q[0]+=dx0*Math.cos(roll);q[1]+=dx0*Math.sin(roll);
 return q;
}
function symmetricBase(point,p,relief=0){
 const [l,r]=outlineRow(point[1]),offset=(l+r)/2-HEAD_CENTER[0];
 const q=artWorld(correctedLift([point[0]-offset,point[1]],relief,p),p),roll=radians(p.ParamAngleZ);
 q[0]+=offset*Math.cos(roll);q[1]+=offset*Math.sin(roll);
 return q;
}
function symmetricFace(point,p,relief=0){
 const [l,r]=outlineRow(point[1]),offset=(l+r)/2-HEAD_CENTER[0];
 const q=faceWorld([point[0]-offset,point[1]],relief,p),roll=radians(p.ParamAngleZ);
 q[0]+=offset*Math.cos(roll);q[1]+=offset*Math.sin(roll);
 return q;
}
// Stage 2: Residual warp matching the 3D-rotated mask edges to 2D reference silhouettes
// while preserving Stage 1's 3D interior arch.
export function outlinePitchShift(point,p){
 const t=clamp((point[1]-218)/247,0,1);
 const weight=correction(p,'CorrOutline')*t*t*(3-2*t);
 const pitch=p.ParamAngleY*correction(p,'CorrTurn');
 if(!pitch||!weight)return [0,0,0];
 const y=clamp(point[1],465,616),[l,r]=outlineRow(y),[frontL,frontR]=outlineRow(point[1]);
 const u=clamp((point[0]-frontL)/(frontR-frontL),0,1);
 const residual=frameValue(p,'outline:'+y,()=>[0,1,2,3].map(component=>gridValue(OUTLINE_POSES,p.ParamAngleX*correction(p,'CorrTurn'),pitch,pose=>{
  if(!pose.pitch)return 0;
  const key={...p,CorrTurn:1,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,ParamAngleZ:0};
  const x=component%2?r:l,axis=component<2?0:1;
  const q=legacyOutline([x,y],symmetricFace([x,y],key),key);
  const s=clamp((y-465)/151,0,1);
  if(axis===0){
   const key0={...key,ParamAngleY:0};
   const qL0=legacyOutline([l,y],symmetricFace([l,y],key0),key0)[0];
   const qR0=legacyOutline([r,y],symmetricFace([r,y],key0),key0)[0];
   const mid0=(qL0+qR0)/2,half0=(qR0-qL0)/2;
   const [,rl,rr]=targetRow(pose,y),[,bl,br]=targetRow(POSE_BY_KEY.get(`${pose.yaw}:0`),y);
   const bw=br-bl,rw=rr-rl;
   const dHalf=clamp((rw-bw)/2,-half0*.44,half0*.56)*smooth(s/.35);
   const midShiftX=pose.yaw?-Math.sign(pose.yaw)*(pose.pitch<0?16:-2)*Math.pow(s,1.15):0;
   return mid0+midShiftX+(component%2?1:-1)*(half0+dHalf)-q[0];
  }
  const normSpan=pose.pitch>0?(pose.yaw===0?166:158):(pose.yaw===0?116:128);
  const sCurve=Math.pow(s,pose.pitch>0?.88:1.08);
  const topEdge=pose.registration.topEdgeY,topMid=(topEdge[0]+topEdge[1])/2;
  const targetY=topMid+normSpan*sCurve+(topEdge[component%2]-topMid)*.35*(1-s);
  return targetY-q[1];
 })));
 const unrolled={...p,ParamAngleZ:0};
 const stage1Pt=legacyOutline(point,symmetricFace(point,unrolled),unrolled);
 const oldPt=legacyOutline(point,symmetricBase(point,unrolled),unrolled);
 const lowerBlend=smooth((point[1]-445)/20);
 const edgeDx=residual[0]+(residual[1]-residual[0])*u;
 const edgeDy=residual[2]+(residual[3]-residual[2])*u;
 const dx=(stage1Pt[0]-oldPt[0])*lowerBlend+edgeDx*weight;
 const dy=(stage1Pt[1]-oldPt[1])*lowerBlend+edgeDy*weight;
 const dz=(stage1Pt[2]-oldPt[2])*lowerBlend;
 const roll=radians(p.ParamAngleZ);
 return [dx*Math.cos(roll)-dy*Math.sin(roll),dx*Math.sin(roll)+dy*Math.cos(roll),dz];
}
export function applyOutline(point,world,p){
 const base=legacyOutline(point,world,p),shift=outlinePitchShift(point,p);
 return base.map((v,i)=>v+shift[i]);
}
export function outlineWorld(point,p,relief=0){
 const plain=artWorld(correctedLift(point,relief,p),p),symmetric=symmetricBase(point,p,relief),weight=correction(p,'CorrOutline');
 return applyOutline(point,plain.map((v,i)=>v+(symmetric[i]-v)*weight),p);
}
