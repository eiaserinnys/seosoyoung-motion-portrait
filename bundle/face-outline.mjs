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
function symmetricBase(point,p,relief=0){
 const [l,r]=outlineRow(point[1]),offset=(l+r)/2-HEAD_CENTER[0];
 const q=artWorld(correctedLift([point[0]-offset,point[1]],relief,p),p),roll=radians(p.ParamAngleZ);
 q[0]+=offset*Math.cos(roll);q[1]+=offset*Math.sin(roll);
 return q;
}
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
  const q=legacyOutline([x,y],symmetricBase([x,y],key),key);
  const s=clamp((y-465)/151,0,1);
  if(axis===0){
   const key0={...key,ParamAngleY:0};
   const qL0=legacyOutline([l,y],symmetricBase([l,y],key0),key0)[0];
   const qR0=legacyOutline([r,y],symmetricBase([r,y],key0),key0)[0];
   const mid0=(qL0+qR0)/2,half0=(qR0-qL0)/2;
   const widthScale=1+(pose.pitch>0?.05:-.05)*Math.sin(Math.PI*s);
   const midShiftX=pose.yaw?-Math.sign(pose.yaw)*(pose.pitch<0?16:3)*s:0;
   return mid0+midShiftX+(component%2?1:-1)*half0*widthScale-q[0];
  }
  const normSpan=pose.pitch>0?(pose.yaw===0?166:158):(pose.yaw===0?124:134);
  const sCurve=Math.pow(s,pose.pitch>0?.84:1.12);
  const topEdge=pose.registration.topEdgeY,topMid=(topEdge[0]+topEdge[1])/2;
  const targetY=topMid+normSpan*sCurve+(topEdge[component%2]-topMid)*(1-s);
  return targetY-q[1];
 })));
 const sPhi=clamp(pitch/30,-1.2,1.2);
 const xi=clamp((point[0]-502)/135,-1,1);
 const midArchY=-10*sPhi*(1-xi*xi)*smooth((point[1]-445)/90)*(1-.65*smooth((point[1]-595)/20));
 const unrolled={...p,ParamAngleZ:0};
 const currentY=legacyOutline(point,symmetricBase(point,unrolled),unrolled)[1];
 const directY=gridValue(OUTLINE_POSES,p.ParamAngleX*correction(p,'CorrTurn'),pitch,pose=>{
  const key={...p,CorrTurn:1,ParamAngleX:pose.yaw,ParamAngleY:0,ParamAngleZ:0};
  if(!pose.pitch)return legacyOutline(point,symmetricBase(point,key),key)[1];
  const s=clamp((y-465)/151,0,1),sCurve=Math.pow(s,pose.pitch>0?.84:1.12);
  const normSpan=pose.pitch>0?(pose.yaw===0?166:158):(pose.yaw===0?124:134);
  const topEdge=pose.registration.topEdgeY,topMid=(topEdge[0]+topEdge[1])/2;
  return topMid+normSpan*sCurve+(topEdge[0]*(1-u)+topEdge[1]*u-topMid)*(1-s);
 });
 // Keep the forehead's rigid motion, joining the normalized lower-face map
 // across the arch start and first measured cheek row without a height step.
 const lowerBlend=smooth((point[1]-445)/(465-445));
 const edgeDy=residual[2]+(residual[3]-residual[2])*u;
 const delta=[residual[0]+(residual[1]-residual[0])*u,edgeDy+(directY-currentY-edgeDy)*lowerBlend+midArchY];
 const roll=radians(p.ParamAngleZ),dx=delta[0]*weight,dy=delta[1]*weight;
 return [dx*Math.cos(roll)-dy*Math.sin(roll),dx*Math.sin(roll)+dy*Math.cos(roll),0];
}
export function applyOutline(point,world,p){
 const base=legacyOutline(point,world,p),shift=outlinePitchShift(point,p);
 return base.map((v,i)=>v+shift[i]);
}
export function outlineWorld(point,p,relief=0){
 const plain=artWorld(correctedLift(point,relief,p),p),symmetric=symmetricBase(point,p,relief),weight=correction(p,'CorrOutline');
 return applyOutline(point,plain.map((v,i)=>v+(symmetric[i]-v)*weight),p);
}
