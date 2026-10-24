import {HEAD_CENTER,HEAD_RADIUS,artWorld,correctedLift,correction,clamp,radians} from './geometry.mjs';
import {OUTLINE_ROWS,OUTLINE_POSES} from './outline-data.mjs';
import {gridValue,frameValue,FRAME_CACHE} from './pose-grid.mjs';
import {faceDepthAt,faceDepthEdgeAt} from './face-depth.mjs';
import {featureWorld} from './face-features.mjs';
import {FEATURE_FIT} from './feature-data.mjs';
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
const longitude=u=>Math.asin(clamp(2*u-1,-1,1));
const latitude=(y,p)=>Math.asin(clamp((y-HEAD_CENTER[1])/(HEAD_RADIUS*correction(p,'CorrVertical')),-1,1));
const sphereEase=(a,a0,a1)=>(1-Math.cos(clamp((a-a0)/Math.max(1e-6,a1-a0),0,1)*Math.PI))*.5;
// Across the full hemisphere this equals u; share it so anchor subtraction
// and the applied rim field use exactly the same spherical weight.
const wLon=u=>sphereEase(longitude(u),-Math.PI/2,Math.PI/2);
function targetRow(pose,y){
 const rows=pose.rows;
 let i=0;while(i<rows.length-2&&y>rows[i+1][0])i++;
 const t=clamp((y-rows[i][0])/(rows[i+1][0]-rows[i][0]),0,1);
 return rows[i].map((v,j)=>v+(rows[i+1][j]-v)*t);
}
// Stage 1: Depth Anything surface calibrated to the measured silhouette edges.
export function faceLift([x,y],relief,p){
 const old=correctedLift([x,y],relief,p);
 if(y<=400)return old;
 const edge=faceDepthEdgeAt(y),depthScale=1+(correction(p,'CorrDepth')/.78-1)*smooth((y-465)/151);
 const depth=edge+(faceDepthAt(x,y)-edge)*depthScale+relief,t=smooth((y-400)/65);
 return [old[0],old[1],old[2]+(depth-old[2])*t];
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
const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
const basePt=(pt,p)=>legacyOutline(pt,symmetricBase(pt,p),p);
export {basePt as outlinePriorWorld};
const facePt=(pt,p)=>legacyOutline(pt,symmetricFace(pt,p),p);
function edgeAt(edges,y){
 let i=0;while(i<edges.length-2&&y>edges[i+1].y)i++;
 const t=clamp((y-edges[i].y)/(edges[i+1].y-edges[i].y),0,1);
 return [mix(edges[i].left,edges[i+1].left,t),mix(edges[i].right,edges[i+1].right,t)];
}
const featureIds={browLeft:'eye-left-brow',browRight:'eye-right-brow',left:'eye-left-sclera',right:'eye-right-sclera',nose:'nose-detail',mouth:'mouth-detail'};
function faceField(p){
 const unrolled={...p,ParamAngleZ:0},neutral={...unrolled,ParamAngleY:0,[FRAME_CACHE]:new Map()};
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
 const edges=OUTLINE_ROWS.map(([y,l,r])=>{
  const residual=(side,point)=>[0,1].map(i=>gridValue(OUTLINE_POSES,yaw,pitch,pose=>{
   if(!pose.pitch)return 0;
   const key={...unrolled,CorrTurn:1,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch};
   const target=targetRow(pose,y),q=facePt(point,key),b=basePt(point,key);
   // Combine stage-one depth displacement and the measured edge residual
   // before interpolating. The sum is the target relative to legacy skin.
   return q[i]-b[i]+target[side==='left'?(i===0?2:1):(i===0?4:3)]-q[i];
  }));
  return {y,left:residual('left',[l,y]),right:residual('right',[r,y])};
 });
 const bands=[[347,['browLeft','browRight']],[431.5,['left','right']],[497,['nose']],[554.5,['mouth']],[616,[]]].map(([y,keys])=>{
  const controls=[[0,[0,0]]];
  for(const key of keys){
   const anchor=FEATURE_FIT.front[key],[l,r]=outlineRow(anchor[1]);
   const u=clamp((anchor[0]-l)/(r-l),0,1),edge=edgeAt(edges,anchor[1]);
   const f=featureWorld(featureIds[key],anchor,unrolled),b=basePt(anchor,unrolled);
   const f0=featureWorld(featureIds[key],anchor,neutral),b0=basePt(anchor,neutral);
   const residual=[0,1].map(i=>f[i]-b[i]-f0[i]+b0[i]);
   controls.push([u,residual.map((v,i)=>v-mix(edge[0],edge[1],wLon(u))[i])]);
  }
  controls.push([1,[0,0]]);controls.sort((a,b)=>a[0]-b[0]);
  return {y,controls};
 });
 return {edges,bands};
}
function interiorAt(band,u){
 if(band.controls.length<=2)return [0,0];
 const lambda=longitude(u),halfPi=Math.PI/2;
 const c=band.controls.slice(1,-1).map(([at,value])=>[longitude(at),value]);
 if(lambda<=c[0][0]){
  const w=Math.sin((lambda+halfPi)/Math.max(1e-6,c[0][0]+halfPi)*halfPi);
  return c[0][1].map(v=>v*w);
 }
 if(lambda>=c.at(-1)[0]){
  const w=Math.sin((halfPi-lambda)/Math.max(1e-6,halfPi-c.at(-1)[0])*halfPi);
  return c.at(-1)[1].map(v=>v*w);
 }
 let i=0;while(i<c.length-2&&lambda>c[i+1][0])i++;
 return mix(c[i][1],c[i+1][1],sphereEase(lambda,c[i][0],c[i+1][0]));
}
// A single residual field joins measured edges and all six internal landmarks.
export function outlinePitchShift(point,p){
 const weight=correction(p,'CorrOutline'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
 if(!pitch||!weight)return [0,0,0];
 const {edges,bands}=frameValue(p,'face-field',()=>faceField(p));
 const [l,r]=outlineRow(point[1]),u=clamp((point[0]-l)/(r-l),0,1);
 const edge=edgeAt(edges,point[1]);
 let i=0;while(i<bands.length-2&&point[1]>bands[i+1].y)i++;
 const a=bands[i],b=bands[i+1],t=sphereEase(latitude(point[1],p),latitude(a.y,p),latitude(b.y,p));
 const residual=mix(edge[0],edge[1],wLon(u)),interior=mix(interiorAt(a,u),interiorAt(b,u),t);
 // Fade only above the brow band; every measured cheek endpoint has full weight.
 const fade=smooth((point[1]-218)/(347-218));
 const dx=(residual[0]+interior[0])*weight*fade,dy=(residual[1]+interior[1])*weight*fade;
 const unrolled={...p,ParamAngleZ:0},old=basePt(point,unrolled),stage1=facePt(point,unrolled);
 const roll=radians(p.ParamAngleZ);
 return [dx*Math.cos(roll)-dy*Math.sin(roll),dx*Math.sin(roll)+dy*Math.cos(roll),stage1[2]-old[2]];
}
export function applyOutline(point,world,p){
 const base=legacyOutline(point,world,p),shift=outlinePitchShift(point,p);
 return base.map((v,i)=>v+shift[i]);
}
export function outlineWorld(point,p,relief=0){
 const plain=artWorld(correctedLift(point,relief,p),p),symmetric=symmetricBase(point,p,relief),weight=correction(p,'CorrOutline');
 return applyOutline(point,plain.map((v,i)=>v+(symmetric[i]-v)*weight),p);
}
