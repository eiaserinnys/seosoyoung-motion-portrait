import {FACE_SOLVE} from './face-solve-data.mjs';
import {faceSolveWorld as baselineProfileWorld,faceSolveAnchorWorld as baselineProfileAnchorWorld} from './profile-baseline/face-solve.mjs';
import {FRAME_CACHE as BASELINE_FRAME_CACHE} from './profile-baseline/pose-grid.mjs';
import {smoothstep} from './face-profile.mjs';
import {artWorld,correction,clamp,radians} from './geometry.mjs';
import {gridValue,frameValue,FRAME_CACHE} from './pose-grid.mjs';
import {contourCoordinates} from './face-contour.mjs';
import {ponytailSkinDelta} from './ponytail-skin.mjs';

const key=point=>point[0]+','+point[1];
const anchors=new Set(Object.values(FACE_SOLVE.anchors));
const nodes=new Map(FACE_SOLVE.points.map((p,i)=>[key(p),i]).filter(([,i])=>!anchors.has(i)));
const queries=new Map();
const evaluationPoints=FACE_SOLVE.points.map(contourCoordinates);
const triangles=FACE_SOLVE.evaluationTriangles.map(indices=>{
 const [a,b,c]=indices.map(i=>evaluationPoints[i]);
 const det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
 return {indices,a,b,c,det};
});
const edgeCounts=new Map();
for(const {indices} of triangles)for(let k=0;k<3;k++){
 const edge=[indices[k],indices[(k+1)%3]].sort((a,b)=>a-b),name=edge.join(',');
 const old=edgeCounts.get(name);edgeCounts.set(name,{edge,count:(old?.count||0)+1});
}
const hull=[...edgeCounts.values()].filter(v=>v.count===1).map(v=>v.edge);
function stencil(point){
 const name=key(point),index=nodes.get(name);
 if(index!==undefined)return [[index,1]];
 if(queries.has(name))return queries.get(name);
 const [x,y]=contourCoordinates(point);
 for(const {indices,a,b,c,det} of triangles){
  const u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/det;
  const v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/det,w=1-u-v;
  if(Math.min(u,v,w)>=-1e-9){const result=indices.map((i,k)=>[i,[u,v,w][k]]);queries.set(name,result);return result;}
 }
 // Continuous extension from the closest hull segment for queries outside the skin.
 let best=Infinity,result;
 for(const [i,j] of hull){
  const a=evaluationPoints[i],b=evaluationPoints[j],dx=b[0]-a[0],dy=b[1]-a[1];
  const t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)));
  const d=(x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2;
  if(d<best){best=d;result=[[i,1-t],[j,t]];}
 }
 queries.set(name,result);return result;
}
const sample=(values,weights)=>weights.reduce((sum,[i,w])=>sum+w*values[i],0);
function frame(p){
 return frameValue(p,'face-solve',()=>{
  const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
  const weights=FACE_SOLVE.poses.map(pose=>gridValue([pose],yaw,pitch,()=>1)),residual=[];
  // A plain query needs only its stencil, not all 740 node residuals. Render
  // frames retain the same per-node values as they are first requested.
  return {residualAt:i=>residual[i]??=([0,1].map(d=>FACE_SOLVE.poses.reduce((sum,pose,k)=>sum+weights[k]*(pose.t[d]+pose.du[i][d]),0)))};
 });
}
function face0(point,p,relief,weights=stencil(point)){
 const z=sample(FACE_SOLVE.Z,weights)+relief;
 const unrolled={...p,ParamAngleZ:0},q=artWorld([point[0]-502,point[1]-326,z],unrolled);
 const {residualAt}=frame(p);
 q[0]+=weights.reduce((sum,[i,w])=>sum+w*residualAt(i)[0],0);
 q[1]+=weights.reduce((sum,[i,w])=>sum+w*residualAt(i)[1],0);
 if(FACE_SOLVE.ponytail&&!weights.every(([i])=>anchors.has(i))){
  const delta=ponytailSkinDelta(point,p.ParamAngleX*correction(p,'CorrTurn'),p.ParamAngleY*correction(p,'CorrTurn'),p);
  q[0]+=delta[0];q[1]+=delta[1];
 }
 // artWorld applies the shared neck-pivot roll after the complete XY solve.
 return artWorld([q[0]-502,q[1]-326,q[2]],{...p,ParamAngleX:0,ParamAngleY:0});
}
export function faceSolveAnchorWorld(name,p,family=FACE_SOLVE.family45){
 const index=FACE_SOLVE.anchors[name];
 if(Math.abs(p.ParamAngleX*correction(p,'CorrTurn'))>45)return evaluateFace45World(FACE_SOLVE.points[index],p,family,0,[[index,1]]);
 return face0(FACE_SOLVE.points[index],p,0,[[index,1]]);
}

// Trigonometric collocation passes through 0, 22.5 and 45 degree increments.
export function turnWeights45(delta){
 const u=radians(clamp(Math.abs(delta),0,45)),u1=Math.PI/8,u2=Math.PI/4;
 const s=Math.sin(u),c=1-Math.cos(u),s1=Math.sin(u1),c1=1-Math.cos(u1),s2=Math.sin(u2),c2=1-Math.cos(u2);
 const det=s1*c2-s2*c1;
 return [(c2*s-s2*c)/det,(-c1*s+s1*c)/det];
}
// Side depths use the same node ordering as face-0, in the evaluated ±45° frame.
function face45Prior(point,p,family,relief,weights,useBaseline=true){
 const gain=correction(p,'CorrTurn'),yaw=p.ParamAngleX*gain,sign=yaw<0?-1:1;
 const edge=frameValue(p,'face45-edge',()=>({...p,ParamAngleX:sign*45/gain,ParamAngleZ:0,[FRAME_CACHE]:new Map()}));
 const rest=frameValue(p,'face45-rest',()=>({...edge,ParamAngleY:0,[FRAME_CACHE]:new Map()}));
 const q=face0(point,edge,relief,weights),p45=face0(point,rest,0,weights);
 const z=family?sample(sign<0?family.ZLeft:family.ZRight,weights):0;
 const delta=yaw-sign*45;
 const local=[p45[0]-502,p45[1]-326,z+relief];
 const shifted0=artWorld(local,{...p,CorrTurn:1,ParamAngleX:delta,ParamAngleY:0,ParamAngleZ:0});
 const [a67,a90]=turnWeights45(delta);
 const pose67=family?.poses.find(pose=>pose.name===(sign<0?'left67':'right67'));
 const pose90=family?.poses.find(pose=>pose.name===(sign<0?'left90':'right90'));
 const pitch=p.ParamAngleY*gain;
 const pitchScale=pitch?frameValue(p,'face45-widths',()=>{
  const flat={...p,ParamAngleY:0,ParamAngleZ:0,[FRAME_CACHE]:new Map()};
  return FACE_SOLVE.samples.map(s=>{
   const a=[s.left,s.right].map(i=>face0(FACE_SOLVE.points[i],rest,0,[[i,1]]));
   const b=[s.left,s.right].map(i=>face45Prior(FACE_SOLVE.points[i],flat,family,0,[[i,1]],false));
   return [(a[0][1]+a[1][1])/2,a[0][0],a[1][0],b[0][0],b[1][0]];
  });
 }):null;
 let composedX;
 if(pitchScale){
  // Compose in the projected 45-degree frame: using the original source row
  // here would add two fields evaluated at different latitudes and can fold.
  let i=0;while(i<pitchScale.length-2&&q[1]>pitchScale[i+1][0])i++;
  const a=pitchScale[i],b=pitchScale[i+1],t=clamp((q[1]-a[0])/(b[0]-a[0]),0,1);
  const row=a.map((v,k)=>v+t*(b[k]-v));
  composedX=row[3]+(q[0]-row[1])*(row[4]-row[3])/(row[2]-row[1]);
 }
 for(const d of [0,1]){
  let residual=0;
  if(family){
   const value=pose=>pose.t[d]+weights.reduce((sum,[i,w])=>sum+w*pose.du[i][d],0);
   residual=a67*value(pose67)+a90*value(pose90);
  }
  const turned=shifted0[d]+residual;
  q[d]=d===0&&pitch?composedX:q[d]+turned-p45[d];
 }
 const current=artWorld([q[0]-502,q[1]-326,q[2]+shifted0[2]-z-relief],{...p,ParamAngleX:0,ParamAngleY:0});
 if(!useBaseline)return current;
 // Beyond 67.5 degrees the authored 6dd absolute profile is retained. At
 // 45 degrees the weight and its first derivative are both zero.
 const baselinePose=frameValue(p,'profile-baseline-pose',()=>({...p,[BASELINE_FRAME_CACHE]:new Map()}));
 const anchorName=weights.length===1&&weights[0][1]===1?Object.keys(FACE_SOLVE.anchors).find(name=>FACE_SOLVE.anchors[name]===weights[0][0]):undefined;
 const old=anchorName===undefined?baselineProfileWorld(point,baselinePose,relief):baselineProfileAnchorWorld(anchorName,baselinePose),w=smoothstep(0,22.5,Math.abs(yaw)-45);
 return current.map((v,d)=>v*(1-w)+old[d]*w);
}
const fixedSideNodes=new Set([...Object.values(FACE_SOLVE.anchors),...FACE_SOLVE.samples.flatMap(s=>[s.left,s.right,s.center])]);
function sideTriangleCorrection(p,family){
 const xy=FACE_SOLVE.points.map((point,i)=>face45Prior(point,p,family,0,[[i,1]]));
 const correction=FACE_SOLVE.points.map(()=>0);
 // With Y fixed, signed triangle area is linear in its three X coordinates.
 // Orthogonal half-space projections move only free skin vertices; every
 // measured rim/anchor stays fixed. The observed fold needs one local move.
 for(let pass=0;pass<16;pass++){
  let changed=false;
  for(const tri of FACE_SOLVE.triangles){
   const [a,b,c]=tri.map(i=>xy[i]),[u,v,w]=tri.map(i=>FACE_SOLVE.points[i]);
   const rest=(v[0]-u[0])*(w[1]-u[1])-(v[1]-u[1])*(w[0]-u[0]);
   const sign=Math.sign(rest),area=sign*((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]));
   const minimum=Math.abs(rest)*1e-6;
   if(area>0)continue;
   const gradient=[b[1]-c[1],c[1]-a[1],a[1]-b[1]].map((g,k)=>fixedSideNodes.has(tri[k])?0:sign*g);
   const norm=gradient.reduce((sum,g)=>sum+g*g,0);
   if(!norm)throw Error('Side triangle conflicts with fixed observations');
   const step=(minimum-area)/norm;
   tri.forEach((i,k)=>{const dx=step*gradient[k];xy[i][0]+=dx;correction[i]+=dx;});
   changed=true;
  }
  if(!changed)return correction;
 }
 throw Error('Side triangle projection did not converge');
}
// The unmixed B position used as the new-frame input to the A interpolation.
// Keep its observations testable independently of the final legacy blend.
export function evaluateFace45BaseWorld(point,p,family=FACE_SOLVE.family45,relief=0,weights=stencil(point)){
 const unrolled={...p,ParamAngleZ:0},q=face45Prior(point,unrolled,family,relief,weights,false);
 return artWorld([q[0]-502,q[1]-326,q[2]],{...p,ParamAngleX:0,ParamAngleY:0});
}
export function evaluateFace45World(point,p,family=FACE_SOLVE.family45,relief=0,weights=stencil(point)){
 const unrolled={...p,ParamAngleZ:0},q=face45Prior(point,unrolled,family,relief,weights);
 if(Math.abs(p.ParamAngleX*correction(p,'CorrTurn'))>45){
  const delta=frameValue(p,'face45-triangle-projection',()=>sideTriangleCorrection(unrolled,family));
  q[0]+=sample(delta,weights);
 }
 return artWorld([q[0]-502,q[1]-326,q[2]],{...p,ParamAngleX:0,ParamAngleY:0});
}
export function faceSolveWorld(point,p,relief=0,family45=FACE_SOLVE.family45){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn');
 return Math.abs(yaw)>45?evaluateFace45World(point,p,family45,relief):face0(point,p,relief);
}
