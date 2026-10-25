import {FACE_SOLVE} from './face-solve-data.mjs';
import {artWorld,correction,clamp,radians} from './geometry.mjs';
import {gridValue,frameValue,FRAME_CACHE} from './pose-grid.mjs';
import {contourCoordinates} from './face-contour.mjs';

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
  const weights=FACE_SOLVE.poses.map(pose=>gridValue([pose],yaw,pitch,()=>1));
  const residual=FACE_SOLVE.points.map((_,i)=>[0,1].map(d=>FACE_SOLVE.poses.reduce((sum,pose,k)=>sum+weights[k]*(pose.t[d]+pose.du[i][d]),0)));
  return {residual};
 });
}
function face0(point,p,relief,weights=stencil(point)){
 const z=sample(FACE_SOLVE.Z,weights)+relief;
 const unrolled={...p,ParamAngleZ:0},q=artWorld([point[0]-502,point[1]-326,z],unrolled);
 const residual=frame(p).residual;
 q[0]+=weights.reduce((sum,[i,w])=>sum+w*residual[i][0],0);
 q[1]+=weights.reduce((sum,[i,w])=>sum+w*residual[i][1],0);
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
export function evaluateFace45World(point,p,family=FACE_SOLVE.family45,relief=0,weights=stencil(point)){
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
 const pitch=p.ParamAngleY*gain,pitchScale=1-.15*Math.min(1,Math.abs(pitch)/30);
 for(const d of [0,1]){
  let residual=0;
  if(family){
   const value=pose=>pose.t[d]+weights.reduce((sum,[i,w])=>sum+w*pose.du[i][d],0);
   residual=a67*value(pose67)+a90*value(pose90);
  }
  q[d]+=(shifted0[d]-p45[d]+residual)*(d===0?pitchScale:1);
 }
 return artWorld([q[0]-502,q[1]-326,q[2]+shifted0[2]-z-relief],{...p,ParamAngleX:0,ParamAngleY:0});
}
export function faceSolveWorld(point,p,relief=0,family45=FACE_SOLVE.family45){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn');
 return Math.abs(yaw)>45?evaluateFace45World(point,p,family45,relief):face0(point,p,relief);
}
