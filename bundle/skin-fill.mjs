import {SKIN_FILL as S} from './skin-fill-data.mjs';
import {faceSolveAnchorWorld} from './face-solve.mjs';
import {deform3D} from './rig.mjs';
import {clamp,correction,radians,NECK_PIVOT} from './geometry.mjs';
import {frameValue,gridValue} from './pose-grid.mjs';

export const skinFillMesh=()=>({id:'exposed-skin',positions:S.neutral,
 uvs:S.neutral.map(p=>p.map(v=>v/1024)),triangles:S.triangles,masks:[]});
// Mask only: fill the upper contour interior, including the forehead/eyes.
export function skinFillClipMesh(){
 const q=clipPositions(S.neutral),m=S.columns;
 return {id:'exposed-skin-clip',positions:q,uvs:q.map(p=>p.slice(0,2).map(v=>v/1024)),
  triangles:[...Array.from({length:m-1},(_,i)=>[m,i,i+1]),[m+1,m+2,m+3],[m+1,m+3,m+4]],masks:[]};
}
function clipPositions(q){
 const outer=q.slice(5*S.columns),l=outer[0],r=outer.at(-1);
 // Clipping preserves the old face everywhere below the attachment line.
 // Closing only the upper fan would cut through pitched eye/cheek fragments
 // whose rest Y is above earHeight but whose world Y lies below that line.
 const slope=(r[1]-l[1])/(r[0]-l[0]),left=[l[0]-2048,l[1]-2048*slope,-200],right=[r[0]+2048,r[1]+2048*slope,-200],bottom=(l[1]+r[1])/2+2048;
 return [...outer,[(l[0]+r[0])/2,(l[1]+r[1])/2,-200],left,right,[right[0],bottom,-200],[left[0],bottom,-200]];
}
export const skinFillClipPositions=(p,face)=>clipPositions(skinFillPositions(p,face));
// Centered least-squares similarity uses both eyes, never a left-eye pivot.
export function similarity2D(src,dst){
 const mean=q=>[0,1].map(d=>q.reduce((s,p)=>s+p[d],0)/q.length),a=mean(src),b=mean(dst);
 let c=0,s=0,den=0;
 for(let i=0;i<src.length;i++){
  const x=src[i][0]-a[0],y=src[i][1]-a[1],u=dst[i][0]-b[0],v=dst[i][1]-b[1];
  c+=x*u+y*v;s+=x*v-y*u;den+=x*x+y*y;
 }
 c/=den;s/=den;return ([x,y])=>[b[0]+c*(x-a[0])-s*(y-a[1]),b[1]+s*(x-a[0])+c*(y-a[1])];
}
const smooth=x=>{const t=clamp(x/0.15,0,1);return t*t*(3-2*t);};
function radialRadius(points,C,theta){
 theta=Math.atan2(Math.sin(theta),Math.cos(theta));
 const polar=points.map(q=>({a:Math.atan2(q[1]-C[1],q[0]-C[0]),r:Math.hypot(q[0]-C[0],q[1]-C[1])})).sort((a,b)=>a.a-b.a);
 let i=polar.findIndex(q=>q.a>theta);if(i<0)i=0;
 const right=polar[i],left=polar[(i+polar.length-1)%polar.length];
 const span=(right.a-left.a+2*Math.PI)%(2*Math.PI),offset=(theta-left.a+2*Math.PI)%(2*Math.PI);
 return left.r+(right.r-left.r)*offset/span;
}
export function skinFillPositions(p,face){
 return frameValue(p,'exposed-skin',()=>{
  const requestedYaw=clamp(p.ParamAngleX*correction(p,'CorrTurn'),-45,45),requestedPitch=clamp(p.ParamAngleY*correction(p,'CorrTurn'),-30,30);
  const yaw=gridValue(S.poses,requestedYaw,requestedPitch,k=>k.sourceYaw),pitch=gridValue(S.poses,requestedYaw,requestedPitch,k=>k.sourcePitch);
  const y=yaw/45,v=pitch/30,a=radians(yaw),b=radians(pitch),B=[1,y,v,y*y,y*v,v*v,y*y*y,y*y*v,y*v*v,v*v*v];
  const rotate=([x,h,z])=>[x*Math.cos(a)+h*Math.sin(a)*Math.sin(b)+z*Math.sin(a)*Math.cos(b),h*Math.cos(b)-z*Math.sin(b)];
  const control=(points,res)=>points.map((point,i)=>rotate(point).map((q,d)=>q+B.reduce((s,c,j)=>s+c*res[j][i][d],0)));
  const A=S.innerRow.map(i=>deform3D(face,face.positions[i],p).slice(0,2)),EL=A[0],ER=A.at(-1),m=S.columns,last=m-1,half=last/2;
  // Anchor helpers return rolled art coordinates; deform3D adds body/breath.
  // Bring the two eyes into that same frame once, before the similarity fit.
  const target=['left','right'].map(k=>{const q=faceSolveAnchorWorld(k,p);return [q[0]+(p.ParamBodyAngleX||0),q[1]-(p.ParamBreath||0)*4];});
  // Fade the central symmetry constraint over six degrees, rather than
  // switching frames only at the exact floating-point yaw zero.
  const sym=1-smooth(Math.abs(requestedYaw)/6*.15),source=control(S.eye3D,S.eyeResidual);
  const roll=radians(p.ParamAngleZ||0),cx=502+(p.ParamBodyAngleX||0),cy=NECK_PIVOT[1]-(p.ParamBreath||0)*4;
  const unroll=([x,y])=>[cx+(x-cx)*Math.cos(roll)+(y-cy)*Math.sin(roll),cy-(x-cx)*Math.sin(roll)+(y-cy)*Math.cos(roll)];
  const reroll=([x,y])=>[cx+(x-cx)*Math.cos(roll)-(y-cy)*Math.sin(roll),cy+(x-cx)*Math.sin(roll)+(y-cy)*Math.cos(roll)];
  const [l,r]=target.map(unroll),eyeHalf=(r[0]-l[0])/2,h=(l[1]+r[1])/2;
  const symDst=[reroll([cx-eyeHalf,h]),reroll([cx+eyeHalf,h])];
  const destination=target.map((q,i)=>q.map((v,d)=>v*(1-sym)+symDst[i][d]*sym));
  const map=similarity2D(source,destination);
  // The scalp outline keeps its front shape while the head turns (card #615): no per-angle depth or residual.
  const center=rotate(S.shell.center.point3D).map((q,d)=>q+S.shell.center.residual[0][d]);
  const radii=S.theta.map((theta,i)=>Math.hypot(...rotate([S.shell.front[i]*Math.cos(theta),S.shell.front[i]*Math.sin(theta),0]))+S.shell.residual[0][i]);
  const C=map(center),shell=S.theta.map((th,i)=>map([center[0]+radii[i]*Math.cos(th),center[1]+radii[i]*Math.sin(th)]));
  let left=Math.atan2(EL[1]-C[1],EL[0]-C[0]),right=Math.atan2(ER[1]-C[1],ER[0]-C[0]);
  if(left>0)left-=2*Math.PI;
  const leftSym=(left-Math.PI-right)/2,rightSym=-Math.PI-leftSym;
  left=left*(1-sym)+leftSym*sym;right=right*(1-sym)+rightSym*sym;
  const angles=Array.from({length:m},(_,j)=>j<=half?left+(-Math.PI/2-left)*j/half:-Math.PI/2+(right+Math.PI/2)*(j-half)/half);
  // Evaluate the boundary in angular coordinates, independent of vertex density.
  const innerPath=A;
  const radiiInner=angles.map(th=>radialRadius(innerPath,C,th,true));
  const outer=angles.map((th,j)=>{
   const w=smooth(j/last)*smooth(1-j/last),r=radiiInner[j],shellRadius=radialRadius(shell,C,th,true);
   return (1-w)*r+w*shellRadius;
  });
  return Array.from({length:6},(_,k)=>angles.map((th,j)=>{
   const w=smooth(j/last)*smooth(1-j/last),skirt=Math.min(radiiInner[j],outer[j])-24*w-6;
   const r=skirt+k/5*(outer[j]-skirt);
   return [C[0]+r*Math.cos(th),C[1]+r*Math.sin(th),-200];
  })).flat();
 });
}
