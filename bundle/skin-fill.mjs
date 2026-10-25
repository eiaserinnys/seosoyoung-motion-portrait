import {SKIN_FILL as S} from './skin-fill-data.mjs';
import {faceSolveAnchorWorld} from './face-solve.mjs';
import {clamp,correction,radians} from './geometry.mjs';
import {frameValue,gridValue} from './pose-grid.mjs';

const offset=[152,20];
const mesh={id:'exposed-skin',positions:S.neutral.map(p=>p.map((v,i)=>v+offset[i])),
 uvs:S.neutral.map(p=>p.map((v,i)=>(v+offset[i])/1024)),triangles:S.triangles,masks:[]};
export const skinFillMesh=()=>mesh;
export function skinFillPositions(p){
 return frameValue(p,'exposed-skin',()=>{
  const requestedYaw=clamp(p.ParamAngleX*correction(p,'CorrTurn'),-45,45),requestedPitch=clamp(p.ParamAngleY*correction(p,'CorrTurn'),-30,30);
  // The authored 45/30 controls correspond to the nearest observed video
  // representatives. Evaluate their measured angles instead of extrapolating
  // the video's 36/21 up-right sample to an unobserved 45/30 camera.
  const yaw=gridValue(S.poses,requestedYaw,requestedPitch,k=>k.sourceYaw),pitch=gridValue(S.poses,requestedYaw,requestedPitch,k=>k.sourcePitch);
  const y=yaw/45,v=pitch/30,a=radians(yaw),b=radians(pitch);
  const B=[1,y,v,y*y,y*v,v*v,y*y*y,y*y*v,y*v*v,v*v*v];
  const rotate=([x,h,z])=>[x*Math.cos(a)+h*Math.sin(a)*Math.sin(b)+z*Math.sin(a)*Math.cos(b),h*Math.cos(b)-z*Math.sin(b)];
  const eyes=S.eye3D.map((point,i)=>rotate(point).map((q,d)=>q+B.reduce((s,c,j)=>s+c*S.eyeResidual[j][i][d],0)));
  const target=['left','right'].map(k=>faceSolveAnchorWorld(k,p));
  const u=eyes[1].map((q,d)=>q-eyes[0][d]),w=target[1].map((q,d)=>q-target[0][d]),den=u[0]**2+u[1]**2;
  const c=(u[0]*w[0]+u[1]*w[1])/den,s=(u[0]*w[1]-u[1]*w[0])/den;
  const map=([x,h])=>[target[0][0]+c*(x-eyes[0][0])-s*(h-eyes[0][1])+(p.ParamBodyAngleX||0),target[0][1]+s*(x-eyes[0][0])+c*(h-eyes[0][1])-(p.ParamBreath||0)*4,-200];
  const center=rotate(S.shell.center.point3D).map((q,d)=>q+B.reduce((sum,c,j)=>sum+c*S.shell.center.residual[j][d],0));
  const radii=S.theta.map((theta,i)=>Math.hypot(...rotate([S.shell.front[i]*Math.cos(theta),S.shell.front[i]*Math.sin(theta),S.shell.depth[i]]))+B.reduce((sum,c,j)=>sum+c*S.shell.residual[j][i],0));
  return [map(center),...Array.from({length:12},(_,ring)=>S.theta.map((theta,i)=>map([center[0]+(ring+1)/12*radii[i]*Math.cos(theta),center[1]+(ring+1)/12*radii[i]*Math.sin(theta)]))).flat()];
 });
}
