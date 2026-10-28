import {CORRECTION_PRESETS} from './corrections.mjs';
export {CORRECTION_PRESETS} from './corrections.mjs';
import {liftEllipsoid} from './head-turn.mjs';
// Orthographic scene geometry. All rotation operates on XYZ vectors, not screen offsets.
export const HEAD_CENTER=[502,326];
export const HEAD_RADIUS=262;
export const NECK_PIVOT=[502,568,0];
export const EYE_BALL_R=54;
export const LEFT_EYE_BALL=[406,438];
export const RIGHT_EYE_BALL=[597,438];
export const radians=value=>value*Math.PI/180;
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function rotateXYZ([x,y,z],yaw=0,pitch=0,roll=0) {
  const a=radians(yaw),b=radians(pitch),c=radians(roll);
  const xx=x*Math.cos(a)+z*Math.sin(a),zz=-x*Math.sin(a)+z*Math.cos(a);
  const yy=y*Math.cos(b)-zz*Math.sin(b),zzz=y*Math.sin(b)+zz*Math.cos(b);
  return [xx*Math.cos(c)-yy*Math.sin(c),xx*Math.sin(c)+yy*Math.cos(c),zzz];
}
function rollAtNeck(local,roll) {
  const pivot=NECK_PIVOT.map((v,i)=>v-(i<2?HEAD_CENTER[i]:0));
  const q=rotateXYZ(local.map((v,i)=>v-pivot[i]),0,0,roll);
  return q.map((v,i)=>v+pivot[i]+(i<2?HEAD_CENTER[i]:0));
}
export function headWorld(local,p) {
  const gain=correction(p,'CorrTurn');
  return rollAtNeck(rotateXYZ(local,p.ParamAngleX*gain,p.ParamAngleY*gain,0),p.ParamAngleZ);
}
export const projectWorld=point=>point.slice(0,2);
// Inside the disk: exact front sphere. Outside: a vertical strip extending from its rim.
// The extension retains the original artwork coordinates and rotates with its attachment.
export function liftHead([x,y],relief=0,back=false) {
  const dx=x-HEAD_CENTER[0],dy=y-HEAD_CENTER[1],radius=HEAD_RADIUS+relief;
  const z=Math.sqrt(Math.max(0,radius*radius-dx*dx-dy*dy));
  return [dx,dy,back?-z:z];
}
export const sphereWarp=(point,relief,p)=>projectWorld(headWorld(liftHead(point,relief),p));
export const spherePoint=sphereWarp;
export function sphereVertex(longitude,latitude,radius=HEAD_RADIUS) {
  return [radius*Math.cos(latitude)*Math.sin(longitude),radius*Math.sin(latitude),radius*Math.cos(latitude)*Math.cos(longitude)];
}
export const headBaseActive=p=>p.CorrHeadBase===1;
// The 3D surface uses the complete face morph without changing the stored slider.
export const correction=(p,key)=>key==='CorrOutline'&&headBaseActive(p)?1:(p[key]??CORRECTION_PRESETS.recommended[key]);
// A tunable art surface; the diagnostic head sphere itself stays rigid and unchanged.
export function correctedLift([x,y],relief,p,back=false) {
  const r=HEAD_RADIUS,ry=r*correction(p,'CorrVertical'),rz=r*correction(p,'CorrDepth');
  const lift=([xx,yy])=>{
    const q=liftEllipsoid([xx,yy],{center:HEAD_CENTER,radii:[r,ry,rz],relief});
    // Extend the outer cap along its tangent, rather than a square-root cliff.
    // XY stays in the artist's rest pose; all attached parts share this depth.
    const rho=Math.hypot((xx-HEAD_CENTER[0])/r,(yy-HEAD_CENTER[1])/ry),join=.60;
    if(rho>join){
      const h=Math.sqrt(1-join*join),depth=rz*(h-(rho-join)*join/h)+relief;
      q[2]+=(depth-q[2])*correction(p,'CorrJaw');
    }
    return q;
  };
  const q=lift([x,y]);
  const join=490,weight=correction(p,'CorrJaw')*clamp((y-join)/45,0,1);
  // A vertical ribbon starts before the south pole. Its original XY contour is retained.
  const ribbon=lift([x,join]);
  q[2]+=(ribbon[2]-q[2])*weight;
  if(back)q[2]=-q[2];
  return q;
}
export function artWorld(local,p) {
  const gain=correction(p,'CorrTurn'),x=p.ParamAngleX*gain,y=p.ParamAngleY*gain,z=p.ParamAngleZ;
  const q=rotateXYZ(local,x,y,0),a=rotateXYZ(local,x,0,0),b=rotateXYZ(local,0,y,0);
  const w=correction(p,'CorrDiagonal')*Math.min(1,Math.abs(x/50*y/40));
  // Remove an adjustable fraction of the measured 3D XY interaction, before roll.
  q[0]+=(a[0]+b[0]-local[0]-q[0])*w;
  q[1]+=(a[1]+b[1]-local[1]-q[1])*w;
  return rollAtNeck(q,z);
}
const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
const unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function eyeFrame(side,p={}) {
  const xy=side==='left'?LEFT_EYE_BALL:RIGHT_EYE_BALL;
  const q=correctedLift(xy,0,p),r=HEAD_RADIUS,ry=r*correction(p,'CorrVertical'),rz=r*correction(p,'CorrDepth');
  const normal=unit([q[0]/(r*r),q[1]/(ry*ry),q[2]/(rz*rz)]);
  const right=unit(cross([0,1,0],normal)),down=cross(normal,right);
  const center=q.map((v,i)=>v-normal[i]*(EYE_BALL_R-3+correction(p,'CorrEyeDepth')));
  return {center,normal,right,down};
}
export const eyeCenter=(side,p={})=>eyeFrame(side,p).center;
export const eyeRootWorld=(side,p)=>artWorld(eyeCenter(side,p),p);
export function eyeballAngles(side,p,factor=1) {
  const [bcx,bcy]=side==='left'?LEFT_EYE_BALL:RIGHT_EYE_BALL;
  const icx=side==='left'?417:587,icy=side==='left'?438:437.5;
  const lon0=Math.asin((icx-bcx)/EYE_BALL_R),lat0=0;
  const gx=p.ParamEyeBallX,gy=p.ParamEyeBallY;
  const gazeLon=side==='left'?gx*(gx<0?24:14):gx*(gx<0?14:24);
  const lon1=clamp(lon0+radians(gazeLon)*factor,radians(-28),radians(28));
  const lat1=clamp(radians(gy*15)*factor,radians(-22),radians(22));
  return {bcx,bcy,icx,icy,lon0,lat0,lon1,lat1};
}
export function eyeWorld(local,side,p,factor=1) {
  const a=eyeballAngles(side,p,factor),f=eyeFrame(side,p);
  const q=rotateXYZ(local,(a.lon1-a.lon0)*180/Math.PI,-(a.lat1-a.lat0)*180/Math.PI,0);
  const oriented=[0,1,2].map(i=>f.right[i]*q[0]+f.down[i]*q[1]+f.normal[i]*q[2]);
  const gain=correction(p,'CorrTurn');
  const rotated=rotateXYZ(oriented,p.ParamAngleX*gain,p.ParamAngleY*gain,p.ParamAngleZ),root=eyeRootWorld(side,p);
  return root.map((v,i)=>v+rotated[i]);
}
export function eyeballVertex([x,y],side,kind,p) {
  const f=eyeFrame(side,p),radius=EYE_BALL_R;
  // Intersect the original artwork ray with the tilted child's sphere, preserving XY.
  const dx=x-502-f.center[0],dy=y-326-f.center[1];
  const ray=[dx,dy,Math.sqrt(Math.max(0,radius*radius-dx*dx-dy*dy))];
  const local=[dot(ray,f.right),dot(ray,f.down),dot(ray,f.normal)];
  return eyeWorld(local,side,p,kind==='highlight'?.65:1);
}
export function makeSphereMesh(radius=HEAD_RADIUS,cols=96,rows=48) {
  const positions=[],uvs=[],triangles=[];
  for(let j=0;j<=rows;j++)for(let i=0;i<=cols;i++){
    positions.push(sphereVertex(-Math.PI+i/cols*Math.PI*2,-Math.PI/2+j/rows*Math.PI,radius));
    uvs.push([i/cols,j/rows]);
  }
  for(let j=0;j<rows;j++)for(let i=0;i<cols;i++){
    const a=j*(cols+1)+i,b=a+1,c=a+cols+1,d=c+1;
    triangles.push([a,b,c],[b,d,c]);
  }
  return {positions,uvs,triangles};
}
