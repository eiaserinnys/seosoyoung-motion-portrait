// Orthographic scene geometry. All rotation operates on XYZ vectors, not screen offsets.
export const HEAD_CENTER=[502,326];
export const HEAD_RADIUS=262;
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
export function headWorld(local,p) {
  const q=rotateXYZ(local,p.ParamAngleX,p.ParamAngleY,p.ParamAngleZ);
  return [HEAD_CENTER[0]+q[0],HEAD_CENTER[1]+q[1],q[2]];
}
export const projectWorld=point=>point.slice(0,2);
// Inside the disk: exact front sphere. Outside: a vertical strip extending from its rim.
// The extension retains the original artwork coordinates and rotates with its attachment.
export function liftHead([x,y],relief=0,back=false) {
  const dx=x-HEAD_CENTER[0],dy=y-HEAD_CENTER[1],radius=HEAD_RADIUS+relief;
  const anchorX=clamp(dx,-radius,radius),edgeY=Math.sqrt(Math.max(0,radius*radius-anchorX*anchorX));
  const anchorY=clamp(dy,-edgeY,edgeY);
  const z=Math.sqrt(Math.max(0,radius*radius-anchorX*anchorX-anchorY*anchorY));
  return [dx,dy,back?-z:z];
}
export const sphereWarp=(point,relief,p)=>projectWorld(headWorld(liftHead(point,relief),p));
export const spherePoint=sphereWarp;
export function sphereVertex(longitude,latitude,radius=HEAD_RADIUS) {
  return [radius*Math.cos(latitude)*Math.sin(longitude),radius*Math.sin(latitude),radius*Math.cos(latitude)*Math.cos(longitude)];
}
export function eyeCenter(side) {
  const [x,y]=side==='left'?LEFT_EYE_BALL:RIGHT_EYE_BALL;
  const q=liftHead([x,y]);
  return [q[0],q[1],q[2]-EYE_BALL_R+3];
}
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
  const a=eyeballAngles(side,p,factor),center=eyeCenter(side);
  // Positive gaze Y points down in artwork coordinates.
  const q=rotateXYZ(local,(a.lon1-a.lon0)*180/Math.PI,-(a.lat1-a.lat0)*180/Math.PI,0);
  return headWorld(q.map((v,i)=>v+center[i]),p);
}
export function eyeballVertex([x,y],side,kind,p) {
  const [cx,cy]=side==='left'?LEFT_EYE_BALL:RIGHT_EYE_BALL;
  const radius=EYE_BALL_R+(kind==='pupil'?.3:kind==='highlight'?.6:0);
  const dx=x-cx,dy=y-cy,z=Math.sqrt(Math.max(0,radius*radius-dx*dx-dy*dy));
  const factor=kind==='highlight'?.65:1;
  const q=eyeWorld([dx,dy,z],side,p,factor);
  // The eyeball is rigid; the moving opening alone performs the blink clipping.
  return q;
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
