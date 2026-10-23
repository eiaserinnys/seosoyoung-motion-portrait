/** Image coordinates: x right, y down, z toward viewer; degrees, yaw then pitch then roll. */
export function rotate([x,y,z],[yaw,pitch,roll]) {
  const [a,b,c]=[yaw,pitch,roll].map(v=>v*Math.PI/180);
  const X=x*Math.cos(a)+z*Math.sin(a),Z=-x*Math.sin(a)+z*Math.cos(a);
  const Y=y*Math.cos(b)-Z*Math.sin(b),zz=y*Math.sin(b)+Z*Math.cos(b);
  return [X*Math.cos(c)-Y*Math.sin(c),X*Math.sin(c)+Y*Math.cos(c),zz];
}
/** Sphere within disk; outside it, a vertical tangent strip. No invented back texture. */
export function lift([x,y],{center:[cx,cy],radius:r,depthSign=1}) {
  if(!(r>0))throw new RangeError('radius must be positive');
  const dx=x-cx,dy=y-cy;
  return [dx,dy,depthSign*Math.sqrt(Math.max(0,r*r-dx*dx-dy*dy))];
}
export function project(local,{center:[cx,cy]},angles) {
  const [x,y]=rotate(local,angles);return [cx+x,cy+y];
}
/** Eye offset is in head-local coordinates; gaze and head transform each occur once. */
export function eyePoint(local,center,gaze,headAngles) {
  return rotate(rotate(local,gaze).map((v,i)=>v+center[i]),headAngles);
}
function checkGrid(grid) {
  if(grid.length!==3||grid.some(row=>row.length!==3||row.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v)))))
    throw new TypeError('expected a finite 3 by 3 grid of 2D vectors');
}
/** Grid order is [pitch + 1][yaw + 1], normalized parameters in [-1,1]. */
export function residualAt(grid,u,v) {
  checkGrid(grid);
  if(!Number.isFinite(u)||!Number.isFinite(v)||Math.abs(u)>1||Math.abs(v)>1)
    throw new RangeError('pose is outside the authored residual domain');
  const x=u<0?0:1,y=v<0?0:1,s=u+1-x,t=v+1-y;
  return [0,1].map(k=>(1-s)*(1-t)*grid[y][x][k]+s*(1-t)*grid[y][x+1][k]+(1-s)*t*grid[y+1][x][k]+s*t*grid[y+1][x+1][k]);
}
/** One field per vertex (or sampled spatial control). Apply residual before rigid roll. */
export function deform(point,head,[yaw,pitch,roll],residual,extent,liftPoint=lift) {
  if(!(extent>0))throw new RangeError('extent must be positive');
  const q=rotate(liftPoint(point,head),[yaw,pitch,0]);
  const d=residualAt(residual,yaw/extent,pitch/extent);
  return project([q[0]+d[0],q[1]+d[1],q[2]],head,[0,0,roll]);
}
/** Targets must be original authored XY keyforms in the same artwork space as point. */
export function bakeResidual(point,head,targets,extent,liftPoint=lift) {
  checkGrid(targets);
  if(!(extent>0))throw new RangeError('extent must be positive');
  return targets.map((row,y)=>row.map((target,x)=>{
    const base=project(liftPoint(point,head),head,[(x-1)*extent,(y-1)*extent,0]);
    return target.map((v,i)=>v-base[i]);
  }));
}

/** Authored attachment weight: 0 at shoulder, 1 at jaw; optional smooth weights between. */
export function attachNeck(point,attachmentDelta,weight) {
  if(!Number.isFinite(weight)||weight<0||weight>1)throw new RangeError('neck weight must be in [0,1]');
  return point.map((v,i)=>v+weight*attachmentDelta[i]);
}

/** Alternative depth surface: independent horizontal, vertical and depth semiaxes. */
export function liftEllipsoid([x,y],{center:[cx,cy],radii:[rx,ry,rz],depthSign=1,relief=0}) {
  if(![rx,ry,rz].every(r=>Number.isFinite(r)&&r>0))throw new RangeError('semiaxes must be positive');
  const dx=x-cx,dy=y-cy;
  return [dx,dy,depthSign*rz*Math.sqrt(Math.max(0,1-(dx/rx)**2-(dy/ry)**2))+relief];
}
