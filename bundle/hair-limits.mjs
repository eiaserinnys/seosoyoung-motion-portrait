import {HAIR_LIMITS as L} from './hair-limits-data.mjs';
import {sphereEase} from './hair-surface.mjs';
import {clamp,correction} from './geometry.mjs';

function interval(nodes,value){
 let lo=0,hi=nodes.length-1;
 while(hi-lo>1){const mid=(lo+hi)>>1;if(value<nodes[mid])hi=mid;else lo=mid;}
 return [lo,clamp((value-nodes[lo])/(nodes[hi]-nodes[lo]),0,1)];
}
// Bilinear vertex scales and spherical pose weights are continuous. Sparse
// entries are exactly neutral outside the triangles touched by the fit.
function field(stage,[x,y],p){
 const [i,u]=interval(L.x,x),[j,v]=interval(L.y,y),width=L.x.length;
 const value=pose=>{
  const values=pose[stage];let delta=0;
  for(const [a,wa] of [[i,1-u],[i+1,u]])for(const [b,wb] of [[j,1-v],[j+1,v]])
   delta+=wa*wb*((values[b*width+a]??L.neutral)-L.neutral);
  return delta;
 };
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
 const row=height=>{
  const poses=L.poses.filter(pose=>pose.pitch===height);
  if(!poses.length)return 0;
  let i=0;while(i<poses.length-2&&yaw>poses[i+1].yaw)i++;
  const a=poses[i],b=poses[Math.min(i+1,poses.length-1)];
  const w=a===b?0:sphereEase((yaw-a.yaw)/(b.yaw-a.yaw));
  return value(a)*(1-w)+value(b)*w;
 };
 const weight=sphereEase(Math.abs(pitch)/30);
 return L.neutral+row(0)*(1-weight)+row(pitch<0?-30:30)*weight;
}
export const hairStageScale=(stage,point,p)=>field(stage,point,p);
