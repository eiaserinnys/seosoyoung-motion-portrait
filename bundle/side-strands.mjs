import {outlineRow} from './face-outline.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {artWorld,HEAD_CENTER,radians,clamp,correction} from './geometry.mjs';
// Canvas-space alpha spans: front restored side-hair, near-view SAM crown's
// connected lock between cheek and ear. The broad posterior hair is supplied
// by the temporal surface, not by stretching this front lock across the ear.
export const STRANDS={
 left:[[407,224,353,357,438],[430,213,353,386,437],[460,227,363,399,438],[490,252,372,406,428],[515,322,351,414,431]],
 right:[[407,657,760,598,634],[430,655,777,598,634],[460,648,767,604,625],[490,631,761,602,618],[515,653,677,597,643]],
};
export function strandDepth([x,y],side){
 const rows=STRANDS[side];let a=rows[0],b=a;
 for(const row of rows){if(row[0]<=y)a=row;if(row[0]>=y){b=row;break;}b=row;}
 const t=b[0]===a[0]?0:clamp((y-a[0])/(b[0]-a[0]),0,1);
 const v=a.map((value,i)=>value+(b[i]-value)*t),u=(x-v[1])/(v[2]-v[1]);
 const target=v[3]+u*(v[4]-v[3]),angle=radians(side==='left'?45:-45);
 return (target-HEAD_CENTER[0]-(x-HEAD_CENTER[0])*Math.cos(angle))/Math.sin(angle);
}
export function strandWorld(point,side,p){
 const rest={...p,ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0};
 // Constrained fit: a lock crossing the ear in the front drawing must stay
 // anterior to it in rest space. This small depth separation rotates with it.
 const z=Math.max(strandDepth(point,side),earWorld(point,side,rest)[2]+4);
 // Intermediate keyform bows the front lock inward along its 3D orbit,
 // exposing the attached ear before the 45-degree keyform. End positions and
 // endpoint velocities are unchanged; d(theta')/d(theta) stays positive.
 const theta=p.ParamAngleX*correction(p,'CorrTurn'),near=theta*(side==='left'?1:-1)>0;
 const advance=near&&Math.abs(theta)<45?12*Math.sin(4*radians(theta))**2:0;
 const pose={...p,ParamAngleX:(theta+Math.sign(theta)*advance)/correction(p,'CorrTurn')};
 const local=[point[0]-HEAD_CENTER[0],point[1]-HEAD_CENTER[1],z];
 const world=artWorld(local,pose);
 if(!near&&theta){
  const farRows=side==='left'?[[407,231,307],[430,229,310],[460,237,318],[490,255,325],[515,331,333]]:[[407,712,764],[430,714,766],[460,706,767],[490,699,762],[515,711,716]];
  const rows=STRANDS[side],y=point[1];let index=0;
  while(index<rows.length-2&&y>rows[index+1][0])index++;
  const a=rows[index],b=rows[index+1],t=clamp((y-a[0])/(b[0]-a[0]),0,1);
  const lerp=(v,w)=>v+(w-v)*t;
  const lo=lerp(a[1],b[1]),hi=lerp(a[2],b[2]),u=(point[0]-lo)/(hi-lo);
  let left=lerp(farRows[index][1],farRows[index+1][1]),right=lerp(farRows[index][2],farRows[index+1][2]);
  const fold=clamp((y-515)/35,0,1),edge=outlineRow(y)[side==='left'?2:5];
  left+=(edge-(side==='right'?2:0)-left)*fold;right+=(edge+(side==='left'?2:0)-right)*fold;
  const target=left+u*(right-left),yaw=side==='left'?-45:45;
  const reference=artWorld(local,{...p,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:0,ParamAngleZ:0});
  const w=(1-Math.cos(radians(Math.min(45,Math.abs(theta)))))/(1-Math.cos(Math.PI/4));
  // Far-lock silhouette residual, measured independently instead of mirroring
  // the near view. Apply before roll/pitch by lifting the residual as a vector.
  const delta=artWorld([w*(target-reference[0]),0,0],{...p,ParamAngleX:0});
  const origin=artWorld([0,0,0],{...p,ParamAngleX:0});
  for(let i=0;i<3;i++)world[i]+=delta[i]-origin[i];
 }
 return world;
}
