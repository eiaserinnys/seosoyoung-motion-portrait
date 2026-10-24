// Independent closed-form check against the public deformed surface at yaw0.
import {HEAD_CENTER,HEAD_RADIUS,correctedLift,artWorld,correction} from '../bundle/geometry.mjs';
import {outlineWorld,outlineRow} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
export function sphericalFaceError(parameters){
 const p={...parameters,ParamAngleX:0,[FRAME_CACHE]:new Map()};
 const p0={...p,ParamAngleY:0,[FRAME_CACHE]:new Map()};
 const base=(pt,q)=>{
  const [l,r]=outlineRow(pt[1]),off=(l+r)/2-HEAD_CENTER[0];
  return artWorld(correctedLift([pt[0]-off,pt[1]],0,q),q)[1];
 };
 const edgeAt=y=>outlineRow(y).slice(0,2).map(x=>outlineWorld([x,y],p)[1]-base([x,y],p));
 const lerp=(a,b,t)=>a+(b-a)*t;
 const bands=[['nose','nose-detail',497],['mouth','mouth-detail',554.5]].map(([key,id,y])=>{
  const pt=FEATURE_FIT.front[key],[l,r]=outlineRow(pt[1]),u=(pt[0]-l)/(r-l),edge=edgeAt(pt[1]);
  const value=featureWorld(id,pt,p)[1]-base(pt,p)-featureWorld(id,pt,p0)[1]+base(pt,p0)-lerp(...edge,u);
  return {y,lambda:Math.asin(2*u-1),value};
 });
 const interior=(band,u)=>{
  const lambda=Math.asin(2*u-1),half=Math.PI/2;
  const angle=lambda<=band.lambda?(lambda+half)/(band.lambda+half):(half-lambda)/(half-band.lambda);
  return band.value*Math.sin(angle*half);
 };
 const lat=y=>Math.asin(Math.max(-1,Math.min(1,(y-HEAD_CENTER[1])/(HEAD_RADIUS*correction(p,'CorrVertical')))));
 let maxError=0;
 for(const y of [497,525,554.5])for(const u of [0,.125,.25,.375,.5,.625,.75,.875,1]){
  const [l,r]=outlineRow(y),pt=[l+(r-l)*u,y],edge=edgeAt(y);
  const t=(1-Math.cos((lat(y)-lat(497))/(lat(554.5)-lat(497))*Math.PI))/2;
  const expected=base(pt,p)+lerp(...edge,u)+lerp(interior(bands[0],u),interior(bands[1],u),t);
  maxError=Math.max(maxError,Math.abs(outlineWorld(pt,p)[1]-expected));
 }
 return maxError;
}
