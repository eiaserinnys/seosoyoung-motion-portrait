import {HEAD_CENTER,artWorld,eyeballVertex,clamp,radians,correction} from './geometry.mjs';
import {FEATURE_FIT} from './feature-data.mjs';
import {gridValue,frameValue} from './pose-grid.mjs';
const landmarks=FEATURE_FIT.landmarks;
const interpolate=(neg,pos,theta)=>(pos-neg)*Math.sin(theta)/(2*Math.sin(Math.PI/4))+(pos+neg)*(1-Math.cos(theta))/(2*(1-Math.cos(Math.PI/4)));
function legacyFeatureWorld(id,point,p){
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const eye=id.startsWith('eye-')||id==='closed-eyes-v1';
 const key=id.endsWith('brow')?(side==='left'?'browLeft':'browRight'):eye?side:id==='nose-detail'?'nose':'mouth';
 const m=landmarks[key];
 const theta=radians(clamp(p.ParamAngleX*correction(p,'CorrTurn'),-45,45)),ref=Math.PI/4;
 const dx=eye?86.2*(side==='left'?-1:1):0,z=(m.pos[0]-m.neg[0])/(2*Math.sin(ref));
 const extra=eye?0:correction(p,id==='nose-detail'?'CorrNose':'CorrMouth')-(id==='nose-detail'?10:4);
 const cx=m.front[0]+interpolate(m.neg[0]-m.front[0],m.pos[0]-m.front[0],theta)+extra*Math.sin(theta);
 const cy=m.front[1]+interpolate(m.neg[1]-m.front[1],m.pos[1]-m.front[1],theta);
 let [x,y]=point;
 if(eye){
  if(/-(iris|pupil|highlight)$/.test(id)){
   const rest={...p,ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0};
   const q=eyeballVertex(point,side,id.split('-').at(-1),rest);x=q[0];y=q[1];
  }else if(id.endsWith('brow'))y-=(side==='left'?p.ParamBrowLY:p.ParamBrowRY)*12;
  else if(id!=='closed-eyes-v1')y=m.front[1]+(y-m.front[1])*(side==='left'?p.ParamEyeLOpen:p.ParamEyeROpen);
  const ratios=landmarks[side].width.map(v=>v<1?1+(v-1)*correction(p,'CorrFarEye')/.18:v);
  x=m.front[0]+(x-m.front[0])*(1+interpolate(ratios[0]-1,ratios[1]-1,theta));
 }else if(id.startsWith('mouth')){
  x=m.front[0]+(x-m.front[0])*(1+p.ParamMouthForm*.12);
  if(id!=='mouth-detail')y=m.front[1]+(y-m.front[1])*(1+.35*p.ParamMouthOpenY);
 }
 return artWorld([cx-HEAD_CENTER[0]+x-m.front[0],cy-HEAD_CENTER[1]+y-m.front[1],-dx*Math.sin(theta)+(z+extra)*Math.cos(theta)+3],{...p,ParamAngleX:0});
}
export function featureWorld(id,point,p){
 const pitch=p.ParamAngleY*correction(p,'CorrTurn');
 if(!pitch)return legacyFeatureWorld(id,point,p);
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const eye=id.startsWith('eye-')||id==='closed-eyes-v1',brow=id.endsWith('brow');
 const key=brow?(side==='left'?'browLeft':'browRight'):eye?side:id==='nose-detail'?'nose':'mouth';
 const anchor=FEATURE_FIT.front[key],yaw=p.ParamAngleX*correction(p,'CorrTurn');
 const rest={...p,ParamAngleY:0,ParamAngleZ:0};
 const neutral={...rest,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamBrowLY:0,ParamBrowRY:0,ParamMouthForm:0,ParamMouthOpenY:0};
 const q=legacyFeatureWorld(id,point,rest);
 const {center,target,width,height}=frameValue(p,'feature:'+id+':'+side,()=>{
  const center=legacyFeatureWorld(id,anchor,neutral);
  const target=[0,1].map(i=>gridValue(FEATURE_FIT.poses,yaw,pitch,pose=>{
   if(pose.pitch)return pose.centers[key][i];
   return legacyFeatureWorld(id,anchor,{...neutral,CorrTurn:1,ParamAngleX:pose.yaw})[i];
  }));
  const scale=i=>eye?gridValue(FEATURE_FIT.poses,yaw,pitch,pose=>pose.scales[side][i]):1;
  const width0=eye?gridValue(FEATURE_FIT.poses,yaw,0,pose=>pose.scales[side][0]):1;
  return {center,target,width:scale(0)/width0,height:brow?1:scale(1)};
 });
 const x=target[0]+(q[0]-center[0])*width;
 const y=target[1]+(q[1]-center[1])*height;
 const depth=legacyFeatureWorld(id,point,{...p,ParamAngleZ:0})[2];
 return artWorld([x-HEAD_CENTER[0],y-HEAD_CENTER[1],depth],{...p,ParamAngleX:0,ParamAngleY:0});
}
