import {HEAD_CENTER,artWorld,eyeballVertex,clamp,radians,correction} from './geometry.mjs';
import {FEATURE_FIT} from './feature-data.mjs';
import {gridValue,frameValue} from './pose-grid.mjs';
const landmarks=FEATURE_FIT.landmarks;
const interpolate=(neg,pos,theta)=>(pos-neg)*Math.sin(theta)/(2*Math.sin(Math.PI/4))+(pos+neg)*(1-Math.cos(theta))/(2*(1-Math.cos(Math.PI/4)));
function legacyFeatureWorld(id,point,p,head=p){
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
   const [ex0,ey0]=landmarks[side].front,u0=x-ex0,v0=y-ey0,r2=(u0*u0+v0*v0)/(42*42);
   const zc=(id.endsWith('highlight')?13:10)*Math.sqrt(Math.max(.15,1-Math.min(.85,r2)));
   const yawE=radians(clamp(head.ParamAngleX*correction(head,'CorrTurn'),-45,45))*.28;
   const pitchE=radians(clamp(head.ParamAngleY*correction(head,'CorrTurn'),-30,30))*.65;
   const z1=-u0*Math.sin(yawE)+zc*Math.cos(yawE);
   x=ex0+u0*Math.cos(yawE)+zc*Math.sin(yawE);
   y=ey0+v0*Math.cos(pitchE)-z1*Math.sin(pitchE);
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
 const pitch=clamp(p.ParamAngleY*correction(p,'CorrTurn'),-30,30);
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const eye=id.startsWith('eye-')||id==='closed-eyes-v1',brow=id.endsWith('brow');
 const key=brow?(side==='left'?'browLeft':'browRight'):eye?side:id==='nose-detail'?'nose':'mouth';
 const anchor=FEATURE_FIT.front[key],yaw=p.ParamAngleX*correction(p,'CorrTurn');
 const rest={...p,ParamAngleY:0,ParamAngleZ:0};
 const neutral={...rest,ParamAngleX:0,ParamAngleY:0,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamBrowLY:0,ParamBrowRY:0,ParamMouthForm:0,ParamMouthOpenY:0};
 const q=legacyFeatureWorld(id,point,rest,p);
 const {center,origin,target,width,height}=frameValue(p,'feature:'+id+':'+side,()=>{
  const center=legacyFeatureWorld(id,anchor,neutral);
  // Remove yaw registration separately from corneal motion. Using a rotated
  // iris as the origin would cancel its head-driven roll inside the socket.
  const origin=legacyFeatureWorld(id,anchor,{...neutral,ParamAngleX:rest.ParamAngleX},neutral);
  const target=[0,1].map(i=>gridValue(FEATURE_FIT.poses,yaw,pitch,pose=>{
   if(pose.pitch)return pose.centers[key][i];
   return legacyFeatureWorld(id,anchor,{...neutral,CorrTurn:1,ParamAngleX:pose.yaw},neutral)[i];
  }));
  const scale=i=>eye?gridValue(FEATURE_FIT.poses,yaw,pitch,pose=>pose.scales[side][i]):1;
  const width0=eye?gridValue(FEATURE_FIT.poses,yaw,0,pose=>pose.scales[side][0]):1;
  return {center,origin,target,width:scale(0)/width0,height:brow?1:scale(1)};
 });
 q[0]+=center[0]-origin[0];q[1]+=center[1]-origin[1];
 const sPhi=clamp(pitch/30,-1.2,1.2),theta=radians(clamp(yaw,-45,45));
 const xi=clamp((point[0]-anchor[0])/46,-1,1),eta=clamp((anchor[1]-point[1])/36,-1,1);
 const vScale=eye&&!brow?1-.08*sPhi*sPhi-.04*Math.max(0,-sPhi):1;
 const lidWeight=sPhi>=0?.55+.45*eta:.35+.65*eta;
 const archY=eye&&!brow&&!/-(iris|pupil|highlight)$/.test(id)?-5*sPhi*(1-xi*xi)*lidWeight:0;
 const dxLocal=(q[0]-center[0])*width;
 const diagTiltY=eye?-dxLocal*Math.sin(theta)*Math.sin(radians(pitch))*.32:0;
 const featurePitchShiftY=id==='nose-detail'?-6*Math.max(0,sPhi)+4*Math.max(0,-sPhi):id.startsWith('mouth')?-12*Math.max(0,sPhi)+6*Math.max(0,-sPhi):0;
 const x=target[0]+dxLocal;
 const y=target[1]+featurePitchShiftY+(q[1]-center[1])*height*vScale+archY+diagTiltY;
 const depth=legacyFeatureWorld(id,point,{...p,ParamAngleZ:0})[2];
 return artWorld([x-HEAD_CENTER[0],y-HEAD_CENTER[1],depth],{...p,ParamAngleX:0,ParamAngleY:0});
}
