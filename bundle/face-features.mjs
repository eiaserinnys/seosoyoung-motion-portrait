import {FEATURE_FIT} from './feature-data.mjs';
import {frameValue} from './pose-grid.mjs';
import {faceSolveAnchorWorld,faceSolveWorld,turnWeights45} from './face-solve.mjs';
import {FACE_SOLVE} from './face-solve-data.mjs';
import {correction,artWorld} from './geometry.mjs';
import {registeredFeatureWorld} from './face-feature-prior.mjs';
import {smoothstep} from './face-profile.mjs';
import {EYE_SAM3} from './eye-sam3-data.mjs';
import {eyeDeformation,achievedAngles} from './eye-sam3-model.mjs';
export {registeredFeatureWorld} from './face-feature-prior.mjs';
export function sam3EyeState(side,p){
 return frameValue(p,'sam3-eye:'+side,()=>{
  const rest=EYE_SAM3[side].rest,unrolled={...p,ParamAngleZ:0};
  const pin=FEATURE_FIT.front[side],a=faceSolveAnchorWorld(side,unrolled);
  const origin=faceSolveWorld(pin,unrolled),center=faceSolveWorld(rest.cO,unrolled);
  const cO=[a[0]+center[0]-origin[0],a[1]+center[1]-origin[1]];
  const gain=correction(p,'CorrTurn'),d=achievedAngles(EYE_SAM3,p.ParamAngleX*gain,p.ParamAngleY*gain);
  const model=eyeDeformation(EYE_SAM3,side,d,cO,[p.ParamEyeBallX,p.ParamEyeBallY]);
  return {...model,achieved:d,depth:center[2]};
 });
}
export function featureWorld(id,point,p){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),angle=Math.abs(yaw);
 if(angle<67&&!id.endsWith('brow')&&(id.startsWith('eye-')||id==='closed-eyes-v1')){
  const S=sam3FeatureWorld(id,point,p);
  if(angle<52)return S;
  const L=legacyFeatureWorld(id,point,p),s=smoothstep(52,67,angle);
  return S.map((value,i)=>value+(L[i]-value)*s);
 }
 return legacyFeatureWorld(id,point,p);
}
export function sam3FeatureWorld(id,point,p){
  const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
  const model=sam3EyeState(side,p);
  const xy=id.endsWith('highlight')?model.highlight(point):/-(iris|pupil)$/.test(id)?model.iris(point):
   model.lid(point,side==='left'?p.ParamEyeLOpen:p.ParamEyeROpen,id==='closed-eyes-v1');
  return artWorld([xy[0]-502,xy[1]-326,model.depth],{...p,ParamAngleX:0,ParamAngleY:0});
}
export function legacyFeatureWorld(id,point,p){
 const q=registeredFeatureWorld(id,point,p);
 // Neutral pitch retains the authored eyelid, gaze, brow and mouth evaluator.
 const yaw=p.ParamAngleX*correction(p,'CorrTurn');
 if(!p.ParamAngleY&&Math.abs(yaw)<=45)return q;
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const key=id.endsWith('brow')?(side==='left'?'browLeft':'browRight'):
  id.startsWith('eye-')||id==='closed-eyes-v1'?side:id==='nose-detail'?'nose':'mouth';
 const canonicalId={left:'eye-left-sclera',right:'eye-right-sclera',browLeft:'eye-left-brow',
  browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'}[key];
 if(Math.abs(yaw)<=45){
  const shift=frameValue(p,'solved-feature:'+key,()=>{
   const anchor=FEATURE_FIT.front[key];
   const neutral={...p,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,
    ParamBrowLY:0,ParamBrowRY:0,ParamMouthOpenY:0,ParamMouthForm:0};
   const origin=registeredFeatureWorld(canonicalId,anchor,neutral),target=faceSolveAnchorWorld(key,p);
   return [target[0]-origin[0],target[1]-origin[1]];
  });
  return [q[0]+shift[0],q[1]+shift[1],q[2]];
 }
 const {origin,target,scale}=frameValue(p,'solved-feature:'+key+':'+id,()=>{
  const anchor=FEATURE_FIT.front[key];
  const neutral={...p,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,
   ParamBrowLY:0,ParamBrowRY:0,ParamMouthOpenY:0,ParamMouthForm:0};
  // Compress in the unrolled feature frame, then apply the shared neck roll.
  const unrolled={...neutral,ParamAngleZ:0};
  const origin=registeredFeatureWorld(canonicalId,anchor,unrolled),target=faceSolveAnchorWorld(key,{...p,ParamAngleZ:0});
  const family=FACE_SOLVE.family45,sign=yaw<0?-1:1,[a67,a90]=turnWeights45(yaw-sign*45);
  let scale=1;
  if(id.startsWith('eye-')||id==='closed-eyes-v1'){
   const s67=family.poses.find(pose=>pose.name===(sign<0?'left67':'right67')).eyeWidthScale[side];
   const s90=family.poses.find(pose=>pose.name===(sign<0?'left90':'right90')).eyeWidthScale[side];
   const near=side===(sign<0?'right':'left');
   scale=1+a67*((id.endsWith('brow')&&near?.70:s67)-1)+a90*((id.endsWith('brow')&&near?.48:s90)-1);
  }else if(id==='nose-detail'||id==='mouth-detail'){
   scale=1+a67*(.55-1)+a90*(.35-1);
  }
  return {origin,target,scale};
 });
 const local=registeredFeatureWorld(id,point,{...p,ParamAngleZ:0});
 let dx=(local[0]-origin[0])*scale,dy=local[1]-origin[1];
 if((id.startsWith('eye-')||id==='closed-eyes-v1')&&!id.endsWith('brow')&&side===(yaw<0?'right':'left')){
  const weight=smoothstep(65,80,Math.abs(yaw)),front=FEATURE_FIT.front[side];
  const anterior=Math.max(-1,Math.min(1,(point[0]-front[0])/45*Math.sign(yaw)));
  // A profile eye is a wedge: its nasal corner contracts into the socket,
  // while the posterior upper arch retains height. All blink/gaze layers
  // use one field, including the closed lid, so no static eye is baked in.
  dx*=1-weight*.28*Math.max(0,anterior);
  dy*=1-weight*.42*Math.max(0,anterior);
 }
 return artWorld([target[0]+dx-502,target[1]+dy-326,local[2]],
  {...p,ParamAngleX:0,ParamAngleY:0});
}
