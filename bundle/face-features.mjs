import {FEATURE_FIT} from './feature-data.mjs';
import {frameValue} from './pose-grid.mjs';
import {faceSolveAnchorWorld,turnWeights45} from './face-solve.mjs';
import {FACE_SOLVE} from './face-solve-data.mjs';
import {correction,artWorld} from './geometry.mjs';
import {registeredFeatureWorld} from './face-feature-prior.mjs';
export {registeredFeatureWorld} from './face-feature-prior.mjs';
export function featureWorld(id,point,p){
 const q=registeredFeatureWorld(id,point,p);
 // Neutral pitch retains the authored eyelid, gaze, brow and mouth evaluator.
 const yaw=p.ParamAngleX*correction(p,'CorrTurn');
 if(!p.ParamAngleY&&Math.abs(yaw)<=45)return q;
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const key=id.endsWith('brow')?(side==='left'?'browLeft':'browRight'):
  id.startsWith('eye-')||id==='closed-eyes-v1'?side:id==='nose-detail'?'nose':'mouth';
 if(Math.abs(yaw)<=45){
  const shift=frameValue(p,'solved-feature:'+key,()=>{
   const anchor=FEATURE_FIT.front[key];
   const neutral={...p,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,
    ParamBrowLY:0,ParamBrowRY:0,ParamMouthOpenY:0,ParamMouthForm:0};
   const origin=registeredFeatureWorld(id,anchor,neutral),target=faceSolveAnchorWorld(key,p);
   return [target[0]-origin[0],target[1]-origin[1]];
  });
  return [q[0]+shift[0],q[1]+shift[1],q[2]];
 }
 const {origin,target,scale}=frameValue(p,'solved-feature:'+key,()=>{
  const anchor=FEATURE_FIT.front[key];
  const neutral={...p,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,
   ParamBrowLY:0,ParamBrowRY:0,ParamMouthOpenY:0,ParamMouthForm:0};
  // Compress in the unrolled feature frame, then apply the shared neck roll.
  const unrolled={...neutral,ParamAngleZ:0};
  const origin=registeredFeatureWorld(id,anchor,unrolled),target=faceSolveAnchorWorld(key,{...p,ParamAngleZ:0});
  let scale=1;
  if(id.startsWith('eye-')||id==='closed-eyes-v1'){
   const family=FACE_SOLVE.family45,sign=yaw<0?-1:1,[a67,a90]=turnWeights45(yaw-sign*45);
   const s67=family.poses.find(pose=>pose.name===(sign<0?'left67':'right67')).eyeWidthScale[side];
   const s90=family.poses.find(pose=>pose.name===(sign<0?'left90':'right90')).eyeWidthScale[side];
   scale=1+a67*(s67-1)+a90*(s90-1);
  }
  return {origin,target,scale};
 });
 const local=registeredFeatureWorld(id,point,{...p,ParamAngleZ:0});
 return artWorld([target[0]+(local[0]-origin[0])*scale-502,target[1]+local[1]-origin[1]-326,local[2]],
  {...p,ParamAngleX:0,ParamAngleY:0});
}
