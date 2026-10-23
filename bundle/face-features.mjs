import {HEAD_CENTER,artWorld,eyeballVertex,clamp,radians,correction} from './geometry.mjs';
const landmarks={
 left:{front:[415.7,432.2],neg:[362.4,429.3],pos:[514.2,430.7],width:[36/44,45/44]},
 right:{front:[588.1,430.8],neg:[525.1,429.2],pos:[670.1,428.9],width:[47/43.9,.72]},
 nose:{front:[508,497],neg:[406.9,497.1],pos:[629.9,490.5]},
 mouth:{front:[504.5,554.5],neg:[434.1,556.2],pos:[598.7,553.3]},
};
const interpolate=(neg,pos,theta)=>(pos-neg)*Math.sin(theta)/(2*Math.sin(Math.PI/4))+(pos+neg)*(1-Math.cos(theta))/(2*(1-Math.cos(Math.PI/4)));
export function featureWorld(id,point,p){
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const eye=id.startsWith('eye-')||id==='closed-eyes-v1';
 const m=landmarks[eye?side:id==='nose-detail'?'nose':'mouth'];
 const theta=radians(clamp(p.ParamAngleX*correction(p,'CorrTurn'),-45,45)),ref=Math.PI/4;
 const dx=m.front[0]-HEAD_CENTER[0],z=(m.pos[0]-m.neg[0])/(2*Math.sin(ref));
 const even=(m.pos[0]+m.neg[0])/2-HEAD_CENTER[0]-dx*Math.cos(ref);
 const extra=eye?0:correction(p,id==='nose-detail'?'CorrNose':'CorrMouth')-(id==='nose-detail'?10:4);
 const cx=extra*Math.sin(theta)+HEAD_CENTER[0]+dx*Math.cos(theta)+z*Math.sin(theta)+even*(1-Math.cos(theta))/(1-Math.cos(ref));
 const cy=m.front[1]+interpolate(m.neg[1]-m.front[1],m.pos[1]-m.front[1],theta);
 let [x,y]=point;
 if(eye){
  if(/-(iris|pupil|highlight)$/.test(id)){
   const rest={...p,ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0};
   const q=eyeballVertex(point,side,id.split('-').at(-1),rest);x=q[0];y=q[1];
  }else if(id.endsWith('brow'))y-=(side==='left'?p.ParamBrowLY:p.ParamBrowRY)*12;
  else if(id!=='closed-eyes-v1')y=m.front[1]+(y-m.front[1])*(side==='left'?p.ParamEyeLOpen:p.ParamEyeROpen);
  const ratios=m.width.map(v=>v<1?1+(v-1)*correction(p,'CorrFarEye')/.18:v);
  x=m.front[0]+(x-m.front[0])*(1+interpolate(ratios[0]-1,ratios[1]-1,theta));
 }else if(id.startsWith('mouth')){
  x=m.front[0]+(x-m.front[0])*(1+p.ParamMouthForm*.12);
  if(id!=='mouth-detail')y=m.front[1]+(y-m.front[1])*(1+.35*p.ParamMouthOpenY);
 }
 return artWorld([cx-HEAD_CENTER[0]+x-m.front[0],cy-HEAD_CENTER[1]+y-m.front[1],-dx*Math.sin(theta)+(z+extra)*Math.cos(theta)+3],{...p,ParamAngleX:0});
}
