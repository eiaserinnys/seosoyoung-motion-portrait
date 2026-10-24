import {HEAD_CENTER,artWorld,eyeballVertex,clamp,radians,correction} from './geometry.mjs';
import {FEATURE_FIT} from './feature-data.mjs';
const landmarks=FEATURE_FIT.landmarks;
const interpolate=(neg,pos,theta)=>(pos-neg)*Math.sin(theta)/(2*Math.sin(Math.PI/4))+(pos+neg)*(1-Math.cos(theta))/(2*(1-Math.cos(Math.PI/4)));
export function featureWorld(id,point,p){
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const eye=id.startsWith('eye-')||id==='closed-eyes-v1',brow=id.endsWith('brow');
 const key=id.endsWith('brow')?(side==='left'?'browLeft':'browRight'):eye?side:id==='nose-detail'?'nose':'mouth';
 const m=landmarks[key];
 const theta=radians(clamp(p.ParamAngleX*correction(p,'CorrTurn'),-45,45)),ref=Math.PI/4;
 const gain=correction(p,'CorrTurn'),phi=radians(clamp((p.ParamAngleY??0)*gain,-30,30));
 const sPhi=clamp(Math.sin(phi)/Math.max(1e-4,Math.sin(radians(30*gain))),-1.2,1.2);
 const dx=eye?86.2*(side==='left'?-1:1):0,z=(m.pos[0]-m.neg[0])/(2*Math.sin(ref));
 const extra=eye?0:correction(p,id==='nose-detail'?'CorrNose':'CorrMouth')-(id==='nose-detail'?10:4);
 const cx=m.front[0]+interpolate(m.neg[0]-m.front[0],m.pos[0]-m.front[0],theta)+extra*Math.sin(theta);
 const cy=m.front[1]+interpolate(m.neg[1]-m.front[1],m.pos[1]-m.front[1],theta);
 let [x,y]=point,zLocal=0;
 if(eye){
  const [ex0,ey0]=m.front,xi=clamp((x-ex0)/48,-1,1),eta=clamp((ey0-y)/38,-1,1);
  if(/-(iris|pupil|highlight)$/.test(id)){
   const rest={...p,ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0};
   const q=eyeballVertex(point,side,id.split('-').at(-1),rest);x=q[0];y=q[1];
   const u0=x-ex0,v0=y-ey0,r2=(u0*u0+v0*v0)/(42*42);
   const zc=(id.endsWith('highlight')?13:10)*Math.sqrt(Math.max(.15,1-Math.min(.85,r2)));
   const yawE=theta*.35,pitchE=phi*.85;
   const u1=u0*Math.cos(yawE)+zc*Math.sin(yawE),z1=-u0*Math.sin(yawE)+zc*Math.cos(yawE);
   x=ex0+u1;y=ey0+v0*Math.cos(pitchE)-z1*Math.sin(pitchE);
   zLocal=z1+14*(1-xi*xi);
  }else if(id.endsWith('brow'))y-=(side==='left'?p.ParamBrowLY:p.ParamBrowRY)*12;
  else if(id!=='closed-eyes-v1')y=m.front[1]+(y-m.front[1])*(side==='left'?p.ParamEyeLOpen:p.ParamEyeROpen);
  if(!/-(iris|pupil|highlight)$/.test(id)){
   const vScale=brow?1:1-.14*sPhi*sPhi-.08*Math.max(0,-sPhi);
   y=ey0+(y-ey0)*vScale;
   zLocal=(brow?14:18)*(1-xi*xi)+(brow?8:14*eta);
  }
  const ratios=landmarks[side].width.map(v=>v<1?1+(v-1)*correction(p,'CorrFarEye')/.18:v);
  x=m.front[0]+(x-m.front[0])*(1+interpolate(ratios[0]-1,ratios[1]-1,theta));
 }else if(id.startsWith('mouth')){
  x=m.front[0]+(x-m.front[0])*(1+p.ParamMouthForm*.12);
  if(id!=='mouth-detail')y=m.front[1]+(y-m.front[1])*(1+.35*p.ParamMouthOpenY);
 }
 const dxVert=dx+(eye?(x-m.front[0])*.85:0);
 return artWorld([cx-HEAD_CENTER[0]+x-m.front[0],cy-HEAD_CENTER[1]+y-m.front[1],-dxVert*Math.sin(theta)+(z+extra+zLocal)*Math.cos(theta)+3],{...p,ParamAngleX:0});
}
