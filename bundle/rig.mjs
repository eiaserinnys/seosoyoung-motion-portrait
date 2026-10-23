import {clamp,radians,eyeballVertex,correctedLift,artWorld,correction} from './geometry.mjs';
export {HEAD_CENTER,HEAD_RADIUS,LEFT_EYE_BALL,RIGHT_EYE_BALL,EYE_BALL_R,spherePoint,sphereWarp,eyeballAngles,headWorld,liftHead,eyeWorld,eyeCenter,rotateXYZ} from './geometry.mjs';
export function interpolate(keys,value) {
  if(value<=keys[0].value)return keys[0].output;
  for(let i=1;i<keys.length;i++)if(value<=keys[i].value){const a=keys[i-1],b=keys[i],t=(value-a.value)/(b.value-a.value);return a.output+(b.output-a.output)*t;}
  return keys.at(-1).output;
}
export function bindRig(rig) {
  const byId=new Map(rig.deformers.map(d=>[d.id,d]));
  for(const mesh of rig.meshes){
    mesh.deformer=byId.get(mesh.deformerId);mesh.chain=[];let node=mesh.deformer;
    while(node){mesh.chain.unshift(node);node=byId.get(node.parentId);}
  }
}
export function deform3D(mesh,point,parameters) {
  const p={...parameters};
  for(const node of mesh.chain||[])for(const [channel,keys] of Object.entries(node.keyforms))if(channel.startsWith('Param'))p[channel]=interpolate(keys,parameters[channel]);
  const id=mesh.id;let [x,y]=point;
  if(mesh.deformerId==='WarpBody') {
    const upper=clamp((1536-y)/960,0,1);
    return [x+p.ParamBodyAngleX*upper,y-p.ParamBreath*4*upper,-500];
  }
  if(id==='neck') {
    const capT=clamp((578-y)/42,0,1);
    const xIn=502+(x-502)*(1-.75*capT),yIn=y-16*capT,sampleY=Math.min(yIn,568);
    // Yaw is a rotation about the neck's vertical axis, never a lateral shear input.
    const head=artWorld(correctedLift([xIn,sampleY],-2,p),{...p,ParamAngleX:0});
    const headDx=head[0]-x,headDy=head[1]-sampleY+yIn-y;
    const u=clamp((680-y)/110,0,1),weight=u*u*(3-2*u),upper=clamp((1536-y)/960,0,1);
    return [x+headDx*weight+p.ParamBodyAngleX*upper*(1-weight),y+headDy*weight-p.ParamBreath*4*upper*(1-weight),-20];
  }
  if(id.startsWith('tail-')) {
    const left=id.includes('-left-'),root=mesh.deformer.pivot;
    const anchor=artWorld(correctedLift(root,0,p),p),dx=x-root[0],dy=y-root[1];
    const s=clamp((y-498)/635,0,1),sagW=s**.62;
    const outward=clamp((left?root[0]-x:x-root[0])/95,0,1);
    const weight=clamp(.42*outward*(1-.55*s)+sagW,0,1);
    const a=radians(p.ParamHairSide*(4.2*weight+3.4*s*s)-p.ParamAngleZ*.35*weight);
    const lift=Math.max(0,root[1]-anchor[1]);
    const sag=(lift*.72+Math.max(0,p.ParamHairBack)*16+Math.abs(p.ParamHairSide)*10)*weight;
    const inward=(left?1:-1)*(lift*.22+Math.abs(p.ParamHairSide)*6)*weight;
    return [anchor[0]+dx*Math.cos(a)-dy*Math.sin(a)+inward,anchor[1]+dx*Math.sin(a)+dy*Math.cos(a)+sag,anchor[2]-s*35];
  }
  if(id.startsWith('back-hair')) {
    x+=p.ParamHairBack*12*clamp((y-120)/500,0,1)**2;
    return artWorld(correctedLift([x,y],-2,p),p);
  }
  let relief=0;
  if(id==='hair-top')relief=2;
  else if(id.startsWith('side-hair'))relief=4;
  else if(id==='hairpin')relief=8;
  else if(id.startsWith('bangs'))relief=6;
  else if(id.startsWith('ear-'))relief=5;
  else if(id==='nose-detail')relief=correction(p,'CorrNose');
  else if(id.startsWith('eye-'))relief=id.endsWith('brow')?5:3;
  else if(id.startsWith('mouth'))relief=correction(p,'CorrMouth');
  else if(id==='closed-eyes-v1')relief=3;
  if(id.startsWith('eye-')||id==='closed-eyes-v1') {
    const left=id==='closed-eyes-v1'?x<502:id.includes('-left-');
    if(/-(iris|pupil|highlight)$/.test(id))return eyeballVertex([x,y],left?'left':'right',id.split('-').at(-1),p);
    if(id.endsWith('brow'))y-=(left?p.ParamBrowLY:p.ParamBrowRY)*12;
    else if(id!=='closed-eyes-v1')y=438+(y-438)*(left?p.ParamEyeLOpen:p.ParamEyeROpen);
    const far=left?p.ParamAngleX<0:p.ParamAngleX>0;
    if(far){const cx=left?406:597;x=cx+(x-cx)*(1-correction(p,'CorrFarEye')*Math.min(1,Math.abs(p.ParamAngleX)/50));}
  }
  if(id.startsWith('mouth')) {
    x=502+(x-502)*(1+p.ParamMouthForm*.12);
    if(id!=='mouth-detail')y=552+(y-552)*(1+.35*p.ParamMouthOpenY);
  }
  if(id.startsWith('side-hair'))x+=p.ParamHairSide*9*clamp((y-180)/400,0,1)**2;
  if(id.startsWith('bangs'))x+=p.ParamHairFront*9*clamp((y-100)/330,0,1)**2;
  return artWorld(correctedLift([x,y],relief,p),p);
}
export const deform=(mesh,point,p)=>deform3D(mesh,point,p).slice(0,2);
export function opacity(mesh,p,state) {
  const id=mesh.id;
  if(state.hidden.has(id))return 0;
  if(state.bangs&&id.startsWith('bangs'))return 0;
  if(state.outfit==='shirt'&&id==='sweater')return 0;
  if(state.outfit==='sweater'&&id==='upper-body-short-sleeve')return 0;
  if(id.startsWith('eye-')&&!id.endsWith('brow')&&(id.includes('-left-')?p.ParamEyeLOpen:p.ParamEyeROpen)<.22)return 0;
  if(id==='closed-eyes-v1'&&p.ParamEyeLOpen>=.22&&p.ParamEyeROpen>=.22)return 0;
  if(id==='mouth-detail')return p.ParamMouthOpenY<.12?1:0;
  if(id==='mouth-open-v1')return p.ParamMouthOpenY>=.12&&p.ParamMouthForm>=-.35?1:0;
  if(id==='mouth-round-v1')return p.ParamMouthOpenY>=.12&&p.ParamMouthForm<-.35?1:0;
  return 1;
}
export class Pendulum {
  constructor(){this.value=0;this.velocity=0;}
  step(target,dt){dt=Math.min(dt,.033);this.velocity+=(target-this.value)*28*dt;this.velocity*=Math.exp(-5*dt);this.value+=this.velocity*dt;return this.value;}
}
