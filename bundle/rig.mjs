import {featureWorld} from './face-features.mjs';
import {BEADS,beadWorld} from './stereo-attachments.mjs';
import {outlineWorld,applyOutline} from './face-outline.mjs';
import {clamp,radians,eyeballVertex,correctedLift,artWorld,correction} from './geometry.mjs';
import {canopyWorld,sideWorld,backWorld,craniumWeight,craniumWorld} from './hair-shell.mjs';
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
  const world=deformLocal(mesh,point,p);
  if(mesh.deformerId==='WarpBody'||mesh.id==='neck'||(!p.ParamBodyAngleX&&!p.ParamBreath))return world;
  return [world[0]+(p.ParamBodyAngleX||0),world[1]-(p.ParamBreath||0)*4,world[2]];
}
function deformLocal(mesh,point,p) {
  const id=mesh.id;let [x,y]=point;
  if(id==='hair-top'||id.startsWith('bangs')||id.startsWith('side-hair')||id==='hairpin')return canopyWorld(point,p);
  if(mesh.deformerId==='WarpBody') {
    const upper=clamp((1536-y)/960,0,1);
    return [x+p.ParamBodyAngleX*upper,y-p.ParamBreath*4*upper,-500];
  }
  if(id==='neck') {
    const capT=clamp((578-y)/42,0,1);
    const xIn=502+(x-502)*(1-.75*capT),yIn=y-16*capT,sampleY=Math.min(yIn,568);
    // Yaw is a rotation about the neck's vertical axis, never a lateral shear input.
    const head=artWorld(correctedLift([xIn,sampleY],-2,p),{...p,ParamAngleX:0,ParamAngleY:0});
    const headDx=head[0]-x,headDy=head[1]-sampleY+yIn-y;
    const u=clamp((680-y)/110,0,1),weight=u*u*(3-2*u),upper=clamp((1536-y)/960,0,1);
    const result=[x+headDx*weight+p.ParamBodyAngleX*upper,y+headDy*weight-p.ParamBreath*4*upper,-20];
    if(p.ParamAngleY){
      const chinDy=outlineWorld([502,616],{...p,ParamAngleZ:0})[1]-616;
      const t=1-clamp((y-585)/95,0,1),attach=t*t*(3-2*t),roll=radians(p.ParamAngleZ);
      result[0]-=chinDy*attach*Math.sin(roll);result[1]+=chinDy*attach*Math.cos(roll);
    }
    return result;
  }
  const feature=id.startsWith('eye-')||id==='closed-eyes-v1'||id==='nose-detail'||id.startsWith('mouth');
  if(feature&&correction(p,'CorrOutline')>0){
    const weight=correction(p,'CorrOutline'),target=featureWorld(id,[x,y],p);
    if(weight===1)return target;
    const original=deformLocal(mesh,point,{...p,CorrOutline:0});
    return original.map((v,i)=>v+(target[i]-v)*weight);
  }
  if(id.startsWith('tail-')) {
    const left=id.includes('-left-'),root=BEADS[left?'left':'right'].front;
    const anchor=beadWorld(left?'left':'right',p),dx=x-root[0],dy=y-root[1];
    const s=clamp((y-root[1])/635,0,1),sagW=s**.62;
    const outward=clamp((left?root[0]-x:x-root[0])/95,0,1);
    const weight=clamp(.42*outward*(1-.55*s)+sagW,0,1);
    const a=radians(p.ParamHairSide*(4.2*weight+3.4*s*s)-p.ParamAngleZ*.35*weight);
    const lift=Math.max(0,root[1]-anchor[1]);
    const sag=(lift*.72+Math.max(0,p.ParamHairBack)*16+Math.abs(p.ParamHairSide)*10)*weight;
    const inward=(left?1:-1)*(lift*.22+Math.abs(p.ParamHairSide)*6)*weight;
    return [anchor[0]+dx*Math.cos(a)-dy*Math.sin(a)+inward,anchor[1]+dx*Math.sin(a)+dy*Math.cos(a)+sag,anchor[2]-s*35];
  }
  if(id.startsWith('side-cranium'))return craniumWorld(point,id.endsWith('left')?'left':'right',p);
  if(id.startsWith('back-hair')) {
    x+=p.ParamHairBack*12*clamp((y-120)/500,0,1)**2;
    return backWorld([x,y],p);
  }
  let relief=0;
  if(id.startsWith('ear-'))relief=5;
  else if(id==='nose-detail')relief=correction(p,'CorrNose');
  else if(id.startsWith('eye-'))relief=id.endsWith('brow')?5:3;
  else if(id.startsWith('mouth'))relief=correction(p,'CorrMouth');
  else if(id==='closed-eyes-v1')relief=3;
  if(id.startsWith('eye-')||id==='closed-eyes-v1') {
    const left=id==='closed-eyes-v1'?x<502:id.includes('-left-');
    if(/-(iris|pupil|highlight)$/.test(id))return applyOutline([x,y],eyeballVertex([x,y],left?'left':'right',id.split('-').at(-1),p),p);
    if(id.endsWith('brow'))y-=(left?p.ParamBrowLY:p.ParamBrowRY)*12;
    else if(id!=='closed-eyes-v1')y=438+(y-438)*(left?p.ParamEyeLOpen:p.ParamEyeROpen);
    const far=left?p.ParamAngleX<0:p.ParamAngleX>0;
    if(far){const cx=left?406:597;x=cx+(x-cx)*(1-correction(p,'CorrFarEye')*Math.min(1,Math.abs(p.ParamAngleX)/50));}
  }
  if(id.startsWith('mouth')) {
    x=502+(x-502)*(1+p.ParamMouthForm*.12);
    if(id!=='mouth-detail')y=552+(y-552)*(1+.35*p.ParamMouthOpenY);
  }
  if(id.startsWith('ear-'))return sideWorld([x,y],id.endsWith('left')?'left':'right',p,true);
  return outlineWorld([x,y],p,relief);
}
export const deform=(mesh,point,p)=>deform3D(mesh,point,p).slice(0,2);
export function opacity(mesh,p,state) {
  const id=mesh.id;
  if(state.hidden.has(id))return 0;
  if(id.startsWith('side-cranium'))return craniumWeight(id.endsWith('left')?'left':'right',p);
  if(id.startsWith('ear-')){
    const far=p.ParamAngleX*(id.endsWith('left')?-1:1)*correction(p,'CorrTurn');
    const t=clamp(far/25,0,1);return 1-t*t*(3-2*t)*clamp(correction(p,'CorrHairOcclusion')/.8,0,1);
  }
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
