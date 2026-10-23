import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {NECK_PIVOT,CORRECTION_PRESETS,correctedLift,artWorld,eyeFrame,eyeCenter,eyeRootWorld,eyeWorld,eyeballVertex,headWorld,liftHead} from '../bundle/geometry.mjs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base=Object.fromEntries(rig.parameters.map(p=>[p.id,p.default]));
let checks=0;
const near=(a,b,message)=>{assert(Math.max(...a.map((v,i)=>Math.abs(v-b[i])))<1e-8,message);checks++;};
for(const preset of Object.values(CORRECTION_PRESETS)){
 const p={...base,...preset};
 for(const pt of [[502,552],[502,617],[330,530],[406,438]])near(artWorld(correctedLift(pt,0,p),p).slice(0,2),pt,'neutral artwork');
 for(const side of ['left','right']){
  const f=eyeFrame(side,p);assert(Math.abs(f.normal.reduce((s,v,i)=>s+v*f.right[i],0))<1e-9);checks++;
 }
}

const pure={...base,...CORRECTION_PRESETS.pure};
for(const pt of [[406,438],[502,552],[330,530]])near(correctedLift(pt,0,pure),liftHead(pt),'zero correction equals sphere');
const normal={...base,...CORRECTION_PRESETS.recommended};
assert(correctedLift([502,617],0,normal)[2]>80,'jaw strip has front depth instead of pole depth zero');checks++;
for(const yaw of [-50,-25,25,50]){
 const p={...normal,ParamAngleX:yaw};
 const neck=rig.meshes.find(m=>m.id==='neck');
 for(const pt of neck.positions)near(deform3D(neck,pt,p),deform3D(neck,pt,normal),'neck yaw independence');
 for(const side of ['left','right']){
  const xy=side==='left'?[417,438]:[587,437.5];
  const q=eyeballVertex(xy,side,'iris',normal);
  const local=[q[0]-502,q[1]-326,q[2]];
  near(eyeballVertex(xy,side,'iris',p),headWorld(local,p),'zero gaze follows parent once');
 }
}
for(const eyeDepth of [-12,12]){
 const p={...normal,CorrEyeDepth:eyeDepth,ParamAngleX:25,ParamAngleY:20};
 const c=eyeRootWorld('left',p);
 for(const q of [[54,0,0],[0,54,0],[0,0,54]]){
  const v=eyeWorld(q,'left',p);assert(Math.abs(Math.hypot(...v.map((x,i)=>x-c[i]))-54)<1e-9);checks++;
 }
}
const eyelid=rig.meshes.find(m=>m.id==='eye-left-sclera'),iris=rig.meshes.find(m=>m.id==='eye-left-iris');
const p={...normal,ParamAngleX:-25,CorrFarEye:0},p2={...p,CorrFarEye:.5};
assert(eyeRootWorld('left',p)[2]<eyeRootWorld('right',p)[2],'compressed eye is farther from camera');checks++;
assert.notDeepEqual(deform3D(eyelid,[370,438],p),deform3D(eyelid,[370,438],p2));checks++;
assert.notDeepEqual(deform3D(iris,[405,432.2],p),deform3D(iris,[405,432.2],p2),'iris shares the eyelid compression');checks++;
near(deform3D(iris,[415.7,432.2],p),deform3D(iris,[415.7,432.2],p2),'compression preserves the measured eye pivot');
console.log(JSON.stringify({status:'PASS',checks}));

for(const roll of [-30,-15,15,30]){
 const p={...normal,ParamAngleZ:roll};
 near(headWorld([0,NECK_PIVOT[1]-326,0],p),NECK_PIVOT,'roll fixes the neck joint');
 const c=headWorld([0,0,0],p);assert(Math.abs(Math.hypot(c[0]-502,c[1]-568)-242)<1e-9);checks++;
}
console.log(JSON.stringify({rollPivot:'PASS',checks}));
