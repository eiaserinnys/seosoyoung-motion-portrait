import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {HEAD_RADIUS,HEAD_CENTER,EYE_BALL_R,makeSphereMesh,headWorld,eyeCenter,eyeWorld,liftHead,eyeballAngles} from '../bundle/geometry.mjs';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base=Object.fromEntries(rig.parameters.map(p=>[p.id,p.default]));
const shell=makeSphereMesh();assert(shell.positions.some(p=>p[2]<-261)&&shell.positions.some(p=>p[2]>261),'closed front and back surface');
const neck=rig.meshes.find(m=>m.id==='neck');let radiusError=0,eyeRadiusError=0,neckYawError=0;
for(const [x,y,z] of [[0,0,0],[-50,0,0],[50,0,0],[0,40,0],[0,-40,0],[50,28,14],[-50,-28,-14]]){
 const p={...base,ParamAngleX:x,ParamAngleY:y,ParamAngleZ:z};
 for(const q of shell.positions){const w=headWorld(q,p);radiusError=Math.max(radiusError,Math.abs(Math.hypot(w[0]-502,w[1]-326,w[2])-HEAD_RADIUS));}
 for(const side of ['left','right']){
  const c=headWorld(eyeCenter(side),p),a=eyeballAngles(side,p);assert.equal(a.lon0,a.lon1);assert.equal(a.lat0,a.lat1);
  for(const q of makeSphereMesh(EYE_BALL_R,16,8).positions){const w=eyeWorld(q,side,{...p,ParamEyeBallX:.8,ParamEyeBallY:-.4});eyeRadiusError=Math.max(eyeRadiusError,Math.abs(Math.hypot(...w.map((v,i)=>v-c[i]))-EYE_BALL_R));}
 }
 if(y===0&&z===0)for(const pt of neck.positions){const a=deform3D(neck,pt,p),b=deform3D(neck,pt,base);neckYawError=Math.max(neckYawError,Math.hypot(...a.map((v,i)=>v-b[i])));}
}
assert(radiusError<1e-9&&eyeRadiusError<1e-9&&neckYawError===0);
for(const pt of [[330,700],[670,700],[502,617],[220,326]]){
 const local=liftHead(pt),a=headWorld(local,base);assert(Math.abs(a[0]-pt[0])<1e-9&&Math.abs(a[1]-pt[1])<1e-9,'strip keeps neutral artwork');
 const b=headWorld(local,{...base,ParamAngleX:50});assert(Math.abs(Math.hypot(...local)-Math.hypot(b[0]-502,b[1]-326,b[2]))<1e-9,'strip is rigidly attached in 3D');
}
assert.equal(opacity(rig.meshes.find(m=>m.id==='upper-body-short-sleeve'),base,{hidden:new Set(),outfit:'sweater'}),0);
console.log(JSON.stringify({status:'PASS',sphereVertices:shell.positions.length,radiusError,eyeRadiusError,neckYawError,closedSphere:true,verticalStrips:true}));
