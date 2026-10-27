import {readFileSync,writeFileSync} from 'node:fs';import {resolve} from 'node:path';import {pathToFileURL} from 'node:url';
const V=resolve(process.argv[2]),T=resolve(process.argv[3]),u=pathToFileURL(T+'/bundle/');
const {earBase,earTransform}=await import(new URL('stereo-attachments.mjs',u));const {FACE_SOLVE}=await import(new URL('face-solve-data.mjs',u));const {EAR_TRANSFORMS}=await import(new URL('ear-transform-data.mjs',u));
const {CORRECTION_PRESETS}=await import(new URL('geometry.mjs',u));const rig=JSON.parse(readFileSync(T+'/bundle/assets/seosoyoung-rig.json'));
const base={...Object.fromEntries(rig.parameters.map(q=>[q.id,q.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0};
let max=0;const poses={};
for(const pose of FACE_SOLVE.poses){
 const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch},sides={};
 for(const side of ['left','right']){
  const center=side==='right'?[671,464.5]:[333,465],c=earBase(center,side,p),ex=earBase([center[0]+1,center[1]],side,p),ey=earBase([center[0],center[1]+1],side,p),bc=[c[0]+pose.t[0],c[1]+pose.t[1]],K=[[ex[0]-c[0],ey[0]-c[0]],[ex[1]-c[1],ey[1]-c[1]]];
  const key=EAR_TRANSFORMS.find(k=>k.yaw===pose.yaw&&k.pitch===pose.pitch)[side],a=key.theta*Math.PI/180;
  for(const q of rig.meshes.find(m=>m.id==='ear-'+side).positions){
   const dx=K[0][0]*(q[0]-center[0])+K[0][1]*(q[1]-center[1]),dy=K[1][0]*(q[0]-center[0])+K[1][1]*(q[1]-center[1]);
   const x=dx*key.s*key.sx,y=dy*key.s,expected=[bc[0]+key.t[0]+Math.cos(a)*x-Math.sin(a)*y,bc[1]+key.t[1]+Math.sin(a)*x+Math.cos(a)*y],actual=earTransform(q,side,p);
   max=Math.max(max,...expected.map((v,i)=>Math.abs(v-actual[i])));
  }
  sides[side]={restCenter:center,Bcenter:bc,K,faceTranslation:pose.t};
 }poses[pose.name]={yaw:pose.yaw,pitch:pose.pitch,sides};
}
const result={poses,maximumRuntimeCompositionError:max,tolerance:1e-9,passed:max<=1e-9};writeFileSync(V+'/ear-bases.json',JSON.stringify(result,null,2));console.log('EAR BASE COMPOSITION CHECK',max);if(max>1e-9)process.exit(2);
