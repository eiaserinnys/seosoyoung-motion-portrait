import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {configureHeadBase,projectModel,modelAngles,phi,fieldAt,headBaseVertices,headSolveWorld} from '../bundle/head-base.mjs';
import {CORRECTION_PRESETS} from '../bundle/corrections.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {registeredFeatureWorld} from '../bundle/face-feature-prior.mjs';
import {outlineWorld} from '../bundle/face-outline.mjs';
import {bindRig,deform3D} from '../bundle/rig.mjs';

const read=path=>readFileSync(new URL(path,import.meta.url));
const metadata=JSON.parse(read('../bundle/assets/head-base.json'));
const binary=read('../bundle/assets/head-base.bin');
configureHeadBase(metadata,binary.buffer.slice(binary.byteOffset,binary.byteOffset+binary.byteLength));
const fixture=JSON.parse(read('./headmodel/fit/results/fit2e/head-base-fixture.json'));
const rig=JSON.parse(read('../bundle/assets/seosoyoung-rig.json'));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrHeadBase:1,CorrOutline:1,CorrTurn:1};
const pose=(yaw=0,pitch=0,extra={})=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,...extra,[FRAME_CACHE]:new Map()});
const skin=rig.meshes.find(m=>m.id==='face-skin');
const feature=rig.meshes.filter(m=>m.id.startsWith('eye-')||m.id==='closed-eyes-v1'||m.id==='nose-detail'||m.id.startsWith('mouth'));
const close=(a,b,tol,label)=>assert(Math.abs(a-b)<=tol,`${label}: ${a} vs ${b}`);
const digest=v=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const oldFeatures={
 '1:0':'80c0e03a75e7d8fdc530f7814323c25a933708fa6c4d6e69a68f824e9326f7f2',
 '1:0.5':'bcdbe31ed6c048b067628dfb4fb7e936524a7f2045a4d782f3baa3543709cf88',
 '1:1':'aa6b7136d84fc56dcba3d375d34c871f142eee458fcf9774ba50899a13a1c51a',
 '0.65:0':'29f256bdce53066bb90bcebe355239c01c19b1cb6d05a662aec93c98ef436f84',
 '0.65:0.5':'f7d7592cc233828d99f27a709c1b4f87d4a3d58bba48ef8c21e372c095885138',
 '0.65:1':'9cfe292c3370db2ba277f50d34bf5323268b5052a2a3ff94e3311b4205d5df70'};
const oldSkin={'1':'643b11c1d22b621f598f6633185d2cd82ee78b03b01f040cb0a4bef5b3048631','0.65':'80388b8f6df4482094cc9e250cde58010b3020732108f266714c148f4241c02c'};

test('N7: Python projection fixture, 100 vertices and 13 poses',{timeout:30000},()=>{
 const vertices=new Float32Array(binary.buffer,binary.byteOffset+metadata.offsets.vertices.offset,metadata.offsets.vertices.length/4);
 for(let k=0;k<fixture.poses.length;k++)for(let j=0;j<fixture.indices.length;j++){
  const i=fixture.indices[j],q=projectModel(Array.from(vertices.slice(i*3,i*3+3)),fixture.poses[k]);
  q.forEach((x,c)=>close(x,fixture.projections[k][j][c],1e-6,`pose ${k} point ${j}`));
 }
});
test('angle field fixture at all nine nodes',{timeout:30000},()=>{
 metadata.theta_k.forEach((theta,k)=>fixture.indices.forEach((index,j)=>fieldAt(theta,index).forEach((x,c)=>close(x,fixture.fields[k][j][c],1e-6,`field ${k} point ${j}`))));
});
test('N1: all rest skin and solve points retain their coordinates',{timeout:30000},()=>{
 const p=pose();for(const point of [...skin.positions,...FACE_SOLVE.points]){
  const q=headSolveWorld(point,p);close(q[0],point[0],1e-3,'rest x');close(q[1],point[1],1e-3,'rest y');
  const via=outlineWorld(point,p);close(via[0],point[0],1e-3,'outline x');close(via[1],point[1],1e-3,'outline y');
 }
});
test('N2: legacy skin and all feature vertices match main hashes',{timeout:30000},()=>{
 for(const gain of [1,.65])for(const outline of [0,.5,1])for(const enabled of [0,1]){
  const values=[],skinValues=[];
  for(const pitch of [-30,0,30])for(const yaw of [-45,0,45]){
   const p=pose(yaw,pitch,{CorrTurn:gain,CorrOutline:outline,CorrHeadBase:enabled});
   for(const mesh of feature)for(const point of mesh.positions)values.push(deform3D(mesh,point,p));
   if(!enabled&&outline===1)for(const point of skin.positions)skinValues.push(deform3D(skin,point,p));
  }
  assert.equal(digest(values),oldFeatures[`${gain}:${outline}`],`features ${gain}/${outline}/${enabled}`);
  if(!enabled&&outline===1)assert.equal(digest(skinValues),oldSkin[gain],`skin ${gain}`);
 }
});
test('N6: skin and head surface are continuous at both 45 degree boundaries',{timeout:30000},()=>{
 for(const sign of [-1,1])for(const pitch of [-30,0,30]){
  const a=pose(sign*44.99,pitch),b=pose(sign*45.01,pitch),va=headBaseVertices(a).surface,vb=headBaseVertices(b).surface;
  for(let i=0;i<va.length;i+=3)assert(Math.hypot(va[i]-vb[i],va[i+1]-vb[i+1])<=.5,`head ${sign}/${pitch}/${i/3}`);
  for(const point of skin.positions){const x=headSolveWorld(point,a),y=headSolveWorld(point,b);assert(Math.hypot(x[0]-y[0],x[1]-y[1])<=.5,`skin ${sign}/${pitch}/${point}`);}
 }
});
test('N8: eye midpoint matches the registered rig at all nine keys',{timeout:30000},()=>{
 for(const pitch of [-30,0,30])for(const yaw of [-45,0,45]){
  const p=pose(yaw,pitch),eye=headSolveWorld(metadata.rest_eye,p),left=registeredFeatureWorld('eye-left-sclera',FEATURE_FIT.front.left,p),right=registeredFeatureWorld('eye-right-sclera',FEATURE_FIT.front.right,p);
  for(let c=0;c<2;c++)close(eye[c],(left[c]+right[c])/2,1e-3,`eye ${yaw}/${pitch}`);
 }
});
test('nine calibrated keys and angular continuity',{timeout:30000},()=>{
 for(const key of metadata.keys){const q=modelAngles(pose(key.yaw,key.pitch));q.forEach((x,i)=>close(x,key.model[i],1e-9,`key ${key.frame}`));}
 for(const pitch of [-30,-15,0,15,30]){
  for(let yaw=-90;yaw<90;yaw++){const a=modelAngles(pose(yaw,pitch)),b=modelAngles(pose(yaw+1,pitch));for(let i=0;i<2;i++)assert(Math.abs(a[i]-b[i])<=2,`step ${yaw}/${pitch}`);}
  for(const sign of [-1,1]){const a=modelAngles(pose(sign*44.99,pitch)),b=modelAngles(pose(sign*45.01,pitch));for(let i=0;i<2;i++)assert(Math.abs(a[i]-b[i])<=.05,`edge ${sign}/${pitch}`);}
 }
});
test('outside attachments retain rest offset and decay it continuously',{timeout:30000},()=>{
 const [key,a]=Object.entries(metadata.attachments).find(([,entry])=>Math.hypot(...entry.offset)>1);
 const point=key.split(',').map(Number),rest=pose(),q=headSolveWorld(point,rest);
 close(q[0],point[0],1e-3,'outside rest x');close(q[1],point[1],1e-3,'outside rest y');
 const p=pose(45,0),frame=headBaseVertices(p),m=headSolveWorld(point,p),surface=[0,0];
 a.vertices.forEach((index,i)=>{for(let c=0;c<2;c++)surface[c]+=a.weights[i]*frame.surface[index*3+c];});
 for(let c=0;c<2;c++)close(m[c]-surface[c],a.offset[c]*frame.offsetWeight,1e-9,'outside decay');
 const unlisted=[250.125,150.25],near=headSolveWorld(unlisted,pose(.01,0));assert(near.every(Number.isFinite));
});
