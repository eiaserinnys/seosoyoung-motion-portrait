import {approvedStageA,stageAScalars} from './approved_stage_a.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {CORRECTION_PRESETS,eyeRootWorld} from '../bundle/geometry.mjs';
import {outlineRow,outlineWorld,faceWorld,legacyOutlineWorld} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {OUTLINE_POSES} from '../bundle/outline-data.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {faceSolveWorld} from '../bundle/face-solve.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
const at=(yaw,pitch)=>({...base,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()});
const near=(a,b,label)=>assert(Math.abs(a-b)<1e-7,`${label}: ${a} vs ${b}`);
test('intermediate corneal dome has spatial curvature while reference anchors stay measured',()=>{
 for(const pitch of [-15,15])for(const yaw of [-22.5,0,22.5]){
  const [x,y]=FEATURE_FIT.front.left,id='eye-left-iris',p=at(yaw,pitch);
  const center=featureWorld(id,[x,y],p),left=featureWorld(id,[x-20,y],p),right=featureWorld(id,[x+20,y],p);
  assert(Math.abs(left[1]+right[1]-2*center[1])>1e-4,`${yaw},${pitch}: planar cornea`);
 }
 for(const f of FEATURE_FIT.poses.filter(p=>p.pitch))for(const [key,id] of Object.entries({left:'eye-left-sclera',right:'eye-right-sclera',nose:'nose-detail',mouth:'mouth-detail'})){
  const q=featureWorld(id,FEATURE_FIT.front[key],at(f.yaw,f.pitch));
  assert(Math.hypot(q[0]-f.centers[key][0],q[1]-f.centers[key][1])<1e-12,f.name+' '+key);
 }
});
test('spherical lid arch grows through the full pitch range without flattening at the endpoint',()=>{
 const [x,y]=FEATURE_FIT.front.left,id='eye-left-sclera',h=20;
 for(const sign of [-1,1]){
  let previous=0;
  for(const pitch of [7.5,15,22.5,30].map(v=>sign*v)){
   const p=at(22.5,pitch),center=featureWorld(id,[x,y],p);
   const left=featureWorld(id,[x-h,y],p),right=featureWorld(id,[x+h,y],p);
   const curvature=left[1]+right[1]-2*center[1];
   const expected=10*(pitch/30)*(pitch>0?.55:.35)*(h/46)**2;
   near(curvature,expected,'full spherical arch '+pitch);
   assert(Math.abs(curvature)>previous);previous=Math.abs(curvature);
  }
 }
});
test('corneal displacement survives canonical origins independent of layer evaluation order',()=>{
 const [x,y]=FEATURE_FIT.front.left;
 for(const yaw of [-90,-67.5,-22.5,0,22.5,67.5,90])for(const pitch of [-30,15,30]){
  const ids=['eye-left-sclera','eye-left-iris','eye-left-pupil','eye-left-highlight'];
  const expected=ids.map(id=>featureWorld(id,[x,y],at(yaw,pitch)));
  for(const order of [ids,[...ids].reverse()]){
   const p=at(yaw,pitch);for(const id of order)featureWorld(id,[x,y],p);
   ids.forEach((id,i)=>featureWorld(id,[x,y],p).forEach((v,k)=>near(v,expected[i][k],id+' cached order')));
  }
  assert(Math.abs(expected[1][1]-expected[0][1])>1,'corneal dome cancelled');
 }
});
test('S1/S2: actual registered chin, model rim and derived sampling floor',()=>{
 for(const f of FEATURE_FIT.poses.filter(p=>p.pitch)){
  const o=OUTLINE_POSES.find(p=>p.name===f.name),r=f.registration;
  near(r.scale,f.yaw?1:1/r.binocularRatio,f.name+' scale');
  const p=at(0,f.pitch),top=outlineRow(431.5).slice(0,2).reduce((s,x)=>s+faceWorld([x,431.5],0,p)[1]/2,0);
  const eyes=['left','right'].reduce((s,k)=>s+eyeRootWorld(k,p)[1]/2,0);
  near(o.mapping.rimOffset,top-eyes,f.name+' rim offset');
  const y3d=y=>outlineRow(y).slice(0,2).reduce((s,x)=>s+faceWorld([x,y],0,p)[1]/2,0);
  near(o.mapping.sFloor,f.pitch>0?(y3d(440)-top)/(y3d(616)-top):0,f.name+' floor');
  for(const side of ['left','right']){
   near(o.mapping.canvasYSpan[side][0],f.centers[side][1]+top-eyes,f.name+' canvas rim');
   near(o.mapping.rawYSpan[side][0],f.rawCenters[side][1]+(top-eyes)/r.scale,f.name+' raw rim');
   near(o.mapping.canvasYSpan[side][1],r.eyeMid[1]+(o.rawYChin-r.referenceEyeMid[1])*r.scale,f.name+' chin');
  }
 }
});
test('T4: independent classified background samples remain exact after shape constraints',()=>{
 for(const p of OUTLINE_POSES.filter(p=>p.pitch))p.rows.forEach((row,i)=>{
  for(const [side,col] of [['left',2],['right',4]])if(p.boundarySamples[i][side].validBG)
   near(row[col],p.boundarySamples[i][side].xCanvas,p.name+' BG '+row[0]);
 });
});
test('S4/S5: measured centerline openings retain the spherical arch, tilt and cornea',()=>{
 for(const f of FEATURE_FIT.poses)for(const side of ['left','right']){
  const p=at(f.yaw,f.pitch),[x,y]=FEATURE_FIT.front[side],id='eye-'+side+'-sclera';
  const a=featureWorld(id,[x-40,y],p),b=featureWorld(id,[x+40,y],p);
  const expected=(b[0]-a[0])*Math.sin(f.yaw*Math.PI/180)*Math.sin(f.pitch*Math.PI/180)*.32;
  near(b[1]-a[1],expected,f.name+' spherical tilt');
  const height=(id,q)=>featureWorld(id,[x,y+20],q)[1]-featureWorld(id,[x,y-20],q)[1];
  near(height(id,p)/height(id,at(f.yaw,0)),f.scales[side][1],f.name+' lid height');
  const s=f.pitch/30,ballRatio=Math.cos(f.pitch*Math.PI/180*.85)*(1-.08*s*s-.04*Math.max(0,-s));
  for(const part of ['iris','pupil','highlight'])near(height('eye-'+side+'-'+part,p)/height('eye-'+side+'-'+part,at(f.yaw,0)),ballRatio,f.name+' ball '+part);
  if(f.pitch){assert(f.measurements[side].openingExtent);near(f.measurements.nose.frontBiasY,5,'nose bias');}
 }
});
test('S5: all 9288 neutral-pitch feature vertices remain bit identical to d6a58ba',()=>{
 const values=[];
 for(const gain of [.65,1])for(const yaw of [-45,0,45])for(const m of rig.meshes.filter(m=>m.id.startsWith('eye-')||m.id==='nose-detail'||m.id.startsWith('mouth')||m.id==='closed-eyes-v1'))for(const pt of m.positions)values.push(featureWorld(m.id,pt,{...base,CorrTurn:gain,ParamAngleX:yaw,ParamAngleY:0}));
 assert.equal(createHash('sha256').update(JSON.stringify(values)).digest('hex'),'9e021c8deabc37ac0befb1d1bffc258d138f36bb8f7c37de6acabf919343ae82');
});
test('neutral-pitch skin vertices match the approved A-stage captured positions',()=>{
 const skin=rig.meshes.find(m=>m.id==='face-skin');
 for(const pose of approvedStageA.poses.filter(p=>p.pitch===0)){
  const expected=pose.meshes.find(m=>m.id==='face-skin').world,p=at(pose.yaw,0);
  skin.positions.forEach((point,i)=>outlineWorld(point,p).forEach((v,d)=>near(v,expected[i][d],pose.name+' neutral skin')));
 }
});
test('joint cheek surface retains the approved A-stage slice and finite spatial curvature',()=>{
 for(const pitch of [-30,30]){
  for(const u of [.25,.5,.75]){
   const values=[480,490,497,510,525].map(y=>{const [l,r]=outlineRow(y);return faceSolveWorld([l+(r-l)*u,y],at(0,pitch))[1];});
   assert(values.every(Number.isFinite));
   assert.deepEqual(values,stageAScalars().cheekY[[pitch,u]],'approved A-stage cheek slice');
  }
  for(const y of [480,497,525]){
   const [l,r]=outlineRow(y);
   const values=Array.from({length:17},(_,i)=>faceSolveWorld([l+(r-l)*i/16,y],at(0,pitch))[1]);
   for(let i=1;i<values.length-1;i++)assert(Number.isFinite(values[i+1]-2*values[i]+values[i-1]));
  }
 }
});
