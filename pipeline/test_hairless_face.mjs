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
test('T4: lower cheeks mirror-average the classified prior profiles without r11 bounds',()=>{
 for(const pitch of [-30,30]){
  const a=OUTLINE_POSES.find(p=>p.pitch===pitch&&p.yaw===-45),b=OUTLINE_POSES.find(p=>p.pitch===pitch&&p.yaw===45);
  for(let i=0;i<a.rows.length;i++){
   const sy=a.rows[i][0];
   const n=pitch<0?1004-b.measuredRows[i][2]:(a.measuredRows[i][4]+1004-b.measuredRows[i][2])/2;
   near(a.rows[i][4],n,'left near '+sy);near(b.rows[i][2],1004-n,'right near '+sy);
  }
 }
});
test('S4/S5: measured opening scales control lids without shear or ball compression',()=>{
 for(const f of FEATURE_FIT.poses)for(const side of ['left','right']){
  const p=at(f.yaw,f.pitch),[x,y]=FEATURE_FIT.front[side],id='eye-'+side+'-sclera';
  near(featureWorld(id,[x+40,y],p)[1]-featureWorld(id,[x-40,y],p)[1],0,f.name+' shear');
  const height=(id,q)=>featureWorld(id,[x,y+20],q)[1]-featureWorld(id,[x,y-20],q)[1];
  near(height(id,p)/height(id,at(f.yaw,0)),f.scales[side][1],f.name+' lid height');
  for(const part of ['iris','pupil','highlight'])near(height('eye-'+side+'-'+part,p)/height('eye-'+side+'-'+part,at(f.yaw,0)),1,f.name+' ball '+part);
  if(f.pitch){assert(f.measurements[side].openingExtent);near(f.measurements.nose.frontBiasY,5,'nose bias');}
 }
});
test('S5: all 9288 neutral-pitch feature vertices remain bit identical to d6a58ba',()=>{
 const values=[];
 for(const gain of [.65,1])for(const yaw of [-45,0,45])for(const m of rig.meshes.filter(m=>m.id.startsWith('eye-')||m.id==='nose-detail'||m.id.startsWith('mouth')||m.id==='closed-eyes-v1'))for(const pt of m.positions)values.push(featureWorld(m.id,pt,{...base,CorrTurn:gain,ParamAngleX:yaw,ParamAngleY:0}));
 assert.equal(createHash('sha256').update(JSON.stringify(values)).digest('hex'),'9e021c8deabc37ac0befb1d1bffc258d138f36bb8f7c37de6acabf919343ae82');
});
test('r11: neutral-pitch skin vertices retain their approved screen positions within one pixel',()=>{
 const skin=rig.meshes.find(m=>m.id==='face-skin');
 for(const gain of [.65,1])for(const yaw of [-45,0,45])for(const pt of skin.positions){
  const p={...at(yaw,0),CorrTurn:gain},actual=outlineWorld(pt,p),expected=legacyOutlineWorld(pt,p);
  assert(Math.hypot(actual[0]-expected[0],actual[1]-expected[1])<=1);
 }
});
test('joint cheek surface preserves vertical order and finite spatial curvature in up/down poses',()=>{
 for(const pitch of [-30,30]){
  for(const u of [.25,.5,.75]){
   const values=[480,490,497,510,525].map(y=>{const [l,r]=outlineRow(y);return faceSolveWorld([l+(r-l)*u,y],at(0,pitch))[1];});
   assert(values.every(Number.isFinite));
   for(let i=1;i<values.length;i++)assert(values[i]>values[i-1],`${pitch},${u}: ${values}`);
  }
  for(const y of [480,497,525]){
   const [l,r]=outlineRow(y);
   const values=Array.from({length:17},(_,i)=>faceSolveWorld([l+(r-l)*i/16,y],at(0,pitch))[1]);
   for(let i=1;i<values.length-1;i++)assert(Number.isFinite(values[i+1]-2*values[i]+values[i-1]));
  }
 }
});
