import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {CORRECTION_PRESETS,eyeRootWorld} from '../bundle/geometry.mjs';
import {outlineRow,outlineWorld,faceWorld} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {OUTLINE_POSES} from '../bundle/outline-data.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {sphericalFaceError} from './hairless_sphere_contract.mjs';
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
test('S3: lower near cheeks follow the less occluded mirrored measurement',()=>{
 for(const pitch of [-30,30]){
  const a=OUTLINE_POSES.find(p=>p.pitch===pitch&&p.yaw===-45),b=OUTLINE_POSES.find(p=>p.pitch===pitch&&p.yaw===45);
  for(let i=0;i<a.rows.length;i++){
   const sy=a.rows[i][0],[l,r]=outlineRow(sy);
   if(sy<=490)continue;
   const n=Math.min(r-a.measuredRows[i][4],b.measuredRows[i][2]-l);
   near(a.rows[i][4],r-n,'left near '+sy);near(b.rows[i][2],l+n,'right near '+sy);
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
test('r11: all 4038 neutral-pitch skin vertices remain bit identical to d6a58ba',()=>{
 const values=[],skin=rig.meshes.find(m=>m.id==='face-skin');
 for(const gain of [.65,1])for(const yaw of [-45,0,45])for(const pt of skin.positions)values.push(outlineWorld(pt,{...base,CorrTurn:gain,ParamAngleX:yaw,ParamAngleY:0}));
 assert.equal(createHash('sha256').update(JSON.stringify(values)).digest('hex'),'2266d76058271ffaaf663ad55ec2226efbf4f7230ad5a0f6bddf8cc260b93dae');
});
test('S6 spherical revision: nose-row cheek curvature stays within approved up10/down4.05 pixels',()=>{
 for(const pitch of [-30,30]){
  const [l,r]=outlineRow(497),dy=u=>{const pt=[l+(r-l)*u,497];return outlineWorld(pt,at(0,pitch))[1]-outlineWorld(pt,at(0,0))[1];};
  assert(Math.abs(dy(.25)-dy(.5))<=(pitch>0?10:4.05),`${pitch},0.25: ${dy(.25)} vs ${dy(.5)}`);
 }
});
test('S6 spherical revision: nose/mouth longitude envelope and latitude bands match the closed form',()=>{
 for(const pitch of [-30,30])assert(sphericalFaceError(at(0,pitch))<1e-6,'sphere field '+pitch);
});
