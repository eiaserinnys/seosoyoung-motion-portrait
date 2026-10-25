import {test} from 'node:test';
import assert from 'node:assert/strict';
import {OUTLINE_POSES} from '../bundle/outline-data.mjs';
import {outlineRow,outlineWorld,outlinePriorWorld} from '../bundle/face-outline.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {readFileSync} from 'node:fs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('T1/T2: pitch profiles record boundary classes, valid measurements and 3D priors',()=>{
 for(const p of OUTLINE_POSES.filter(p=>p.pitch)){
  assert.equal(p.boundarySamples?.length,16,p.name+' classified rows');
  assert.equal(p.priorRows?.length,16,p.name+' prior rows');
  for(const row of p.boundarySamples)for(const side of ['left','right']){
   assert(['BG','HAIR','SKIN'].includes(row[side].class));
   if(row[side].validBG)assert.equal(row[side].class,'BG');
   assert.equal(row[side].neckBounded,row[side].class==='SKIN');
  }
 }
});
test('T2: prior X retains the approved yaw projection before measured pitch residuals',()=>{
 for(const f of OUTLINE_POSES.filter(p=>p.pitch)){
  const p={...base,ParamAngleX:f.yaw,ParamAngleY:f.pitch,[FRAME_CACHE]:new Map()},p0={...p,ParamAngleY:0};
  for(const [y,l,r] of f.priorRows)for(const [i,x] of outlineRow(y).slice(0,2).entries()){
   const value=outlinePriorWorld([x,y],p)[0];
   assert(Math.abs(value-[l,r][i])<1e-7);
   assert(Math.abs(value-outlineWorld([x,y],p0)[0])<1e-7);
  }
 }
});
test('T3/T6: all lower cheeks narrow monotonically in target and runtime',()=>{
 for(const f of OUTLINE_POSES.filter(p=>p.pitch)){
  const p={...base,ParamAngleX:f.yaw,ParamAngleY:f.pitch,[FRAME_CACHE]:new Map()};
  const rows=f.rows.filter(r=>r[0]>=(f.name==='up'?465:480));
  for(let i=1;i<rows.length;i++){
   const prev=rows[i-1],row=rows[i];
   assert(row[4]-row[2]<=prev[4]-prev[2]+.5,`${f.name} target bulge ${prev[0]}→${row[0]}`);
   const width=y=>{const [l,r]=outlineRow(y);return outlineWorld([r,y],p)[0]-outlineWorld([l,y],p)[0];};
   assert(width(row[0])<=width(prev[0])+.5,`${f.name} runtime bulge ${prev[0]}→${row[0]}`);
  }
 }
});
test('T3/T6: upper up cheeks retain the rotated prior instead of side-hair cutouts',()=>{
 for(const p of OUTLINE_POSES.filter(p=>p.pitch>0)){
  assert(p.priorRows,p.name+' priors');
  p.rows.forEach((row,i)=>{if(row[0]<=465)assert(Math.abs(row[4]-row[2]-(p.priorRows[i][2]-p.priorRows[i][1]))<=8,p.name+' upper width '+row[0]);});
 }
 const up=OUTLINE_POSES.find(p=>p.name==='up'),width=y=>{const row=up.rows.find(r=>r[0]===y);return row[4]-row[2];};
 assert(width(450)>width(490)+8,'up pear inversion');
});
test('T4: pitched profiles preserve independent background edges and measured chin',()=>{
 for(const p of OUTLINE_POSES.filter(p=>p.pitch)){
  p.rows.forEach((r,i)=>{
   for(const [side,col] of [['left',2],['right',4]])if(p.boundarySamples[i][side].validBG)
    assert(Math.abs(r[col]-p.boundarySamples[i][side].xCanvas)<.5,p.name+' '+side);
  });
  const reg=p.registration,target=reg.eyeMid[1]+(p.rawYChin-reg.referenceEyeMid[1])*reg.scale;
  assert(Math.abs(p.yChin-target)<.5,p.name+' chin');
 }
});
