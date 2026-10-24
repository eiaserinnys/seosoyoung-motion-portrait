import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {fringeSurface} from '../bundle/hair-shell.mjs';
import {pitchHairDepth,sphereEase,domeSection} from '../bundle/hair-surface.mjs';
import {correctedLift} from '../bundle/geometry.mjs';
import {DOME} from '../bundle/dome-data.mjs';
import {frozenRig} from './frozen_rig.mjs';

const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('r15 neutral geometry stays exact on the reduced mesh',{timeout:60000},async()=>{
 const old=await frozenRig();
 const oldRig=structuredClone(rig);old.bindRig(oldRig);
 try{
 const fixture=JSON.parse(readFileSync(new URL('./fixtures/r15-hair-boundary.json',import.meta.url)));
 for(const {yaw,pitch,sha256} of fixture.cases.filter(c=>c.pitch===0)){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},h=createHash('sha256');
  const oldP={...base,ParamAngleX:yaw,ParamAngleY:0,[old.FRAME_CACHE]:new Map()};
  for(const [i,m] of rig.meshes.entries()){
   const actual=m.positions.map(pt=>deform3D(m,pt,p));
   const expected=m.positions.map(pt=>old.deform3D(oldRig.meshes[i],pt,oldP));
   for(let j=0;j<actual.length;j++)assert(Math.hypot(...actual[j].map((v,k)=>v-expected[j][k]))<1e-9,`${m.id} neutral drift`);
   h.update(JSON.stringify([m.id,actual]));
  }
  assert.equal(h.digest('hex'),sha256,`r15 boundary drift at ${yaw},${pitch}`);
 }
 }finally{old.dispose();}
});
test('H1 depth is a single bounded cosine latitude arc and keeps the lower lift',()=>{
 assert(DOME.pitchArc.topZ>=-130&&DOME.pitchArc.topZ<=-80);
 const p={...base,ParamAngleY:30};
 for(const y of [8,70,150,230,309])for(const x of [330,502,670]){
  const baseZ=correctedLift([x,y],0,p)[2],expected=baseZ+(DOME.pitchArc.topZ-baseZ)*sphereEase((310-y)/302);
  assert(Math.abs(pitchHairDepth([x,y],p)-expected)<1e-8);
 }
 for(const y of [310,350,420])for(const x of [300,502,700])assert.equal(pitchHairDepth([x,y],p),correctedLift([x,y],0,p)[2]);
});
test('H1 removes all per-row pitch affine residuals and H5 bounds tip lag',()=>{
 assert(!('pitchRows' in DOME));assert(!('pitchPoses' in DOME));assert(!('frontHair' in DOME));
 assert(DOME.tipLag.kappa>=.35&&DOME.tipLag.kappa<=.85);
});
