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

const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const front=['hair-top','hairpin','bangs-left','bangs-right','side-hair-left','side-hair-right'];
test('r15 neutral geometry stays exact',{timeout:60000},()=>{
 const fixture=JSON.parse(readFileSync(new URL('./fixtures/r15-hair-boundary.json',import.meta.url)));
 for(const {yaw,pitch,sha256} of fixture.cases.filter(c=>c.pitch===0)){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},h=createHash('sha256');
  const far=pitch>0&&yaw?(yaw<0?'left':'right'):null;
  for(const m of rig.meshes.filter(m=>(!front.includes(m.id)||m.id.startsWith('side-hair'))&&!(far&&m.id.startsWith('tail-'+far+'-')))){
   const points=m.positions.filter(pt=>!m.id.startsWith('side-hair')||pt[1]>=310);
   h.update(JSON.stringify([m.id,points.map(pt=>deform3D(m,pt,p))]));
  }
  assert.equal(h.digest('hex'),sha256,`r15 boundary drift at ${yaw},${pitch}`);
 }
});
test('H1 depth is a single bounded cosine latitude arc and keeps the lower lift',()=>{
 assert(DOME.pitchArc.topZ>=-130&&DOME.pitchArc.topZ<=-80);
 const p={...base,ParamAngleY:30},z310=correctedLift([502,310],0,p)[2];
 for(const y of [8,70,150,230,309]){
  const expected=z310+(DOME.pitchArc.topZ-z310)*sphereEase((310-y)/302);
  assert(Math.abs(pitchHairDepth([502+domeSection(y).cx,y],p)-expected)<1e-8);
 }
 for(const y of [310,350,420])for(const x of [300,502,700])assert.equal(pitchHairDepth([x,y],p),correctedLift([x,y],0,p)[2]);
});
test('H1 removes all per-row pitch affine residuals and H5 bounds tip lag',()=>{
 assert(!('pitchRows' in DOME));assert(!('pitchPoses' in DOME));assert(!('frontHair' in DOME));
 assert(DOME.tipLag.kappa>=.35&&DOME.tipLag.kappa<=.85);
});
