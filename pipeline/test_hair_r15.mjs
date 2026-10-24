import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {fringeSurface} from '../bundle/hair-shell.mjs';
import {fringeRotation} from '../bundle/hair-surface.mjs';
import {DOME} from '../bundle/dome-data.mjs';

const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const front=['hair-top','hairpin','bangs-left','bangs-right','side-hair-left','side-hair-right'];
test('r15 faces, rear surfaces, lower side locks and physical attachments retain every vertex',{timeout:60000},()=>{
 const fixture=JSON.parse(readFileSync(new URL('./fixtures/r15-hair-boundary.json',import.meta.url)));
 for(const {yaw,pitch,sha256} of fixture.cases){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},h=createHash('sha256');
  for(const m of rig.meshes.filter(m=>!front.includes(m.id)||m.id.startsWith('side-hair'))){
   const points=m.positions.filter(pt=>!m.id.startsWith('side-hair')||pt[1]>=310);
   h.update(JSON.stringify([m.id,points.map(pt=>deform3D(m,pt,p))]));
  }
  assert.equal(h.digest('hex'),sha256,`r15 boundary drift at ${yaw},${pitch}`);
 }
});
test('raw pitch rotates the crown without an additional downward horizon turn',()=>{
 const y=fringeRotation([502,70],{...base,ParamAngleY:-30})[1];
 assert(y>100&&y<110,`r15 raw crown anchor was pre-rotated to ${y}`);
});
test('learned 3D relief affects the pitched crown interior and preserves neutral projection',()=>{
 const point=[502,180],p={...base,ParamAngleY:30},prior=DOME.depthPrior.relief;
 const real=fringeSurface(point,p),neutral=fringeSurface(point,base);
 try{
  DOME.depthPrior.relief=prior.map(row=>row.map(()=>0));
  const flat=fringeSurface(point,p);
  assert(Math.hypot(...real.map((v,i)=>v-flat[i]))>.01,'learned depth is disconnected from the rendered hair');
  assert.deepEqual(fringeSurface(point,base),neutral);
 }finally{DOME.depthPrior.relief=prior;}
});
test('close-up width measurements never expand the approved silhouette',()=>{
 assert.equal(DOME.pitchPoses.length,6);
 for(const pose of DOME.pitchPoses)for(const row of pose.crown.rows)assert(row[2]<=0,`${pose.name}: width expansion ${row[2]}`);
});
