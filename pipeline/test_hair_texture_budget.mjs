import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {referenceHairSample,posteriorLocal,domeSection} from '../bundle/hair-surface.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {DOME} from '../bundle/dome-data.mjs';
const baseline='83d4a80cfe2cd419d8daae90ee2783c2b9b1cbf9';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('T2 uses the near 45-degree texture through the ear-top latitude',()=>{
 const {cx,rx}=domeSection(400),point=[502+cx+rx*Math.sin(Math.PI/3),400];
 assert.equal(referenceHairSample(point,{...base,ParamAngleX:0}).weight,0);
 assert.equal(referenceHairSample(point,{...base,ParamAngleX:45}).weight,0);
 assert.equal(referenceHairSample(point,{...base,ParamAngleX:-45}).weight,1);
 assert(referenceHairSample(point,{...base,ParamAngleX:-15}).weight>0);
 assert.equal(referenceHairSample([point[0],490],{...base,ParamAngleX:-45}).weight,0);
});
test('V2 rear supplementation leaves all rows above 200 unchanged',()=>{
 const original=DOME.rearPitch.poses;
 DOME.rearPitch.poses=original.map(p=>p.pitch>0?{...p,rx:35,rz:10,y:0}:p);
 try{
 for(const yaw of [-45,45])for(const y of [60,100,199,200]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:30};
  assert.deepEqual(posteriorLocal(y,2.5,p),posteriorLocal(y,2.5,p,false));
 }
 }finally{DOME.rearPitch.poses=original;}
});
test('P3 limits vertices while preserving every non-hair mesh and all rig metadata',()=>{
 const old=JSON.parse(execFileSync('git',['show',baseline+':bundle/assets/seosoyoung-rig.json'],{maxBuffer:20*1024*1024}));
 const hair=m=>/^(hair-top|hairpin|bangs-|side-hair-|side-cranium-|back-hair-|tail-)/.test(m.id);
 assert.equal(rig.meshes.filter(hair).length,15);
 assert.deepEqual(rig.meshes.filter(m=>!hair(m)),old.meshes.filter(m=>!hair(m)));
 assert.deepEqual({...rig,meshes:[]},{...old,meshes:[]});
 execFileSync('git',['diff','--exit-code',baseline,'--','parts/','bundle/assets/texture_00.png']);
 assert(rig.meshes.reduce((s,m)=>s+m.positions.length,0)<=10000);
});
