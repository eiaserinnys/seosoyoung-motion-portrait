// Regenerate the neutral contract from the frozen formula at current vertices.
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {frozenRig,BASELINE} from './frozen_rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
const path=new URL('./fixtures/r15-hair-boundary.json',import.meta.url);
const fixture=JSON.parse(readFileSync(path));
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const old=await frozenRig();old.bindRig(rig);
try{
 const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
 fixture.baseline=BASELINE;fixture.scope='All current mesh vertices at neutral pitch, evaluated with frozen baseline formulas';
 fixture.cases=[-45,0,45].map(yaw=>{
  const p={...base,ParamAngleX:yaw,ParamAngleY:0,[old.FRAME_CACHE]:new Map()},hash=createHash('sha256');
  for(const m of rig.meshes)hash.update(JSON.stringify([m.id,m.positions.map(pt=>old.deform3D(m,pt,p))]));
  return {yaw,pitch:0,sha256:hash.digest('hex')};
 });
 writeFileSync(path,JSON.stringify(fixture,null,2)+'\n');
}finally{old.dispose();}
