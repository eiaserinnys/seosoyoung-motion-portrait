import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {canopyWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {hairCoordinates,hairSolveField,applyHairSolve} from '../bundle/hair-solve.mjs';
import {hairSolveMetrics} from './hair_solve_metrics.mjs';
import {HAIR_SOLVE} from '../bundle/hair-solve-data.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('the three approved prior modules are byte-identical to main',()=>{
 for(const name of ['hair-shell.mjs','hair-surface.mjs','dome-data.mjs'])assert.deepEqual(readFileSync(new URL('../bundle/'+name,import.meta.url)),execFileSync('git',['show','0b75a02:bundle/'+name]));
});
test('shared residual preserves the upper seam and protects neutral/lower geometry',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const y of [60,230,325,435,470,584])for(const sign of [-1,1]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},{cx,rx}=domeSection(y),edge=[502+cx+sign*rx,y];
  const front=applyHairSolve(canopyWorld(edge,p),edge,p),pt=[...edge,sign*Math.PI/2];
  const prior=craniumWorld(pt,sign<0?'left':'right',p),rear=applyHairSolve(prior,pt,p,true);
  if(y<=325)assert(Math.hypot(...front.map((v,i)=>v-rear[i]))<1e-8);
  if(!pitch||y>=435)assert.deepEqual(hairSolveField(sign,y,p),[0,0]);
  assert.equal(hairCoordinates(pt,true)[0],sign);
 }
});
test('residual chart stays above the approved determinant floor',()=>assert(hairSolveMetrics().pass));
test('chart gate rejects an injected contraction',()=>{
 const saved=HAIR_SOLVE.poses[0].du;
 try{HAIR_SOLVE.poses[0].du=HAIR_SOLVE.y.flatMap(y=>HAIR_SOLVE.s.map(s=>[-s*1000,0]));assert(!hairSolveMetrics().pass);}
 finally{HAIR_SOLVE.poses[0].du=saved;}
});
