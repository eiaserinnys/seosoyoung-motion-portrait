import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {canopyWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {hairCoordinates,hairSolveField,applyHairSolve,hairPoseWeights} from '../bundle/hair-solve.mjs';
import * as split from '../bundle/hair-solve.mjs';
import {BEADS,beadWorld} from '../bundle/stereo-attachments.mjs';
import {hairSolveMetrics} from './hair_solve_metrics.mjs';
import {HAIR_SOLVE} from '../bundle/hair-solve-data.mjs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('hairpin keeps the canopy shape under a nonuniform residual field',()=>{
 bindRig(rig);const pin=rig.meshes.find(m=>m.id==='hairpin');
 const saved=HAIR_SOLVE.poses.map(p=>p.du);
 try{
  for(const yaw of [-90,-67.5,-45,0,45,67.5,90])for(const pitch of [-30,0,30]){
   const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};
   for(const pose of HAIR_SOLVE.poses)pose.du=HAIR_SOLVE.y.flatMap(()=>HAIR_SOLVE.s.map(()=>[0,0]));
   const prior=pin.positions.map(pt=>deform3D(pin,pt,p));
   // Inject compression/shear over the unchanged canopy + profile projection.
   for(const pose of HAIR_SOLVE.poses)pose.du=HAIR_SOLVE.y.flatMap(y=>HAIR_SOLVE.s.map(s=>[s*20,-y*.25]));
   const offsets=pin.positions.map((pt,n)=>{const a=deform3D(pin,pt,p);return a.map((v,i)=>v-prior[n][i]);});
   for(const delta of offsets)for(let i=0;i<3;i++)assert(Math.abs(delta[i]-offsets[0][i])<1e-9,`${yaw}/${pitch} pin shape`);
  }
 }finally{HAIR_SOLVE.poses.forEach((p,i)=>p.du=saved[i]);}
});
test('hair weights preserve nine keys, cross neutral smoothly and fade before profile parts',()=>{
 const weights=(yaw,pitch=30)=>hairPoseWeights({...base,ParamAngleX:yaw,ParamAngleY:pitch});
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])
  assert.deepEqual(weights(yaw,pitch),HAIR_SOLVE.poses.map(p=>p.yaw===yaw&&p.pitch===pitch?1:0));
 const e=1e-4,a=weights(-e),b=weights(0),c=weights(e);
 assert(Math.hypot(...a.map((v,i)=>(c[i]-2*b[i]+v)/e))<1e-6);
 assert.deepEqual(weights(60),weights(45));
 for(const yaw of [-90,-88,88,90])assert(weights(yaw).every(v=>v===0));
 assert(Math.abs(weights(74).reduce((a,b)=>a+b,0)-.5)<1e-12);
});
test('the three approved prior modules are byte-identical to main',()=>{
 for(const name of ['hair-shell.mjs','hair-surface.mjs','dome-data.mjs'])assert.deepEqual(readFileSync(new URL('../bundle/'+name,import.meta.url)),execFileSync('git',['show','0b75a02:bundle/'+name]));
});
test('shared residual preserves the upper seam and protects neutral/lower geometry',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const y of [60,230,325,435,470,584])for(const sign of [-1,1]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},{cx,rx}=domeSection(y),edge=[502+cx+sign*rx,y];
  const front=applyHairSolve(canopyWorld(edge,p),edge,p),pt=[...edge,sign*Math.PI/2];
  const prior=craniumWorld(pt,sign<0?'left':'right',p),rear=applyHairSolve(prior,pt,p,true);
  if(y<=325)assert(Math.hypot(...front.map((v,i)=>v-rear[i]))<1e-8);
  if(!pitch||y>=545)assert.deepEqual(hairSolveField(sign,y,p),[0,0]);
  assert.equal(hairCoordinates(pt,true)[0],sign);
 }
});
test('residual chart stays above the approved determinant floor',()=>assert(hairSolveMetrics().pass));
test('chart gate rejects an injected contraction',()=>{
 const saved=HAIR_SOLVE.poses[0].du;
 try{HAIR_SOLVE.poses[0].du=HAIR_SOLVE.y.flatMap(y=>HAIR_SOLVE.s.map(s=>[-s*1000,0]));assert(!hairSolveMetrics().pass);}
 finally{HAIR_SOLVE.poses[0].du=saved;}
});
test('lower triangle gate rejects an injected tail reflection',()=>{
 const saved=HAIR_SOLVE.lower.poses.map(p=>p.left);
 try{
  for(const pose of HAIR_SOLVE.lower.poses)pose.left=HAIR_SOLVE.lower.tau.map(()=>[0,-2]);
  assert(!hairSolveMetrics().lower.pass);
 }finally{HAIR_SOLVE.lower.poses.forEach((p,i)=>p.left=saved[i]);}
});
test('bead split extends upper coverage and independently fits both hanging locks',()=>{
 assert.equal(split.hairFade(500),1);
 assert.equal(split.hairFade(545),0);
 assert.equal(HAIR_SOLVE.y.at(-1),545);
 assert.equal(typeof split.applyTailSolve,'function');
 assert.equal(HAIR_SOLVE.lower.poses.length,6);
 for(const side of ['left','right'])for(const yaw of [-90,-67.5,-45,0,45,67.5,90])for(const pitch of [-30,0,30]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},root=BEADS[side].front,anchor=beadWorld(side,p);
  assert.deepEqual(split.applyTailSolve(anchor,root,side,p),anchor);
  const point=[root[0]+20,680],world=[anchor[0]+20,anchor[1]+136,anchor[2]];
  if(!pitch)assert.deepEqual(split.applyTailSolve(world,point,side,p),world);
 }
 for(const pose of HAIR_SOLVE.poses){
  const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch};
  assert(Math.hypot(...hairSolveField(1.5,480,p))>1e-4);
  for(const side of ['left','right']){
   const root=BEADS[side].front,anchor=beadWorld(side,p),point=[root[0]+20,680],world=[anchor[0]+20,anchor[1]+136,anchor[2]];
   assert(Math.abs(split.applyTailSolve(world,point,side,p)[0]-world[0])>1e-4);
  }
 }
});
