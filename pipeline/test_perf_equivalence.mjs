import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {FRAME_CACHE,frameValue} from '../bundle/pose-grid.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {Renderer} from '../bundle/renderer.mjs';

const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('P1/P2 cache all nine poses without moving any vertex',{timeout:60000},async()=>{
 let beforeMs=0,afterMs=0,maxDisplacement=0;
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30]){
  const plain={...base,ParamAngleX:yaw,ParamAngleY:pitch};
  let start=performance.now();
  const before=rig.meshes.map(m=>m.positions.map(pt=>deform3D(m,pt,plain)));
  beforeMs+=performance.now()-start;
  const p={...plain,[FRAME_CACHE]:new Map()};start=performance.now();
  const after=rig.meshes.map(m=>frameValue(p,'pos:'+m.id,()=>m.positions.map(pt=>deform3D(m,pt,p))));
  afterMs+=performance.now()-start;
  for(let i=0;i<rig.meshes.length;i++){
   const mesh=rig.meshes[i];
   assert(p[FRAME_CACHE].has('params:'+mesh.id),'mesh parameters must be cached');
   assert.strictEqual(frameValue(p,'pos:'+mesh.id,()=>assert.fail('duplicate mask deformation')),after[i]);
   for(let j=0;j<after[i].length;j++)maxDisplacement=Math.max(maxDisplacement,Math.hypot(...after[i][j].map((v,k)=>v-before[i][j][k])));
  }
  assert(maxDisplacement<1e-9,`vertex displacement ${maxDisplacement}`);
 }
 console.log(JSON.stringify({poses:9,beforeMs,afterMs,maxDisplacement,displacementPx:maxDisplacement.toFixed(3)}));
});

test('P2 mask and main draw reuse positions only within their current frame',{timeout:60000},()=>{
 const gl=new Proxy({}, {get:()=>()=>{}});
 const renderer={gl,loc:{},rig};
 const mesh=rig.meshes.find(m=>m.id==='face-skin');
 const p={...base,ParamAngleX:45,ParamAngleY:30,[FRAME_CACHE]:new Map()};
 const mask=Renderer.prototype.draw.call(renderer,mesh,p,1);
 assert.strictEqual(Renderer.prototype.draw.call(renderer,mesh,p,1),mask);
 const next={...p,ParamAngleX:-45,[FRAME_CACHE]:new Map()};
 assert.notDeepEqual(Renderer.prototype.draw.call(renderer,mesh,next,1),mask);
});
