import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
const area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const rows=[];
for(const [x,y] of [[-50,-28],[-50,28],[50,-28],[50,28],[-25,0],[25,0]]){
 const p={...base,ParamAngleX:x,ParamAngleY:y};
 for(const id of ['face-skin','hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right']){
  const mesh=rig.meshes.find(m=>m.id===id),points=mesh.positions.map(pt=>deform3D(mesh,pt,p));
  const folded=mesh.triangles.filter(t=>area(t.map(i=>points[i]))*area(t.map(i=>mesh.positions[i]))<0).length;
  rows.push({pose:[x,y],id,folded});
 }
}
console.log(JSON.stringify(rows));
assert(rows.every(row=>row.folded===0),'artwork ribbons must not fold into themselves');
console.log('36 attachment cases PASS');
