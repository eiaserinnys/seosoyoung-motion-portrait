import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
const area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const area3=([a,b,c])=>{
 const u=b.map((v,i)=>v-a[i]),v=c.map((v,i)=>v-a[i]);
 return Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);
};
const rows=[];
for(const gain of [.65,1])for(const [x,y] of [[-50,-28],[-50,28],[50,-28],[50,28],[-50,0],[50,0],[-25,0],[25,0],[0,-20],[0,20]]){
 const p={...base,CorrTurn:gain,ParamAngleX:x,ParamAngleY:y};
 for(const id of ['face-skin','hair-top','bangs-left','bangs-right','hairpin','side-hair-left','side-hair-right','back-hair-left','back-hair-center','back-hair-right']){
  const mesh=rig.meshes.find(m=>m.id===id),points=mesh.positions.map(pt=>deform3D(mesh,pt,p));
  assert(points.every(q=>q.every(Number.isFinite)),id+' finite vertices');
  const reversed=mesh.triangles.filter(t=>area(t.map(i=>points[i]))*area(t.map(i=>mesh.positions[i]))<0).length;
  const solid=id==='hair-top'||id==='hairpin'||id.startsWith('bangs')||id.startsWith('side-hair')||id.startsWith('back-hair');
  // A surface turning away from the camera is a back face, not a collapsed
  // ArtMesh. These surfaces use CULL_FACE; keep the unculled facial ribbons'
  // original screen-orientation contract and check 3D area for the surfaces.
  let minAreaRatio=Infinity;
  for(const t of mesh.triangles){
   const original=Math.abs(area(t.map(i=>mesh.positions[i])));
   if(original>1e-8)minAreaRatio=Math.min(minAreaRatio,area3(t.map(i=>points[i]))/original);
  }
  assert(minAreaRatio>1e-4,`${id}: collapsed 3D triangle ${minAreaRatio}`);
  if(!solid)assert.equal(reversed,0,`${id}: front artwork fold`);
  rows.push({gain,pose:[x,y],id,backFacing:solid?reversed:0,minAreaRatio});
 }
}
console.log(JSON.stringify({cases:rows.length,minAreaRatio:Math.min(...rows.map(r=>r.minAreaRatio)),maxBackFacing:Math.max(...rows.map(r=>r.backFacing))}));
console.log(`${rows.length} attachment cases PASS (back faces culled; no collapsed 3D triangles or inverted facial ribbons)`);
