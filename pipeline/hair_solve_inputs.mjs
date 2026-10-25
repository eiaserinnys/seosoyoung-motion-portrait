import {readFileSync} from 'node:fs';
import {canopyWorld,backWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {craniumFacing} from '../bundle/hair-surface.mjs';
import {hairCoordinates} from '../bundle/hair-solve.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {beadWorld} from '../bundle/stereo-attachments.mjs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const bare=process.argv.includes('--bare-face');if(bare)bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const poses=[];
for(const pitch of [30,-30])for(const yaw of [-45,0,45]){
 const name=(pitch>0?'up':'down')+(yaw<0?'-left':yaw>0?'-right':'');
 const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},vertices=[],triangles=[];
 if(bare){
  const m=rig.meshes.find(m=>m.id==='face-skin');
  poses.push({name,yaw,pitch,triangles:m.triangles,vertices:m.positions.map((pt,i)=>({xy:deform3D(m,pt,p).slice(0,2),uv:m.uvs[i],front:false,cull:true,facing:1,order:1}))});
  continue;
 }
 for(const m of rig.meshes){
  const posterior=m.id.startsWith('side-cranium'),back=m.id.startsWith('back-hair');
  const front=m.id==='hair-top'||m.id.startsWith('bangs')||m.id.startsWith('side-hair')||m.id==='hairpin';
  if(!posterior&&!back&&!front)continue;
  const indices=new Map();
  for(const [index,point] of m.positions.entries()){
   if(point[1]>=545)continue;
   const world=posterior?craniumWorld(point,m.id.endsWith('left')?'left':'right',p):back?backWorld(point,p):canopyWorld(point,p);
   const xy=world.slice(0,2);
   const [s,y]=hairCoordinates(point,posterior);
   indices.set(index,vertices.length);
   vertices.push({s,y,x:point[0],xy,front,cull:front||back,clipFromY:back?(m.id==='back-hair-center'?407:310):null,uv:m.uvs[index],facing:posterior?craniumFacing(point,p):1,order:m.drawOrder});
  }
  for(const ids of m.triangles)if(ids.every(i=>indices.has(i)))triangles.push(ids.map(i=>indices.get(i)));
 }
 poses.push({name,yaw,pitch,vertices,triangles,anchors:['left','right'].map(side=>beadWorld(side,p))});
}
process.stdout.write(JSON.stringify(poses));
