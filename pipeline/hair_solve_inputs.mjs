import {readFileSync} from 'node:fs';
import {canopyWorld,backWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {craniumFacing} from '../bundle/hair-surface.mjs';
import {hairCoordinates} from '../bundle/hair-solve.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const poses=[];
for(const pitch of [30,-30])for(const yaw of [-45,0,45]){
 const name=(pitch>0?'up':'down')+(yaw<0?'-left':yaw>0?'-right':'');
 const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},vertices=[];
 for(const m of rig.meshes){
  const posterior=m.id.startsWith('side-cranium'),back=m.id.startsWith('back-hair');
  const front=m.id==='hair-top'||m.id.startsWith('bangs')||m.id.startsWith('side-hair')||m.id==='hairpin';
  if(!posterior&&!back&&!front)continue;
  for(const point of m.positions){
   if(point[1]>=435||posterior&&craniumFacing(point,p)<=0)continue;
   const world=posterior?craniumWorld(point,m.id.endsWith('left')?'left':'right',p):back?backWorld(point,p):canopyWorld(point,p);
   const xy=world.slice(0,2);
   const [s,y]=hairCoordinates(point,posterior);
   vertices.push({s,y,xy,front});
  }
 }
 poses.push({name,yaw,pitch,vertices});
}
process.stdout.write(JSON.stringify(poses));
