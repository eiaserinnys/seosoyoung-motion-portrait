// Bake the same face-skin triangles at the actual reference angles, never the UI gain.
import {readFileSync} from 'node:fs';
import {faceSolveWorld} from '../bundle/face-solve.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const mesh=rig.meshes.find(m=>m.id==='face-skin');
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const poses=[-67.5,-90,67.5,90].map(yaw=>{
 const p={...base,ParamAngleX:yaw,[FRAME_CACHE]:new Map()};
 return {yaw,positions:mesh.positions.map(q=>faceSolveWorld(q,p).slice(0,2))};
});
process.stdout.write(JSON.stringify({rest:mesh.positions,triangles:mesh.triangles,poses}));
