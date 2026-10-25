// Small merge-time diagnostic: track the dome and both attachment endpoints.
import {readFileSync} from 'node:fs';
import {skinFillPositions} from '../bundle/skin-fill.mjs';
import {SKIN_FILL as S} from '../bundle/skin-fill-data.mjs';
import {bindRig} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const face=rig.meshes.find(m=>m.id==='face-skin');
const outer=yaw=>skinFillPositions({...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:-30,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,[FRAME_CACHE]:new Map()},face).slice(5*S.columns);
const samples=[-6,-3,0,3,6].map(yaw=>{const q=outer(yaw);return {yaw,top:q.reduce((a,b)=>a[1]<b[1]?a:b),left:q[0],right:q.at(-1)};});
const epsilonSteps=[-6,0,6].flatMap(yaw=>[-1,1].map(side=>{
 const a=outer(yaw),b=outer(yaw+side*.001);
 return {yaw,deltaYaw:side*.001,maxDisplacement:Math.max(...a.map((q,i)=>Math.hypot(q[0]-b[i][0],q[1]-b[i][1])))};
}));
process.stdout.write(JSON.stringify({pitch:-30,samples,epsilonSteps}));
