// Approved r11 yaw + pure 3D pitch, before any measured pitch residual.
import {readFileSync} from 'node:fs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {outlinePriorWorld} from '../bundle/face-outline.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const rows=JSON.parse(readFileSync(0,'utf8'));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleZ:0};
const result={};
for(const f of FEATURE_FIT.poses.filter(p=>p.pitch)){
 const p={...base,ParamAngleX:f.yaw,ParamAngleY:f.pitch};
 result[f.name]=rows.map(([y,l,r])=>[y,outlinePriorWorld([l,y],p)[0],outlinePriorWorld([r,y],p)[0]]);
}
process.stdout.write(JSON.stringify(result));
