import {readFileSync} from 'node:fs';
import {skinFillMesh,skinFillPositions} from '../bundle/skin-fill.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {SKIN_FILL as S} from '../bundle/skin-fill-data.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const face=rig.meshes.find(m=>m.id==='face-skin'),mesh=skinFillMesh();
const params=(yaw,pitch)=>({...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,[FRAME_CACHE]:new Map()});
if(process.argv.includes('--neutral'))process.stdout.write(JSON.stringify(skinFillPositions(params(0,0),face).map(p=>p.slice(0,2))));
else process.stdout.write(JSON.stringify(FEATURE_FIT.poses.map(f=>{const p=params(f.yaw,f.pitch);return {name:f.name,triangles:mesh.triangles,innerRow:S.innerRow,earHeight:S.earHeight,
 positions:skinFillPositions(p,face),face:face.positions.map(q=>deform3D(face,q,p)),faceRest:face.positions,faceTriangles:face.triangles};})));
