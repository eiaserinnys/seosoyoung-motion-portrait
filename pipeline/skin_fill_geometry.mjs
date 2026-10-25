import {skinFillMesh,skinFillPositions} from '../bundle/skin-fill.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const mesh=skinFillMesh();
process.stdout.write(JSON.stringify(FEATURE_FIT.poses.map(f=>({name:f.name,triangles:mesh.triangles,
 positions:skinFillPositions({...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:f.yaw,ParamAngleY:f.pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,[FRAME_CACHE]:new Map()})}))));
