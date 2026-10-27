import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {faceSolveAnchorWorld} from '../bundle/face-solve.mjs';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
import {faceSolveAnchorWorld as legacyAnchor} from '../bundle/profile-baseline/face-solve.mjs';
import {FRAME_CACHE as LEGACY_CACHE} from '../bundle/profile-baseline/pose-grid.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(q=>[q.id,q.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0};
test('legacy anchor contract preserves all six named anchors in absolute profile poses',{timeout:60000},()=>{
 for(const yaw of [-90,-78.75,-67.5,67.5,78.75,90])for(const pitch of Math.abs(yaw)===67.5?[-30,0,30]:[0]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()};
  const q={...base,ParamAngleX:yaw,ParamAngleY:pitch,[LEGACY_CACHE]:new Map()};
  for(const name of Object.keys(FACE_SOLVE.anchors)){
   const actual=faceSolveAnchorWorld(name,p),expected=legacyAnchor(name,q);
   const error=Math.hypot(actual[0]-expected[0],actual[1]-expected[1]);
   assert(error<1e-12,`legacy anchor contract ${yaw}/${pitch}/${name}: ${error}`);
  }
 }
});
