// The existing r11 child-eye pivots anchor reference camera translation.
import {readFileSync} from 'node:fs';
import {CORRECTION_PRESETS,eyeRootWorld,artWorld,correctedLift} from '../bundle/geometry.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
const row=trace.rows.find(r=>r[0]===465);
const half=(row[2]-row[1])/2;
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const poses=[];
for(const pitch of [-30,30])for(const yaw of [-45,0,45]){
 const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},eyes=['left','right'].map(side=>eyeRootWorld(side,p));
 const topEdgeY=[502-half,502+half].map(x=>artWorld(correctedLift([x,465],0,p),p)[1]);
 poses.push({yaw,pitch,eyeMid:[0,1].map(i=>(eyes[0][i]+eyes[1][i])/2),topEdgeY,topY:(topEdgeY[0]+topEdgeY[1])/2});
}
process.stdout.write(JSON.stringify(poses));
