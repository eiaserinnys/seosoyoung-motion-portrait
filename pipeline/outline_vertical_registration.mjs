// The yaw-zero 3D silhouette owns vertical fractions for every yaw keyform.
import {readFileSync} from 'node:fs';
import {CORRECTION_PRESETS,eyeRootWorld} from '../bundle/geometry.mjs';
import {faceWorld,outlineRow} from '../bundle/face-outline.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const rows=JSON.parse(readFileSync(0,'utf8'));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:0,ParamAngleZ:0};
const result={};
for(const pitch of [-30,30]){
 const p={...base,ParamAngleY:pitch};
 const y3d=(y,edges=outlineRow(y).slice(0,2))=>edges.reduce((sum,x)=>sum+faceWorld([x,y],0,p)[1]/2,0);
 const top=y3d(431.5),span=y3d(616)-top;
 const eyeMidY=['left','right'].reduce((sum,side)=>sum+eyeRootWorld(side,p)[1]/2,0);
 result[pitch]={s3d:rows.map(([y,l,r])=>(y3d(y,[l,r])-top)/span),
  rimOffset:top-eyeMidY,sFloor:pitch>0?(y3d(440)-top)/span:0};
}
process.stdout.write(JSON.stringify(result));
