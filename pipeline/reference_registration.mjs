// Approved r11 yaw registration owns X; child-eye rotation supplies only pitch Y.
import {readFileSync} from 'node:fs';
import {CORRECTION_PRESETS,eyeRootWorld} from '../bundle/geometry.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const neutral=JSON.parse(readFileSync(0,'utf8'));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const poses=[];
for(const pitch of [-30,30])for(const yaw of [-45,0,45]){
 const centers=neutral.find(p=>p.yaw===yaw).centers;
 const r11Mid=[0,1].map(i=>(centers.left[i]+centers.right[i])/2);
 const eyeY=angle=>['left','right'].reduce((s,side)=>s+eyeRootWorld(side,{...base,ParamAngleX:yaw,ParamAngleY:angle})[1]/2,0);
 const r11EyeDist=Math.hypot(...centers.right.map((v,i)=>v-centers.left[i]));
 poses.push({yaw,pitch,r11Mid,r11EyeDist,eyeMid:[r11Mid[0],r11Mid[1]+eyeY(pitch)-eyeY(0)]});
}
process.stdout.write(JSON.stringify(poses));
