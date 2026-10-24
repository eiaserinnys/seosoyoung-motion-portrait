import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {canopyWorld,craniumWorld,backWorld} from '../bundle/hair-shell.mjs';
import {pitchHairDepth,sphereEase} from '../bundle/hair-surface.mjs';
import {DOME} from '../bundle/dome-data.mjs';
import {STRANDS} from '../bundle/side-strands.mjs';
import {beadWorld,BEADS} from '../bundle/stereo-attachments.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
test('all three hanging hair layers reach gravity-frame XY at y506',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const side of ['left','right']){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},neutral={...p,ParamAngleY:0},sign=side==='left'?-1:1;
  for(const y of [506,520,544])for(const [surface,pt] of [
   [canopyWorld,[side==='left'?300:720,y]],
   [(pt,p)=>craniumWorld(pt,side,p),[0,y,sign*2.3]],
   [backWorld,[side==='left'?300:720,y]],
  ]){
   const a=surface(pt,p),b=surface(pt,neutral);
   assert(Math.abs(a[1]-b[1])<1e-8,`${yaw},${pitch},${side},${y} gravity Y`);
   if(surface!==canopyWorld||yaw*(side==='left'?-1:1)<=0)assert(Math.abs(a[0]-b[0])<1e-8,`${yaw},${pitch},${side},${y} gravity X`);
  }
 }
});
test('upper pole follows the bounded symmetric yaw depth arc',()=>{
 assert.equal(DOME.pitchArc.topZ,-90);
 assert(DOME.pitchArc.diagonalTopZ>=-35&&DOME.pitchArc.diagonalTopZ<=-25);
 for(const yaw of [-45,-22.5,0,22.5,45]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:30};
  const expected=DOME.pitchArc.topZ+(DOME.pitchArc.diagonalTopZ-DOME.pitchArc.topZ)*sphereEase(Math.abs(yaw)/45);
  for(const x of [300,502,700])assert(Math.abs(pitchHairDepth([x,8],p)-expected)<1e-8);
 }
});
test('tip lag and rear supplementation share mirrored key coefficients',()=>{
 for(const direction of ['up','down']){
  const tips=DOME.tipLag.poses.filter(p=>p.name===`${direction}-left`||p.name===`${direction}-right`);
  assert.equal(tips[0].kappa,tips[1].kappa);
  const rear=DOME.rearPitch.poses.filter(p=>p.name===`${direction}-left`||p.name===`${direction}-right`);
  assert.deepEqual(rear.map(p=>[p.rx,p.rz,p.y])[0],rear.map(p=>[p.rx,p.rz,p.y])[1]);
  if(direction==='up')assert.deepEqual([rear[0].rx,rear[0].rz,rear[0].y],[0,0,0]);
 }
});
test('far hanging ribbons span the authored central 60 percent of each tie',()=>{
 for(const side of ['left','right'])for(const pitch of [-30,30]){
  const p={...base,ParamAngleX:side==='left'?-45:45,ParamAngleY:pitch},y=506;
  const [a,b]=STRANDS[side].slice(-2),t=(y-a[0])/(b[0]-a[0]);
  const lo=a[1]+(b[1]-a[1])*t,hi=a[2]+(b[2]-a[2])*t;
  const left=canopyWorld([lo,y],p),right=canopyWorld([hi,y],p);
  const root=beadWorld(side,p)[0]-BEADS[side].front[0];
  const [l,r]=side==='left'?[250,366]:[632,746];
  assert(left[0]<=root+l+(r-l)*.2+1e-8);
  assert(right[0]>=root+r-(r-l)*.2-1e-8);
 }
});
test('far ribbon expansion is continuous with a shared tangent across zero yaw',()=>{
 for(const x of [296.8,358.56,645.08,707.24])for(const pitch of [-30,30]){
  const sample=yaw=>canopyWorld([x,506],{...base,ParamAngleX:yaw,ParamAngleY:pitch});
  const zero=sample(0),h=1e-5,left=sample(-h),right=sample(h);
  assert(Math.hypot(...sample(1e-8).map((v,i)=>v-zero[i]))<1e-5);
  assert(Math.hypot(...zero.map((v,i)=>(right[i]-2*v+left[i])/h))<.001);
 }
});
