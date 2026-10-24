import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {artWorld,CORRECTION_PRESETS,correctedLift} from '../bundle/geometry.mjs';
import {hairLift,fringeSurface,canopyWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {beadWorld,beadRotation} from '../bundle/stereo-attachments.mjs';
import {domeSection,craniumPosition,pitchHairDepth} from '../bundle/hair-surface.mjs';
import {DOME} from '../bundle/dome-data.mjs';
import {measureHairBounds} from './hair_bounds.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const near=(a,b)=>assert(Math.max(...a.map((v,i)=>Math.abs(v-b[i])))<1e-6,`${a} != ${b}`);
test('runtime dome data matches its canonical source',()=>{
 assert.deepEqual(DOME,JSON.parse(readFileSync(new URL('../source/viewer/dome-fit.json',import.meta.url))));
 const config=JSON.parse(readFileSync(new URL('../config/rig-config.json',import.meta.url)));
 for(const key of ['pitchArc','tipLag','rearPitch'])assert.deepEqual(DOME[key],config[key]);
 const upperRight=config.rearPitch.poses.find(p=>p.name==='up-right');
 assert.deepEqual([upperRight.rx,upperRight.rz,upperRight.y],[0,0,0]);
});
test('H1 every pitched fringe X/Z stays on the yaw dome, including diagonals',{timeout:60000},()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,-15,15,30])for(const y of [8,70,150,230,310,400])for(const x of [300,400,502,600,700]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},a=artWorld(hairLift([x,y],p),p),q=fringeSurface([x,y],p);
  near([q[0],q[2]],[a[0],a[2]]);
 }
});
test('H4 both bead roots remain in the yaw-only gravity frame at every pitch',{timeout:60000},()=>{
 for(const yaw of [-45,0,45])for(let pitch=-30;pitch<=30;pitch++)for(const side of ['left','right']){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};near(beadWorld(side,p),beadRotation(side,{...p,ParamAngleY:0,ParamAngleZ:0}));
 }
});
test('H2 pitched posterior retains true X/Z and joins the canopy seam',{timeout:60000},()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const sign of [-1,1])for(const y of [80,200,340,465,540]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},side=sign<0?'left':'right',lambda=sign*Math.PI/2,{cx,rx}=domeSection(y);
  const seam=craniumWorld([0,y,lambda],side,p),canopy=canopyWorld([502+cx+sign*rx,y],p);
  near([seam[0],seam[2]],[canopy[0],canopy[2]]);
  const pt=[0,y,sign*130*Math.PI/180],a=craniumWorld(pt,side,p),b=craniumPosition(pt,p);
  near([a[0],a[2]],[b[0],b[2]]);
 }
});
test('H5 the last fringe rows lag vertically without changing X/Z',{timeout:60000},()=>{
 const point=[502,420],p={...base,ParamAngleY:30},local=hairLift(point,p),neutral=artWorld(local,{...p,ParamAngleY:0})[1];
 const raw=artWorld([local[0],local[1],correctedLift(point,0,p)[2]],p)[1];
 const k=(fringeSurface(point,p)[1]-neutral)/(raw-neutral);assert(k>=.35&&k<=.85,`kappa ${k}`);
});
test('H1 every longitude converges to the single top depth at the pole',()=>{
 for(const x of [300,502,536,700])assert(Math.abs(pitchHairDepth([x,8],base)-DOME.pitchArc.topZ)<1e-6);
});
test('H2 posterior supplementation stays within 45px of stage a, including intermediate poses',{timeout:60000},()=>{
 const poses=DOME.rearPitch.poses;
 try{
  for(const yaw of [-45,-22.5,0,22.5,45])for(const pitch of [-30,-15,0,15,30])for(const y of [8,80,200,340,465,540])for(const lambda of [-Math.PI,-2.2,-Math.PI/2,Math.PI/2,2.2,Math.PI]){
   const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};
   DOME.rearPitch.poses=poses;const q=craniumPosition([0,y,lambda],p);
   DOME.rearPitch.poses=[];const a=craniumPosition([0,y,lambda],p);
   assert(Math.hypot(...q.map((v,i)=>v-a[i]))<=45+1e-6);
   if(!pitch||!yaw||Math.abs(lambda)===Math.PI/2)near(q,a);
  }
  DOME.rearPitch.poses=[{yaw:45,pitch:30,rx:1000,rz:1000,y:1000}];
  const p={...base,ParamAngleX:45,ParamAngleY:30},q=craniumPosition([0,300,2.5],p);
  DOME.rearPitch.poses=[];const a=craniumPosition([0,300,2.5],p);
  assert(Math.hypot(...q.map((v,i)=>v-a[i]))<=45+1e-6,'injected balloon is capped');
 }finally{DOME.rearPitch.poses=poses;}
});
test('bounded acceptance rejects a configuration outside the approved scalar limits',{timeout:60000},()=>{
 assert(measureHairBounds().pass);
 const top=DOME.pitchArc.topZ,kappa=DOME.tipLag.kappa;
 try{
  DOME.pitchArc.topZ=-200;assert(!measureHairBounds().pass);
  DOME.pitchArc.topZ=top;DOME.tipLag.kappa=.9;assert(!measureHairBounds().pass);
 }finally{DOME.pitchArc.topZ=top;DOME.tipLag.kappa=kappa;}
});
test('rear seam and hanging locks cross zero pitch with continuous position and tangent',{timeout:60000},()=>{
 for(const yaw of [-45,45])for(const y of [340,350,360,370,380])for(const sign of [-1,1]){
  const side=sign<0?'left':'right',pt=[0,y,sign*Math.PI/2];
  const sample=pitch=>craniumWorld(pt,side,{...base,ParamAngleX:yaw,ParamAngleY:pitch});
  const zero=sample(0),tiny=sample(1e-8);assert(Math.hypot(...zero.map((v,i)=>v-tiny[i]))<1e-4);
  const h=1e-4,left=sample(-h),right=sample(h);
  assert(Math.hypot(...zero.map((v,i)=>(right[i]-2*v+left[i])/h))<.001,'C1 pitch tangent');
 }
 const sample=pitch=>canopyWorld([648,420],{...base,ParamAngleX:-45,ParamAngleY:pitch});
 near(sample(0),sample(1e-8));
});
test('pointwise upper depth preserves convexity and joins each lower-depth column',{timeout:60000},()=>{
 for(const y of [60,120,200,280,309]){
  const center=pitchHairDepth([502,y],base);
  for(const x of [300,330,400,600,670,700])assert(center>=pitchHairDepth([x,y],base)-1e-9,`convex depth ${x},${y}`);
 }
 for(const x of [300,330,400,502,600,670,700]){
  const lower=pitchHairDepth([x,310],base),upper=pitchHairDepth([x,310-1e-6],base);
  assert(Math.abs(lower-upper)<1e-5,`depth step at ${x}: ${lower-upper}`);
 }
});
test('rear horizon excludes raw skull pitch drop and adds only bounded posterior Y',{timeout:60000},()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const sign of [-1,1])for(const y of [80,200,340,360,465]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},side=sign<0?'left':'right',{cx,rx}=domeSection(y),edge=[502+cx+sign*rx,y];
  const t=Math.max(0,Math.min(1,(y-345)/35)),blend=1-(1-Math.cos(t*Math.PI))*.5;
  const fringe=fringeSurface(edge,p),canopy=canopyWorld(edge,p),horizon=fringe[1]+(canopy[1]-fringe[1])*blend;
  for(const deg of [90,100,105,120,150,180]){
   const pt=[0,y,sign*deg*Math.PI/180],raw=craniumPosition(pt,p),noRear=craniumPosition(pt,p,false);
   const actual=craniumWorld(pt,side,p);
   assert(Math.abs(actual[1]-horizon-(raw[1]-noRear[1]))<1e-7,`${yaw},${pitch},${y},${deg}`);
  }
 }
});
