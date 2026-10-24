// Acceptance for the approved low-DOF hair model. Raster residuals stay in
// the H0 report; this measures the geometric limits that permit bounded pass.
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {DOME} from '../bundle/dome-data.mjs';
import {artWorld,CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {hairLift,fringeSurface} from '../bundle/hair-shell.mjs';
import {craniumPosition,pitchEase,sphereEase} from '../bundle/hair-surface.mjs';
import {beadWorld,beadRotation} from '../bundle/stereo-attachments.mjs';

export function measureHairBounds(){
 const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
 const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
 const checks=[];
 const check=(gate,pass,value)=>checks.push({gate,pass:!!pass,value});
 check('H1 front and diagonal bounded depth scalars',Object.keys(DOME.pitchArc).length===2&&DOME.pitchArc.topZ>=-130&&DOME.pitchArc.topZ<=-80&&DOME.pitchArc.diagonalTopZ>=-35&&DOME.pitchArc.diagonalTopZ<=-5,DOME.pitchArc);
 const tips=[DOME.tipLag.kappa,...DOME.tipLag.poses.map(p=>p.kappa)];
 check('H5 bounded tip coefficients',DOME.tipLag.poses.length<=6&&tips.every(k=>k>=.35&&k<=.85),tips);
 const paired=(poses,fields)=>['up','down'].every(direction=>{
  const a=poses.find(p=>p.name===direction+'-left'),b=poses.find(p=>p.name===direction+'-right');
  return a&&b&&fields.every(key=>a[key]===b[key]);
 });
 check('H5 mirrored diagonal tip coefficients',paired(DOME.tipLag.poses,['kappa']),null);
 check('no affine rows or free front-hair fit',!('pitchRows' in DOME)&&!('pitchPoses' in DOME)&&!('frontHair' in DOME),null);
 const rear=DOME.rearPitch.poses;
 check('H2 at most three rear scalars per diagonal pose',rear.length<=4&&rear.every(p=>p.yaw&&p.pitch&&Object.keys(p).sort().join()===['name','pitch','rx','rz','y','yaw'].sort().join()&&[p.rx,p.rz,p.y].every(Number.isFinite)),rear);
 check('H2 mirrored rear coefficients',paired(rear,['rx','rz','y'])&&rear.every(p=>p.rx>=0&&p.rx<=45&&p.rz>=0&&p.rz<=45&&p.y===0),null);
 let maxXZ=0,maxRear=0,maxBead=0,samples=0;
 for(const yaw of [-45,-22.5,0,22.5,45])for(const pitch of [-30,-15,0,15,30]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};
  for(const y of [8,70,150,230,280,310,420])for(const x of [300,400,502,536,600,700]){
   const a=artWorld(hairLift([x,y],p),p),b=fringeSurface([x,y],p);
   maxXZ=Math.max(maxXZ,Math.abs(a[0]-b[0]),Math.abs(a[2]-b[2]));
  }
  for(const y of [8,40,80,120,200,300,340,400,465,500,540])for(const deg of [-180,-150,-120,-90,90,120,150,180]){
   const point=[0,y,deg*Math.PI/180];
   DOME.rearPitch.poses=rear;const a=craniumPosition(point,p);
   DOME.rearPitch.poses=[];const b=craniumPosition(point,p);
   maxRear=Math.max(maxRear,Math.hypot(...a.map((v,i)=>v-b[i])));samples++;
  }
  DOME.rearPitch.poses=rear;
 }
 for(const yaw of [-45,0,45])for(let pitch=-30;pitch<=30;pitch++)for(const side of ['left','right']){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};
  const a=beadWorld(side,p),b=beadRotation(side,{...p,ParamAngleY:0,ParamAngleZ:0});
  maxBead=Math.max(maxBead,Math.hypot(...a.map((v,i)=>v-b[i])));
 }
 check('H1 yaw dome X/Z invariant',maxXZ<=1e-6,maxXZ);
 check('H2 rear XYZ and projected shift cap 45px',maxRear<=45+1e-6,{maximumSampled: maxRear,samples,limit:45});
 check('H4 gravity-frame beads <=0.5px',maxBead<=.5,maxBead);
 check('H6 spherical pitch activation',[.1,.25,.5,.75,.9].every(t=>{
  const smooth=t*t*(3-2*t),expected=smooth+(sphereEase(t)-smooth)*sphereEase(.5);
  return pitchEase(t,{...base,ParamAngleY:15})===expected;
 }),null);
 return {pass:checks.every(c=>c.pass),checks};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)console.log(JSON.stringify(measureHairBounds()));
