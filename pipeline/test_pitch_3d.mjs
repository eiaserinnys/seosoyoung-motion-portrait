import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift} from '../bundle/geometry.mjs';
import {outlineWorld,outlineRow} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {canopyWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {domeSection,craniumPosition} from '../bundle/hair-surface.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const pose=(yaw,pitch)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch});
test('face mesh follows its alpha silhouette instead of a coarse rectangle',()=>{
 const mesh=rig.meshes.find(m=>m.id==='face-skin');
 assert.equal(mesh.topology,'alpha-silhouette');assert(mesh.positions.length>132);
});
test('r12 yaw-only face XYZ remains unchanged at all sampled rows and both gains',()=>{
 const {cases}=JSON.parse(readFileSync(new URL('./fixtures/r12-yaw-outline.json',import.meta.url)));
 for(const {gain,yaw,point,world} of cases){
  const q=outlineWorld(point,{...pose(yaw,0),CorrTurn:gain});
  assert(Math.hypot(...q.map((v,i)=>v-world[i]))<1e-9,`${gain},${yaw},${point}`);
 }
});
test('crown has a round volumetric dome rather than a flattened pole ribbon',()=>{
 for(const [y,min] of [[80,85],[120,135],[160,165],[200,180]]){
  assert(canopyWorld([502,y],base)[2]>min,`crown y=${y}`);
 }
});
test('upper canopy meets longitude 90 exactly while rear interior retains full volume',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const sign of [-1,1]){
  const p=pose(yaw,pitch),side=sign<0?'left':'right';
  for(const y of [80,120,200,300,340]){
   const {cx,rx}=domeSection(y),a=canopyWorld([502+cx+sign*rx,y],p),b=craniumWorld([0,y,sign*Math.PI/2],side,p);
   assert(Math.hypot(...a.map((v,i)=>v-b[i]))<1e-9,`seam ${yaw},${pitch},${y}`);
  }
  for(const y of [120,300,400,465])for(const deg of [100,135,150,180]){
   const pt=[0,y,sign*deg*Math.PI/180],a=craniumWorld(pt,side,p),b=craniumPosition(pt,p);
   for(const i of [0,2])assert(Math.abs(a[i]-b[i])<1e-9,`volume ${yaw},${pitch},${y},${deg}`);
  }
 }
});
test('pitch raises and lowers the chin with U/V jaw perspective instead of stretching the face',()=>{
 const neutral=outlineWorld([502,616],base)[1];
 const jaw=p=>{const [l]=outlineRow(550);return outlineWorld([502,616],p)[1]-outlineWorld([l,550],p)[1];};
 const up=pose(0,30),down=pose(0,-30);
 assert(outlineWorld([502,616],up)[1]<neutral-45,'up chin rises');
 assert(outlineWorld([502,616],down)[1]>neutral+20,'down chin descends');
 assert(jaw(up)<45,'up jaw becomes round U');assert(jaw(down)>80,'down jaw becomes V');
});
test('diagonal face geometry is continuous through zero pitch',()=>{
 for(const yaw of [-45,45])for(const y of [420,465,510,550,590,616])for(const x of [...outlineRow(y).slice(0,2),502]){
  const a=outlineWorld([x,y],pose(yaw,-.000001)),b=outlineWorld([x,y],pose(yaw,.000001));
  assert(Math.hypot(...a.map((v,i)=>v-b[i]))<.0001,`${yaw},${x},${y}`);
 }
});
test('central fringe remains in front of the forehead without a temple indentation',()=>{
 for(const y of [245,280,310,340]){
  const {cx,rx}=domeSection(y);
  for(const s of [-.7,-.4,0,.4,.7]){
   const x=502+cx+s*rx,hair=canopyWorld([x,y],base);
   // Compare the unrotated authored surfaces at their shared canvas point.
   const face=correctedLift([x,Math.min(y,350)],0,base);
   assert(hair[2]>=face[2]+6-1e-9,`${x},${y}`);
  }
 }
});
test('socket shortens vertically and tilts along the diagonal latitude',()=>{
 for(const side of ['left','right']){
  const id=`eye-${side}-sclera`,[x,y]=FEATURE_FIT.landmarks[side].front;
  const height=p=>featureWorld(id,[x,y+25],p)[1]-featureWorld(id,[x,y-25],p)[1];
  assert(height(pose(0,30))<height(base)*.96);assert(height(pose(0,-30))<height(base)*.8);
  for(const yaw of [-45,45])for(const pitch of [-30,30]){
   const p=pose(yaw,pitch),dy=featureWorld(id,[x+40,y],p)[1]-featureWorld(id,[x-40,y],p)[1];
   assert(Math.abs(dy)>10,`socket slope ${side},${yaw},${pitch}`);
   assert(Math.sign(dy)===Math.sign(yaw*pitch));
  }
 }
});
test('iris rotates inside its socket with head pitch and yaw and retains independent gaze',()=>{
 for(const side of ['left','right']){
  const [x,y]=FEATURE_FIT.landmarks[side].front,id=`eye-${side}-iris`,socket=`eye-${side}-sclera`;
  const offset=p=>featureWorld(id,[x,y],p).map((v,i)=>v-featureWorld(socket,[x,y],p)[i]);
  assert(offset(pose(0,30))[1]<-2,'up iris rolls up');
  assert(offset(pose(0,-30))[1]>2,'down iris rolls down');
  assert(offset(pose(-45,0))[0]<-1,'left iris rolls left');
  assert(offset(pose(45,0))[0]>1,'right iris rolls right');
  assert.notDeepEqual(featureWorld(id,[x,y],{...pose(45,30),ParamEyeBallX:1}),featureWorld(id,[x,y],pose(45,30)));
 }
});
test('neck top tracks chin pitch displacement while body attachment stays fixed',()=>{
 const mesh=rig.meshes.find(m=>m.id==='neck');
 for(const pitch of [-30,30]){
  const p=pose(0,pitch),dy=outlineWorld([502,616],p)[1]-616;
  const neutral=deform3D(mesh,[502,585],base),q=deform3D(mesh,[502,585],p);
  assert(Math.abs(q[1]-neutral[1]-dy)<1e-8);
  assert.deepEqual(deform3D(mesh,[502,700],p),deform3D(mesh,[502,700],base));
 }
});
