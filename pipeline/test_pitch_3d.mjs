import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift} from '../bundle/geometry.mjs';
import {outlineWorld,outlineRow,outlinePitchShift} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {canopyWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {domeSection,craniumPosition} from '../bundle/hair-surface.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {strandWorld} from '../bundle/side-strands.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const pose=(yaw,pitch)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch});
test('diagonal jaw slope stays near the feature plane instead of twisting the lower face',()=>{
 assert.equal(CORRECTION_PRESETS.recommended.CorrDiagonal,.65);
 for(const yaw of [-45,45])for(const pitch of [-30,30]){
  const p=pose(yaw,pitch),[l,r]=outlineRow(465);
  const dy=outlineWorld([r,465],p)[1]-outlineWorld([l,465],p)[1];
  assert(Math.abs(dy)>15&&Math.abs(dy)<25,`diagonal cheek slope ${yaw},${pitch}: ${dy}`);
 }
});
test('pitch jaw widths stay within five percent of approved yaw shapes and converge at the chin',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const y of [465,510,550,590,610,616]){
  const [l,r]=outlineRow(y),neutral=[l,r].map(x=>outlineWorld([x,y],pose(yaw,0))[0]);
  const current=[l,r].map(x=>outlineWorld([x,y],pose(yaw,pitch))[0]),s=(y-465)/151;
  const scale=1+(pitch>0?.05:-.05)*Math.sin(Math.PI*s);
  const shift=yaw?-Math.sign(yaw)*(pitch<0?16:3)*s:0;
  assert(Math.abs((current[1]-current[0])-(neutral[1]-neutral[0])*scale)<1e-8,`width ${yaw},${pitch},${y}`);
  assert(Math.abs((current[0]+current[1]-neutral[0]-neutral[1])/2-shift)<1e-8,`midpoint ${yaw},${pitch},${y}`);
 }
});
test('right hanging lock tip follows its face attachment under upward left yaw',()=>{
 const point=[648,544],p=pose(-45,30),strand=strandWorld(point,'right',p),shift=outlinePitchShift(point,p);
 const t=22/30,join=1-t*t*(3-2*t),world=canopyWorld(point,p);
 assert(Math.hypot(shift[0]*join,shift[1]*join)>1);
 for(const i of [0,1])assert(Math.abs(world[i]-strand[i]-shift[i]*join)<1e-8);
 assert.equal(world[2],strand[2]);
});
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
test('normalized pitch spans preserve a curved jaw and mouth-to-chin space',()=>{
 const mouth=[504.5,554.5];
 for(const yaw of [-45,0,45])for(const pitch of [-30,30]){
  const p=pose(yaw,pitch),chin=outlineWorld([502.5,616],p),lip=featureWorld('mouth-detail',mouth,p);
  assert(chin[1]>540&&chin[1]<640,`chin ${yaw},${pitch}: ${chin[1]}`);
  assert(chin[1]-lip[1]>45&&chin[1]-lip[1]<85,`mouth clearance ${yaw},${pitch}: ${chin[1]-lip[1]}`);
 }
 const up=pose(0,30),jaw=outlineWorld([outlineRow(550)[0],550],up),chin=outlineWorld([502.5,616],up);
 assert(chin[1]-jaw[1]>45,'low-angle jaw keeps curvature instead of a horizontal brick');
});
test('all upper rear longitudes share crown row Y under downward diagonals',()=>{
 for(const yaw of [-45,45])for(const sign of [-1,1])for(const y of [60,120,200,300,340]){
  const p=pose(yaw,-30),{cx,rx}=domeSection(y),seam=canopyWorld([502+cx+sign*rx,y],p);
  for(const deg of [100,120,150,180]){
   const q=craniumWorld([0,y,sign*deg*Math.PI/180],sign<0?'left':'right',p);
   assert(Math.abs(q[1]-seam[1])<1e-9,`crown Y ${yaw},${sign},${y},${deg}`);
  }
 }
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
  for(const pitch of [-30,30]){const ratio=height(pose(0,pitch))/height(base);assert(ratio>.7&&ratio<1.05,`natural socket height ${pitch}: ${ratio}`);}
  for(const yaw of [-45,45])for(const pitch of [-30,30]){
   const p=pose(yaw,pitch),dy=featureWorld(id,[x+40,y],p)[1]-featureWorld(id,[x-40,y],p)[1];
   assert(Math.abs(dy)>5&&Math.abs(dy)<11,`socket slope ${side},${yaw},${pitch}: ${dy}`);
   assert(Math.sign(dy)===-Math.sign(yaw*pitch));
  }
 }
});
test('iris rotates inside its socket with head pitch and yaw and retains independent gaze',()=>{
 for(const side of ['left','right']){
  const [x,y]=FEATURE_FIT.landmarks[side].front,id=`eye-${side}-iris`,socket=`eye-${side}-sclera`;
  const offset=p=>featureWorld(id,[x,y],p).map((v,i)=>v-featureWorld(socket,[x,y],p)[i]);
  assert(offset(pose(0,30))[1]<-.1,'up iris rolls up relative to its arched socket');
  assert(offset(pose(0,-30))[1]>.5,'down iris rolls down relative to its arched socket');
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
