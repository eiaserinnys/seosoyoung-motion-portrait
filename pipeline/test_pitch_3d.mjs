import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift,artWorld} from '../bundle/geometry.mjs';
import {outlineWorld,outlineRow,outlinePitchShift,faceLift,faceWorld} from '../bundle/face-outline.mjs';
import {faceDepthAt,faceDepthEdgeAt} from '../bundle/face-depth.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {canopyWorld,craniumWorld,hairLift} from '../bundle/hair-shell.mjs';
import * as hairShell from '../bundle/hair-shell.mjs';
import {domeSection,craniumPosition} from '../bundle/hair-surface.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {strandWorld} from '../bundle/side-strands.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const pose=(yaw,pitch)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()});
const smooth=value=>{const t=Math.max(0,Math.min(1,value));return t*t*(3-2*t);};
test('diagonal cheek edge heights follow the independently mapped left and right eyes',()=>{
 assert.equal(CORRECTION_PRESETS.recommended.CorrDiagonal,.65);
 for(const yaw of [-45,45])for(const pitch of [-30,30]){
  const p=pose(yaw,pitch),[l,r]=outlineRow(465);
  const dy=outlineWorld([r,465],p)[1]-outlineWorld([l,465],p)[1];
  const fit=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
  const row=fit.poses.find(p=>p.yaw===yaw&&p.pitch===pitch).rows.find(r=>r[0]===465);
  assert(Math.abs(dy-(row[3]-row[1]))<1e-7,`diagonal cheek slope ${yaw},${pitch}: ${dy}`);
 }
});
test('depth surface keeps exact outline edge depths and a smooth convex interior',()=>{
 for(const y of [465,510,530,570,616]){
  const [l,r]=outlineRow(y),offset=(l+r)/2-502,edge=faceDepthEdgeAt(y);
  assert(Math.abs(faceDepthAt(l-offset,y)-edge)<1e-8,`left registered edge depth at ${y}`);
  assert(Math.abs(faceDepthAt(r-offset,y)-edge)<1e-8,`right registered edge depth at ${y}`);
  if(y<616)assert(faceDepthAt(502,y)>edge,`convex face interior at ${y}`);
 }
 assert(Math.abs(faceDepthAt(502,616)-faceDepthEdgeAt(616))<1e-8,'chin center joins the measured edge depth');
 for(const y of [465,530,570,610])for(const x of [352,424,480,500]){
  assert(Math.abs(faceDepthAt(x,y)-faceDepthAt(1004-x,y))<1e-8,`depth symmetry at ${x},${y}`);
 }
});
test('3D pitch face preserves neutral projection and uses four-landmark reference maps',()=>{
 const traces=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 for(const sample of traces.poses.filter(q=>q.pitch)){
  assert.deepEqual(sample.mapping.restY,[431.5,497,554.5,616]);
  const feature=FEATURE_FIT.poses.find(p=>p.name===sample.name);
  assert.equal(sample.mapping.rawYLeft[0],feature.rawCenters.left[1]);
  assert.equal(sample.mapping.rawYRight[0],feature.rawCenters.right[1]);
  assert.equal(sample.mapping.canvasYLeft[1],feature.centers.nose[1]+(sample.pitch>0?-6:4));
  assert.equal(sample.mapping.canvasYLeft[2],feature.centers.mouth[1]+(sample.pitch>0?-12:6));
  assert(Math.abs(sample.mapping.canvasYLeft[3]-sample.mapping.canvasYLeft[2]-(sample.pitch>0?64:50))<1e-8);
 }
 for(const y of [465,530,616]){
  const edge=faceDepthEdgeAt(y),depthScale=1+(base.CorrDepth/.78-1)*smooth((y-465)/151),expected=edge+(faceDepthAt(502,y)-edge)*depthScale;
  assert(Math.abs(faceLift([502,y],0,base)[2]-expected)<1e-7,`depth grid drives faceLift at ${y}`);
 }
 const cheekDepth=correctedLift([502,465],0,base)[2];
 assert(Math.abs(faceDepthAt(502,465)-cheekDepth)<.01,'depth prior meets the existing upper-face center at y=465');
 assert(Math.abs(faceLift([502,465],0,base)[2]-cheekDepth)<1e-8,'faceLift has no centerline depth cliff at y=465');
 assert.equal(faceLift([502,400],0,base)[2],correctedLift([502,400],0,base)[2],'y=400 retains the original surface');
 for(const yaw of [-45,0,45])for(const point of [[502,465],[502,550],[502,616]]){
  const p=pose(yaw,0),expected=artWorld(correctedLift(point,0,p),p),actual=faceWorld(point,0,p);
  assert(Math.hypot(actual[0]-expected[0],actual[1]-expected[1])<1e-9,`neutral pitch projection ${yaw},${point}`);
 }
});
test('right hanging lock tip follows its face attachment under upward left yaw',()=>{
 const point=[648,544],p=pose(-45,30),strand=strandWorld(point,'right',p),shift=outlinePitchShift(point,p);
 const t=22/30,join=1-t*t*(3-2*t),world=canopyWorld(point,p);
 assert(Math.abs(shift[0]*join)>1e-9);
 assert(Math.abs(world[0]-strand[0]-shift[0]*join)<1e-8);
 assert.equal(world[1],strand[1]);
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
test('pitch crown stays convex while yaw-only dome projection is unchanged',()=>{
 for(const yaw of [-45,0,45])for(const y of [60,80,100,120,200,300]){
  const {cx,rx}=domeSection(y),center=[502+cx,y],p=pose(yaw,0);
  assert.deepEqual(hairShell.fringeSurface(center,p),artWorld(hairLift(center,p),p));
  if(y<=120){
   const down=pose(yaw,-30),mid=canopyWorld(center,down);
   const sides=[-1,1].map(sign=>canopyWorld([502+cx+sign*rx,y],down)[1]);
   // The frontal arc has at most 6px local curvature, replacing the old
   // 60–100px depression. Diagonals stay above the far-side horizon.
   assert(mid[1]<=Math.max(...sides)+(yaw?0:6),`crown depression ${yaw},${y}`);
  }
 }
});
test('rear shell horizon increases continuously through the lower crown',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const sign of [-1,1])for(const deg of [90,120,150,180]){
  const p=pose(yaw,pitch),side=sign<0?'left':'right';let previous=-Infinity;
  for(let y=60;y<=465;y++){
   const q=craniumWorld([0,y,sign*deg*Math.PI/180],side,p);
   assert(q[1]>=previous-1e-8,`rear horizon inversion ${yaw},${pitch},${sign},${deg},${y}`);previous=q[1];
   if(y>=380){const {cx,rx}=domeSection(y);assert(Math.abs(q[1]-hairShell.fringeSurface([502+cx+sign*rx,y],p)[1])<1e-8);}
  }
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
test('landmark-mapped jaw preserves the specified mouth-to-chin space',()=>{
 const mouth=[504.5,554.5];
 const traces=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 for(const yaw of [-45,0,45])for(const pitch of [-30,30]){
  const p=pose(yaw,pitch),chin=outlineWorld([502.5,616],p),lip=featureWorld('mouth-detail',mouth,p);
  const target=traces.poses.find(q=>q.yaw===yaw&&q.pitch===pitch);
  const featurePose=FEATURE_FIT.poses.find(q=>q.yaw===yaw&&q.pitch===pitch);
  // Measured endpoints are exact; the interior keeps its 3D curvature (as in
  // the approved yaw row), rather than adding an unrequested chin-center pin.
  for(const x of outlineRow(616).slice(0,2))assert(Math.abs(outlineWorld([x,616],p)[1]-target.yChin)<1e-7);
  assert(Math.abs(chin[1]-target.yChin)<1,`chin retains the endpoint height within its curved interior ${yaw},${pitch}`);
  assert(Math.abs(target.yChin-featurePose.centers.mouth[1]-(pitch>0?52:56))<1e-7,`registered mouth-to-chin space ${yaw},${pitch}`);
  assert(chin[1]-lip[1]>45&&chin[1]-lip[1]<85,`rendered mouth clearance ${yaw},${pitch}: ${chin[1]-lip[1]}`);
 }
 const up=pose(0,30),jaw=outlineWorld([outlineRow(550)[0],550],up),chin=outlineWorld([502.5,616],up);
 assert(chin[1]-jaw[1]>35,'low-angle jaw keeps curvature instead of a horizontal brick');
});
test('downward pitch keeps the central cheek free of projected face-mask folds',()=>{
 const face=rig.meshes.find(m=>m.id==='face-skin');
 const area=points=>(points[1][0]-points[0][0])*(points[2][1]-points[0][1])-(points[1][1]-points[0][1])*(points[2][0]-points[0][0]);
 for(const yaw of [-45,0,45]){
  const p=pose(yaw,-30),world=face.positions.map(point=>deform3D(face,point,p));
  let reversed=0;
  for(const triangle of face.triangles){
   const source=triangle.map(i=>face.positions[i]),centerX=(source[0][0]+source[1][0]+source[2][0])/3,centerY=(source[0][1]+source[1][1]+source[2][1])/3;
   if(centerY<430||centerY>550||Math.abs(centerX-502)>=130)continue;
   if(area(triangle.map(i=>world[i]))*area(source)<0)reversed++;
  }
  assert.equal(reversed,0,`central cheek projected folds at yaw=${yaw}, pitch=-30`);
 }
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
   assert(Math.sign(dy)===Math.sign(yaw*pitch));
  }
 }
});
test('iris rotates inside its socket with head pitch and yaw and retains independent gaze',()=>{
 for(const side of ['left','right']){
  const [x,y]=FEATURE_FIT.landmarks[side].front,id=`eye-${side}-iris`,socket=`eye-${side}-sclera`;
  const offset=p=>featureWorld(id,[x,y],p).map((v,i)=>v-featureWorld(socket,[x,y],p)[i]);
  assert(offset(pose(0,30))[1]<-3.5,'up iris rolls up relative to its arched socket');
  assert(offset(pose(0,-30))[1]>3.5,'down iris rolls down relative to its arched socket');
  for(const yaw of [-45,45]){
   assert(Math.abs(offset(pose(yaw,0))[0])<1e-6,'iris keeps its registered horizontal center');
   const width=(id,p)=>featureWorld(id,[x+20,y],p)[0]-featureWorld(id,[x-20,y],p)[0];
   assert(Math.abs(width(id,pose(yaw,0))/width(socket,pose(yaw,0))-Math.cos(yaw*Math.PI/180*.28))<1e-8,'corneal width foreshortens around the registered center');
  }
  assert.notDeepEqual(featureWorld(id,[x,y],{...pose(45,30),ParamEyeBallX:1}),featureWorld(id,[x,y],pose(45,30)));
 }
});
test('near upper lash stays inside the face skin under downward diagonals',()=>{
 const face=rig.meshes.find(m=>m.id==='face-skin');
 const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
 // Opaque upper-arc samples (source alpha >200), rather than the outer
 // wisps covered by the temple hair in the final composition.
 for(const [yaw,side,point] of [[-45,'right',[639,397]],[45,'left',[376,390]]]){
  const p=pose(yaw,-30),tip=featureWorld(`eye-${side}-upper-lashes`,point,p),skin=face.positions.map(pt=>outlineWorld(pt,p));
  assert(face.triangles.some(t=>{const [a,b,c]=t.map(i=>skin[i]),s=[cross(a,b,tip),cross(b,c,tip),cross(c,a,tip)];return s.every(v=>v>=-1e-6)||s.every(v=>v<=1e-6);}),`near lash outside skin ${yaw}`);
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
test('head meshes and the upper neck inherit one continuous breath and body sway offset',()=>{
 const bodyX=5,breath=.75,shift=[bodyX,-breath*4,0];
 const still={...pose(0,0),ParamBodyAngleX:0,ParamBreath:0};
 const moving={...still,ParamBodyAngleX:bodyX,ParamBreath:breath};
 const assertShift=(mesh,point,label)=>{
  const a=deform3D(mesh,point,moving),b=deform3D(mesh,point,still);
  assert(Math.hypot(...a.map((v,i)=>v-b[i]-shift[i]))<1e-8,label);
 };
 for(const [id,points] of [['face-skin',[[502,465],[435,300],[650,560]]],['hair-top',[[502,120],[390,245],[610,300]]]]){
  const mesh=rig.meshes.find(m=>m.id===id);
  for(const point of points)assertShift(mesh,point,id+' '+point);
 }
 const neck=rig.meshes.find(m=>m.id==='neck'),top=neck.positions.filter(point=>point[1]<=576);
 assert(top.length>0,'neck mesh has upper attachment samples');
 for(const point of top)assertShift(neck,point,'neck '+point);
 const sclera=rig.meshes.find(m=>m.id==='eye-left-sclera'),point=[415,432];
 const partialMoving={...moving,CorrOutline:.5},partialStill={...still,CorrOutline:.5};
 const a=deform3D(sclera,point,partialMoving),b=deform3D(sclera,point,partialStill);
 assert(Math.hypot(...a.map((v,i)=>v-b[i]-shift[i]))<1e-8,'partial outline fallback adds body motion exactly once');
});
