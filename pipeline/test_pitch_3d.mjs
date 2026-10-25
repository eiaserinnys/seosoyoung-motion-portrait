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
import {domeSection,craniumPosition,sphereEase,pitchEase} from '../bundle/hair-surface.mjs';
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
test('3D pitch face preserves neutral projection and uses yaw-zero vertical fractions',()=>{
 const traces=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 for(const sample of traces.poses.filter(q=>q.pitch)){
  assert.deepEqual(sample.mapping.restY,[431.5,616]);
  const feature=FEATURE_FIT.poses.find(p=>p.name===sample.name);
  for(const side of ['left','right'])assert.equal(sample.mapping.rawYSpan[side][0],feature.rawCenters[side][1]+sample.mapping.rimOffset/feature.registration.scale);
  const reg=feature.registration,chin=reg.eyeMid[1]+(sample.rawYChin-reg.referenceEyeMid[1])*reg.scale;
  assert(Math.abs(sample.mapping.canvasYSpan.left[1]-chin)<1e-8);
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
test('right hanging lock tip retains Z and gravity-frame XY under upward left yaw',()=>{
 const point=[648,544],p=pose(-45,30),strand=strandWorld(point,'right',p),shift=outlinePitchShift(point,p);
 const join=1-sphereEase(22/30),world=canopyWorld(point,p);
 // The s3d near edge can have zero X residual after the r11 width bound.
 // A nonzero skin Y shift makes this a direct regression for M6's Y decoupling.
 assert(Math.abs(shift[1]*join)>1);
 assert.equal(world[0],canopyWorld(point,pose(-45,0))[0]);
 assert.equal(world[1],canopyWorld(point,pose(-45,0))[1]);
 assert.equal(world[2],strand[2]);
});
test('face mesh follows its alpha silhouette instead of a coarse rectangle',()=>{
 const mesh=rig.meshes.find(m=>m.id==='face-skin');
 assert.equal(mesh.topology,'alpha-silhouette');assert(mesh.positions.length>132);
});
test('r12 skin vertices retain one-pixel XY and interior queries retain 1.2-pixel chord accuracy',()=>{
 const {cases}=JSON.parse(readFileSync(new URL('./fixtures/r12-yaw-outline.json',import.meta.url)));
 for(const {gain,yaw,point,world} of cases){
  if(point[1]<347)continue; // Outside the valid skin: only continuous hull extension is specified.
  const q=outlineWorld(point,{...pose(yaw,0),CorrTurn:gain});
  const mesh=rig.meshes.find(m=>m.id==='face-skin');
  const isVertex=mesh.positions.some(pt=>pt[0]===point[0]&&pt[1]===point[1]);
  assert(Math.hypot(q[0]-world[0],q[1]-world[1])<=(isVertex?1:1.2),`${gain},${yaw},${point}`);
  assert(Number.isFinite(q[2]),'optimized depth is finite');
 }
 for(const gain of [.65,1])for(const yaw of [-45,45])for(const y of [350,420,465,510,550,590]){
  const row=cases.filter(c=>c.gain===gain&&c.yaw===yaw&&c.point[1]===y).sort((a,b)=>a.point[0]-b.point[0]);
  const depths=row.map(c=>outlineWorld(c.point,{...pose(yaw,0),CorrTurn:gain})[2]);
  for(let i=1;i<row.length;i++)assert((depths[i]-depths[i-1])*(row[i].world[2]-row[i-1].world[2])>0,'yaw depth ordering');
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
   const up=pose(yaw,30),mid=canopyWorld(center,up);
   const sides=[-1,1].map(sign=>canopyWorld([502+cx+sign*rx,y],up)[1]);
   assert(mid[1]<=Math.max(...sides)+1e-8,`upward crown depression ${yaw},${y}`);
  }
 }
});
test('rear shell uses the front horizon plus only posterior pitch displacement',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const sign of [-1,1])for(const deg of [90,120,150,180]){
  const p=pose(yaw,pitch),side=sign<0?'left':'right';let previous=-Infinity;
  for(const y of [60,80,120,200,300,340,350,360,370,380,400]){
   const pt=[0,y,sign*deg*Math.PI/180],q=craniumWorld(pt,side,p),{cx,rx}=domeSection(y),edge=[502+cx+sign*rx,y];
   const a=hairShell.fringeSurface(edge,p),b=canopyWorld(edge,p),blend=1-pitchEase((y-345)/35,p);
   const delta=craniumPosition(pt,p)[1]-craniumPosition(pt,p,false)[1];
   assert(Math.abs(q[1]-a[1]-(b[1]-a[1])*blend-delta)<1e-8);
   assert(q.every(Number.isFinite));previous=q[1];
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
  for(const y of [120,300,400])for(const deg of [105,135,150,180]){
   const pt=[0,y,sign*deg*Math.PI/180],a=craniumWorld(pt,side,p),b=craniumPosition(pt,p);
   for(const i of [0,2])assert(Math.abs(a[i]-b[i])<1e-9,`volume ${yaw},${pitch},${y},${deg}`);
  }
 }
});
test('landmark-mapped jaw uses the actual registered chin without a synthetic lip span',()=>{
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
  const reg=featurePose.registration,actualChin=reg.eyeMid[1]+(target.rawYChin-reg.referenceEyeMid[1])*reg.scale;
  const mirror=traces.poses.find(q=>q.yaw===-yaw&&q.pitch===pitch),other=FEATURE_FIT.poses.find(q=>q.yaw===-yaw&&q.pitch===pitch).registration;
  const mirrorChin=other.eyeMid[1]+(mirror.rawYChin-other.referenceEyeMid[1])*other.scale;
  assert(Math.abs(target.yChin-actualChin)<1e-7,`registered actual chin ${yaw},${pitch}`);
  assert(Math.abs(lip[1]-featurePose.centers.mouth[1])<1e-7,`registered mouth ${yaw},${pitch}`);
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
test('rear Y stays on the shared crown horizon throughout the former collar',()=>{
 for(const yaw of [-45,45])for(const sign of [-1,1])for(const y of [60,120,200,300,340]){
  const p=pose(yaw,-30),{cx,rx}=domeSection(y),seam=canopyWorld([502+cx+sign*rx,y],p);
  for(const deg of [100,120,150,180]){
   const q=craniumWorld([0,y,sign*deg*Math.PI/180],sign<0?'left':'right',p);
   const pt=[0,y,sign*deg*Math.PI/180],raw=craniumPosition(pt,p),base=craniumPosition(pt,p,false);
   assert(Math.abs(q[1]-(seam[1]+raw[1]-base[1]))<1e-9,`crown Y ${yaw},${sign},${y},${deg}`);
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
test('socket uses measured opening height with spherical diagonal tilt',()=>{
 for(const side of ['left','right']){
  const id=`eye-${side}-sclera`,[x,y]=FEATURE_FIT.landmarks[side].front;
  const height=p=>featureWorld(id,[x,y+25],p)[1]-featureWorld(id,[x,y-25],p)[1];
  for(const pitch of [-30,30]){
   const ratio=height(pose(0,pitch))/height(base),fit=FEATURE_FIT.poses.find(p=>p.yaw===0&&p.pitch===pitch);
   assert(Math.abs(ratio-fit.scales[side][1])<1e-8,`measured socket height ${pitch}: ${ratio}`);
  }
  for(const yaw of [-45,45])for(const pitch of [-30,30]){
   const p=pose(yaw,pitch),a=featureWorld(id,[x-40,y],p),b=featureWorld(id,[x+40,y],p);
   const expected=(b[0]-a[0])*Math.sin(yaw*Math.PI/180)*Math.sin(pitch*Math.PI/180)*.32;
   assert(Math.abs(b[1]-a[1]-expected)<1e-8,`spherical socket tilt ${side},${yaw},${pitch}`);
  }
 }
});
test('iris rotates inside its socket with head pitch and yaw and retains independent gaze',()=>{
 for(const side of ['left','right']){
  const [x,y]=FEATURE_FIT.landmarks[side].front,id=`eye-${side}-iris`,socket=`eye-${side}-sclera`;
  const offset=p=>featureWorld(id,[x,y],p).map((v,i)=>v-featureWorld(socket,[x,y],p)[i]);
  for(const pitch of [-30,30]){
   const s=pitch/30,vScale=1-.08*s*s-.04*Math.max(0,-s);
   const expected=-11*Math.sin(pitch*Math.PI/180*.85)*vScale;
   assert(Math.abs(offset(pose(0,pitch))[1]-expected)<1e-8,'corneal dome survives the canonical socket origin');
  }
  for(const yaw of [-45,45]){
   assert(Math.abs(offset(pose(yaw,0))[0])<1e-6,'iris keeps its registered horizontal center');
   const width=(id,p)=>featureWorld(id,[x+20,y],p)[0]-featureWorld(id,[x-20,y],p)[0];
   assert(Math.abs(width(id,pose(yaw,0))/width(socket,pose(yaw,0))-Math.cos(yaw*Math.PI/180*.28))<1e-8,'corneal width foreshortens around the registered center');
  }
  assert.notDeepEqual(featureWorld(id,[x,y],{...pose(45,30),ParamEyeBallX:1}),featureWorld(id,[x,y],pose(45,30)));
 }
});
test('near upper lash tips stay within the measured T-profile 4.9px silhouette tolerance',()=>{
 const face=rig.meshes.find(m=>m.id==='face-skin');
 const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
 // T1..T6 remove the ear-bound skin expansion; measured maximum is 4.84746px.
 const accept=(distance,label)=>assert(distance<=4.9,`${label}: ${distance}px`);
 accept(4.9,'allowed endpoint');assert.throws(()=>accept(4.900001,'injected protrusion'),assert.AssertionError);
 // Opaque upper-arc samples (source alpha >200), rather than the outer
 // wisps covered by the temple hair in the final composition.
 for(const [yaw,side,point] of [[-45,'right',[639,397]],[45,'left',[376,390]]]){
  const p=pose(yaw,-30),tip=featureWorld(`eye-${side}-upper-lashes`,point,p),skin=face.positions.map(pt=>outlineWorld(pt,p));
  const inside=face.triangles.some(t=>{const [a,b,c]=t.map(i=>skin[i]),s=[cross(a,b,tip),cross(b,c,tip),cross(c,a,tip)];return s.every(v=>v>=-1e-6)||s.every(v=>v<=1e-6);});
  const distance=(a,b)=>{
   const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
   const t=length?Math.max(0,Math.min(1,((tip[0]-a[0])*dx+(tip[1]-a[1])*dy)/length)):0;
   return Math.hypot(tip[0]-a[0]-dx*t,tip[1]-a[1]-dy*t);
  };
  const protrusion=inside?0:Math.min(...face.triangles.flatMap(t=>t.map((n,i)=>distance(skin[n],skin[t[(i+1)%3]]))));
  accept(protrusion,`near lash protrusion ${yaw}`);
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
 const moving={...still,ParamBodyAngleX:bodyX,ParamBreath:breath,[FRAME_CACHE]:new Map()};
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
 const partialMoving={...moving,CorrOutline:.5,[FRAME_CACHE]:new Map()},partialStill={...still,CorrOutline:.5,[FRAME_CACHE]:new Map()};
 const a=deform3D(sclera,point,partialMoving),b=deform3D(sclera,point,partialStill);
 assert(Math.hypot(...a.map((v,i)=>v-b[i]-shift[i]))<1e-8,'partial outline fallback adds body motion exactly once');
});
