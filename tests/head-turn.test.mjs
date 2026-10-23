import {test} from 'node:test';
import assert from 'node:assert/strict';
import {rotate, lift, eyePoint, project, residualAt, deform, bakeResidual, attachNeck, liftEllipsoid} from '../src/head-turn.mjs';
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-9,`${a} != ${b}`));};
const head={center:[0,0],radius:1};
test('neutral preserves sphere and vertical tape artwork',()=>{
 for(const p of [[0,0],[.5,.2],[.5,2],[2,2]])close(project(lift(p,head),head,[0,0,0]),p);
});
test('yaw is a rigid rotation with correct depth sign',()=>{
 close(rotate([0,0,1],[90,0,0]),[1,0,0]);close(rotate([1,0,0],[90,0,0]),[0,0,-1]);
 const q=rotate([.3,.8,2],[25,-20,17]);assert.ok(Math.abs(Math.hypot(...q)-Math.hypot(.3,.8,2))<1e-12);
});
test('pitch and roll follow image y-down coordinates',()=>{
 close(rotate([0,0,1],[0,90,0]),[0,-1,0]);close(rotate([1,0,0],[0,0,90]),[0,1,0]);
});
test('tape rotates its extension, not a screen-space offset',()=>{
 close(project(lift([0,2],head),head,[0,60,0]),[0,1]);
});
test('eye child rotates locally then around the head',()=>{
 const q=eyePoint([0,0,.2],[.4,.1,.5],[90,0,0],[0,0,90]);close(q,[-.1,.6,.5]);
});
const field=Array.from({length:3},(_,y)=>Array.from({length:3},(_,x)=>[x-1,2*(y-1)]));
test('residual grid interpolates all quadrants and knots',()=>{
 close(residualAt(field,.5,-.25),[.5,-.5]);close(residualAt(field,-1,1),[-1,2]);
});
test('joint residual vanishes on axes but survives in the corner',()=>{
 const joint=field.map((row,y)=>row.map((_,x)=>[(x-1)*(y-1),0]));
 close(residualAt(joint,.5,0),[0,0]);close(residualAt(joint,.5,.5),[.25,0]);
});
test('bake target minus 3D baseline and recover every authored knot',()=>{
 const p=[.2,.1],targets=field.map((row,y)=>row.map((v,x)=>[p[0]+v[0],p[1]+v[1]]));
 const r=bakeResidual(p,head,targets,30);
 for(let y=0;y<3;y++)for(let x=0;x<3;x++)close(deform(p,head,[(x-1)*30,(y-1)*30,0],r,30),targets[y][x]);
});
test('roll rotates the XY correction exactly once',()=>{
 const r=field.map(row=>row.map(()=>[.2,0]));
 close(deform([0,0],head,[0,0,90],r,30),[0,.2]);
});
test('reject malformed authoring inputs and out-of-range poses',()=>{
 assert.throws(()=>lift([0,0],{center:[0,0],radius:0}));
 assert.throws(()=>residualAt(field,1.1,0));
 assert.throws(()=>bakeResidual([0,0],head,[[[0,0]]],30));
});

test('rear hemisphere and neck endpoints have separate attachments',()=>{
  close(lift([0,0],{...head,depthSign:-1}),[0,0,-1]);
  close(attachNeck([0,2],[.2,-.1],0),[0,2]);
  close(attachNeck([0,1],[.2,-.1],1),[.2,.9]);
});

test('ellipsoid is shallow independently of artwork width and height',()=>{
 const h={center:[0,0],radii:[2,3,.5],depthSign:1};
 close(liftEllipsoid([0,0],h),[0,0,.5]);
 close(liftEllipsoid([2,0],h),[2,0,0]);
 const targets=field.map((row,y)=>row.map((_,x)=>[.2+(x-1)*.1,.1+(y-1)*.1]));
 const r=bakeResidual([.2,.1],h,targets,30,liftEllipsoid);
 close(deform([.2,.1],h,[30,30,0],r,30,liftEllipsoid),targets[2][2]);
});
