import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {EYE_SAM3 as data} from '../bundle/eye-sam3-data.mjs';
import {eyeDeformation,eyeResidual,gazeAngles,rotateEye,compression} from '../bundle/eye-sam3-model.mjs';
const measurement=JSON.parse(readFileSync(new URL('../source/viewer/sam3-eye-measurements.json',import.meta.url)));
const audit=JSON.parse(readFileSync(new URL('../source/references/seedance/eye-audit-v291.json',import.meta.url))).evaluation;
const near=(a,b,tol=1e-6)=>assert(Math.abs(a-b)<=tol,`${a} != ${b}`);
const vector=(a,b)=>a.forEach((v,i)=>near(v,b[i]));
const report={};

test('corrected delta0 contract preserves neutral centers and every part transform',{timeout:60000},()=>{
 report.front=[];
 for(const side of ['left','right']){
  const rest=data[side].rest,m=eyeDeformation(data,side,[0,0],rest.cO,[0,0]);
  const error=Math.hypot(...m.cI.map((v,i)=>v-rest.cI[i]));
  near(error,0);near(data[side].w0,measurement.K[side].w);vector(data[side].delta0,measurement.rest[side].delta);
  for(const point of [rest.cI,rest.cO,rest.cH,[352,432],[654,472]]){
   vector(m.lid(point,1),point);vector(m.iris(point),point);vector(m.highlight(point),point);
  }
  report.front.push({side,errorPx:error});
 }
});

test('R1: 24 achieved poses reproduce independent audit predictions without beta',{timeout:60000},()=>{
 let maximum=0;
 for(const row of measurement.records){
  const {side,yaw,pitch}=row,b=data[side],f=audit.fit[side];
  const theta=Math.max(-45,Math.min(45,side==='left'?-yaw:yaw)),t=theta*Math.PI/180,p=pitch/30;
  const table=fit=>{
   const c=fit.coefficients;
   let v=0;for(let j=0;j<3;j++)v+=-Math.max(0,Math.min(1,theta/15-j))*c[j]+Math.max(0,Math.min(1,-theta/15-j))*c[j+3];
   return Math.exp(v);
  };
  const poly=c=>c[0]*p+c[1]*p*p+c[2]*t*t+c[3]*t*p;
  const expected=[table(f.dTemp),table(f.dNas),Math.exp(poly(f.h.coefficients)),poly(f.deltaY.coefficients),Math.cos(t)**f.a.coefficients[0],Math.cos(pitch*Math.PI/180)];
  const r=eyeResidual(data,side,[yaw,pitch]);
  const actual=[r.rT,r.rN,Math.exp(r.logH),r.deltaY,Math.exp(r.logA),Math.exp(r.logB)];
  const scales=[b.rest.dTemp,b.rest.dNas,b.rest.h,b.w0,b.rest.a,b.rest.bI];
  actual.forEach((v,i)=>{const error=Math.abs(v-expected[i])*scales[i];maximum=Math.max(maximum,error);assert(error<=.1);});
 }
 report.R1={maxErrorPx:maximum};
});

test('temporal split uses screen-right tau times outward for both eyes',{timeout:60000},()=>{
 for(const side of ['left','right']){
  const b=data[side],r=b.rest,outward=side==='left'?-1:1,phi=r.phi;
  const m=eyeDeformation(data,side,[30,0],r.cO,[0,0]);
  for(const sign of [-1,1]){
   const point=r.cO.map((v,i)=>v+sign*10*[Math.cos(phi),Math.sin(phi)][i]);
   const q=m.lid(point),ratio=sign*outward>0?m.internal.rT:m.internal.rN;
   near(Math.hypot(...q.map((v,i)=>v-r.cO[i])),10*ratio);
  }
 }
});

test('retained gaze rotation and compression contracts',{timeout:60000},()=>{
 for(const z of [.2,.3]){
  const h=1e-7;assert(Math.abs((compression(z)-compression(z-h))/h-(compression(z+h)-compression(z))/h)<1e-5);
 }
 for(const side of ['left','right'])for(const d of [[0,0],[37,0],[-37,20]])for(const input of [[-1,0],[1,0],[0,-1],[0,1]]){
  const [gx,gy]=gazeAngles(data,side,input),[yaw,pitch]=d.map(v=>v*Math.PI/180);
  const x=Math.sin(gx)*Math.cos(gy),y=-Math.sin(gy),z=Math.cos(gx)*Math.cos(gy);
  const expected=[Math.cos(yaw)*x+Math.sin(yaw)*(Math.sin(pitch)*y+Math.cos(pitch)*z)-Math.sin(yaw)*Math.cos(pitch),Math.cos(pitch)*y-Math.sin(pitch)*z+Math.sin(pitch)].map(v=>v*data.rho*data[side].w0);
  const m=eyeDeformation(data,side,d,data[side].rest.cO,input),neutral=eyeDeformation(data,side,d,data[side].rest.cO,[0,0]);
  vector(m.cI.map((v,i)=>v-neutral.cI[i]),expected);
 }
 vector(rotateEye([0,0,1],[0,0]),[0,0,1]);
});

test('save targeted numerical evidence',{timeout:60000},()=>{
 writeFileSync(new URL('../debug/sam3-eye-v291/runtime-contracts.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
});
