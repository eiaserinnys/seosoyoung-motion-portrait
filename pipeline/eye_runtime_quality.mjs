// Independent audit derivatives bound each requested R6 interval in pixel units.
import {HEAD_RADIUS} from '../bundle/geometry.mjs';
const rad=Math.PI/180;
const values=(r,b)=>[b.rest.dTemp*r.rT,b.rest.dNas*r.rN,b.rest.h*Math.exp(r.logH),b.w0*r.deltaY,b.rest.a*Math.exp(r.logA)];
function slopeBounds(f,b,side,yaw0,yaw1,pitch){
 const sign=side==='left'?-1:1,[lo,hi]=[sign*yaw0,sign*yaw1].sort((a,b)=>a-b),mid=(lo+hi)/2,p=pitch/30;
 const table=(fit,base)=>{
  const j=Math.min(5,Math.floor((mid+45)/15)),m=(fit.knotLogs[j+1]-fit.knotLogs[j])/15;
  return Math.max(...[lo,hi].map(t=>base*Math.exp(fit.knotLogs[j]+m*(t-fit.knotsDeg[j]))*Math.abs(m)));
 };
 const h=f.h.coefficients,A=h[2]*rad*rad,B=h[3]*p*rad,C=h[0]*p+h[1]*p*p;
 const candidates=[lo,hi];
 // Extrema of d exp(A theta²+B theta+C)/d theta, inside this interval.
 if(A<0)for(const s of [-1,1]){
  const x=(s*Math.sqrt(-2*A)-B)/(2*A);if(x>lo&&x<hi)candidates.push(x);
 }
 const height=Math.max(...candidates.map(x=>b.rest.h*Math.exp(A*x*x+B*x+C)*Math.abs(2*A*x+B)));
 const d=f.deltaY.coefficients;
 const delta=b.w0*Math.max(...[lo,hi].map(x=>Math.abs(2*d[2]*rad*rad*x+d[3]*p*rad)));
 const k=f.a.coefficients[0];
 const iris=b.rest.a*Math.max(...[lo,hi].map(x=>Math.abs(k*rad*Math.sin(x*rad)*Math.cos(x*rad)**(k-1))));
 return [table(f.dTemp,b.rest.dTemp),table(f.dNas,b.rest.dNas),height,delta,iris];
}
export function continuity(data,audit,evaluate){
 let maxExcess=-Infinity,maxFineStep=0,maxDegreeStep=0,samples=0;const failures=[];
 for(const side of ['left','right'])for(const pitch of [-20,0,20]){
  const b=data[side],f=audit.fit[side];let previous;
  for(let i=0;i<=9000;i++){
   const yaw=-45+i/100,now=values(evaluate(data,side,[yaw,pitch]),b);samples++;
   if(previous){
    const bounds=slopeBounds(f,b,side,yaw-.01,yaw,pitch);
    now.forEach((v,j)=>{
     const step=Math.abs(v-previous[j]),excess=step-bounds[j]*.01-1e-6;
     maxFineStep=Math.max(maxFineStep,step);maxExcess=Math.max(maxExcess,excess);
     if(excess>0)failures.push({side,pitch,yaw,field:j,step,bound:bounds[j]*.01+1e-6});
    });
   }
   previous=now;
  }
  for(let yaw=-44;yaw<=45;yaw++){
   const a=values(evaluate(data,side,[yaw-1,pitch]),b),b1=values(evaluate(data,side,[yaw,pitch]),b);
   b1.forEach((v,j)=>maxDegreeStep=Math.max(maxDegreeStep,Math.abs(v-a[j])));
  }
 }
 return {samples,pitchDeg:[-20,0,20],stepDeg:.01,maxFineStepPx:maxFineStep,maxExcessPx:maxExcess,
  maxDegreeStepPx:maxDegreeStep,headRadiusPx:HEAD_RADIUS,source:'bundle/geometry.mjs HEAD_RADIUS',
  speedLimitPx:HEAD_RADIUS*rad,passed:!failures.length&&maxDegreeStep<=HEAD_RADIUS*rad,failures};
}
