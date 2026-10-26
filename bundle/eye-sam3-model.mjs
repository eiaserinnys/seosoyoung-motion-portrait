// SAM 3 v2.9.1: per-eye log tables, neutral delta0 and spherical gaze increment.
const rad=Math.PI/180;
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const scale=(a,s)=>a.map(v=>v*s);
const rot=a=>[[Math.cos(a),-Math.sin(a)],[Math.sin(a),Math.cos(a)]];
const transpose=a=>[[a[0][0],a[1][0]],[a[0][1],a[1][1]]];
const apply=(a,v)=>a.map(row=>row[0]*v[0]+row[1]*v[1]);
const mul=(a,b)=>a.map(row=>[0,1].map(j=>row[0]*b[0][j]+row[1]*b[1][j]));
const inverse=a=>{
 const det=a[0][0]*a[1][1]-a[0][1]*a[1][0];
 return [[a[1][1]/det,-a[0][1]/det],[-a[1][0]/det,a[0][0]/det]];
};
const I=[[1,0],[0,1]];

export function compression(z){
 if(z<=.20)return .25;
 if(z>=.30)return z;
 return .25+.05*((z-.20)/.10)**2;
}
export function rotateEye([x,y,z],[yaw,pitch]){
 const a=yaw*rad,b=pitch*rad;
 return [x*Math.cos(a)+(y*Math.sin(b)+z*Math.cos(b))*Math.sin(a),
  y*Math.cos(b)-z*Math.sin(b),-x*Math.sin(a)+(y*Math.sin(b)+z*Math.cos(b))*Math.cos(a)];
}
const headGaze=([x,y])=>[Math.sin(x)*Math.cos(y),-Math.sin(y),Math.cos(x)*Math.cos(y)];
function foreshorten(v){
 const n=Math.hypot(v[0],v[1]);
 if(n<1e-8)return I;
 const x=v[0]/n,y=v[1]/n,q=compression(v[2])-1;
 return [[1+q*x*x,q*x*y],[q*x*y,1+q*y*y]];
}
export function gazeAngles(data,side,[x,y]){
 const sigma=side==='left'?-1:1,u=sigma*x,g=data.gaze[side];
 return [sigma*(g.a*u+g.b*u*u),g.Gy*y];
}
export function eyeResidual(data,side,[yaw,pitch]){
 const b=data[side],theta=Math.max(-45,Math.min(45,side==='left'?-yaw:yaw));
 const t=theta*rad,p=pitch/30,B=[p,p*p,t*t,t*p];
 const polynomial=c=>c.reduce((sum,v,i)=>sum+v*B[i],0);
 const table=logs=>{
  const j=Math.min(5,Math.floor((theta+45)/15)),u=(theta-b.knotsDeg[j])/15;
  return Math.exp(logs[j]*(1-u)+logs[j+1]*u);
 };
 return {theta,rT:table(b.r_T),rN:table(b.r_N),logH:polynomial(b.h),deltaY:polynomial(b.deltaNu),
  logA:b.kappa*Math.log(Math.cos(t)),logB:Math.log(Math.cos(pitch*rad))};
}
export function achievedAngles(data,yaw,pitch){
 const y=yaw/45,p=pitch/30,c=data.achieved.c;
 return [c[0]*y+c[1]*y*p,c[2]*p+c[3]*y*y+c[4]*p*p];
}
export function eyeDeformation(data,side,d,cO,input){
 const block=data[side],rest=block.rest,r=eyeResidual(data,side,d),E=rot(rest.phi),E0T=transpose(rest.E);
 const height=Math.exp(r.logH),outward=side==='left'?-1:1;
 const ellipse=(psi,a,b)=>mul(rot(psi),[[a,0],[0,b]]);
 const B0=ellipse(rest.psi,rest.a,rest.bI);
 const B=ellipse(rest.psi,rest.a*Math.exp(r.logA),rest.bI*Math.exp(r.logB));
 const T0=mul(mul(mul(E,B),inverse(B0)),E0T);
 const delta=add(block.delta0,[0,r.deltaY]);
 const cIbase=add(cO,scale(apply(E,delta),block.w0));
 const v0=rotateEye([0,0,1],d),v=rotateEye(headGaze(gazeAngles(data,side,input)),d);
 const gazeDelta=scale(sub(v.slice(0,2),v0.slice(0,2)),data.rho*block.w0);
 const cI=add(cIbase,gazeDelta);
 const T=mul(mul(foreshorten(v),inverse(foreshorten(v0))),T0);
 const cH=add(add(cIbase,apply(T0,sub(rest.cH,rest.cI))),scale(gazeDelta,data.lambdaH));
 return {cO,cI,cIbase,cH,T,T0,internal:r,descriptors:{dTemp:rest.dTemp*r.rT,dNas:rest.dNas*r.rN,
  aUp:rest.aUp*height,bLow:rest.bLow*height,phi:rest.phi},
  lid(point,open=1,closed=false){
   const blink=[point[0],closed?point[1]:rest.anchor[1]+(point[1]-rest.anchor[1])*open];
   const [t,n]=apply(E0T,sub(blink,rest.cO));
   return add(cO,apply(E,[t*(t*outward>0?r.rT:r.rN),n*height]));
  },
  iris:point=>add(cI,apply(T,sub(point,rest.cI))),
  highlight:point=>add(cH,apply(T0,sub(point,rest.cH)))};
}
