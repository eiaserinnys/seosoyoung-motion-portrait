import {gridValue,frameValue} from './pose-grid.mjs';
import {correction,NECK_PIVOT} from './geometry.mjs';

export const HEAD_BASE = Object.freeze({
 pitchWeight:1.4, earRelief:5, earTiltDegrees:55, neckAxisTolerance:5,
 earFadeDegrees:20, earMinMargin:2, faceRimWidth:12, rimStep:6,
 facingStart:.05, facingEnd:.20,
 foreheadFadeYawStart:45, foreheadFadeYawEnd:60, foreheadRimWidth:48,
 foreheadFadeTop:255, foreheadFadeBottom:330,
});

let model;
const rad=x=>x*Math.PI/180;
const pointKey=point=>point.join(',');
export const getHeadBase=()=>model;

export function configureHeadBase(metadata,buffer){
 const array=(name,Type)=>{const {offset,length}=metadata.offsets[name];return new Type(buffer,offset,length/Type.BYTES_PER_ELEMENT);};
 model={...metadata,vertices:array('vertices',Float32Array),normals:array('normals',Float32Array),
  fields:array('fields',Float32Array),triangles:array('triangles',Uint32Array),queries:new Map()};
 model.thetaRest=model.keys.find(k=>k.yaw===0&&k.pitch===0).model;
 model.reference=projectedVertices(model.thetaRest);
 for(let i=0;i<model.vertexCount;i++){
  model.reference[i*3]+=model.rest_translation[0];model.reference[i*3+1]+=model.rest_translation[1];
 }
 model.earData=Object.fromEntries(Object.entries(model.ear_roots).map(([side,root])=>{
  const center=ringCenter(model.reference,root.ring);
  return [side,{...root,center,rest:root.ring.map(i=>[0,1,2].map(c=>model.reference[i*3+c]-center[c])),
   delta:[root.anchor[0]-center[0],root.anchor[1]-center[1],0]}];
 }));
 return model;
}

export async function loadHeadBase(url='./assets/head-base.json'){
 const response=await fetch(url);if(!response.ok)throw Error(`두상 정보 로드 실패 (${response.status})`);
 const metadata=await response.json();const binaryUrl=new URL('head-base.bin',new URL(url,globalThis.location.href));
 const binary=await fetch(binaryUrl);if(!binary.ok)throw Error(`두상 자산 로드 실패 (${binary.status})`);
 return configureHeadBase(metadata,await binary.arrayBuffer());
}

export function projectModel([x,y,z],[yaw,pitch,roll,logscale,tx,ty]){
 const a=rad(yaw),b=rad(pitch),c=rad(roll),cy=Math.cos(a),sy=Math.sin(a),cp=Math.cos(b),sp=Math.sin(b),cr=Math.cos(c),sr=Math.sin(c);
 const yp=-cp*y-sp*z,zp=-sp*y+cp*z,xp=cy*x+sy*zp,depth=-sy*x+cy*zp,scale=Math.exp(logscale);
 return [(cr*xp-sr*yp)*scale+tx,(sr*xp+cr*yp)*scale+ty,depth];
}

export function modelAngles(p){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
 const edge=Math.max(-45,Math.min(45,yaw));
 const y=gridValue(model.keys,edge,pitch,k=>k.model[0]);
 const v=gridValue(model.keys,edge,pitch,k=>k.model[1]);
 return [Math.abs(yaw)<=45?y:y+Math.sign(yaw)*(Math.abs(yaw)-45)*(90-Math.abs(y))/45,v];
}

export function phi(theta){
 const logits=model.theta_k.map(k=>-((theta[0]-k[0])**2+(HEAD_BASE.pitchWeight*(theta[1]-k[1]))**2)/(2*model.sigma**2));
 const max=Math.max(...logits),values=logits.map(v=>Math.exp(v-max)),total=values.reduce((a,b)=>a+b,0);
 return values.map(v=>v/total);
}

export function fieldAt(theta,index){
 const weights=phi(theta),n=model.vertexCount,result=[0,0,0];
 for(let k=0;k<weights.length;k++)for(let c=0;c<3;c++)result[c]+=weights[k]*model.fields[(k*n+index)*3+c];
 return result;
}

function projectedVertices(theta){
 const weights=phi(theta),n=model.vertexCount,out=new Float64Array(n*3),scale=Math.exp(model.logscale_front);
 const pose=[theta[0],theta[1],0,model.logscale_front,0,0];
 for(let i=0;i<n;i++){
  const xyz=[0,1,2].map(c=>{
   let v=model.vertices[i*3+c];
   for(let k=0;k<weights.length;k++)v+=weights[k]*model.fields[(k*n+i)*3+c]/model.s_ref;
   return v;
  });
  const q=projectModel(xyz,pose);out.set([q[0],q[1],q[2]*scale],i*3);
 }
 return out;
}

const weighted=(coords,attachment)=>attachment.vertices.reduce((out,index,i)=>{
 for(let c=0;c<3;c++)out[c]+=attachment.weights[i]*coords[index*3+c];return out;
},[0,0,0]);

export function headBaseVertices(p){
 return frameValue(p,'head-base',()=>{
  const theta=modelAngles(p),surface=projectedVertices(theta);
  const [dx,dy]=model.rest_translation;
  const angle=rad(p.ParamAngleZ),c=Math.cos(angle),s=Math.sin(angle);
  for(let i=0;i<model.vertexCount;i++){
   const j=i*3,x=surface[j]+dx-NECK_PIVOT[0],y=surface[j+1]+dy-NECK_PIVOT[1];
   surface[j]=NECK_PIVOT[0]+c*x-s*y;surface[j+1]=NECK_PIVOT[1]+s*x+c*y;
  }
  const distance2=(theta[0]-model.thetaRest[0])**2+(HEAD_BASE.pitchWeight*(theta[1]-model.thetaRest[1]))**2;
  return {surface,theta,offsetWeight:Math.exp(-distance2/(2*model.sigma**2))};
 });
}

function triangleAt(point,insideOnly){
 const q=model.reference,tri=model.triangles;let best=null,bestDepth=-Infinity,bestDistance=Infinity,bestArea=0;
 for(let i=0;i<tri.length;i+=3){
  const ids=[tri[i],tri[i+1],tri[i+2]],a=ids.map(j=>[q[j*3],q[j*3+1],q[j*3+2]]);
  const det=(a[1][1]-a[2][1])*(a[0][0]-a[2][0])+(a[2][0]-a[1][0])*(a[0][1]-a[2][1]);
  if(det>=-1e-8)continue;
  const u=((a[1][1]-a[2][1])*(point[0]-a[2][0])+(a[2][0]-a[1][0])*(point[1]-a[2][1]))/det;
  const v=((a[2][1]-a[0][1])*(point[0]-a[2][0])+(a[0][0]-a[2][0])*(point[1]-a[2][1]))/det,w=1-u-v;
  if(Math.min(u,v,w)>=-1e-9){
   const z=u*a[0][2]+v*a[1][2]+w*a[2][2];
   if(z>bestDepth){bestDepth=z;best={vertices:ids,weights:[u,v,w],offset:[0,0]};}
  }else if(!insideOnly&&bestDepth===-Infinity){
   for(const [first,last] of [[0,1],[1,2],[2,0]]){
    const dx=a[last][0]-a[first][0],dy=a[last][1]-a[first][1],t=Math.max(0,Math.min(1,((point[0]-a[first][0])*dx+(point[1]-a[first][1])*dy)/(dx*dx+dy*dy)));
    const near=[a[first][0]+t*dx,a[first][1]+t*dy],d=(point[0]-near[0])**2+(point[1]-near[1])**2;
    if(d<bestDistance-1e-12||(Math.abs(d-bestDistance)<=1e-12&&Math.abs(det)>bestArea)){
     const weights=[0,0,0];weights[first]=1-t;weights[last]=t;bestDistance=d;bestArea=Math.abs(det);
     best={vertices:ids,weights,offset:[point[0]-near[0],point[1]-near[1]],nearest:near};
    }
   }
  }
 }
 if(!insideOnly&&best?.nearest){
  const visible=triangleAt(best.nearest,true);
  if(visible){visible.offset=best.offset;return visible;}
  delete best.nearest;
 }
 return best;
}

function attachment(point){
 const key=pointKey(point),stored=model.attachments[key];
 if(stored)return stored;
 if(!model.queries.has(key))model.queries.set(key,triangleAt(point,false));
 return model.queries.get(key);
}

const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=v=>{const length=Math.hypot(...v);return v.map(x=>x/length);};
function triangleFrame(coords,vertices){
 const p=vertices.map(i=>[coords[i*3],coords[i*3+1],coords[i*3+2]]);
 const a=p[1].map((x,i)=>x-p[0][i]),b=p[2].map((x,i)=>x-p[0][i]);
 const t1=unit(a),n=unit(cross(b,a)),t2=cross(n,t1);
 return [t1,t2,n];
}

export function headSurfaceFacingDelta(point,p){
 const a=attachment(point),current=headBaseVertices(p).surface;
 return triangleFrame(current,a.vertices)[2][2]-triangleFrame(model.reference,a.vertices)[2][2];
}

function ringCenter(coords,ring){
 const center=[0,0,0];for(const i of ring)for(let c=0;c<3;c++)center[c]+=coords[i*3+c];
 return center.map(x=>x/ring.length);
}

function rootRotation(reference,current,center){
 const h=Array.from({length:3},()=>[0,0,0]);
 for(let i=0;i<reference.length;i++)for(let a=0;a<3;a++)for(let b=0;b<3;b++)h[a][b]+=reference[i][a]*(current[i][b]-center[b]);
 const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]=h;
 const n=[[xx+yy+zz,yz-zy,zx-xz,xy-yx],[yz-zy,xx-yy-zz,xy+yx,zx+xz],
  [zx-xz,xy+yx,-xx+yy-zz,yz+zy],[xy-yx,zx+xz,yz+zy,-xx-yy+zz]];
 const vectors=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>Number(i===j)));
 for(let sweep=0;sweep<12;sweep++)for(let p=0;p<4;p++)for(let q=p+1;q<4;q++){
  const value=n[p][q];if(Math.abs(value)<1e-12)continue;
  const tau=(n[q][q]-n[p][p])/(2*value),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.sqrt(1+tau*tau));
  const c=1/Math.sqrt(1+t*t),s=t*c;n[p][p]-=t*value;n[q][q]+=t*value;n[p][q]=n[q][p]=0;
  for(let i=0;i<4;i++){
   if(i!==p&&i!==q){const a=n[i][p],b=n[i][q];n[i][p]=n[p][i]=c*a-s*b;n[i][q]=n[q][i]=s*a+c*b;}
   const a=vectors[i][p],b=vectors[i][q];vectors[i][p]=c*a-s*b;vectors[i][q]=s*a+c*b;
  }
 }
 let largest=0;for(let i=1;i<4;i++)if(n[i][i]>n[largest][largest])largest=i;
 const [w,x,y,z]=vectors.map(row=>row[largest]);
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
}

function earRootTransform(side,p){
 return frameValue(p,'ear-root:'+side,()=>{
  const root=model.earData[side],coords=headBaseVertices(p).surface,center=ringCenter(coords,root.ring);
  const current=root.ring.map(i=>[coords[i*3],coords[i*3+1],coords[i*3+2]]);
  return {center,R:rootRotation(root.rest,current,center),root};
 });
}

export function headEarWorld(point,side,p){
 const {center,R,root}=earRootTransform(side,p);
 const dx=point[0]-root.anchor[0];
 // Scale the tilted plane by 1/cos(alpha): its rest x projection stays exact.
 const depth=(side==='left'?dx:-dx)*Math.tan(rad(HEAD_BASE.earTiltDegrees));
 const offset=[root.delta[0]+dx,root.delta[1]+point[1]-root.anchor[1],HEAD_BASE.earRelief+depth];
 return center.map((x,i)=>x+R[i].reduce((sum,v,j)=>sum+v*offset[j],0));
}

export function headSolveWorld(point,p,relief=0){
 const a=attachment(point),frame=headBaseVertices(p),out=weighted(frame.surface,a);
 const angle=rad(p.ParamAngleZ),c=Math.cos(angle),s=Math.sin(angle),[ox,oy]=a.offset;
 out[0]+=(c*ox-s*oy)*frame.offsetWeight;out[1]+=(s*ox+c*oy)*frame.offsetWeight;
 if(relief){
  const xyz=a.vertices.map(i=>[0,1,2].map(j=>frame.surface[i*3+j]));
  const u=xyz[1].map((x,i)=>x-xyz[0][i]),v=xyz[2].map((x,i)=>x-xyz[0][i]);
  const n=[u[2]*v[1]-u[1]*v[2],u[0]*v[2]-u[2]*v[0],u[1]*v[0]-u[0]*v[1]],length=Math.hypot(...n);
  for(let i=0;i<3;i++)out[i]+=relief*n[i]/length;
 }
 return out;
}
