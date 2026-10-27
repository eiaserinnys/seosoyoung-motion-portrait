import {gridValue,frameValue} from './pose-grid.mjs';
import {correction} from './geometry.mjs';

let model;
const rad=x=>x*Math.PI/180;
const pointKey=point=>point.join(',');
export const getHeadBase=()=>model;

export function configureHeadBase(metadata,buffer){
 const array=(name,Type)=>{const {offset,length}=metadata.offsets[name];return new Type(buffer,offset,length/Type.BYTES_PER_ELEMENT);};
 model={...metadata,vertices:array('vertices',Float32Array),normals:array('normals',Float32Array),
  fields:array('fields',Float32Array),triangles:array('triangles',Uint32Array),queries:new Map(),restFrames:new Map()};
 model.thetaRest=model.keys.find(k=>k.yaw===0&&k.pitch===0).model;
 model.reference=projectedVertices(model.thetaRest,false);
 for(let i=0;i<model.vertexCount;i++){
  model.reference[i*3]+=model.rest_translation[0];model.reference[i*3+1]+=model.rest_translation[1];
 }
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
 const logits=model.theta_k.map(k=>-((theta[0]-k[0])**2+(1.4*(theta[1]-k[1]))**2)/(2*model.sigma**2));
 const max=Math.max(...logits),values=logits.map(v=>Math.exp(v-max)),total=values.reduce((a,b)=>a+b,0);
 return values.map(v=>v/total);
}

export function fieldAt(theta,index){
 const weights=phi(theta),n=model.vertexCount,result=[0,0,0];
 for(let k=0;k<weights.length;k++)for(let c=0;c<3;c++)result[c]+=weights[k]*model.fields[(k*n+index)*3+c];
 return result;
}

function projectedVertices(theta,inset){
 const weights=phi(theta),n=model.vertexCount,out=new Float64Array(n*3),scale=Math.exp(model.logscale_front);
 const pose=[theta[0],theta[1],0,model.logscale_front,0,0],shift=inset?model.inset_px/model.s_ref:0;
 for(let i=0;i<n;i++){
  const xyz=[0,1,2].map(c=>{
   let v=model.vertices[i*3+c]-model.normals[i*3+c]*shift;
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
  const theta=modelAngles(p),surface=projectedVertices(theta,false),inset=projectedVertices(theta,true);
  const [dx,dy]=model.rest_translation;
  const angle=rad(p.ParamAngleZ),c=Math.cos(angle),s=Math.sin(angle);
  for(const points of [surface,inset])for(let i=0;i<model.vertexCount;i++){
   const j=i*3,x=points[j]+dx-502,y=points[j+1]+dy-568;
   points[j]=502+c*x-s*y;points[j+1]=568+s*x+c*y;
  }
  const distance2=(theta[0]-model.thetaRest[0])**2+(1.4*(theta[1]-model.thetaRest[1]))**2;
  return {surface,inset,theta,offsetWeight:Math.exp(-distance2/(2*model.sigma**2))};
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

export function attachedTransform(point,p){
 const key=pointKey(point),a=attachment(point),current=headBaseVertices(p).surface;
 if(!model.restFrames.has(key))model.restFrames.set(key,triangleFrame(model.reference,a.vertices));
 const rest=model.restFrames.get(key),frame=triangleFrame(current,a.vertices);
 const R=[0,1,2].map(i=>[0,1,2].map(j=>frame.reduce((sum,v,k)=>sum+v[i]*rest[k][j],0)));
 return {A:headSolveWorld(point,p),R};
}

export function headEarWorld(point,side,p){
 const anchor=side==='left'?[369,465]:[636,464.5],{A,R}=attachedTransform(anchor,p);
 const offset=[point[0]-anchor[0],point[1]-anchor[1],5];
 return A.map((x,i)=>x+R[i].reduce((sum,v,j)=>sum+v*offset[j],0));
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
