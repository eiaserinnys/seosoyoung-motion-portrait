import {deform3D} from './rig.mjs';
import {frameValue} from './pose-grid.mjs';
import {headBaseVertices,modelAngles,projectModel} from './head-base.mjs';

const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const smooth=v=>{const t=clamp(v,0,1);return t*t*(3-2*t);};

function basis(theta,roll){
 const a=roll*Math.PI/180,c=Math.cos(a),s=Math.sin(a),pose=[theta[0],theta[1],0,0,0,0];
 return [[1,0,0],[0,1,0],[0,0,1]].map(point=>{
  const [x,y,z]=projectModel(point,pose);return [c*x-s*y,s*x+c*y,z];
 });
}

function rotation(model,p){
 const start=basis(model.thetaRest,0),end=basis(modelAngles(p),p.ParamAngleZ);
 return [0,1,2].map(i=>[0,1,2].map(j=>end.reduce((sum,v,k)=>sum+v[i]*start[k][j],0)));
}

export function createNeckTube(neck,model,rest){
 const {open_boundary:ring,reference:ref,neck_tube:spec}=model,n=ring.length;
 const [x0,y0,width,height]=spec.bbox,center=[0,2].map(c=>ring.reduce((sum,i)=>sum+ref[i*3+c],0)/n);
 const rows=spec.rings,bottom=spec.bottom_y,top=spec.first_opaque_y;
 const vertices=[],positions=[],uvs=[],triangles=[];
 for(let row=0;row<rows;row++){
  const t=row/(rows-1),w=smooth((t-2/3)*3),rawY=bottom+(top-bottom)*t;
  const q=clamp(rawY-y0,0,spec.profile.length-1),low=Math.floor(q),high=Math.min(low+1,spec.profile.length-1),f=q-low;
  const [left,right]=[0,1].map(k=>spec.profile[low][k]*(1-f)+spec.profile[high][k]*f),middle=(left+right)/2,radius=(right-left)/2;
  const axis=Math.abs(middle-center[0])>5?middle:center[0];
  for(let j=0;j<n;j++){
   const index=ring[j],target=[ref[index*3],ref[index*3+1],ref[index*3+2]];
   const angle=Math.atan2(target[2]-center[1],target[0]-center[0]);
   const raw=[axis+radius*Math.cos(angle),rawY],body=deform3D(neck,raw,rest);
   const cylinder=[body[0],body[1],center[1]+radius*Math.sin(angle)];
   const point=cylinder.map((x,k)=>x+(target[k]-x)*w);
   vertices.push({index,raw,body,point,target,w});positions.push(point.slice(0,2));
   uvs.push([clamp((point[0]-x0)/width,0,1),clamp((point[1]-y0)/height,0,1)]);
  }
 }
 for(let row=0;row<rows-1;row++)for(let j=0;j<n;j++){
  const a=row*n+j,b=row*n+(j+1)%n,c=(row+1)*n+j,d=(row+1)*n+(j+1)%n;
  triangles.push([a,c,b],[b,c,d]);
 }
 return {id:'neck-tube',positions,uvs,triangles,masks:[],vertices,neck,model,ringCount:n,rows,bottom};
}

export function neckTubePositions(tube,p){
 return frameValue(p,'neck-tube',()=>{
  const head=headBaseVertices(p).surface,R=rotation(tube.model,p);
  return tube.vertices.map(v=>{
   const body=deform3D(tube.neck,v.raw,p),fixed=v.point.map((x,k)=>x+body[k]-v.body[k]);
   const offset=v.point.map((x,k)=>x-v.target[k]);
   const attached=[0,1,2].map(k=>head[v.index*3+k]+R[k].reduce((sum,x,j)=>sum+x*offset[j],0));
   return fixed.map((x,k)=>x+(attached[k]-x)*v.w);
  });
 });
}
