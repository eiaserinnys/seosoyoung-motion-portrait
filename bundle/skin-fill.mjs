import {SKIN_FILL as S} from './skin-fill-data.mjs';
import {deform3D} from './rig.mjs';
import {clamp} from './geometry.mjs';
import {frameValue} from './pose-grid.mjs';

export const skinFillMesh=()=>({id:'exposed-skin',positions:S.neutral,
 uvs:S.neutral.map(p=>p.map(v=>v/1024)),triangles:S.triangles,masks:[]});
// Mask only: fill the upper contour interior, including the forehead/eyes.
export function skinFillClipMesh(){
 const q=clipPositions(S.neutral),m=S.columns;
 return {id:'exposed-skin-clip',positions:q,uvs:q.map(p=>p.slice(0,2).map(v=>v/1024)),
  triangles:[...Array.from({length:m-1},(_,i)=>[m,i,i+1]),[m+1,m+2,m+3],[m+1,m+3,m+4]],masks:[]};
}
function clipPositions(q){
 const outer=q.slice(5*S.columns),l=outer[0],r=outer.at(-1);
 // Clipping preserves the old face everywhere below the attachment line.
 // Closing only the upper fan would cut through pitched eye/cheek fragments
 // whose rest Y is above earHeight but whose world Y lies below that line.
 const slope=(r[1]-l[1])/(r[0]-l[0]),left=[l[0]-2048,l[1]-2048*slope,-200],right=[r[0]+2048,r[1]+2048*slope,-200],bottom=(l[1]+r[1])/2+2048;
 return [...outer,[(l[0]+r[0])/2,(l[1]+r[1])/2,-200],left,right,[right[0],bottom,-200],[left[0],bottom,-200]];
}
export const skinFillClipPositions=(p,face)=>clipPositions(skinFillPositions(p,face));
// The ribbon moves with the face mask (card #615). Each vertex follows the mask's upper
// contour through a distance-weighted similarity, so the ribbon keeps its front-view
// relation to the mask at every pose without folding between contour vertices.
export function skinFillPositions(p,face){
 return frameValue(p,'exposed-skin',()=>{
  const rest=S.innerRow.map(i=>face.positions[i]),posed=S.innerRow.map(i=>deform3D(face,face.positions[i],p));
  return S.neutral.map(v=>{
   const w=rest.map(q=>1/Math.max(1e-9,(q[0]-v[0])**2+(q[1]-v[1])**2)),total=w.reduce((a,b)=>a+b,0);
   const mean=points=>[0,1].map(d=>points.reduce((sum,q,i)=>sum+w[i]*q[d],0)/total),a=mean(rest),b=mean(posed);
   let c=0,s=0,norm=0;
   for(let i=0;i<rest.length;i++){
    const x=rest[i][0]-a[0],y=rest[i][1]-a[1],u=posed[i][0]-b[0],t=posed[i][1]-b[1];
    c+=w[i]*(x*u+y*t);s+=w[i]*(x*t-y*u);norm+=w[i]*(x*x+y*y);
   }
   c/=norm;s/=norm;const x=v[0]-a[0],y=v[1]-a[1];
   return [b[0]+c*x-s*y,b[1]+s*x+c*y,-200];
  });
 });
}

// Temple strips (card #615): beside each ear a strip joins the face edge to the ear.
// One copy of the edge moves with the face and one with the ear. The copies coincide
// in the front view and open into the side of the head only when the face and ear part.
// The edge runs from the ear top to the ear bottom in the art (the ears span y 410-520).
const TEMPLE_TOP=405,TEMPLE_BOTTOM=513,TEMPLE_TAPER=24,TEMPLE_UV_INSET=14;
const ease=t=>t*t*(3-2*t);
const templeCache=new WeakMap();
function temple(face){
 let cached=templeCache.get(face);if(cached)return cached;
 const count=new Map(),next=new Map();
 for(const t of face.triangles)for(const [a,b] of [[t[0],t[1]],[t[1],t[2]],[t[2],t[0]]]){const key=a<b?`${a},${b}`:`${b},${a}`;count.set(key,(count.get(key)||0)+1);}
 for(const [key,n] of count)if(n===1){const [a,b]=key.split(',').map(Number);for(const [i,j] of [[a,b],[b,a]]){if(!next.has(i))next.set(i,[]);next.get(i).push(j);}}
 // Up the ribbon's own contour to the ear top, then down the face boundary to the ear bottom.
 const edge=row=>{
  const up=[];for(const i of row){if(face.positions[i][1]<TEMPLE_TOP)break;up.push(i);}
  const down=[];
  for(let prev=row[1],cur=row[0];;){const n=next.get(cur).find(i=>i!==prev);if(face.positions[n][1]>TEMPLE_BOTTOM)break;down.push(n);prev=cur;cur=n;}
  return [...up.reverse(),...down].map(i=>face.positions[i]);
 };
 const edges={left:edge(S.innerRow),right:edge([...S.innerRow].reverse())},positions=[],uvs=[],triangles=[];
 for(const [side,points] of Object.entries(edges)){
  const base=positions.length,n=points.length,inward=side==='left'?1:-1;
  for(let copy=0;copy<2;copy++)for(const q of points){positions.push(q);uvs.push([(q[0]+inward*TEMPLE_UV_INSET)/1024,q[1]/1024]);}
  for(let k=0;k<n-1;k++)triangles.push([base+k,base+k+1,base+n+k],[base+k+1,base+n+k+1,base+n+k]);
 }
 cached={edges,mesh:{id:'temple-fill',positions,uvs,triangles,masks:[]}};
 templeCache.set(face,cached);return cached;
}
export const templeFillMesh=face=>temple(face).mesh;
export function templeFillPositions(p,face){
 return frameValue(p,'temple-fill',()=>Object.entries(temple(face).edges).flatMap(([side,points])=>{
  // The ears share the face-skin deformer chain, so the face mesh stands in for the ear here.
  const ear={...face,id:'ear-'+side},top=points[0][1],onFace=points.map(q=>deform3D(face,q,p));
  // The top of the strip stays on the face so it does not stick out above the ear.
  const onEar=points.map((q,k)=>{
   const e=deform3D(ear,q,p),t=ease(clamp((q[1]-top)/TEMPLE_TAPER,0,1));
   return [onFace[k][0]+(e[0]-onFace[k][0])*t,onFace[k][1]+(e[1]-onFace[k][1])*t];
  });
  return [...onFace,...onEar].map(q=>[q[0],q[1],-200]);
 }));
}
