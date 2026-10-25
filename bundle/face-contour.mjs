import {OUTLINE_ROWS} from './outline-data.mjs';
// Cubic Hermite interpolation keeps the sampled mask residual smooth in height.
export function outlineRow(y){
 if(y<=OUTLINE_ROWS[0][0])return OUTLINE_ROWS[0].slice(1);
 for(let i=1;i<OUTLINE_ROWS.length;i++)if(y<=OUTLINE_ROWS[i][0]){
  const a=OUTLINE_ROWS[i-1],b=OUTLINE_ROWS[i],prev=OUTLINE_ROWS[Math.max(0,i-2)],next=OUTLINE_ROWS[Math.min(OUTLINE_ROWS.length-1,i+1)];
  const h=b[0]-a[0],t=(y-a[0])/h;
  return a.slice(1).map((v,j)=>{
   const k=j+1,m0=(b[k]-prev[k])/(b[0]-prev[0]),m1=(next[k]-a[k])/(next[0]-a[0]);
   return (2*t**3-3*t*t+1)*v+(t**3-2*t*t+t)*h*m0+(-2*t**3+3*t*t)*b[k]+(t**3-t*t)*h*m1;
  });
 }
 return OUTLINE_ROWS.at(-1).slice(1);
}
export function contourCoordinates([x,y]){
 const [l,r]=outlineRow(y);
 return [300*(x-l)/(r-l),y];
}
