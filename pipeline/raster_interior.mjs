// Near-side coverage follows the actual bridged geometry. The far side is
// cropped by source hair alpha and uses the measured main opaque spans.
import {canopyWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {domeSection,craniumFacing} from '../bundle/hair-surface.mjs';
export function projectedHeadInterval(y,p) {
 const {cx,rx}=domeSection(y),xs=[];
 for(let i=0;i<=32;i++)xs.push(canopyWorld([502+cx+rx*(i/16-1),y],p)[0]);
 for(const sign of [-1,1])for(let deg=88;deg<=180;deg+=2){
  const point=[0,y,sign*deg*Math.PI/180];
  if(craniumFacing(point,p)>1e-6)xs.push(craniumWorld(point,sign<0?'left':'right',p)[0]);
 }
 return [Math.min(...xs),Math.max(...xs)];
}
export function mainOpaqueRuns(rgba,width,height,sy,scale) {
 const runs=[];let start=-1;
 for(let x=0;x<=width;x++){
  const opaque=x<width&&rgba[((height-1-sy)*width+x)*4+3]>200;
  if(opaque&&start<0)start=x;
  if(!opaque&&start>=0){
   if(x-start>=Math.ceil(20*scale))runs.push([start,x]);
   start=-1;
  }
 }
 return runs.length?[runs[0][0],runs.at(-1)[1]]:null;
}
export function headInteriorInterval(y,p,l,r,ox,scale) {
 const [projLeft,projRight]=projectedHeadInterval(y,p),yaw=p.ParamAngleX*p.CorrTurn;
 const {rx,rz,cz,cx}=domeSection(y),angle=yaw*Math.PI/180;
 const center=502+cx*Math.cos(angle)+cz*Math.sin(angle),radius=Math.hypot(rx*Math.cos(angle),rz*Math.sin(angle));
 const domeLeft=ox+(center-radius+15)*scale,domeRight=ox+(center+radius-15)*scale,inset=15*scale;
 return [yaw>0?ox+(projLeft+15)*scale:Math.max(l+inset,yaw<0?domeLeft:0),
         yaw<0?ox+(projRight-15)*scale:Math.min(r-inset,yaw>0?domeRight:Infinity)];
}
export function rowAlphaGaps(rgba,width,height,sy,left,right) {
 let count=0;const locations=[];
 for(let x=Math.max(0,Math.ceil(left));x<Math.min(width,right);x++)if(rgba[((height-1-sy)*width+x)*4+3]<10){
  count++;if(locations.length<20)locations.push(x);
 }
 return {count,locations};
}
