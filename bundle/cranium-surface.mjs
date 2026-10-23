import {DOME} from './dome-data.mjs';
import {clamp,radians,correction,artWorld,HEAD_CENTER} from './geometry.mjs';
export const domeRadius=y=>DOME.rx*Math.sqrt(Math.max(0,1-((y-DOME.cy)/DOME.ry)**2));
export const projectedLongitude=(x,y)=>Math.asin(clamp((x-DOME.cx)/domeRadius(y),-1,1));
export function craniumWeight(side,p){
 const t=clamp(p.ParamAngleX*correction(p,'CorrTurn')*(side==='right'?-1:1)/45,0,1);
 return t*t*(3-2*t);
}
export function craniumWorld(point,side,p){
 const [x,y]=point,r=domeRadius(y),reference=radians(side==='right'?-45:45);
 const atReference=projectedLongitude(x,y),rest=atReference-reference;
 const yaw=radians(p.ParamAngleX*correction(p,'CorrTurn'));
 const longitude=clamp(rest+yaw,-Math.PI/2,Math.PI/2);
 // SAM silhouette residual only where the fitted ellipse misses drawn hair.
 // It vanishes smoothly at rest and recovers the authored reference exactly.
 const residual=x-DOME.cx-r*Math.sin(atReference);
 const local=[r*Math.sin(longitude)+residual*craniumWeight(side,p),y-HEAD_CENTER[1],r*Math.cos(longitude)];
 return artWorld(local,{...p,ParamAngleX:0});
}
