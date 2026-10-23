import {craniumPosition} from './hair-surface.mjs';
export {domeRadius,projectedLongitude} from './hair-surface.mjs';
export const craniumWeight=()=>1;
export const craniumWorld=(point,side,p)=>craniumPosition(point,p);
