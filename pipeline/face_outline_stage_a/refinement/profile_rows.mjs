// 7.18: every intersecting row uses the same projected skin mesh boundary.
export function profileRows(samples,rest,oldRest,points,triangles){
 const edges=new Map();for(const tri of triangles)for(let k=0;k<3;k++){
  const pair=[tri[k],tri[(k+1)%3]].sort((a,b)=>a-b),key=pair.join(',');const edge=edges.get(key);edges.set(key,{pair,count:(edge?.count??0)+1});
 }
 const boundary=[...edges.values()].filter(e=>e.count===1).map(e=>e.pair);
 return samples.map(s=>{
  const before=[oldRest[s.left][0],oldRest[s.right][0]],after=[rest[s.left][0],rest[s.right][0]],maximum=Math.max(...after.map((x,i)=>Math.abs(x-before[i])));let selected=after;
  const xs=[];for(const [i,j]of boundary){const a=points[i],b=points[j];if(s.y<Math.min(a[1],b[1])||s.y>Math.max(a[1],b[1]))continue;
    if(a[1]===b[1]){xs.push(rest[i][0],rest[j][0]);continue;}const t=(s.y-a[1])/(b[1]-a[1]);xs.push(rest[i][0]+t*(rest[j][0]-rest[i][0]));}
  if(xs.length)selected=[Math.min(...xs),Math.max(...xs)];
  return {y:s.y,before,after,selected,maxChange:maximum,replaced:xs.length>0,definition:xs.length?'mesh-boundary':'unchanged-upper-rim'};
 });
}
