// Identify uses the existing >127 white/alpha rule. The visible silhouette is
// independent: a normal frame with all eye meshes hidden, alpha exactly zero.
export function outsideVisibleOutline(identified,outline,width){
 const coordinates=[];
 for(let i=0;i<identified.length;i+=4){
  if(identified[i]>127&&identified[i+1]>127&&identified[i+2]>127&&identified[i+3]>127&&outline[i+3]===0){
   const pixel=i/4;coordinates.push([pixel%width,Math.floor(pixel/width)]);
  }
 }
 return {pixels:coordinates.length,coordinates};
}
