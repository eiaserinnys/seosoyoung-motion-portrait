import {CORRECTION_CONTROLS,CORRECTION_PRESETS} from './corrections.mjs';
import {Renderer} from './renderer.mjs';
import {Pendulum} from './rig.mjs';
import {viewerPose,sliderForYaw} from './viewer-pose.mjs';
import {loadHeadBase} from './head-base.mjs';
const $=id=>document.getElementById(id);
const loadImage=src=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('이미지 로드 실패: '+src));im.src=src;});
const eyeIcon='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>';
async function main(){
  const response=await fetch('./assets/seosoyoung-rig.json');if(!response.ok)throw Error('리그 로드 실패');const rig=await response.json();
  const images=await Promise.all(['texture_00.png','texture_before_00.png','head-surface.png','head-surface-before.png','face-profile.png','ponytail-skin-alpha.png','exposed-skin.png'].map(file=>loadImage('./assets/'+file)));
  const p=Object.fromEntries(rig.parameters.map(param=>[param.id,param.default]));
  Object.assign(p,CORRECTION_PRESETS.recommended);
  let headBaseError=null;
  try{await loadHeadBase();}catch(error){headBaseError=error.message;p.CorrHeadBase=0;}
  const state={selected:'face-skin',selectedIds:new Set(['face-skin']),hidden:new Set(),mode:'all',before:false,bangs:false,outfit:'sweater',wire:false,grid:false,masks:false,auto:false,track:false,tab:'parts'};
  const renderer=new Renderer($('stage'),$('overlay'),rig,images);
  const inputs={};
  const label={AngleX:'고개 좌우',AngleY:'고개 상하',AngleZ:'고개 기울임',EyeBallX:'시선 좌우',EyeBallY:'시선 상하',EyeLOpen:'왼눈 뜨기',EyeROpen:'오른눈 뜨기',BrowLY:'왼쪽 눈썹',BrowRY:'오른쪽 눈썹',MouthOpenY:'입 벌림',MouthForm:'입 모양',Breath:'호흡',HairFront:'앞머리',HairSide:'옆머리',HairBack:'뒷머리',BodyAngleX:'몸 기울임'};
  for(const param of rig.parameters){
    const row=document.createElement('div');row.className='param';
    row.innerHTML=`<label for="${param.id}" title="${param.id}">${label[param.id.slice(5)]}</label><output for="${param.id}"></output><input id="${param.id}" aria-label="${param.id}" type="range" min="${param.min}" max="${param.max}" step="${param.max>2?.1:.01}" value="${param.default}">`;
    inputs[param.id]=row.querySelector('input');inputs[param.id].oninput=e=>{stopMotion();p[param.id]=+e.target.value;sync();};$('params').append(row);
  }
  for(const param of CORRECTION_CONTROLS){
    const row=document.createElement('div');row.className='param';
    row.innerHTML=`<label for="${param.id}">${param.label}</label><output for="${param.id}"></output><input id="${param.id}" aria-label="${param.label}" type="range" min="${param.min}" max="${param.max}" step="${param.step}" value="${p[param.id]}">`;
    inputs[param.id]=row.querySelector('input');inputs[param.id].disabled=param.id==='CorrHeadBase'&&!!headBaseError;inputs[param.id].oninput=e=>{p[param.id]=+e.target.value;sync();};$('corrections').append(row);
  }
  function setCorrection(name){Object.assign(p,CORRECTION_PRESETS[name]);if(headBaseError)p.CorrHeadBase=0;sync();}
  for(const name of ['pure','recommended'])$('correction-'+name).onclick=()=>setCorrection(name);
  function sync(){const view=viewerPose(p);for(const [name,values]of Object.entries(CORRECTION_PRESETS))pressed('correction-'+name,Object.entries(values).every(([k,v])=>Math.abs(p[k]-v)<.00001));for(const [key,input]of Object.entries(inputs)){input.value=p[key];input.previousElementSibling.value=p[key].toFixed(2).replace(/\.00$/,'');}$('angles').textContent=`X ${(view.ParamAngleX*view.CorrTurn).toFixed(Math.abs(p.ParamAngleX)>45?1:0).replace(/\.0$/,'')}° / Y ${(p.ParamAngleY*p.CorrTurn).toFixed(0)}° / Z ${p.ParamAngleZ.toFixed(0)}°`;}
  function pressed(id,value){$(id).setAttribute('aria-pressed',String(value));}
  function stopMotion(){state.auto=false;state.track=false;pressed('auto',false);pressed('track',false);}
  function setPose(x,y,z,name){stopMotion();p.ParamAngleX=x;p.ParamAngleY=y;p.ParamAngleZ=z;$('pose-label').textContent=name;sync();}
  const presets=[['정면 0°',0,0,0],['좌 -25°',-25,0,0],['우 +25°',25,0,0],['상 +20°',0,20,0],['하 -20°',0,-20,0],['기울임 ±15°',0,0,15],['3/4 좌상',-25,18,0],['3/4 우하',25,-18,0]];
  for(const [name,x,y,z]of presets){const b=document.createElement('button');b.textContent=name;b.onclick=()=>setPose(x,y,name==='기울임 ±15°'&&p.ParamAngleZ===15?-15:z,name);$('presets').append(b);}
  for(const [name,yaw]of [['좌 67.5°',-67.5],['우 67.5°',67.5],['좌 90°',-90],['우 90°',90]]){
    const b=document.createElement('button');b.textContent=name;b.onclick=()=>setPose(sliderForYaw(yaw,p.CorrTurn),0,0,name);$('presets').append(b);
  }
  for(const name of ['auto','track','wire','grid','masks','bangs'])$(name).onclick=()=>{state[name]=!state[name];if(name==='auto'&&state.auto){state.track=false;pressed('track',false);}if(name==='track'&&state.track){state.auto=false;pressed('auto',false);}pressed(name,state[name]);};
  const headBaseButton=document.createElement('button'),hairButton=document.createElement('button');
  headBaseButton.id='head-base-toggle';headBaseButton.textContent='두상 바탕';
  hairButton.id='hair-hide-toggle';hairButton.textContent='머리카락 숨김';$('bangs').after(headBaseButton,hairButton);
  const hairIds=rig.meshes.filter(m=>m.id==='hair-top'||m.id==='hairpin'||/^(bangs-|side-hair-|back-hair-|side-cranium-|tail-)/.test(m.id)).map(m=>m.id);
  let hiddenHairBefore=null;
  headBaseButton.onclick=()=>{state.hidden.has('head-base')?state.hidden.delete('head-base'):state.hidden.add('head-base');renderTree();};
  hairButton.onclick=()=>{if(hiddenHairBefore===null){hiddenHairBefore=new Set(hairIds.filter(id=>state.hidden.has(id)));for(const id of hairIds)state.hidden.add(id);}
    else{for(const id of hairIds)if(!hiddenHairBefore.has(id))state.hidden.delete(id);hiddenHairBefore=null;}renderTree();};
  $('compare').onclick=()=>{state.before=!state.before;pressed('compare',state.before);$('compare').textContent=state.before?'복원 전 (기존 파츠)':'복원 후 (SAM3+Inpaint)';$('version-label').textContent=state.before?'BEFORE':'AFTER';};
  for(const name of ['shirt','sweater'])$(name).onclick=()=>{state.outfit=name;pressed('shirt',name==='shirt');pressed('sweater',name==='sweater');};
  $('reset').onclick=()=>{stopMotion();for(const param of rig.parameters)p[param.id]=param.default;$('pose-label').textContent='정면 0°';sync();};
  document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));});
  function descendants(id){const ids=new Set([id]);let changed=true;while(changed){changed=false;for(const d of rig.deformers)if(ids.has(d.parentId)&&!ids.has(d.id)){ids.add(d.id);changed=true;}}return rig.meshes.filter(m=>ids.has(m.deformerId)).map(m=>m.id);}
  function choose(id,ids=[id]){state.selected=id;state.selectedIds=new Set(ids);const mesh=rig.meshes.find(m=>m.id===id)||rig.meshes.find(m=>ids.includes(m.id));if(mesh){state.inspect=mesh.id;$('selected-label').textContent=rig.deformers.some(d=>d.id===id)?id:mesh.label;updateInspector(mesh);}else if(id==='head-base')$('selected-label').textContent='두상 바탕';renderTree();}
  function treeRow(id,text,ids){const row=document.createElement('div');row.className='tree-row';row.dataset.active=state.selected===id;row.dataset.hidden=ids.every(key=>state.hidden.has(key));const select=document.createElement('button');select.className='select';select.title=id;const span=document.createElement('span');span.textContent=text;select.append(span);select.onclick=()=>choose(id,ids);const eye=document.createElement('button');eye.className='eye';eye.innerHTML=eyeIcon;eye.setAttribute('aria-label',text+' 가시성');eye.setAttribute('aria-pressed',String(!ids.every(key=>state.hidden.has(key))));eye.onclick=()=>{const hide=!ids.every(key=>state.hidden.has(key));for(const key of ids)hide?state.hidden.add(key):state.hidden.delete(key);renderTree();};row.append(select,eye);return row;}
  function renderTree(){const query=$('search').value.toLowerCase();$('tree').replaceChildren();$('counts').textContent=state.tab==='parts'?`${rig.parts.length} Part / ${rig.meshes.length} ArtMesh`:`${rig.deformers.length} Deformer`;
    headBaseButton.setAttribute('aria-pressed',String(!state.hidden.has('head-base')));hairButton.setAttribute('aria-pressed',String(hiddenHairBefore!==null));
    if(state.tab==='parts'&&'두상 바탕 head-base'.includes(query))$('tree').append(treeRow('head-base','두상 바탕',['head-base']));
    if(state.tab==='parts')for(const part of rig.parts){const members=rig.meshes.filter(m=>m.partId===part.id&&(`${m.id} ${m.label} ${part.label}`).toLowerCase().includes(query));if(!members.length)continue;const group=document.createElement('details');group.open=true;const title=document.createElement('summary');title.textContent=part.label+' / '+members.length;group.append(title);for(const mesh of members)group.append(treeRow(mesh.id,mesh.label,[mesh.id]));$('tree').append(group);}
    else for(const d of rig.deformers){if(!d.id.toLowerCase().includes(query))continue;const row=treeRow(d.id,d.id,descendants(d.id));let depth=0,parent=d.parentId;while(parent){depth++;parent=rig.deformers.find(item=>item.id===parent)?.parentId;}row.style.paddingLeft=`${Math.min(depth,4)*8}px`;$('tree').append(row);}}
  for(const [id,tab]of [['part-tab','parts'],['deformer-tab','deformers']])$(id).onclick=()=>{state.tab=tab;$('part-tab').setAttribute('aria-selected',String(tab==='parts'));$('deformer-tab').setAttribute('aria-selected',String(tab==='deformers'));renderTree();};$('search').oninput=renderTree;
  function crop(canvas,mesh,image){const ctx=canvas.getContext('2d'),[x,y,w,h]=mesh.atlas;ctx.clearRect(0,0,canvas.width,canvas.height);const scale=Math.min(200/w,200/h);ctx.drawImage(image,x,y,w,h,(220-w*scale)/2,(220-h*scale)/2,w*scale,h*scale);}
  function updateInspector(mesh){$('inspect-label').textContent=mesh.label;crop($('before-crop').querySelector('canvas'),mesh,images[1]);crop($('after-crop').querySelector('canvas'),mesh,images[0]);drawAtlas();}
  function drawAtlas(){const c=$('atlas'),ctx=c.getContext('2d'),s=c.width/2048;ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(images[0],0,0,c.width,c.height);ctx.strokeStyle='#9be6ff55';ctx.lineWidth=.5;ctx.beginPath();for(const m of rig.meshes)for(const tri of m.triangles){tri.forEach((n,i)=>{const uv=m.uvs[n];i?ctx.lineTo(uv[0]*c.width,uv[1]*c.height):ctx.moveTo(uv[0]*c.width,uv[1]*c.height);});ctx.closePath();}ctx.stroke();for(const m of rig.meshes.filter(m=>state.selectedIds.has(m.id))){const [x,y,w,h]=m.atlas;ctx.fillStyle='#6bffd522';ctx.fillRect(x*s,y*s,w*s,h*s);ctx.strokeStyle='#a5ffe1';ctx.lineWidth=3;ctx.strokeRect(x*s,y*s,w*s,h*s);}}
  $('atlas').onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*2048,y=(e.clientY-r.top)/r.height*2048;const m=rig.meshes.find(m=>{const[a,b,w,h]=m.atlas;return x>=a&&x<=a+w&&y>=b&&y<=b+h;});if(m)choose(m.id);};
  function showImage(src,title){$('lightbox-title').textContent=title;$('lightbox-image').src=src;$('lightbox-image').alt=title;$('lightbox-image').classList.remove('zoomed');$('lightbox').showModal();}
  $('close-modal').onclick=()=>$('lightbox').close();$('lightbox').onclick=e=>{if(e.target===$('lightbox'))$('lightbox').close();};$('lightbox-image').onclick=e=>e.currentTarget.classList.toggle('zoomed');
  for(const [id,image]of [['before-crop',images[1]],['after-crop',images[0]]])$(id).onclick=()=>{const m=rig.meshes.find(m=>m.id===(state.inspect||'face-skin')),c=document.createElement('canvas'),[x,y,w,h]=m.atlas;c.width=w;c.height=h;c.getContext('2d').drawImage(image,x,y,w,h,0,0,w,h);showImage(c.toDataURL(),m.label+' / '+(id==='before-crop'?'복원 전':'복원 후'));};
  document.querySelectorAll('[data-sheet]').forEach(b=>b.onclick=()=>showImage('./assets/'+b.dataset.sheet,b.textContent.replace(' ↗','')));$('atlas-zoom').onclick=()=>showImage($('atlas').toDataURL(),'UV 아틀라스');
  for(const [id,atlas]of [['compare-tab',false],['atlas-tab',true]])$(id).onclick=()=>{$('comparison').hidden=atlas;$('atlas-panel').hidden=!atlas;$('compare-tab').setAttribute('aria-selected',String(!atlas));$('atlas-tab').setAttribute('aria-selected',String(atlas));drawAtlas();};
  const pointer={x:0,y:0};$('viewport').onpointermove=e=>{const r=e.currentTarget.getBoundingClientRect();pointer.x=Math.max(-1,Math.min(1,(e.clientX-r.left)/r.width*2-1));pointer.y=Math.max(-1,Math.min(1,1-(e.clientY-r.top)/r.height*2));};$('viewport').onpointerleave=()=>{pointer.x=pointer.y=0;};
  const springs=[new Pendulum(),new Pendulum(),new Pendulum()];let last=performance.now();
  function frame(now){const dt=Math.min((now-last)/1000,.033),t=now/1000;last=now;
    if(state.auto){p.ParamAngleX=Math.sin(t*.48)*50;p.ParamAngleY=Math.sin(t*.96)*28;p.ParamAngleZ=Math.sin(t*.64+1.1)*14;const phase=t%4.3,blink=phase<.18?Math.abs(phase-.09)/.09:1;p.ParamEyeLOpen=p.ParamEyeROpen=blink;p.ParamBreath=(1+Math.sin(t*1.6))*.5;p.ParamBodyAngleX=Math.sin(t*.48-.4)*6;p.ParamEyeBallX=0;p.ParamEyeBallY=0;p.ParamHairFront=springs[0].step(Math.sin(t*.48+.5)*.75-p.ParamAngleZ/30,dt);p.ParamHairSide=springs[1].step(-p.ParamAngleX/50-p.ParamAngleZ/28,dt);p.ParamHairBack=springs[2].step(.30+Math.sin(t*.96-.5)*.55+Math.abs(p.ParamAngleZ)/28,dt);sync();}
    if(state.track){const f=1-Math.exp(-dt*9);for(const[key,target]of [['ParamAngleX',pointer.x*30],['ParamAngleY',pointer.y*24],['ParamEyeBallX',pointer.x],['ParamEyeBallY',-pointer.y]])p[key]+=(target-p[key])*f;sync();}
    if(state.auto||state.track)renderer.render(viewerPose(p),state);requestAnimationFrame(frame);
  }
  const redraw=()=>renderer.render(viewerPose(p),state);
  document.addEventListener('click',redraw);document.addEventListener('input',redraw);window.addEventListener('resize',redraw);
  choose('face-skin');sync();redraw();$('status').textContent=headBaseError?'두상 바탕 로드 실패: '+headBaseError:rig.meshes.length+' 파츠 / 준비됨';requestAnimationFrame(frame);
  window.studio={rig,p,state,renderer,choose,setPose,setCorrection,sync,ready:true};
}
main().catch(error=>{$('status').textContent=error.message;console.error(error);});
