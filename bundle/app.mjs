import {Rig,defaultPose} from './rig.mjs';
import {Renderer} from './renderer.mjs';

const byId=id=>document.getElementById(id),status=byId('status'),canvas=byId('stage'),viewport=byId('viewport');
const numeric=value=>Number(value).toLocaleString('ko-KR',{maximumSignificantDigits:3});
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));

let rig,renderer,pose=defaultPose(),hidden=new Set(),auto=false,track=false,raf=0,last=0,phase=0;
const controls=new Map();

function setStatus(value){status.textContent=value;}

function poseText(){
  const yaw=pose.u<0?pose.u*35.8:pose.u*37.7,pitch=pose.v<0?pose.v*16.3:pose.v*24.8;
  const direction=[];
  if(Math.abs(yaw)>.5)direction.push(`${yaw<0?'왼쪽':'오른쪽'} ${numeric(yaw)}°`);
  if(Math.abs(pitch)>.5)direction.push(`${pitch<0?'아래':'위'} ${numeric(pitch)}°`);
  byId('pose-label').textContent=direction.join(' · ')||'정면';
}

function render(){
  if(!renderer)return;
  renderer.render(pose,{hidden});poseText();
}

function updateControl(id){
  const item=controls.get(id);if(!item)return;
  item.input.value=String(pose[id]);item.output.value=numeric(pose[id]);
}

function setPose(next,{sync=true,draw=true}={}){
  pose={...pose,...next};
  if(sync)for(const id of Object.keys(next))updateControl(id);
  if(draw)render();
}

function addControl(parameter){
  const row=document.createElement('div');row.className='param';
  const label=document.createElement('label'),output=document.createElement('output'),input=document.createElement('input');
  label.textContent=parameter.label;output.value=numeric(parameter.default);
  input.type='range';input.min=String(parameter.min);input.max=String(parameter.max);input.step='any';input.value=String(parameter.default);input.setAttribute('aria-label',parameter.label);
  input.addEventListener('input',()=>{pose[parameter.id]=Number(input.value);output.value=numeric(pose[parameter.id]);render();});
  row.append(label,output,input);byId('params').append(row);controls.set(parameter.id,{input,output});
}

function visibleMeshes(data){
  const hair=new Set(data.hair);
  return data.meshes.filter(mesh=>!hair.has(mesh.id)&&!mesh.id.startsWith('still-fill-hair-'));
}

function addParts(data){
  const visible=visibleMeshes(data),visibleIds=new Set(visible.map(mesh=>mesh.id)),meshById=new Map(data.meshes.map(mesh=>[mesh.id,mesh]));
  for(const group of data.parts){
    const entries=group.meshes.filter(id=>visibleIds.has(id));if(!entries.length)continue;
    const details=document.createElement('details');details.className='part-group';
    const summary=document.createElement('summary');summary.textContent=group.label;details.append(summary);
    for(const id of entries){
      const mesh=meshById.get(id),label=document.createElement('label');label.className='part-row';
      const input=document.createElement('input');input.type='checkbox';input.checked=true;input.setAttribute('aria-label',mesh.label);
      const name=document.createElement('span');name.textContent=mesh.label;
      input.addEventListener('change',()=>{if(input.checked)hidden.delete(id);else hidden.add(id);render();});
      label.append(input,name);details.append(label);
    }
    byId('parts-tree').append(details);
  }
}

function setToggle(button,value){button.setAttribute('aria-pressed',String(value));}

function animate(now){
  if(!auto&&!track){raf=0;last=now;return;}
  if(!last)last=now;
  const dt=Math.min((now-last)/1000,.1);last=now;phase+=dt;
  if(auto){
    const t=phase,blinkPhase=t%4.3,open=blinkPhase<.18?Math.abs(blinkPhase-.09)/.09:1;
    pose.u=Math.sin(t*.48);pose.v=Math.sin(t*.96);pose.roll=Math.sin(t*.64+1.1)*14;
    pose.openL=pose.openR=open;pose.gazeX=pose.gazeY=0;pose.breath=(1+Math.sin(t*1.6))*.5;pose.bodyAngleX=Math.sin(t*.48-.4)*6;
    for(const id of Object.keys(pose))updateControl(id);
  }
  render();raf=requestAnimationFrame(animate);
}

function beginAnimation(){if(!raf)raf=requestAnimationFrame(animate);}

async function load(){
  try{
    const response=await fetch('./assets/seosoyoung-rig.json');if(!response.ok)throw new Error(`데이터 요청 실패 (${response.status})`);
    const data=await response.json(),image=new Image();image.src='./assets/texture_00.png';await image.decode();
    rig=new Rig(data);renderer=new Renderer(canvas,rig,image);
    for(const parameter of data.parameters)addControl(parameter);
    addParts(data);canvas.hidden=false;byId('loading').hidden=true;
    render();setStatus('준비됨');
    window.viewer={rig,renderer,get pose(){return pose;},get hidden(){return hidden;},render(next={},options={}){setPose(next,{sync:false,draw:false});renderer.render(pose,{hidden:options.hidden===undefined?hidden:new Set(options.hidden),hair:!!options.hair});},setPose,readPixels:()=>renderer.readPixels()};
  }catch(error){
    byId('loading').classList.add('load-error');byId('loading').textContent='화면을 불러오지 못했습니다. 새로고침해 주세요.';setStatus('불러오기 실패');console.error(error);
  }
}

byId('reset').addEventListener('click',()=>{setPose(defaultPose());});
byId('auto').addEventListener('click',event=>{auto=!auto;setToggle(event.currentTarget,auto);if(auto){track=false;setToggle(byId('track'),false);}beginAnimation();});
byId('track').addEventListener('click',event=>{track=!track;setToggle(event.currentTarget,track);if(track){auto=false;setToggle(byId('auto'),false);}beginAnimation();});
viewport.addEventListener('pointermove',event=>{
  if(!track||!renderer)return;const rect=viewport.getBoundingClientRect();
  setPose({u:clamp((event.clientX-rect.left)/rect.width*2-1,-1,1),v:clamp(1-(event.clientY-rect.top)/rect.height*2,-1,1)});
});
viewport.addEventListener('pointerleave',()=>{if(track)setPose({u:0,v:0});});
load();
