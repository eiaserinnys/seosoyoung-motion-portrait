// Refit only the face. Hair, cranium, attachments, atlas and rig stay authored.
import {execFileSync} from 'node:child_process';
import {copyFileSync} from 'node:fs';
const root=new URL('./',import.meta.url);
for(const script of ['fit_features.py','fit_outline.py']){
 execFileSync('python3',[new URL('pipeline/'+script,root).pathname],{cwd:root,stdio:'inherit'});
}
copyFileSync(new URL('source/viewer/index.html',root),new URL('bundle/index.html',root));
console.log('PASS face-only fits and viewer HTML; hair/atlas/rig untouched');
