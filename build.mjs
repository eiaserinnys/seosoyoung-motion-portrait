// Refit only the face. Hair, cranium, attachments, atlas and rig stay authored.
import {execFileSync} from 'node:child_process';
const root=new URL('./',import.meta.url);
execFileSync('python3',[new URL('pipeline/build_face_only.py',root).pathname],{cwd:root,stdio:'inherit'});
