"""Additional hidden-row Dirichlet constraints in du only; Z and face t frozen."""
import numpy as np

def recover_hidden(system,z,tface,hidden):
 d=system['d']-np.asarray(z)[:,None]*system['b'];w=np.array(system['pose']['weights']);m=system['r']+np.diag(w)
 fixed=np.zeros(len(z),bool);fixed[system['pose']['pins']]=True
 for i in hidden:fixed[i]=True
 du=np.zeros((len(z),2));du[fixed]=(d-tface)[fixed]
 for i,target in hidden.items():du[i]=target-system['a'][i]-z[i]*system['b']-tface
 free=~fixed
 if free.any():
  rhs=w[:,None]*(d-tface);du[free]=np.linalg.solve(m[np.ix_(free,free)],rhs[free]-m[np.ix_(free,fixed)]@du[fixed])
 return du

def restore(system,z,tface,initial_du,rest,boundary,ear_lines,max_iterations=3):
 from earmath import edge_at
 du=np.array(initial_du).copy();hidden={};history=[];observed=set(system['pose']['pins']);observed_overrides=[]
 for iteration in range(max_iterations):
  world=system['a']+np.asarray(z)[:,None]*system['b']+tface+du;added=[]
  for side,line,yc in ear_lines:
   ids=[i for i in boundary if (rest[i,0]>502 if side=='right' else rest[i,0]<502) and i not in hidden];ys=world[ids,1];cx=edge_at(line,ys,side)
   for j,i in enumerate(ids):
    if not np.isfinite(cx[j]) or ys[j]<yc:continue
    limit=cx[j]+6 if side=='right' else cx[j]-6
    violated=world[i,0]<limit if side=='right' else world[i,0]>limit
    if violated:
     hidden[i]=np.array([limit,world[i,1]]);added.append(i)
     if i in observed:observed_overrides.append(i)
  if not added:break
  du=recover_hidden(system,z,np.asarray(tface),hidden);history.append({'iteration':iteration+1,'newPinIds':added,'newPins':len(added),'totalHiddenPins':len(hidden)})
 return du,{'iterations':len(history),'hiddenPinCount':len(hidden),'history':history,'hiddenTargets':{str(i):q.tolist() for i,q in hidden.items()},'observedPinsAlsoActive':observed_overrides,'faceTranslationFrozen':True,'ZFrozen':True}
