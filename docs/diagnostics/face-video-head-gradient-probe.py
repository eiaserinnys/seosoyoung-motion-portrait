import resource,sys,json
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
sys.path.insert(0,'/home/eias/workspace/.projects/seosoyoung-motion-portrait--feat-face-video-refit-f0ed41f9/pipeline')
import numpy as np
from face_video_contour import project
from face_video_contour_fit import edge_incidence
from face_video_head_fit import FrozenProblem,evaluate,actual_objective,apply_head
rest=np.array([[492.,316.],[512.,315.],[513.,336.],[491.,337.]])
tri=np.array([[0,1,2],[0,2,3]]);z=np.array([20.,21.,19.,20.]);pose=[10.,-8.,3.]
base=project(np.column_stack((rest-[502,326],z)),pose)
target=apply_head(base,[.7,-.4,2.62])
obs=np.array([(1-t)*target[a]+t*target[b] for a,b in [(0,1),(1,2),(2,3),(3,0)] for t in np.linspace(0,1,13,endpoint=False)])
rows=[dict(frame=1,pose=pose,pointsArt=obs.tolist(),keep=[True]*len(obs))]
inc=edge_incidence(rest,tri);factor=np.zeros((1,1));strength=.1
field=np.zeros((1,4,3));head=np.zeros((1,3));x=np.r_[field.ravel(),head.ravel()]
frames=evaluate(rest,z,tri,rows,field,head);f=FrozenProblem(rest,z,frames,inc,factor,strength)
gf=f.jacobian(x).T@f.residual(x)
def actual(v):
 c,h=f.unpack(v);fs=evaluate(rest,z,tri,rows,c,h)
 return actual_objective(fs,c,h,strength,inc,factor),fs
out={'shape':[1,4,3],'base_pair_counts':[(v['pairs']['forward_count'],v['pairs']['reverse_count']) for v in frames],'base_objective':actual(x)[0]}
for eps in [1e-3,1e-4,1e-5]:
 ga=[];counts=set()
 for j in range(len(x)):
  e=np.zeros_like(x);e[j]=eps
  plus,fp=actual(x+e);minus,fm=actual(x-e);ga.append((plus-minus)/(2*eps))
  for fr in fp+fm:counts.add((fr['pairs']['forward_count'],fr['pairs']['reverse_count']))
 ga=np.array(ga);idx=int(np.argmax(abs(ga-gf)))
 d=-gf;out[str(eps)]={'max_grad_difference':float(abs(ga-gf).max()),'largest_index':idx,'frozen_gradient':float(gf[idx]),'actual_gradient':float(ga[idx]),'pair_counts':sorted(counts),'descent_frozen':float(gf@d),'descent_actual':float(ga@d),'frozen_norm':float(np.linalg.norm(gf)),'actual_norm':float(np.linalg.norm(ga))}
 direction=ga-gf*(ga@gf)/(gf@gf)
 out[str(eps)]['orthogonal_direction_frozen']=float(gf@direction);out[str(eps)]['orthogonal_direction_actual']=float(ga@direction)
print(json.dumps(out,indent=2))
