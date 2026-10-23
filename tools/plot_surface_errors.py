"""Export an aggregate scientific figure; inputs contain scores only, not source meshes."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
files=sys.argv[1:5];out=Path(sys.argv[5])
shapes=['plane','sphere','cylinder','ellipsoid','superellipse4','ellipsoid_relief']
labels=['Plane','Sphere','Cylinder','Ellipsoid','Flat-center cap','Ellipsoid + relief']
fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharex=True)
for i,ax in enumerate(axs):
 for j,label in enumerate(['Divina','Sakura']):
  d=json.loads(Path(files[2*i+j]).read_text())['scopes']['face_and_features']
  values=[100*d[s]['heldout_rms'] for s in shapes];y=np.arange(len(shapes))+(j-.5)*.34
  ax.barh(y,values,height=.3,label=label,color=['#396ca5','#cc793b'][j])
  for yy,v in zip(y,values):ax.text(v+.08,yy,f'{v:.2f}',va='center',fontsize=9)
 ax.set_yticks(np.arange(len(shapes)),labels);ax.invert_yaxis();ax.set_xlim(0,11)
 ax.set_xlabel('Held-out RMS / neutral face width (%)')
 ax.set_title(['Angles calibrated','Same physical angles'][i]);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
axs[0].legend(loc='lower right',frameon=False)
fig.suptitle('Face and features: 4 cardinal poses fitted, 20 other poses evaluated',fontsize=12)
fig.tight_layout();fig.savefig(out,metadata={'Description':'Aggregate numerical errors only. No third-party geometry or artwork.'})

if out.suffix.lower() == '.svg':
 out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines()) + '\n')
