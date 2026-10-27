"""Cartesian disk coordinates keep Powell evaluations feasible and incumbents valid."""
import numpy as np

def encode(theta):
 q=np.asarray(theta,float).copy();extent=np.sqrt(max(0.,6400.-q[0]*q[0]));q[1]=q[1]/extent if extent>0 else 0.;return q

def decode(x):
 q=np.asarray(x,float).copy();extent=np.sqrt(max(0.,6400.-q[0]*q[0]));q[1]=np.nextafter(extent,0.)*q[1];return q

class Incumbent:
 def __init__(self):self.score=-1.;self.theta=None
 def observe(self,theta,score):
  if score>self.score:self.score=float(score);self.theta=np.array(theta,float).copy()
