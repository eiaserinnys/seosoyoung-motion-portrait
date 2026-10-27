"""Coordinator correction2: thick dark components; condition2 only."""
import numpy as np,cv2
from scipy.ndimage import binary_opening,binary_dilation

def make_hair_mass(rgb):
 gray=rgb.astype(float).mean(2);dark=gray<90;y,x=np.mgrid[-3:4,-3:4];opened=binary_opening(dark,structure=x*x+y*y<=9)
 n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(opened),connectivity=8);ids=[i for i in range(1,n) if int(stats[i,cv2.CC_STAT_AREA])>=2000]
 kept=np.isin(labels,ids);dilated=binary_dilation(binary_dilation(kept,iterations=2),iterations=4)
 return dilated,{'grayDefinition':'mean RGB, same as existing band','grayThreshold':90,'openingDiskRadius':3,'minimumComponentArea':2000,'connectivity':8,'keptComponentAreas':[int(stats[i,cv2.CC_STAT_AREA]) for i in ids],'darkPixels':int(dark.sum()),'openedPixels':int(opened.sum()),'keptPixels':int(kept.sum()),'dilatedPixels':int(dilated.sum()),'dilationStages':[2,4]}
