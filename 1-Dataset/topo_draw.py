'''draw network'''
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

path='D:\\0-Data\\9-stcleak4\\'
error='0100'
pipeinfo=pd.read_excel(path+'netcsv\\pipeinfo.xlsx')
for i in range(len(pipeinfo)):
    cord=eval(pipeinfo.loc[i,'坐标'])
    x1,y1,x2,y2=float(cord[0][0]),float(cord[0][1]),float(cord[-1][0]),float(cord[-1][1])
    plt.plot([x1,x2], [y1,y2])

name='topo'
plt.title(name)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(path+'pic\\'+name+'.png')
plt.show()