
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from data import trainAp11,trainAp115,trainAp12,trainAp125,trainAp13,trainAp14,trainAp15,trainAp16,trainAp,testAp,apTestFull,apTrainFull

##MDS

md2=apTrainFull.drop(['SK_ID_CURR'],axis=1)
###randomly choose 3% instances
###because so big the dataset is 
resample=md2.sample(frac=0.03,replace=False)
x=resample.drop('TARGET',axis=1)
y=resample['TARGET']
y=y.reset_index(drop=True)
mds= MDS(n_components=2)
X_transformed = mds.fit_transform(x)

v1=list()
v2=list()
for i in X_transformed:
    v1.append(i[0])
    v2.append(i[1])
mds_dt=pd.DataFrame()
mds_dt['v1']=v1
mds_dt['v2']=v2
mds_dt['TARGET']=y

gp=mds_dt.groupby('TARGET')
for i1,i2 in gp:
    plt.plot(i2.v1,i2.v2,'o',label=i1,alpha=0.5)
plt.legend()
plt.show()

##Maybe there exist high dimention trends.
##3-D 
mds3= MDS(n_components=3)
X_transformed = mds3.fit_transform(x)
v1=list()
v2=list()
v3=list()
for i in X_transformed:
    v1.append(i[0])
    v2.append(i[1])
    v3.append(i[2])
mds_dt3=pd.DataFrame()
mds_dt3['v1']=v1
mds_dt3['v2']=v2
mds_dt3['v3']=v3
mds_dt3['TARGET']=y

gp3=mds_dt3.groupby('TARGET')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i1,i2 in gp3:
    ax.scatter( i2['v3'],i2['v1'],i2['v2'],label=i1,alpha=0.3)
plt.legend()
plt.show()
