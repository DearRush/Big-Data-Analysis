#使用DBSCAN算法进行聚类
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np

#首先提取数据
data=pd.read_csv("iris_data.txt",header=0,sep=",")
olddata=data
type=data.iloc[0:,-1]
typelist=[]
data=data.iloc[0:,0:-1]
data=data.to_numpy()
#为精确确定eps的值，计算组内距离均值
row,_=data.shape
dis=[]
for i in range(0,row):
    for j in range(i+1,row):
        first=data[i,:]
        second=data[j,:]
        if type[i]==type[j]:
            first=first.reshape(1,-1)
            second=second.reshape(1,-1)
            distance=euclidean_distances(first,second)
            dis.append(distance[0][0])
dis=np.array(dis)
ave=np.average(dis)
#初始化聚类器
db=DBSCAN(eps=ave,min_samples=20).fit(data)
#探索从ave/2~ave的最佳eps值
'''
oriave=(float)(ave/2)
while oriave<=ave:
    db=DBSCAN(eps=oriave,min_samples=20).fit(data)
    print(db.labels_)
'''
#输出保存
print(db.labels_)
olddata["Predict"]=db.labels_
olddata.to_csv("iris_data_predict2.csv",index=False)
