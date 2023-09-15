#使用K-means进行聚类
from sklearn.cluster import KMeans
import pandas as pd
#提取数据
data=pd.read_csv("iris_data.txt",header=0,sep=",")
olddata=data
type=data.iloc[0:,-1]
type=type.to_list()
data=data.iloc[0:,0:-1]
data=data.to_numpy()
#进行训练
classifyer=KMeans(3,init="random",n_init=3).fit(data)
#输出结果
print(classifyer.labels_)
print(classifyer.cluster_centers_)
#保存结果于文件中
olddata["Predict"]=classifyer.labels_
olddata.to_csv("iris_data_predict.csv",index=False)





