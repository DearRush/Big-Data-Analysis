#使用KNN算法执行分类任务
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve

#数据清洗
heart=pd.read_csv("heart-disease_data.txt",sep=",",header=0,na_values="?")#heart-dataframe object
data=heart.to_numpy()#data:numpy-array
imputer=SimpleImputer(strategy='median')#使用中位数代替缺失值
imputer.fit(data)
data=imputer.transform(data)
rowsize,colsize=data.shape
x_data=data[0:,1:-2]
y_data_ori=data[:,-1]
y_data=[]
for i in y_data_ori:
    if i>0:
        y_data.append(0)
    else :
        y_data.append(1)
y_data=np.array(y_data)

#简单随机分组
selector=ShuffleSplit(n_splits=1,test_size=0.25)
for train_index,test_index in selector.split(x_data):
    x_train,y_train=x_data[train_index],y_data[train_index]
    x_test,y_test=x_data[test_index],y_data[test_index]

#拟合
classifier=KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train,y_train)

#输出拟合结果
time=0
predict_list=[]
proba=[]
for i in x_test:
    predict=classifier.predict(i.reshape(1,-1))[0]
    predict_list.append(predict)
    proba.append(classifier.predict_proba(i.reshape(1,-1))[0][1])
print("Accuracy:%.2f"%accuracy_score(y_test,predict_list))
for i in range(0,len(proba)):#为了做ROC图，根据正例概率大小对测试集样本进行排序
    for j in range(0,len(proba)-1-i):
        if proba[j+1]>proba[j]:
            tmp=proba[j+1]
            proba[j+1]=proba[j]
            proba[j]=tmp
            tmp=y_test[j+1]
            y_test[j+1]=y_test[j]
            y_test[j]=tmp
'''
使用roc_curve()函数直接求ROC图坐标(不精确)
fpr,tpr,threshold=roc_curve(y_test,proba,pos_label=1)
for i in range(0,len(fpr)):
    print("FPR:%.2f TPR:%.2f"%(fpr[i],tpr[i]))
'''

x=0
y=0
x_list=[]
y_list=[]
auc=0
prex=0
positive_number=(y_test==1).sum()
negative_number=(y_test==0).sum()
for i in range(0,len(y_test)):#计算ROC曲线上的各个点,同时计算AUC值
    x_list.append(x)
    y_list.append(y)
    if y_test[i]==1:
        y+=(float)(1/positive_number)
    else :
        x+=(float)(1/negative_number)
print("FPR TPR")
for i in range(0,len(y_test)):
    print("%.2f %.2f"%(x_list[i],y_list[i]))
print("AUC Value:%.2f"%roc_auc_score(y_test,proba))


