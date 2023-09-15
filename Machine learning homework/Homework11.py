#使用决策树算法进行分类任务
import pandas as pd
import numpy as np
import sklearn.tree as tree
import sklearn.impute as imp
import sklearn.model_selection as ms

#数据清洗
heart=pd.read_csv("heart-disease_data.txt",sep=",",header=0,na_values="?")#heart-dataframe object
data=heart.to_numpy()#data:numpy-array
imputer=imp.KNNImputer(n_neighbors=8)#为了处理缺失值,使用KNN算法预测缺失值
data=imputer.fit_transform(data)
row_number,col_number=data.shape
x_data=data[0:,1:-2]#x_data:numpy-2Darray
y_data_ori=data[0:,-1:]
y_data=[]
for i in y_data_ori:
    if i>0:
        y_data.append(1)
    else :
        y_data.append(0)
y_data=np.array(y_data)#y_data:numpy-1Darray

#随机分组（training and testing）
Shuffle=ms.StratifiedShuffleSplit(test_size=0.3,n_splits=1)#为实现分层随机抽样
for train_index, test_index in Shuffle.split(x_data, y_data):
    x_train, x_test = x_data[train_index], x_data[test_index]#x_train,x_test 2Darray
    y_train, y_test = y_data[train_index], y_data[test_index]#y_train;y_test都是一维数组

#拟合
desiciontree=tree.DecisionTreeClassifier()#构建决策树
desiciontree=desiciontree.fit(x_train.astype(int),y_train.astype(int))

#结果统计
total=len(y_test)
correct=0
proba=[]#预测为正例的概率
time=0
for i in x_test:#统计正确预测的概率
    predict=desiciontree.predict(i.reshape(1,-1))[0]
    if predict==y_test[time]:
        correct=correct+1
    proba.append(desiciontree.predict_proba(i.reshape(1,-1))[0][1])
    time+=1
print("Total Accuracy:%.2f"%(correct/total))
for i in range(0,total):#为了做ROC图，根据正例概率大小对测试集样本进行排序
    for j in range(0,total-1-i):
        if proba[j+1]>proba[j]:
            tmp=proba[j+1]
            proba[j+1]=proba[j]
            proba[j]=tmp
            tmp=y_test[j+1]
            y_test[j+1]=y_test[j]
            y_test[j]=tmp
#计算ROC图点坐标
x=0
y=0
x_list=[]
y_list=[]
auc=0
prex=0
positive_number=(y_test==1).sum()
negative_number=(y_test==0).sum()
for i in range(0,total):
    x_list.append(x)
    y_list.append(y)
    if y_test[i]==1:
        y+=(float)(1/positive_number)
    else :
        x+=(float)(1/negative_number)
#输出绘制AUC图时的点坐标
print("FPR TPR")
for i in range(0,len(y_test)):
    print("%.2f %.2f"%(x_list[i],y_list[i]))
#计算、输出AUC值
for i in range(0,total-1):
    if y_list[i]!=y_list[i+1]:
        auc+=y_list[i]*(x_list[i]-prex)
        prex=x_list[i]
print("AUC:%.2f"%(auc))








