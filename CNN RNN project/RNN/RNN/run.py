from pandas import DataFrame,read_csv
from numpy import zeros,array,linspace
from model import RNN,Normal
from torch import from_numpy,zeros as torch_zeros,float as torch_float 
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt

def accurate(label,output,dict,real_label):#根据网络输出的张量output预测结果序列，同时根据真实序列real_label计算预测准确率并返回
    label=label.numpy()
    output=output.numpy()
    predict_sequence=""
    i=0
    j=0
    max_value_two=-999
    max_position_two=0
    while i<len(label):
        if real_label[i]==' ':
            break
        while j<len(label[0]):
            if output[i][j]>max_value_two:#认为每一行最大值对应的字符就是结果字符
                max_value_two=output[i][j]
                max_position_two=j    
            j+=1            
        predict_sequence=predict_sequence+dict[max_position_two]
        max_position_two=0
        max_value_two=-999
        i+=1
        j=0
    #print("Predict:%s\n"%predict_sequence) #能够打印比较真实结果与最终预测结果
    #print("Real:%s\n"%real_label[0:i])
    count=0
    for i in range(0,len(predict_sequence)):
        if predict_sequence[i]==real_label[i]:
            count+=1
    
    return (float)(count/len(predict_sequence)),predict_sequence,real_label



plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

data=read_csv("sst3.csv",header=0,index_col=0,sep=",")
sequence_text=data[0:]["seq"].tolist()
label_text=data[0:]["sst3"].tolist()

sequence_chars=set(''.join(sequence_text))
label_chars=set(''.join(label_text))


int2char_sequence=dict(enumerate(sequence_chars))
int2char_label=dict(enumerate(label_chars))

char2int_sequence={char: ind for ind, char in int2char_sequence.items()}
char2int_label={char: ind for ind, char in int2char_label.items()}

maxlen = len(max(sequence_text, key=len))
for i in range(0,len(sequence_text)):
    while len(sequence_text[i])<maxlen:
        sequence_text[i]+=" "
        label_text[i]+=" "

sequence_array=[]
label_array=[]

for i,j in zip(sequence_text,label_text):#将序列转换为矩阵，进而转换为张量
    sequence_ar=zeros((maxlen,21),dtype=int)
    label_ar=zeros((maxlen,3),dtype=int)
    k=0
    for charone,chartwo in zip(i,j):
        if charone==" ":
            break
        sequence_ar[k][char2int_sequence[charone]]=1
        label_ar[k][char2int_label[chartwo]]=1
        k+=1
    sequence_array.append(sequence_ar)
    label_array.append(label_ar)

sequence_array=array(sequence_array)
label_array=array(label_array)
rs=ShuffleSplit(n_splits=1,test_size=0.3)#指定随机分类器
for train,test in rs.split(sequence_array):#获取训练集和测试集
    train_sequence=sequence_array[train]
    train_label=label_array[train]
    test_sequence=sequence_array[test]
    test_label=label_array[test]


inputsize=21
outputsize=3
hiddensize=12
layer=2
i=0
j=0
l=[]
loss_sum=0
iter_num=1
rnn=RNN(inputsize=inputsize,output_size=outputsize,hidden_dim=hiddensize,n_layers=layer)#初始化RNN神经网络
hidden0=torch_zeros(layer,maxlen,hiddensize,dtype=torch_float)#初始化隐藏状态


with open("output.text",'w') as f:
    while i<iter_num:
        for sequence_vec,label_vec in zip(train_sequence,train_label):
            rnn.optimizerone.zero_grad()
            rnn.optimizertwo.zero_grad()
            sequence_vec=from_numpy(sequence_vec).float().view(1,maxlen,21)
            label_vec=from_numpy(label_vec).float().view(maxlen,3)
            #print(sequence_vec,sequence_vec.dim())  
            output,hidden0=rnn.forward(sequence_vec,hidden0)  #计算输出向量
            hidden0=hidden0.detach()    
            loss=rnn.criterion(output,label_vec)
            loss.backward(retain_graph=True)
            rnn.optimizerone.step()#step更新参数
            rnn.optimizertwo.step()
            if j%1000==0 and j>0:
                l.append(loss_sum)
                print("RNN:Already trained sequence:%d------Loss:%.4f"%(j,loss_sum))
                loss_sum=0
                #print(output)
                #print(label_vec)
                ''
                for k in range(0,len(output)):#能够输出张量到文件，用于查看RNN效率
                    f.write(str(output[k])+"\n")
                    f.write(str(label_vec[k])+"\n")
                f.write("\n")
                ''
                
            j+=1
            loss_sum+=loss.item()      
        i+=1
        j=0


x_tick_text=range(1000,1000+11000*iter_num,1000)#绘制训练时每1000条序列的损失总和图像
x_dis=range(1000,1000+11000*iter_num,1000)
plt.plot(x_dis,l)
plt.xticks(x_dis,x_tick_text)
plt.xlabel('训练序列数')
plt.ylabel('loss')
plt.title('RNN训练段损失曲线')
plt.savefig("lossone.png")
plt.close()


accuracy=[]
count=0


with open("output.text",'w') as f:
    for sequence_vec,label_vec in zip(test_sequence,test_label):
        sequence_vec=from_numpy(sequence_vec).float().view(1,maxlen,21)
        label_vec=from_numpy(label_vec).float().view(maxlen,3)
        hidden0=torch_zeros(layer,maxlen,hiddensize,dtype=torch_float)#初始化隐藏状态
        c0=torch_zeros(layer,maxlen,hiddensize,dtype=torch_float)
        output,hidden_this=rnn.forward(sequence_vec,hidden0)#对每一个测试集序列，计算获得输出
        output=output.detach()
        ''#输出张量到文件，用于细致查看张量状态
        if count%1000==0 and count>0:
            for i in range(0,len(output)):
                f.write(str(output[i])+"\n")
                f.write(str(label_vec[i])+"\n")
            f.write("\n")
        ''
        tmp,_,_=accurate(label_vec,output,int2char_label,label_text[count])
        accuracy.append(tmp)
        count+=1

print("Overall accuracy for RNN is:%.4f"%(sum(accuracy)/len(accuracy)))#计算输出总预测精度
#print(accuracy)#用于查看每个序列的准确率

#查看某几个序列及其预测结果
''
count=0
for sequence_vec,label_vec in zip(test_sequence,test_label):
    sequence_vec=from_numpy(sequence_vec).float().view(1,maxlen,21)
    label_vec=from_numpy(label_vec).float().view(maxlen,3)
    hidden0=torch_zeros(layer,maxlen,hiddensize,dtype=torch_float)
    output,_=rnn.forward(sequence_vec,hidden0)#对每一个测试集序列，计算获得输出
    output=output.detach()
    tmp,predict,real=accurate(label_vec,output,int2char_label,label_text[count])
    if count>6:
        break;
    count+=1
    print("Predict:%s"%predict)
    print("Truth:%s"%real)
''


print("\n")


#接下来训练MLP
normal=Normal(inputsize,outputsize,70,20)
i=0
j=0
k=0
iter_num=1
l=[]
loss_sum=0
f=open("output_normal.text","w")
while i<iter_num:
    normal.optimizer.zero_grad()
    for sequence_vec,label_vec in zip(train_sequence,train_label):
        sequence_vec=from_numpy(sequence_vec).float().view(1,maxlen,21)
        label_vec=from_numpy(label_vec).float().view(maxlen,3)
        output=normal.forward(sequence_vec)
        loss=normal.criterion(label_vec,output)
        loss.backward()
        normal.optimizer.step()
        normal.zero_grad()
        if j%1000==0 and j>0:
            l.append(loss_sum)
            print("MLP:Already trained sequence:%d------Loss:%.4f"%(j,loss_sum))
            loss_sum=0
            #print(output)
            #print(label_vec)
            ''
            for k in range(0,len(output)):#能够输出张量到文件，用于查看网络效率
                f.write(str(output[k])+"\n")
                f.write(str(label_vec[k])+"\n")
            f.write("\n")
            ''
                
        j+=1
        loss_sum+=loss.item()  
    j=0
    i+=1
    loss_sum=0

x_tick_text=range(1000,1000+11000*iter_num,1000)#绘制训练时每1000条序列的损失总和图像
x_dis=range(1000,1000+11000*iter_num,1000)
plt.plot(x_dis,l)
plt.xticks(x_dis,x_tick_text)
plt.xlabel('训练序列数')
plt.ylabel('loss')
plt.title('MLP训练段损失曲线')
plt.savefig("losstwo.png")

count=0
accuracy=[]
for sequence_vec,label_vec in zip(test_sequence,test_label):
        sequence_vec=from_numpy(sequence_vec).float().view(1,maxlen,21)
        label_vec=from_numpy(label_vec).float().view(maxlen,3)
        output=normal.forward(sequence_vec)#对每一个测试集序列，计算获得输出
        output=output.detach()
        ''#输出张量到文件，用于细致查看张量状态
        if count%1000==0 and count>0:
            for i in range(0,len(output)):
                f.write(str(output[i])+"\n")
                f.write(str(label_vec[i])+"\n")
            f.write("\n")
        ''
        #if count%1000==0:
        tmp,_,_=accurate(label_vec,output,int2char_label,label_text[count])
        accuracy.append(tmp)
        count+=1

print("Overall accuracy for MLP is:%.4f"%(sum(accuracy)/len(accuracy)))#计算输出总预测精度
#print(accuracy)#用于查看每个序列的准确率

#查看某几个序列及其预测结果


count=0
for sequence_vec,label_vec in zip(test_sequence,test_label):
    sequence_vec=from_numpy(sequence_vec).float().view(1,maxlen,21)
    label_vec=from_numpy(label_vec).float().view(maxlen,3)
    output=normal.forward(sequence_vec)#对每一个测试集序列，计算获得输出
    output=output.detach()
    tmp,predict,real=accurate(label_vec,output,int2char_label,label_text[count])
    if count>6:
        break;
    count+=1
    print("Predict:%s"%predict)
    print("Truth:%s"%real)









