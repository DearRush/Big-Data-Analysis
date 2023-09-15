import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class CNNnet(nn.Module):
    def __init__(self) :
        super(CNNnet,self).__init__()#继承公用类nn.Module
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fulcon1=nn.Linear(16*5*5,120)
        self.fulcon2=nn.Linear(120,84)
        self.fulcon3=nn.Linear(84,10)
        self.lossfunc=nn.CrossEntropyLoss()#该损失函数可以用于两长度不同向量的比较
        self.optimise=optim.SGD(self.parameters(),lr=0.001,momentum=0.9)#发现momentum=0.9极高地提升了网络预测性能
    
    def forward(self,x):
        x=f.max_pool2d(f.relu(self.conv1(x)),2)
        x=f.max_pool2d(f.relu(self.conv2(x)),2)
        x=x.view(-1,16*5*5)
        x=f.relu(self.fulcon1(x))
        x=f.relu(self.fulcon2(x))
        x=self.fulcon3(x)

        return x
        
