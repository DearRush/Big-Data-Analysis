import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.functional import log_softmax,nll_loss
from os.path import exists
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


img_size = 150*150
outputsize = 14
iteration = 1
dict={0:'+',1:'/',2:'8',3:'5',4:'4',5:'*',6:'9',7:'1',8:'7',9:'6',10:'-',11:'3',12:'2',13:'0'}#用于解出真实值

trasformer=torchvision.transforms.Compose([torchvision.transforms.Grayscale(1),torchvision.transforms.ToTensor()])
train_dataset = ImageFolder('./final_symbols_split_ttv/train',transform=trasformer)
test_dataset = ImageFolder('./final_symbols_split_ttv/test',transform=trasformer)
batchsize=64


train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batchsize,shuffle=True)


class MLP(torch.nn.Module):
    def __init__(self,inputsize,hiddenone,hiddentwo,hiddenthree,outputsize):
        super(MLP,self).__init__()
        self.linear1 = torch.nn.Linear(inputsize,hiddenone)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hiddenone,hiddentwo)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hiddentwo,hiddenthree)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(hiddenthree,outputsize)
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.linear4(self.relu3(x))
        return log_softmax(x)
    
network = MLP(img_size,1000,225,25,outputsize)
train_losses=[]
train_number=[]

optimizer = torch.optim.SGD(network.parameters(),lr=0.01)

if exists("model_mlp.pth")==True:
    network_state_dict = torch.load('model_mlp.pth')
    network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('optimizer_mlp.pth')
    optimizer.load_state_dict(optimizer_state_dict)
else:
    for iteration in range(iteration):
        loss_sum= 0
        for index, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view(data.size(0),-1)
            outputs = network(data)
            loss = nll_loss(outputs,target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_number.append((index * batchsize) + ((iteration) * len(train_loader.dataset)))
            loss_sum += loss.data
            print('MLP Train Iteration: %d ----%d/%d (%.2f%%)----Loss: %.6f'%(iteration+1,index * len(data),len(train_loader.dataset),100.0* index / len(train_loader),loss.item()))
            if index%20==0 and index>0:
                torch.save(network.state_dict(), './model_mlp.pth')
                torch.save(optimizer.state_dict(), './optimizer_mlp.pth')
    torch.save(network.state_dict(), './model_mlp.pth')
    torch.save(optimizer.state_dict(), './optimizer_mlp.pth')
    print( "After %d epoch , training loss is %.2f"%(iteration,loss_sum))
    figure = plt.figure()#绘制loss图像
    plt.plot(train_number, train_losses, color='red')
    plt.xlabel('Number of training images')
    plt.ylabel('Loss of MLP')
    plt.title("MLP训练损失函数")
    figure.show()
    plt.savefig("Loss_mlp.png")
    plt.close()

accuracy = 0
count=0
for data,label in test_loader:
    img = data.view(data.size(0),-1)
    output = network(img)
    loss = nll_loss(output,label)
    _,pred = output.data.max(1)
    num_correct = pred.eq(label).sum()
    accuracy += num_correct.data
    count+=1
    print("MLP accuracy :%d/%d(%.4f%%)"%(accuracy,count*batchsize,100*accuracy.float()/(count*batchsize)))
print('MLP Overall accuracy: %.4f%%'%(100*accuracy.float()/(len(test_dataset)))) 

images = enumerate(test_loader)
index, (matrix, label) = next(images)
input=matrix.view(matrix.size(0),-1)
output = network(input)
fig = plt.figure()
for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.tight_layout()
    plt.imshow(matrix[i][0])
    plt.title("Prediction: the symbol is %s"%(dict[output.data.max(1, keepdim=True)[1][i].item()]))
    plt.xticks([])
    plt.yticks([])
fig.savefig("show_mlp.png")
plt.show()




  
