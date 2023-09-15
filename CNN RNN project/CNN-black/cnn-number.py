import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from os.path import exists
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

trasformer=torchvision.transforms.Compose([torchvision.transforms.Grayscale(1),torchvision.transforms.ToTensor()])#图片为黑白图片，因此不需要三层像素
batchsize=64
train_loader = DataLoader(torchvision.datasets.ImageFolder('./final_symbols_split_ttv/train',transform=trasformer),batch_size=batchsize,shuffle=True)
test_loader = DataLoader(torchvision.datasets.ImageFolder('./final_symbols_split_ttv/test',transform=trasformer),batch_size=batchsize,shuffle=True)
dict={0:'+',1:'/',2:'8',3:'5',4:'4',5:'*',6:'9',7:'1',8:'7',9:'6',10:'-',11:'3',12:'2',13:'0'}#用于解出真实值


'''用于查看数据格式
images = enumerate(train_loader)
index, (imagedata, real_label) = next(images)
print(imagedata)
print(real_label)
'''

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=11)#卷积层1
        self.conv2 = nn.Conv2d(6, 12, kernel_size=11)#卷积层2
        self.conv3 = nn.Conv2d(12, 24, kernel_size=11)#卷积层3
        self.fc1 = nn.Linear(384, 70)#全连接层1
        self.fc2 = nn.Linear(70, 14)#全连接层2

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 5))
        x = x.view(-1, 384)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

network = Cnn()
iteration = 1
learning = 0.01
sumnumber = 100
train_losses = []
train_counter = []
optimizer = optim.SGD(network.parameters(), lr=learning,momentum=0.5)


def train(iteration):
    for index, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data) 
        loss = F.nll_loss(output, target)#使用专用于分类任务的损失函数
        loss.backward()
        optimizer.step()
        print('Train Iteration: %d ----%d/%d (%.2f%%)----Loss: %.6f'%(iteration,index * len(data),len(train_loader.dataset),100.0* index / len(train_loader),loss.item()))
        train_losses.append(loss.item())
        train_counter.append((index * batchsize) + ((iteration - 1) * len(train_loader.dataset)))
        torch.save(network.state_dict(), './model.pth')
        torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    success_num = 0
    count=0
    for data, target in test_loader:
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1]
        success_num += pred.eq(target.data.view_as(pred)).sum()
        count+=batchsize
        if count>0:
            print("Test accuracy :%d/%d(%.4f)"%(success_num,count,(float)(success_num/count)))
    print("Test overall accuracy:%d/%d(%.2f)"%(success_num, len(test_loader.dataset), (float)(success_num / len(test_loader.dataset))))

if exists("model.pth")==False:#如果之前从未训练、保存网络
    for i in range(1,iteration+1):
        train(i)
    figure = plt.figure()#绘制loss图像
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('Number of training images')
    plt.ylabel('Loss')
    plt.title("Cnn的迭代损失函数")
    figure.show()
    plt.savefig("Loss_cnn.png")
    plt.close()
    test() 
else:
    network_state_dict = torch.load('model.pth')
    network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('optimizer.pth')
    optimizer.load_state_dict(optimizer_state_dict)   
    test()

#展示预测结果
images = enumerate(test_loader)
index, (matrix, label) = next(images)
output = network(matrix)
fig = plt.figure()
for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.tight_layout()
    plt.imshow(matrix[i][0])
    plt.title("Prediction: the symbol is %s"%(dict[output.data.max(1, keepdim=True)[1][i].item()]))
    plt.xticks([])
    plt.yticks([])
plt.savefig("show_cnn.png")
plt.show()
