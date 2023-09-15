import torch
import time
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import CNNdefine

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./CNN/data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True)

testset=torchvision.datasets.CIFAR10(root='./CNN/data',download=True,train=False,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True)

i=0
for index,j in enumerate(testloader,0):
    print(index)
    print(j)
    i+=1
    if i>=1:
        break


'''
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net=CNNdefine.CNNnet()
times=2
start=time.time()


for i in range(0,times):
    runtimeloss=0
    for index,j in enumerate(trainloader,0):
        data,label=j
        output=net(data)
        loss=net.lossfunc(output,label)
        runtimeloss+=loss.item()
        net.optimise.zero_grad()
        loss.backward()
        net.optimise.step()
        if index%3000==2999:
            print("Round %d-[1:%d] loss: %f"%(i+1,index,runtimeloss))
            runtimeloss=0

print("Finish training in %.2f seconds"%(time.time()-start))

correct=0
total=0
for index, j in enumerate(testloader,0):
    inputdata,label=j
    output=net(inputdata)
    _,predicted=torch.max(output,dim=1)
    correct+=(predicted==label).sum().item()
    total=total+label.size(0)
print("The overall accuracy is %f%%"%(100*correct/total))
'''






