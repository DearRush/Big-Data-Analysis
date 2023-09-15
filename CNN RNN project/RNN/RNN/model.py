from torch import nn,optim,float
from torch.nn.functional import log_softmax

class RNN(nn.Module):#RNN和单层感知机
    def __init__(self,inputsize,output_size, hidden_dim, n_layers):
        super(RNN,self).__init__()#继承公用类nn.Module
        self.hidden_dim=hidden_dim
        self.layer=n_layers
        self.inputsize=inputsize
        self.outputsize=output_size
        self.rnn=nn.RNN(inputsize,hidden_dim,n_layers)
        self.fc=nn.Linear(hidden_dim,output_size)
        self.criterion=nn.CrossEntropyLoss()
        self.optimizerone=optim.RMSprop(self.rnn.parameters(),lr=0.1,momentum=0.7)
        self.optimizertwo=optim.SGD(self.fc.parameters(),lr=0.1,momentum=0.7)
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0,std=0.001)
        for p in self.fc.parameters():
            nn.init.normal_(p,mean=0,std=0.001)

    def forward(self,x,hidden0):
        x,hidden0=self.rnn(x,hidden0)
        x=x.view(-1,self.hidden_dim)
        x=self.fc(x)
        x=log_softmax(x)
        x=x.view(-1,self.outputsize)
        return x,hidden0

class Normal(nn.Module):#三层感知机
    def __init__(self,inputsize,output_size,hide_one,hide_two) :
        super(Normal,self).__init__()
        self.layer1=nn.Linear(inputsize,hide_one)
        self.acti1=nn.ReLU()
        self.layer2=nn.Linear(hide_one,hide_two)
        self.acti2=nn.Sigmoid()
        self.layer3=nn.Linear(hide_two,output_size)
        self.criterion=nn.MSELoss()
        self.optimizer=optim.SGD(self.parameters(),lr=0.003,momentum=0.9)
        self.outputsize=output_size
    def forward(self,x):
        x=self.layer1(x)
        x=self.acti1(x)
        x=self.layer2(x)
        x=self.acti2(x)
        x=self.layer3(x)
        x=x.view(-1,self.outputsize)

        return x



