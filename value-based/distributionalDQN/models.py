import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    '''DQN Network.
    
    Attribute:
        fc1: the first fully connected layer
        fc2: the second fully connected layer
        output_layer: the output layer
        
    Arg:
        input_dim: the shape of inputs
        out_dim: the shape of output
        hidden1: the number of neural cell in fc1
        hidden2: the number of neural cell in fc2
        '''
    def __init__(self,input_dim,out_dim,atom_size,support):
        super(Network,self).__init__()
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,128)
        self.output_layer=nn.Linear(128,out_dim*atom_size)
        

        self._initialize()

    def _initialize(self):
        # xavier intialization
        # see <<understanding the difficulty of training deep feedforward neural networks>>
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data,0.0)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.constant_(self.fc2.bias.data,0.0)
        nn.init.xavier_uniform_(self.output_layer.weight.data)
        nn.init.constant_(self.output_layer.bias.data,0.0)




    def forward(self,inputs):
        dist=self.distributional(inputs)
        return torch.sum(dist * self.support,dim=2)

    def distributional(self,inputs):
        '''get distribution for atoms'''
        fc1out=F.relu(self.fc1(inputs),inplace=True)
        fc2out=F.relu(self.fc2(fc1out),inplace=True)
        outputs=self.output_layer(fc2out)
        q_atoms=outputs.view(-1,self.out_dim,self.atom_size)
        dist=F.softmax(q_atoms,dim=2)
        return dist


if __name__ == '__main__':
    support=torch.linspace(0.0,8.0,4)
    print(support)
    net=Network(16,2,4,support)
    optim=torch.optim.Adam(net.parameters(),lr=0.005)
    loss_func=nn.MSELoss()
    for i in range(24):
        state=torch.randn(2,16).to(torch.float32)
        action=net.distributional(state)
    
    with torch.no_grad():
        state=torch.randn(2,16).to(torch.float32)
        dist=net.distributional(state)
        action=net.forward(state).argmax(1)
        print("action")
        print(action)
        print("dist")
        print(dist)
        print(dist[range(2),action])

