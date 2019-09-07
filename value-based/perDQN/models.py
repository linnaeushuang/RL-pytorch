import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self,input_dim,out_dim,hidden1=128,hidden2=128):
        super(Network,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden1)
        self.fc2=nn.Linear(hidden1,hidden2)
        self.output_layer=nn.Linear(hidden2,out_dim)
        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias.data,0.0)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.constant_(self.fc2.bias.data,0.0)
        nn.init.xavier_uniform_(self.output_layer.weight.data)
        nn.init.constant_(self.output_layer.bias.data,0.0)




    def forward(self,inputs):
        fc1out=F.relu(self.fc1(inputs),inplace=True)
        fc2out=F.relu(self.fc2(fc1out),inplace=True)
        return self.output_layer(fc2out)


if __name__ == '__main__':
    net=Network(16,1)
    optim=torch.optim.Adam(net.parameters(),lr=0.005)
    loss_func=nn.MSELoss()
    for i in range(24):
        state=torch.randn(16).to(torch.float32)
        action=net.forward(state)
        print(action)

