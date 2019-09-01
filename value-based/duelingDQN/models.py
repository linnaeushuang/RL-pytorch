import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self,input_dim,out_dim,hidden1=128,hidden2=128):
        super(Network,self).__init__()

        self.feature_layer = nn.Linear(input_dim,hidden1)
        self.feature_layer2= nn.Linear(hidden1,hidden2)

        # see figure1 in original paper
        self.advantage_layer = nn.Linear(hidden1,hidden2)
        self.advantage_output_layer = nn.Linear(hidden2,out_dim)

        self.value_layer = nn.Linear(hidden1,hidden2)
        self.value_output_layer = nn.Linear(hidden2,1)

        self._initialize()

    def _initialize(self):

        nn.init.xavier_uniform_(self.feature_layer.weight.data)
        nn.init.constant_(self.feature_layer.bias.data,0.0)
        nn.init.xavier_uniform_(self.feature_layer2.weight.data)
        nn.init.constant_(self.feature_layer2.bias.data,0.0)

        nn.init.xavier_uniform_(self.advantage_layer.weight.data)
        nn.init.constant_(self.advantage_layer.bias.data,0.0)
        nn.init.xavier_uniform_(self.advantage_output_layer.weight.data)
        nn.init.constant_(self.advantage_output_layer.bias.data,0.0)
        
        nn.init.xavier_uniform_(self.value_layer.weight.data)
        nn.init.constant_(self.value_layer.bias.data,0.0)
 
        nn.init.xavier_uniform_(self.value_output_layer.weight.data)
        nn.init.constant_(self.value_output_layer.bias.data,0.0)





    def forward(self,inputs):
        feature = F.relu(self.feature_layer(inputs),inplace=True)
        feature = F.relu(self.feature_layer2(feature),inplace=True)

        value = F.relu(self.value_layer(feature),inplace=True)
        value = self.value_output_layer(value)

        advantage = F.relu(self.advantage_layer(feature),inplace=True)
        advantage = self.advantage_output_layer(advantage)

        # equation(9) in original paper
        return value + advantage - advantage.mean(dim=-1,keepdim=True)

if __name__ == '__main__':
    net=Network(16,3)
    optim=torch.optim.Adam(net.parameters(),lr=0.005)
    loss_func=nn.MSELoss()
    for i in range(24):
        state=torch.randn(11,16).to(torch.float32)
        action=net.forward(state)
        print(action)

