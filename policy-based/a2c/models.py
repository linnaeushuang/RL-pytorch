import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    ''' actor policy'''
    def __init__(self,input_dim,out_dim):
        super(ActorNetwork,self).__init__()
        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,out_dim)
                    
    def forward(self,inputs):
        out=F.relu(self.fc1(inputs),inplace=True)
        out=F.relu(self.fc2(out),inplace=True)
        out=F.softmax(self.fc3(out))
        return out
class CriticNetwork(nn.Module):
    '''critic policy,state value'''
    def __init__(self,input_dim):
        super(CriticNetwork,self).__init__()
        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,1)
    def forward(self,inputs):
        out=F.relu(self.fc1(inputs),inplace=True)
        out=F.relu(self.fc2(out),inplace=True)
        out=self.fc3(out)
        return out
