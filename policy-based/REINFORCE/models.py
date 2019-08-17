import torch
import torch.nn as nn
import torch.nn.functional as F


class discreteNetwork(nn.Module):
    ''' discrete policy'''
    def __init__(self,input_dim,out_dim):
        super(discreteNetwork,self).__init__()
        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,out_dim)
                    
    def forward(self,inputs):
        out=F.relu(self.fc1(inputs),inplace=True)
        out=F.relu(self.fc2(out),inplace=True)
        out=F.softmax(self.fc3(out))
        return out
class continuousNetwork(nn.Module):
    '''continuous policy'''
    def __init__(self,input_dim,out_dim):
        super(continuousNetwork,self).__init__()
        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,out_dim)
        self.fc3=nn.Linear(128,out_dim)
    def forward(self,inputs):
        out=F.relu(self.fc1(inputs),inplace=True)
        mu=self.fc2(out)
        sigma_sq=self.fc3(out)
        return mu,sigma_sq
