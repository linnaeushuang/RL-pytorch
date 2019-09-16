
import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(Network,self).__init__()
        self.critic=CriticNetwork(input_dim)
        self.actor=ActorNetwork(input_dim,out_dim)

    def forward(self,inputs):
        value=self.critic(inputs)
        mean,sigma=self.actor(inputs)
        return value,mean,sigma

class ActorNetwork(nn.Module):
    ''' actor policy'''
    def __init__(self,input_dim,out_dim):
        super(ActorNetwork,self).__init__()
        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,128)
        self.action_mean=nn.Linear(128,out_dim)
        self.sigma_log=nn.Parameter(torch.zeros(1,out_dim))
                    
    def forward(self,inputs):
        out=F.relu(self.fc1(inputs),inplace=True)
        out=F.relu(self.fc2(out),inplace=True)
        mean=self.action_mean(x)
        sigma_log=self.sigma_log.expand_as(mean)
        sigma=torch.exp(sigma_log)
        return mean,sigma
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

