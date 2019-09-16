import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLayer(nn.Module):

    def __init__(self,in_featrues,out_features,std_init=0.5):
        super(NoisyLayer,self).__init__()

        self.in_featrues=in_featrues
        self.out_features=out_features
        self.std_init=std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features,in_featrues))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features,in_featrues))
        self.register_buffer("weight_epsilon",torch.Tensor(out_features,in_featrues))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon",torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / (self.in_featrues**0.5)
        self.weight_mu.data.uniform_(-mu_range,mu_range)
        self.weight_sigma.data.fill_(self.std_init/(self.in_featrues**0.5))

        self.bias_mu.data.uniform_(-mu_range,mu_range)
        self.bias_sigma.data.fill_(self.std_init/(self.in_featrues**0.5))

    def reset_noise(self):
        epsilon_in=self.scale_noise(self.in_featrues)
        epsilon_out=self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self,inputs):
        outputs = F.linear(inputs,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return outputs

    @staticmethod
    def scale_noise(size):
        x=torch.randn(size)
        y=x.abs().sqrt()
        return x.sign()*y

class Network(nn.Module):
    def __init__(self,input_dim,out_dim,atom_size,support):
        super(Network,self).__init__()
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # common feature layer
        self.feature_layer=nn.Linear(input_dim,128)

        # advantage layer
        self.advantage_hidden_layer=NoisyLayer(128,128)
        self.advantage_layer=NoisyLayer(128,out_dim*atom_size)

        # value layer
        self.value_hidden_layer=NoisyLayer(128,128)
        self.value_layer=NoisyLayer(128,atom_size)
        

        self._initialize()

    def _initialize(self):
        # xavier intialization
        # see <<understanding the difficulty of training deep feedforward neural networks>>
        nn.init.xavier_uniform_(self.feature_layer.weight.data)
        nn.init.constant_(self.feature_layer.bias.data,0.0)


    def forward(self,inputs):
        dist=self.distributional(inputs)
        return torch.sum(dist * self.support,dim=2)

    def distributional(self,inputs):
        '''get distribution for atoms'''
        feature_layerout=F.relu(self.feature_layer(inputs),inplace=True)
        adv_hid=F.relu(self.advantage_hidden_layer(feature_layerout))
        val_hid=F.relu(self.value_hidden_layer(feature_layerout))
        # distributional
        advantage=self.advantage_layer(adv_hid).view(-1,self.out_dim,self.atom_size)
        value=self.value_layer(val_hid).view(-1,1,self.atom_size)

        # dueling
        q_atoms=value+advantage-advantage.mean(dim=1,keepdim=True)
        # distributional
        dist=F.softmax(q_atoms,dim=-1)
        return dist



    def reset_noise(self):
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()



if __name__ == '__main__':
    support=torch.linspace(0.0,8.0,4)
    net=Network(16,2,4,support)
    optim=torch.optim.Adam(net.parameters(),lr=0.005)
    loss_func=nn.MSELoss()
    for i in range(24):
        state=torch.randn(2,16).to(torch.float32)
        #action=net.distributional(state)
    
    with torch.no_grad():
        state=torch.randn(2,16).to(torch.float32)
        print('state')
        print(state)
        action=net.forward(state).argmax(1)
        print('action')
        print(action)

