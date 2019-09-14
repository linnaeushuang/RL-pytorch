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
        mu_range = 1 / (self.in_featrues**0.5)
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
        '''
        x=torch.randn(size)
        y=x.abs().sqrt()
        return x.sign()*y
        '''
        x=torch.from_numpy(np.random.normal(loc=0.0,scale=1.0,size=size))
        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(Network,self).__init__()

        self.feature_layer = nn.Linear(input_dim,128)
        self.feature_layer2= nn.Linear(128,128)
        self.noisy_layer1 = NoisyLayer(128,128)
        self.noisy_layer2 = NoisyLayer(128,out_dim)

        nn.init.xavier_uniform_(self.feature_layer.weight.data)
        nn.init.constant_(self.feature_layer.bias.data,0.0)

        nn.init.xavier_uniform_(self.feature_layer2.weight.data)
        nn.init.constant_(self.feature_layer2.bias.data,0.0)


    def forward(self,inputs):
        feature = F.relu(self.feature_layer(inputs),inplace=True)
        feature = F.relu(self.feature_layer2(feature),inplace=True)
        noisy_feature = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(noisy_feature)
        return out

    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

if __name__ == '__main__':
    net=Network(16,3)
    optim=torch.optim.Adam(net.parameters(),lr=0.005)
    loss_func=nn.MSELoss()
    for i in range(24):
        state=torch.randn(11,16).to(torch.float32)
        action=net.forward(state)
        print(action)

