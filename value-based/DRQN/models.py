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
    def __init__(self,input_dim,out_dim,batch_size):
        super(Network,self).__init__()

        self.batch_size=batch_size

        self.fc1=nn.Linear(input_dim,128)
        self.fc2=nn.Linear(128,128)
        # lstm inputs is (x,h,c)
        # x :(seq_len,batch,in_dims)
        # outputs is (out,h,c)
        # out:(seq_len,batch,out_dims)

        # batch_first make input output (batch,seq_len,dim)
        #
        # fully connection layer's out is (batch,feature_dim)
        # we don't think of feature as sequence,
        # because output of linearlayer has no temporal coorelation.
        # feature_dim as input_dim,then seq_len is 1.
        # input (batch,128,1)
        # output (batch,128,out_dim) (batch,128,64)
        self.rnn=nn.LSTM(128,128,batch_first=True)

        self.output_layer=nn.Linear(128,out_dim)
        
        # lstm need batch_dim
        # but these is a different batch_size in selec(one-batch) and train(minibatch)
        #
        #
        # todo: is using different (h,c) correct?
        # are we need to reset h_train or restore h_train?
        self.h_train=torch.zeros(1,batch_size,128)
        self.c_train=torch.zeros(1,batch_size,128)

        self.h_selec=torch.zeros(1,1,128)
        self.c_selec=torch.zeros(1,1,128)

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




    def forward(self,inputs,is_selce):
        '''
        # ---pixel normalize---------
        inputs=inputs/255.0
        net=F.relu(self.conv1(inputs),inplace=True)
        net=F.relu(self.conv2(net),inplace=True)
        net=F.relu(self.conv3(net),inplace=True)
        net=net.permute(0,2,3,1)
        net=net.view(-1,22*16,64)
        # h : (num_layer*dic,batchsize,hiddensize)
        # if init in __init__ need batch_size
        h=torch.zeros(1,net.shape[0],512)
        c=torch.zeros(1,net.shape[0],512)
        net,(h,c)=self.rnn(net,(h,c))
        net=net.contiguous()
        net=net.view(net.shape[0],-1)
        net=F.relu(self.q_values(net),inplace=True)
        return net
        '''



        net=F.relu(self.fc1(inputs),inplace=True)
        net=F.relu(self.fc2(net),inplace=True)
        net=net.unsqueeze(1)


        if is_selce:
            net,(self.h_selec,self.c_selec)=self.rnn(net,(self.h_selec,self.c_selec))
            # output (batch,seqlen,outdim) (batch,128,64)
        else:
            net,(self.h_train,self.c_train)=self.rnn(net,(self.h_train,self.c_train))
            # output (batch,seqlen,outdim) (batch,128,64)
        net=net.contiguous()
        net=net.view(inputs.shape[0],-1)

        return self.output_layer(net)


if __name__ == '__main__':
    batch_size=24
    input_dim=16
    out_dim=3
    net=Network(input_dim,out_dim,batch_size)
    optim=torch.optim.Adam(net.parameters(),lr=0.005)
    loss_func=nn.MSELoss()
    for i in range(24):
        state=torch.randn(batch_size,input_dim).to(torch.float32)
        action=net.forward(state)
        print(action.shape)

