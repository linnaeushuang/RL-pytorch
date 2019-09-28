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



class ActorNetwork_lstm(nn.Module):

    def __init__(self,in_size,hidden_size,out_size):
        super(ActorNetwork_lstm, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size,out_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        #x = F.log_softmax(x,2)
        x=F.softmax(x,2)
        return x, hidden

class CriticNetwork_lstm(nn.Module):

    def __init__(self,in_size,hidden_size):
        super(CriticNetwork_lstm, self).__init__()

        self.lstm = nn.LSTM(in_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x, hidden):

        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


