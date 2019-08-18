import gym

from models import discreteNetwork,continuousNetwork 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical


class agent():
    def __init__(
        self,
        scenario,
        seed=123,
        learning_rate=0.001,
        entropy_weight=0.0001,
        entropy_eps=1e-6,
        gamma=0.99,
        train_mode=True,
        discrete=True):

        self.env=gym.make(scenario).unwrapped
    
        self.entropy_weight=entropy_weight
        self.entropy_eps=entropy_eps
        self.gamma=gamma
        self.train_mode=train_mode
        self.step_num=0
        self.discrete=discrete


        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.discrete:
            self.net=discreteNetwork(self.env.observation_space.shape[0],self.env.action_space.n).to(self.device)
        else:
            self.net=continuousNetwork(observation_shape,self.env.action_space.n).to(self.device)
        if self.train_mode:
            self.optimizer=torch.optim.Adam(self.net.parameters(),lr=learning_rate)
        else:
            self.net.eval()



    def select_action(self,state):
        state=torch.from_numpy(state)
        if self.discrete:
            # two way can work,second faster
            probs=self.net(state.to(self.device))
            '''
            action = probs.multinomial(1).data
            return action.squeeze(0).detach().cpu().numpy()
            '''
            m=Categorical(probs)
            action=m.sample()
            return action.item()
        else:
            mu,sigma_sq=self.net(state)
            sigma_sq=F.softplus(sigma_sq)
            eps=torch.randn(mu.shape)
            action=(mu+sigma_sq.sqrt()*eps).data
            return action.squeeze(0).detach().cpu().numpy()

    def update(self,states,actions,rewards):

        self.optimizer.zero_grad()
        states=torch.from_numpy(states).to(self.device)
        actions=torch.from_numpy(actions).to(self.device)
        '''
        R_batch = torch.zeros(rewards.shape)
        R_batch[-1,0]=rewards[-1,0]

        for t in reversed(range(rewards.shape[0]-1)):
            R_batch[t,-1]=rewards[t,-1]+self.gamma*R_batch[t+1,-1]
        '''
        R_batch = torch.zeros(rewards.shape[0])
        R_batch[-1]=rewards[-1]
        for t in reversed(range(rewards.shape[0]-1)):
            R_batch[t]=rewards[t]+self.gamma*R_batch[t+1]
        # returns standardized
        # R_batch=(R_batch-R_batch.mean())/(R_batch.std()+self.entropy_eps)

        if self.discrete:
            probs=self.net(states)
            '''
            probs=probs.gather(1,actions)
            print("states.shape:"+str(states.shape)+'\n'+
                    "actions.shape:"+str(actions.shape)+'\n'+
                    "rewards.shape:"+str(rewards.shape)+'\n'+
                    "R_batch.shape:"+str(R_batch.shape)+'\n'
                    "probs.shape:"+str(probs.shape))
            '''
            m=Categorical(probs)
            log_probs=m.log_prob(actions)

        else:
            # continuous action case has not been tested.
            # we don't have the environment.
            # test later.
            # pay attention to run step,
            # in particular, dim of reward batch
            mu,sigma_sq=self.net(states)
            # ------normal------
            a = (-1*(actions-mu).pow(2)/(2*sigma_sq)).exp()
            b = 1/(2*sigma_sq*np.pi).sqrt()
            probs = a*b
                

        '''
        loss=torch.sum(torch.log(probs)*(-R_batch))+self.entropy_weight*torch.sum(probs*torch.log(probs+self.entropy_eps))
        loss=loss/rewards.shape[0]
        '''
        loss=torch.sum(log_probs*(-R_batch)) # no entropy

        loss.backward()
        # gradinet clipping
        # https://pytorch.prg/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_grad_norm_(self.net.parameters(),40)
        self.optimizer.step()

    def train(self,num_episode):
        if not self.train_mode:
            return None

        step=0
        for i in range(num_episode):
            s_batch=[]
            r_batch=[]
            a_batch=[]
            state=self.env.reset().astype(np.float32)
            done=False
            while not done:
                step+=1
                action=self.select_action(state)
                next_state,reward,done,_=self.env.step(action)
                next_state=next_state.astype(np.float32)
                s_batch.append(state)
                a_batch.append(action)
                r_batch.append(reward)
                state=next_state
            print("episode: "+str(i)+" reward_sum: "+str(np.sum(r_batch)))
            '''

            self.update(np.stack(s_batch,axis=0),np.vstack(a_batch),np.vstack(r_batch))
            '''
            self.update(np.stack(s_batch,axis=0),np.array(a_batch),np.array(r_batch))
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            


                
    def test(self,model_path=None,seedlist=None):
        if self.train_mode:
            return None
        if model_path is None:
            print("no model to test")
            return None
        if seedlist is None:
            seedlist=[111,123,1234]
        self._load(model_path)
        for s in seedlist:
            self.env.seed(s)
            r_batch=[]
            state=self.env.reset().astype(np.float32)
            done=False
            while not done:
                step+=1
                action=self.select_action(state)
                next_state,reward,done,_=self.env.step(action)
                next_state=next_state.astype(np.float32)
                self.store_transition(state,action,reward,next_state,done)
                r_batch.append(reward)
                state=next_state

            print("seed: "+str(s)+" reward_sum: "+str(np.sum(r_batch)))
            del r_batch[:]





if __name__ =='__main__':
    torch.set_num_threads(3)

    seed=123
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_agent=agent('CartPole-v0',seed=seed)
    train_agent.train(100)
