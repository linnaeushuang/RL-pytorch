import gym

from models import ActorNetwork,CriticNetwork
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
        train_mode=True):

        self.env=gym.make(scenario)
    
        self.entropy_weight=entropy_weight
        self.entropy_eps=entropy_eps
        self.gamma=gamma
        self.train_mode=train_mode
        self.step_num=0


        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_net=ActorNetwork(self.env.observation_space.shape[0],self.env.action_space.n).to(self.device)
        if self.train_mode:
            #self.loss_func=nn.SmoothL1Loss()
            self.loss_func=nn.MSELoss()

            self.critic_net=CriticNetwork(self.env.observation_space.shape[0]).to(self.device)
            self.optimizer=torch.optim.Adam([{'params':self.actor_net.parameters()},{'params':self.critic_net.parameters()}],lr=learning_rate)
        else:
            self.actor_net.eval()



    def select_action(self,state):
        state=torch.from_numpy(state)
        with torch.no_grad():
            probs=self.actor_net(state.to(self.device))
            m=Categorical(probs)
            action=m.sample()
            return action.item()

    def update(self,states,actions,rewards):

        self.optimizer.zero_grad()
        states=torch.from_numpy(states).to(self.device)
        actions=torch.from_numpy(actions).to(self.device)

        R_batch = torch.zeros(rewards.shape[0])
        R_batch[-1]=rewards[-1]
        for t in reversed(range(rewards.shape[0]-1)):
            R_batch[t]=rewards[t]+self.gamma*R_batch[t+1]

        # returns standardized
        # R_batch=(R_batch-R_batch.mean())/(R_batch.std()+self.entropy_eps)


        probs=self.actor_net(states)
        m=Categorical(probs)
        log_probs=m.log_prob(actions)

        s_values=self.critic_net(states).squeeze(1)

        advantage=R_batch-s_values
                

        # see original paper,chapter 4
        # we have not used gamma^k * v(s_{t+k})
        actor_loss=torch.sum(log_probs*(-advantage)) # no entropy

        critic_loss=self.loss_func(s_values,R_batch)

        # backward together
        loss=actor_loss+critic_loss

        loss.backward()
        # gradinet clipping
        # https://pytorch.prg/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_grad_norm_(self.actor_net.parameters(),40)
        clip_grad_norm_(self.critic_net.parameters(),40)
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


    # not meetiing expectations
    torch.set_num_threads(3)

    seed=123
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_agent=agent('CartPole-v0',seed=seed)
    train_agent.train(300)
 
