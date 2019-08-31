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
        lambd=0.997,
        train_mode=True):

        self.env=gym.make(scenario)
    
        self.entropy_weight=entropy_weight
        self.entropy_eps=entropy_eps
        self.gamma=gamma
        self.lambd=lambd
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
        #--------new-----------
        values=self.critic_net(states).squeeze(1)
        R_batch=torch.from_numpy(rewards).to(self.device)
        batch_size=rewards.shape[0]
        returns=torch.Tensor(batch_size).to(self.device)
        deltas=torch.Tensor(batch_size).to(self.device)
        advantage=torch.Tensor(batch_size).to(self.device)

        prev_return=0
        prev_value=0
        prev_advantage=0
        probs=self.actor_net(states)
        m=Categorical(probs)
        log_probs=m.log_prob(actions)
        entropy_loss=m.entropy()
        for i in reversed(range(batch_size)):
            returns[i]=rewards[i]+self.gamma*prev_return
            deltas[i]=rewards[i]+self.gamma*prev_value-values[i].data.numpy()
            # ref: https://arxiv.org/pdf/1506.02438.pdf(GAE)
            advantage[i]=deltas[i]+self.gamma*self.lambd*prev_advantage

            prev_return=returns[i]
            prev_value=values[i].data.numpy()
            prev_advantage=advantage[i]
        advantage = (advantage - advantage.mean())/(advantage.std()+self.entropy_eps)
        loss_policy=torch.mean(-log_probs*advantage)
        loss_value=torch.mean((values-returns).pow(2))
        loss=loss_policy+loss_value
        loss.backward()

        # gradinet clipping
        # https://pytorch.prg/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_grad_norm_(self.actor_net.parameters(),40)
        clip_grad_norm_(self.critic_net.parameters(),40)
        self.optimizer.step()
        return loss.item()



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
    train_agent.train(600)
 
