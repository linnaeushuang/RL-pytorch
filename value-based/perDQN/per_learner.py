
import gym


from memory import PrioritizedReplayBuffer 
from models import Network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class agent():

    def __init__(
        self,
        scenario,
        seed=123,
        stack_size=1,
        replay_capacity=4096,
        batch_size=64,
        learning_rate=0.0001,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=128,
        update_period=1,
        target_update_period=32,
        epsilon_train_start=1,
        epsilon_train_end=0.01,
        epsilon_eval=0.001,
        epsilon_decay=0.0001,
        # PER param
        alpha=0.6,
        beta=0.4,
        beta_iters=1000,
        prior_eps=1e-6,
        train_mode=True):

        self.env=gym.make(scenario)
        self.env.seed(seed)
        self.batch_size=batch_size
        self.update_period=update_period
        self.target_update_period=target_update_period
        self.gamma=gamma
        self.train_mode=train_mode
        if min_replay_history<batch_size:
            self.min_replay_history=batch_size
        else:
            self.min_replay_history=min_replay_history

        self.action_num=self.env.action_space.n

        # PER param
        self.beta=beta
        self.prior_eps=prior_eps
        self.beta_iters=beta_iters

        if self.train_mode:
            self.epsilon=epsilon_train_start
            self.epsilon_decay=epsilon_decay
            self.epsilon_train_start=epsilon_train_start
            self.epsilon_train_end=epsilon_train_end
        else:
            self.epsilon=epsilon_eval
        
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.net=Network(self.env.observation_space.shape[0],self.env.action_space.n).to(self.device)
        if self.train_mode:
            self.memory=PrioritizedReplayBuffer(replay_capacity,self.env.observation_space.shape[0],alpha)
            self.target_net=Network(self.env.observation_space.shape[0],self.env.action_space.n).to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()
            self.optimizer=torch.optim.RMSprop(self.net.parameters(),lr=learning_rate,alpha=0.9,eps=1e-10)
            self.loss_func=nn.MSELoss()
        else:
            self.net.eval()


    def select_action(self,state):

        if self.epsilon < np.random.random():
            with torch.no_grad():
                state=torch.from_numpy(state)
                action=self.net(state.to(self.device)).detach().cpu().numpy()
                #return the index of action
                return action.argmax()
        else:
            return np.random.randint(self.action_num)


    def store_transition(self,obs,action,reward,next_obs,done):
        self.memory.append(obs,action,reward,next_obs,done)


    def update(self,beta=0.4):
        self.optimizer.zero_grad()
        samples=self.memory.sample(self.batch_size,beta)
        state=torch.from_numpy(samples["obs"]).to(self.device)
        action=torch.from_numpy(samples["action"].reshape(-1,1)).to(self.device)
        reward=torch.from_numpy(samples["reward"].reshape(-1,1)).to(self.device)
        next_state=torch.from_numpy(samples["next_obs"]).to(self.device)
        done=torch.from_numpy(samples["done"].reshape(-1,1)).to(self.device)

        # PER, we don't use double DQN.
        weights=torch.from_numpy(samples["weights"].reshape(-1,1)).to(self.device)
        indices=samples["indices"]
        print(len(indices))

        q_value=self.net(state).gather(1,action)
        next_q_value=self.target_net(
                next_state
                ).max(dim=1,keepdim=True)[0].detach()
        mask=1-done
        target=(reward + self.gamma * next_q_value * mask).to(self.device)
        #-----------PER algorithm 1 ,page 5 in original paper---------------
        elementwise_loss=(target-q_value)                                   # line 11,compute td-error,using first variant
        loss_for_prior = elementwise_loss.detach().abs().cpu().numpy()      # line 12
        new_priorities=loss_for_prior+self.prior_eps                        # line 12
        print(new_priorities.shape)
        self.memory.update_priorities(indices,new_priorities)               # line 12
        loss=torch.mean(elementwise_loss.pow(2)*weights)                    # line 13,elementwise_loss.pow(2) is MSEloss
        loss.backward()
        # gradinet clipping
        # https://pytorch.prg/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_grad_norm_(self.net.parameters(),1.0,norm_type=1)
        self.optimizer.step()

        return loss.item()

    def target_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def train(self,num_episode):
        if not self.train_mode:
            return None
        step=0
        for i in range(num_episode):
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
                if self.memory.size>=self.min_replay_history and step%self.update_period==0:
                    # start learn
                    self.epsilon=max(self.epsilon_train_end,self.epsilon-(self.epsilon_train_start-self.epsilon_train_end)*self.epsilon_decay)
                    #------per increase------------
                    # update beta must after learning_start
                    # see openai baseline deepq.py line 293-297
                    fraction=min(step/self.beta_iters,1.0)
                    beta=self.beta+fraction*(1.0-self.beta)
                    self.update(beta)
                if step % self.target_update_period==0:
                    self.target_update()

            print("episode: "+str(i)+" reward_sum: "+str(np.sum(r_batch)))
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



 
    def _restore(self,path):
        if self.train_mode:
            torch.save(self.net.state_dict(),path)
        else:
            print("testing model,cannot save models")
    def _load(self,path):
        if self.train_mode:
            print("training model,cannot load models")
        else:
            self.net.load_state_dict(torch.load(path))


    def reset(self):
        self.memory.ptr=0
        self.memory.size=0
    


if __name__ =='__main__':
    torch.set_num_threads(3)

    seed=123
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_agent=agent('CartPole-v0',seed=seed)
    train_agent.train(1000)
 
            
        

        



