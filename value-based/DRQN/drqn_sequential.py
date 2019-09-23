
import gym


from memory import ReplayBuff3
from models import Network2
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
        replay_capacity=4096000,
        batch_size=24,
        learning_rate=0.0001,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=512,
        lstm_history=24,
        update_period=10,
        target_update_period=1,
        epsilon_train_start=1,
        epsilon_train_end=0.01,
        epsilon_eval=0.001,
        epsilon_decay=0.0001,
        lstm_size=128,
        train_mode=True):

        self.env=gym.make(scenario)
        self.env.seed(seed)
        self.batch_size=batch_size
        self.update_period=update_period
        self.target_update_period=target_update_period
        self.gamma=gamma
        self.lstm_size=lstm_size
        self.train_mode=train_mode


        self.hidden_state=None
        self.cell_state=None
        self.lstm_history=lstm_history

        if min_replay_history<batch_size:
            self.min_replay_history=batch_size
        else:
            self.min_replay_history=min_replay_history

        self.action_num=self.env.action_space.n

        if self.train_mode:
            self.epsilon=epsilon_train_start
            self.epsilon_decay=epsilon_decay
            self.epsilon_train_start=epsilon_train_start
            self.epsilon_train_end=epsilon_train_end
        else:
            self.epsilon=epsilon_eval
        
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.net=Network2(self.env.observation_space.shape[0],self.env.action_space.n,lstm_size).to(self.device)
        if self.train_mode:
            self.memory=ReplayBuff3(replay_capacity,256,self.env.observation_space.shape[0])
            self.target_net=Network2(self.env.observation_space.shape[0],self.env.action_space.n,lstm_size).to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
            self.target_net.eval()
            self.optimizer=torch.optim.Adam(self.net.parameters(),lr=learning_rate)
            self.loss_func=nn.MSELoss()
        else:
            self.net.eval()


    def select_action(self,state):

        if self.epsilon < np.random.random():
            with torch.no_grad():
                state=torch.from_numpy(state)
                state=state.view(1,1,-1)
                action,self.hidden_state,self.cell_state=self.net(state.to(self.device),self.hidden_state,self.cell_state)
                action=action.detach().cpu().squeeze().numpy()
                return action.argmax()
        else:
            return np.random.randint(self.action_num)


    def store_transition(self,obs,action,reward,next_obs,done):
        self.memory.append(obs,action,reward,next_obs,done)


    def update(self):
        #print('init update')
        self.optimizer.zero_grad()

        samples=self.memory.sample(self.batch_size,self.lstm_history)
        '''
        state=torch.from_numpy(samples["obs"]).to(self.device)
        action=torch.from_numpy(samples["action"]).to(self.device)
        reward=torch.from_numpy(samples["reward"]).to(self.device)
        next_state=torch.from_numpy(samples["next_obs"]).to(self.device)
        done=torch.from_numpy(samples["done"]).to(self.device)
        end=samples["end"]
        #hidden_batch=torch.from_numpy(samples["hidden_batch"]).to(self.device)
        #cell_batch=torch.from_numpy(samples["cell_batch"]).to(self.device)
        print(state.shape)
        print(action.shape)
        print(reward.shape)
        print(next_state.shape)
        print(done.shape)
        print(action.dtype)
        state=state.unsqueeze(0)
        action=action.view(1,-1,1)
        reward=reward.view(1,-1,1)
        done=done.view(1,-1,1)
        '''
        batch=len(samples['obs'])
        for i in range(batch):
            state=torch.Tensor(samples["obs"][i:i+1])
            action=torch.Tensor(samples["action"][i:i+1]).unsqueeze(2)
            action=action.long()
            reward=torch.Tensor(samples["reward"][i:i+1]).unsqueeze(2)
            next_state=torch.Tensor(samples["next_obs"][i:i+1])
            done=torch.Tensor(samples["done"][i:i+1]).unsqueeze(2)
            '''
            print(state.shape)
            print(action.shape)
            print(reward.shape)
            print(next_state.shape)
            print(done.shape)
            '''

            hidden_batch=torch.zeros(1,1,self.lstm_size)
            cell_batch=torch.zeros(1,1,self.lstm_size)

            q_value,_,_=self.net(state,hidden_batch,cell_batch)
            q_value=q_value.gather(2,action)
            with torch.no_grad():
                next_q_value,_,_=self.target_net(state,hidden_batch,cell_batch)
                next_q_value=next_q_value.max(dim=2,keepdim=True)[0]
            mask=1-done
            target=(reward+self.gamma*next_q_value*mask).to(self.device)
            loss=self.loss_func(target,q_value)
            #print(loss.item())
            loss.backward()
            #clip_grad_norm_(self.net.parameters(),1.0,norm_type=1)
        self.optimizer.step()

    def target_update(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def train(self,num_episode):
        if not self.train_mode:
            return None
        step=0
        for i in range(num_episode):
            self.hidden_state=torch.zeros(1,1,self.lstm_size)
            self.cell_state=torch.zeros(1,1,self.lstm_size)
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
                if step>self.min_replay_history and step%self.update_period==0:
                    self.epsilon=max(self.epsilon_train_end,self.epsilon-(self.epsilon_train_start-self.epsilon_train_end)*self.epsilon_decay)
                    self.update()
            self.memory.new_episode()
            #self.update()
            #if i%self.target_update_period==0:
            self.target_update()
            #self.hidden_state=torch.zeros(1,1,self.env.action_space.n)
            #self.cell_state=torch.zeros(1,1,self.env.action_space.n)
            print("episode: "+str(i)+" reward_sum: "+str(np.sum(r_batch)))
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


   


if __name__ =='__main__':
    torch.set_num_threads(1)

    seed=111
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    train_agent=agent('CartPole-v0')
    train_agent.train(3000)
 
            
        

        



