
import gym


from memory import ReplayBuff
from models import Network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class agent():
    '''DQN Agent.
    
    Attribute:
        memory: replay buffer to store transition
        batch_size: batch size for sampling
        epsilon: parameter for epsilon greedy policy
        epsilon_decay: step size to decrease epsilon
        epsilon_train_start: initial(max) value of epsilon
        epsilon_train_end: end(min) value of epsilon
        target_update_period: period for target model's hard update
        gamma: discount factor
        net: RL model to train and select actions
        target_net: target model to hard update
        optimizer: optimizer for training dqn
        train_mode: train a new model of test a model
        step_num: current timestep

        
    Arg:
        observation_space: shape of observation from environment
        action_num: the number of optional discrete actions
        replay_capacity: length of memory
        batch_size: batch size for sampling
        target_update_period: period for target model's hard update
        learning_rate: learning rate
        epsilon_train_start: max value of epsilon
        epsilon_train_end: min value of epsilon
        gamma: discout factor
        train_mode: train a new model of test a model
        '''

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
        # distributional param
        v_min = 0.0,    # from preliminary experiments over the training games,see section5
        v_max = 200.0,
        atom_size=51,    # see section5,footnote2,in original paper
        train_mode=True):

        self.env=gym.make(scenario)
        self.env.seed(seed)
        self.batch_size=batch_size
        self.update_period=update_period
        self.target_update_period=target_update_period
        self.gamma=gamma

        #-----dist
        self.v_min=v_min
        self.v_max=v_max
        self.atom_size=atom_size


        self.train_mode=train_mode
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

        # distributional DQN
        # section4.1
        # support is the set of atoms
        self.support=torch.linspace(self.v_min,self.v_max,self.atom_size).to(self.device)
        self.delta_z=float(self.v_max-self.v_min)/(self.atom_size-1)

        self.net=Network(self.env.observation_space.shape[0],self.env.action_space.n,atom_size,self.support).to(self.device)
        if self.train_mode:
            self.memory=ReplayBuff(replay_capacity,self.env.observation_space.shape[0])
            self.target_net=Network(self.env.observation_space.shape[0],self.env.action_space.n,atom_size,self.support).to(self.device)
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


    def update(self):
        self.optimizer.zero_grad()
        samples=self.memory.sample(self.batch_size)
        state=torch.from_numpy(samples["obs"]).to(self.device)
        action=torch.from_numpy(samples["action"].reshape(-1,1)).to(self.device)
        reward=torch.from_numpy(samples["reward"].reshape(-1,1)).to(self.device)
        next_state=torch.from_numpy(samples["next_obs"]).to(self.device)
        done=torch.from_numpy(samples["done"].reshape(-1,1)).to(self.device)


        # ---------------distribution dqn--------------
        with torch.no_grad():
            # see algorithms 1 in original paper
            # next_action is index
            next_action=self.target_net(next_state).argmax(1)          # line2-3
            next_dist=self.target_net.distributional(next_state)       # nextwork output
            next_dist=next_dist[range(self.batch_size),next_action]    # get distribution by next_action(argmax policy)

            t_z=reward+(1-done)*self.gamma*self.support                # line7 
                                                                       # reward.shape is (batchsize,1);support,shape is (atomssize);t_z.shape is (batchszie,atomsize)
            t_z=t_z.clamp(min=self.v_min,max=self.v_max)               # line7
            b=(t_z-self.v_min)/self.delta_z                            # line8
            l=b.floor().long()                                         # line9
            u=b.ceil().long()                                          # line9
            offset=(
                    torch.linspace(
                        0,(self.batch_size-1)*self.atom_size,self.batch_size
                        ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size,self.atom_size)
                    .to(self.device)
                    )
            proj_dist = torch.zeros(next_dist.size(),device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)) # line11
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)) # line12

        dist=self.net.distributional(state)                            # line14, p_i(x_t,a_t)
        log_p=torch.log(dist[range(self.batch_size),action])           # line14
        loss = -(proj_dist*log_p).sum(1).mean()                        # line14
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

                    self.epsilon=max(self.epsilon_train_end,self.epsilon-(self.epsilon_train_start-self.epsilon_train_end)*self.epsilon_decay)
                    self.update()
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
    train_agent.train(15)
 
            
        

        



