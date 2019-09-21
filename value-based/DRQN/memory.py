import numpy as np


class ReplayBuff(object):
    '''replay buffer.
    
    Attribute:
        max_size: max capacity of buffer
        observations: state buffer
        actions: action buffer
        rewards: reward buffer
        next_observations: next state buffer
        terminals: terminal buffer
        size: length of buffer
        ptr: the pionter of last transition in buffer.(circular-queue)

    Arg:
        observation_shape: shape of state
        '''
    def __init__(self,max_size,observation_shape):
        self.max_size=max_size
        self.observations=np.zeros([max_size,observation_shape],dtype=np.float32)
        # action is the index of vector of optional action
        # type is np.int in python2.7 because of using tensor.gather().see line80 in dqn_learner.py
        # it can be np.float in python>=3.5
        self.actions=np.zeros([max_size],dtype=np.int)
        self.rewards=np.zeros([max_size],dtype=np.float32)
        self.next_observations=np.zeros([max_size,observation_shape],dtype=np.float32)
        self.terminals=np.zeros(max_size,dtype=np.float32)
        self.size=0
        self.ptr=0

    def append(self,obs,action,reward,next_obs,terminal):
        self.observations[self.ptr]=obs
        self.actions[self.ptr]=action
        self.rewards[self.ptr]=reward
        self.next_observations[self.ptr]=next_obs
        self.terminals[self.ptr]=terminal
        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)

    def sample(self,batch_size):
        if batch_size > self.size:
            # return all transitions in buffer
            # batch_idxs: a set of indexs of transitions
            batch_idxs=np.arange(self.size)
        else:
            batch_idxs=np.random.choice(self.size, size=batch_size,replace=False)
        return dict(obs=self.observations[batch_idxs],
                    action=self.actions[batch_idxs],
                    reward=self.rewards[batch_idxs],
                    next_obs=self.next_observations[batch_idxs],
                    done=self.terminals[batch_idxs])
    

    def __len__(self):
        return self.size

class ReplayBuff2(object):
    def __init__(self,max_size,observation_shape,hidden_size):
        self.max_size=max_size
        self.observations=np.zeros([max_size,observation_shape],dtype=np.float32)
        self.actions=np.zeros([max_size],dtype=np.long)
        self.rewards=np.zeros([max_size],dtype=np.float32)
        self.next_observations=np.zeros([max_size,observation_shape],dtype=np.float32)
        self.terminals=np.zeros(max_size,dtype=np.float32)
        #self.hidden_state=np.zeros([max_size,hidden_size],dtype=np.float32)
        #self.cell_state=np.zeros([max_size,hidden_size],dtype=np.float32)
        self.size=0
        self.ptr=0

    def append(self,obs,action,reward,next_obs,terminal):
        self.observations[self.ptr]=obs
        self.actions[self.ptr]=action
        self.rewards[self.ptr]=reward
        self.next_observations[self.ptr]=next_obs
        self.terminals[self.ptr]=terminal

        #self.hidden_state[self.ptr]=h
        #self.cell_state[self.ptr]=c

        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)

    def sample(self,batch_size,horizon):
        
        batch_start_idxs=np.random.choice(self.size-horizon, size=batch_size,replace=False)

        #print(batch_start_idxs)
        return dict(obs_batch=np.array([self.observations[start:start+horizon] for start in batch_start_idxs]),
                    act_batch=np.array([self.actions[start:start+horizon] for start in batch_start_idxs]),
                    rew_batch=np.array([self.rewards[start:start+horizon] for start in batch_start_idxs]),
                    nx_obs_batch=np.array([self.next_observations[start:start+horizon] for start in batch_start_idxs]),
                    done_batch=np.array([self.terminals[start:start+horizon] for start in batch_start_idxs]))
                    #hidden_batch=np.array([self.hidden_state[start] for start in batch_start_idxs]),
                    #cell_batch=np.array([self.cell_state[start] for start in batch_start_idxs]))
    

    def __len__(self):
        return self.size

    def reset(self):
        self.size=0
        self.ptr

class ReplayBuff3(object):
    def __init__(self,max_size,observation_shape):
        self.max_size=max_size
        self.observations=np.zeros([max_size,observation_shape],dtype=np.float32)
        self.actions=np.zeros([max_size],dtype=np.long)
        self.rewards=np.zeros([max_size],dtype=np.float32)
        self.next_observations=np.zeros([max_size,observation_shape],dtype=np.float32)
        self.terminals=np.zeros(max_size,dtype=np.float32)
        self.size=0
        self.ptr=0

    def append(self,obs,action,reward,next_obs,terminal):
        self.observations[self.ptr]=obs
        self.actions[self.ptr]=action
        self.rewards[self.ptr]=reward
        self.next_observations[self.ptr]=next_obs
        self.terminals[self.ptr]=terminal
        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)

    def sample(self,batch_size,horizon):
   
        batch_idxs=np.arange(self.size)
        return dict(obs=self.observations[batch_idxs],
                    action=self.actions[batch_idxs],
                    reward=self.rewards[batch_idxs],
                    next_obs=self.next_observations[batch_idxs],
                    done=self.terminals[batch_idxs])
 
    def __len__(self):
        return self.size

    def reset(self):
        self.size=0
        self.ptr



if __name__=='__main__':
    rb=ReplayBuff2(512,6,4)
    for i in range(50):
        rb.append(np.random.randn(6),np.random.randn(),3.7,np.random.randn(6),3.3,np.random.randn(4),np.random.randn(4))
    rb.sample(4,2)
    #print("sample test\n sample return type:"+str(type(rb.sample(1))))
    
    #print(rb.sample(128))
