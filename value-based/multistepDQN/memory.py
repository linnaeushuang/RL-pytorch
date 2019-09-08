import numpy as np
from collections import deque


class multistepReplayBuff(object):
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
        n_step:
        gamma:
        '''
    def __init__(self,max_size,observation_shape,n_step,gamma):
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

        # for multi-step dqn
        self.multi_step_buffer = deque(maxlen=n_step)
        self.n_step=n_step
        self.gamma=gamma

    def append(self,obs,action,reward,next_obs,terminal):

        transtion = (obs,action,reward,next_obs,terminal)
        self.multi_step_buffer.append(transtion)

        if len(self.multi_step_buffer) >= self.n_step:

            reward,next_obs,done = self._get_n_step_info()
            obs,action = self.multi_step_buffer[0][:2]

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
                    done=self.terminals[batch_idxs],
                    indices=batch_idxs)

    def _get_n_step_info(self):
        for index in range(self.n_step):
            if self.multi_step_buffer[index][-1]:
                break
        reward, next_obs, done = self.multi_step_buffer[index][-3:]
        # if index>0
        if index:
            # from index-1 to 0
            for transition in reversed(list(self.multi_step_buffer)[:index]):
                r = transition[2]
                reward = r + self.gamma * reward

        return reward, next_obs, done

    

    def __len__(self):
        return self.size

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
    def sample_from_idxs(self,batch_idxs):
        return dict(obs=self.observations[batch_idxs],
                        action=self.actions[batch_idxs],
                        reward=self.rewards[batch_idxs],
                        next_obs=self.next_observations[batch_idxs],
                        done=self.terminals[batch_idxs],
                        indices=batch_idxs)



    def __len__(self):
        return self.size



if __name__=='__main__':
    rb=ReplayBuff(512,6,3,0.9)
    for i in range(50):
        rb.append(np.random.randn(6),np.random.randn(),np.random.randn(),np.random.randn(6),False)
    #print("sample test\n sample return type:"+str(type(rb.sample(1))))
    
    print(rb.sample(1))
