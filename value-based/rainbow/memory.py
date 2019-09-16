import numpy as np

from collections import deque
from segment_tree import SumSegmentTree,MinSegmentTree

class ReplayBuff(object):
    def __init__(self,max_size,observation_shape):
        self.max_size=max_size
        self.observations=np.zeros([max_size,observation_shape],dtype=np.float32)
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

class PrioritizedReplayBuff(ReplayBuff):
    def __init__(self,max_size,observation_shape,alpha=0.6):
        assert alpha>=0
        super(PrioritizedReplayBuff,self).__init__(max_size,observation_shape)
        self.max_priority=1.0
        self.tree_ptr=0
        self.alpha=alpha

        tree_capacity=1
        while tree_capacity < self.max_size:
            tree_capacity*=2
        self.sum_tree=SumSegmentTree(tree_capacity)
        self.min_tree=MinSegmentTree(tree_capacity)

    def append(self,obs,action,reward,next_obs,terminal):
        super(PrioritizedReplayBuff,self).append(obs,action,reward,next_obs,terminal)
        self.sum_tree[self.tree_ptr]=self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr]=self.max_priority ** self.alpha
        self.tree_ptr=(self.tree_ptr+1)%self.max_size

    def sample(self,batch_size,beta=0.4):
        assert beta>0
        if batch_size>self.size:
            batch_size=self.size
        batch_idxs=self._sample_proportional(batch_size)
        weights=np.array([self._calculate_weight(i,beta) for i in batch_idxs],dtype=np.float32)
        return dict(obs=self.observations[batch_idxs],
                    action=self.actions[batch_idxs],
                    reward=self.rewards[batch_idxs],
                    next_obs=self.next_observations[batch_idxs],
                    done=self.terminals[batch_idxs],
                    weights=weights,
                    indices=batch_idxs)
    


    def update_priorities(self,idxs,priorities):
        assert len(idxs) == len(priorities)
        for idx,priority in zip(idxs,priorities):
            assert priority>0
            assert 0<=idx<len(self)
            self.sum_tree[idx]=priority**self.alpha
            self.min_tree[idx]=priority**self.alpha
            self.max_priority=max(self.max_priority,priority)


    def _sample_proportional(self,batch_size):
        batch_idxs=[]
        p_total=float(self.sum_tree.sum(0,len(self)-1))
        segment=p_total/batch_size
        for i in range(batch_size):
            upperbound=np.random.uniform(segment*i,segment*(i+1))
            batch_idxs.append(self.sum_tree.retrieve(upperbound))
        return batch_idxs
    def _calculate_weight(self,idx,beta):

        p_min=float(self.min_tree.min())/self.sum_tree.sum()
        max_weight=(p_min*len(self))**(-beta)

        p_sample=self.sum_tree[idx]/float(self.sum_tree.sum())
        weight=(p_sample*len(self))**(-beta)
        weight=weight/max_weight
        return weight
class multistepReplayBuff(object):
    def __init__(self,max_size,observation_shape,n_step,gamma):
        self.max_size=max_size
        self.observations=np.zeros([max_size,observation_shape],dtype=np.float32)
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

    def append(self,obs,action,reward,next_obs,done):

        transtion = (obs,action,reward,next_obs,done)
        self.multi_step_buffer.append(transtion)

        if len(self.multi_step_buffer) >= self.n_step:

            reward,next_obs,done = self._get_n_step_info()
            obs,action = self.multi_step_buffer[0][:2]

        self.observations[self.ptr]=obs
        self.actions[self.ptr]=action
        self.rewards[self.ptr]=reward
        self.next_observations[self.ptr]=next_obs
        self.terminals[self.ptr]=done
        self.ptr=(self.ptr+1)%self.max_size
        self.size=min(self.size+1,self.max_size)

    def sample(self,batch_size):
        if batch_size > self.size:
            batch_idxs=np.arange(self.size)
        else:
            batch_idxs=np.random.choice(self.size, size=batch_size,replace=False)
        return dict(obs=self.observations[batch_idxs],
                    action=self.actions[batch_idxs],
                    reward=self.rewards[batch_idxs],
                    next_obs=self.next_observations[batch_idxs],
                    done=self.terminals[batch_idxs])
    def sample_from_indexs(self,batch_idxs):
        return dict(obs=self.observations[batch_idxs],
                    action=self.actions[batch_idxs],
                    reward=self.rewards[batch_idxs],
                    next_obs=self.next_observations[batch_idxs],
                    done=self.terminals[batch_idxs])

    def _get_n_step_info(self):
        for index in range(self.n_step):
            if self.multi_step_buffer[index][-1]:
                break
        reward, next_obs, done = self.multi_step_buffer[index][-3:]
        if index:
            for transition in reversed(list(self.multi_step_buffer)[:index]):
                r = transition[2]
                reward = r + self.gamma * reward

        return reward, next_obs, done

    

    def __len__(self):
        return self.size


if __name__=='__main__':
    rb=ReplayBuff(512,6)
    for i in range(50):
        rb.append(np.random.randn(6),np.random.randn(),3.7,np.random.randn(6),3.3)
    #print("sample test\n sample return type:"+str(type(rb.sample(1))))
    
    #print(rb.sample(128))
