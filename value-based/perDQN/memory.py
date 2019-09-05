import numpy as np

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

class PrioritizedReplayBuffer(ReplayBuff):
    def __init__(self,max_size,observation_shape,alpha=0.6):
        assert alpha>=0
        super(PrioritizedReplayBuffer,self).__init__(max_size,observation_shape)
        self.max_priority=1.0
        #pointer of final transition in array(tree array)
        self.tree_ptr=0
        self.alpha=alpha

        # in segment tree,element only in leaf.
        # segment tree is full binary tree.N node,log(N) leaf
        # if we have max_size element(transition),we have 2^max_size node
        # so tree_capacity is power of 2 for full binary tree finally.
        tree_capacity=1
        while tree_capacity < self.max_size:
            tree_capacity*=2

        # sum_tree is denominator of equation(1) in original paper
        # element of sum_tree is the priorty
        # sum_tree.sum(a,b) is the sum priorty of memory[a:b]
        self.sum_tree=SumSegmentTree(tree_capacity)
        self.min_tree=MinSegmentTree(tree_capacity)

    def append(self,obs,action,reward,next_obs,terminal):
        # store in buff
        super(PrioritizedReplayBuffer,self).append(obs,action,reward,next_obs,terminal)
        # store in segment tree
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
        '''I don't know why use prefixsum'''
        batch_idxs=[]
        p_total=float(self.sum_tree.sum(0,len(self)-1))
        segment=p_total/batch_size
        for i in range(batch_size):
            upperbound=np.random.uniform(segment*i,segment*(i+1))
            batch_idxs.append(self.sum_tree.find_prefixsum_idx(upperbound))
        return batch_idxs
    def _calculate_weight(self,idx,beta):
        ''' see section 3.4'''

        p_min=float(self.min_tree.min())/self.sum_tree.sum()
        max_weight=(p_min*len(self))**(-beta)

        # p_sample is equation(1), len(self) is N
        p_sample=self.sum_tree[idx]/float(self.sum_tree.sum())
        weight=(p_sample*len(self))**(-beta)
        # normalize weights by 1/max(w)
        weight=weight/max_weight
        return weight

if __name__=='__main__':
    rb=ReplayBuff(512,6)
    for i in range(50):
        rb.append(np.random.randn(6),np.random.randn(),3.7,np.random.randn(6),3.3)
    #print("sample test\n sample return type:"+str(type(rb.sample(1))))
    
    #print(rb.sample(128))
