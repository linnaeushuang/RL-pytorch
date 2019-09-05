import operator

class SegmentTree(object):
    def __init__(self,capacity,operation,init_value):


        assert capacity > 0 and capacity & (capacity-1)==0
        "capacity must be postitive and a power of 2."
        self._capacity=capacity
        self._tree=[init_value for _ in range(2*capacity)]
        self._operation=operation

    def _operate_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._tree[node]
        mid = (node_start+node_end) // 2
        if end<=mid:
            return self._operate_helper(start,end,2*node,node_start,mid)
        else:
            if mid+1 <= start:
                return self._operate_helper(start, end, 2*node+1, mid+1, node_end)
            else:
                return self._operation(
                        self._operate_helper(start,mid,2*node,node_start,mid),
                        self._operate_helper(mid+1,end,2*node+1,mid+1,node_end),
                        )
    def operate(self,start=0,end=None):
        if end is None:
            end=self._capacity
        if end<0:
            end+=self._capacity
        end-=1
        return self._operate_helper(start,end,1,0,self._capacity-1)

    def __setitem__(self,idx,val):
        '''update, O(log(n))'''
        idx+=self._capacity
        self._tree[idx]=val
        idx //= 2
        while idx >=1:
            self._tree[idx]=self._operation(self._tree[2*idx],self._tree[2*idx+1])
            idx //=2

    def __getitem__(self,idx):
        assert 0<=idx<self._capacity
        return self._tree[self._capacity+idx]

class SumSegmentTree(SegmentTree):
    def __init__(self,capacity):
        super(SumSegmentTree,self).__init__(capacity=capacity,operation=operator.add,init_value=0.0)

    def sum(self,start=0,end=None):
        return super(SumSegmentTree,self).operate(start,end)
    def find_prefixsum_idx(self,upperbound):
        '''
        upperbound is the max sum priority in memory[0,n]
        '''
        assert 0<=upperbound<=self.sum()+1e-5
        "upperbound:{}.format(upperbound)"
        idx=1
        while idx<self._capacity:
            if self._tree[2*idx]>upperbound:
                idx=2*idx
            else:
                upperbound-=self._tree[2*idx]
                idx=2*idx+1
        return idx-self._capacity

class MinSegmentTree(SegmentTree):
    def __init__(self,capacity):
        super(MinSegmentTree,self).__init__(
                capacity=capacity,
                operation=min,
                init_value=float("inf")
                )
    def min(self,start=0,end=None):
        return super(MinSegmentTree,self).operate(start,end)

        
