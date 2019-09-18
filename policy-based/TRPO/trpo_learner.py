
import gym
import numpy as np
from models import ActorNetwork,CriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_ 
from torch.distributions.normal import Normal





class learner():
    def __init__(
        self,
        scenario,
        seed=123,
        learning_rate=5e-4,
        gamma=0.99,
        total_timesteps=1e6,
        nsteps=1024,
        batch_size=64,
        vf_itrs=25,
        Lambda=0.97,
        damping=0.1,
        max_kl=0.01,
        entropy_eps=1e-6,
        cgrad_update_steps=10,
        accept_ratio=0.1,
        train_mode=True
        ):

        self.env=gym.make(scenario)
        self.gamma=gamma
        self.train_mode=train_mode
        self.max_kl=max_kl
        self.Lambda=Lambda
        self.vf_itrs=vf_itrs
        self.batch_size=batch_size
        self.nsteps=nsteps
        self.total_timesteps=total_timesteps
        self.entropy_eps=entropy_eps
        self.cgrad_update_steps=cgrad_update_steps
        self.accept_ratio=accept_ratio
        self.damping=damping

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor=ActorNetwork(self.env.observation_space.shape[0],self.env.action_space.shape[0])
        self.old_actor=ActorNetwork(self.env.observation_space.shape[0],self.env.action_space.shape[0])
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic=CriticNetwork(self.env.observation_space.shape[0])
        self.optimizer=torch.optim.Adam(self.critic.parameters(),lr=learning_rate)

    def select_action(self,state):
        state=torch.from_numpy(state)
        with torch.no_grad():
            mean,std=self.actor(state)
            normal_dist=Normal(mean,std) 
            action=normal_dist.sample()
            return action.detach().numpy().squeeze()

    def update(self,states,actions,rewards):

        self.optimizer.zero_grad()
        states=torch.from_numpy(states).to(self.device)
        actions=torch.from_numpy(actions).to(self.device)
        rewards=torch.from_numpy(rewards).to(self.device)

        with torch.no_grad():
            values=self.critic(states)

        batch_size=rewards.shape[0]
        returns=torch.Tensor(batch_size).to(self.device)
        deltas=torch.Tensor(batch_size).to(self.device)
        advantage=torch.Tensor(batch_size).to(self.device)

        prev_return=0
        prev_value=0
        prev_advantage=0
        for i in reversed(range(batch_size)):
            returns[i]=rewards[i]+self.gamma*prev_return
            deltas[i]=rewards[i]+self.gamma*prev_value-values[i].data
            # ref: https://arxiv.org/pdf/1506.02438.pdf(GAE)
            advantage[i]=deltas[i]+self.gamma*self.Lambda*prev_advantage

            prev_return=returns[i]
            prev_value=values[i].data
            prev_advantage=advantage[i]
        advantage = (advantage - advantage.mean())/(advantage.std()+self.entropy_eps)

        values=self.critic(states)

        #-----------------------------------
        with torch.no_grad():
            old_mean,old_std=self.old_actor(states)

        #--------get surr grad------------
        mean,std=self.actor(states)
        normal_dist = Normal(mean, std)
        log_prob=normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
        old_normal_dist = Normal(mean, std)
        old_log_prob=old_normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
        # weight sample
        surr_loss=-torch.exp(log_prob-old_log_prob)*advantage
        surr_loss=surr_loss.mean()
        #-----------------------------

        surr_grad=torch.autograd.grad(surr_loss,self.actor.parameters())
        flat_surr_grad=torch.cat([grad.view(-1) for grad in surr_grad]).data

        fisher_matrix=self._fisher_matrix(-flat_surr_grad,states,old_mean,old_std)
        nature_grad=self._conjugated_gradient(-flat_surr_grad,self.cgrad_update_steps,fisher_matrix)
        non_fmatrix=self._fisher_matrix(nature_grad,states,old_mean,old_std)
        non_scale_kl=0.5*(nature_grad * non_fmatrix).sum(0,keepdim=True)
        scale_ratio=torch.sqrt(non_scale_kl/self.max_kl)
        final_nature_grad=nature_grad/scale_ratio[0]

        expected_improve=(-flat_surr_grad*nature_grad).sum(0,keepdim=True)/scale_ratio[0]

        prev_params=torch.cat([param.data.view(-1) for param in self.actor.parameters()])




        #---------update actor
        for _n_backtracks,stepfrac in enumerate(0.5**np.arange(self.cgrad_update_steps)):
            new_loss = prev_params + stepfrac*final_nature_grad
            self._set_flat_params_by(new_loss)
            with torch.no_grad():
                new_mean,new_std=self.actor(states)
                new_normal_dist = Normal(new_mean, new_std)
                new_log_prob=normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
                new_surr_loss=-torch.exp(new_log_prob-old_log_prob)*advantage
                new_surr_loss=new_surr_loss.mean()

            actual_improve=surr_loss-new_surr_loss
            e_improve=expected_improve*stepfrac
            ratio = actual_improve /e_improve 
            if ratio.item()>self.accept_ratio and actual_improve.item()>0:
                break


        #---------update critic
        for _ in range(self.vf_itrs):
            if self.batch_size>states.shape[0]:
                batch_idxs=np.arange(states.shape[0])
            else:
                batch_idxs=np.random.choice(states.shape[0],size=self.batch_size,replace=True)
            mini_states=states[batch_idxs]
            mini_returns=returns[batch_idxs]
            update_value=self.critic(mini_states)
            v_loss=(mini_returns-update_value).pow(2).mean()
            self.optimizer.zero_grad()
            v_loss.backward()
            self.optimizer.step()



    def _fisher_matrix(self,v,obs,old_mean,old_std):
        kl=self._get_kl(obs,old_mean,old_std)
        kl=kl.mean()

        kl_grads=torch.autograd.grad(kl,self.actor.parameters(),create_graph=True)
        #kl_grads=torch.gard(kl,self.actor.parameters(),create_graph=True)
        flat_kl_grads=torch.cat([grad.view(-1) for grad in kl_grads])

        kl_v=(flat_kl_grads*torch.autograd.Variable(v)).sum()
        kl_second_grads=torch.autograd.grad(kl_v,self.actor.parameters())
        flat_kl_second_grads=torch.cat([grad.contiguous().view(-1) for grad in kl_second_grads])

        return flat_kl_second_grads+self.damping*v

    def _get_kl(self,obs,old_mean,old_std):
        mean,std=self.actor(obs)
        kl=-torch.log(std/old_std)+(std.pow(2)+(mean-old_mean).pow(2))/(2*old_std.pow(2))-0.5
        return kl.sum(1,keepdim=True)

    def _conjugated_gradient(self,surr_grad,update_steps,fmatrix,residual_limit=1e-10):
        r=surr_grad.clone()
        p=surr_grad.clone()
        r_dot_r=torch.dot(r,r)
        x=torch.zeros(surr_grad.size()).to(self.device)

        for i in range(update_steps):
            alpha=r_dot_r/torch.dot(p,fmatrix)
            x=x+alpha*p
            r=r-alpha*fmatrix
            new_r_dot_r=torch.dot(r,r)
            beta=new_r_dot_r/r_dot_r
            p=r+beta*p
            r_dot_r=new_r_dot_r
            if r_dot_r<residual_limit:
                break
        return x
    def _set_flat_params_by(self,flat_params):
        prev_idx=0
        for param in self.actor.parameters():
            flat_size=int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_idx:prev_idx+flat_size].view(param.size()))
            prev_idx+=flat_size




if __name__=='__main__':
    obs_dim=3
    act_dim=1
    horizon=128
    states=np.random.randn(horizon,obs_dim).astype('float32')
    actions=np.random.randn(horizon,act_dim).astype('float32')
    returns=np.random.randn(horizon)
    agent=learner('Pendulum-v0')
    agent.update(states,actions,returns)

