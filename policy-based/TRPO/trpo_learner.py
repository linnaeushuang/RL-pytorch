
import gym
import numpy as np
from models import Network
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
        tau=0,97,
        damping=0,1,
        max_kl=0.01,
        entropy_eps=1e-6,
        train_mode=True
        ):

        self.env=gym.make(scenario)
        self.gamma=gamma
        self.train_mode=train_mode
        self.max_kl=max_kl
        self.tau=tau
        self.vf_itrs=self.vf_itrs
        self.batch_size=batch_size
        self.nsteps=nsteps
        self.total_timesteps=total_timesteps
        self.entropy_eps=entropy_eps

        self.net=Network(slef.env.observation_space.shape[0],self.env.action_space.shape[0])
        self.old_net=Network(slef.env.observation_space.shape[0],self.env.action_space.shape[0])
        self.old_net.load_state_dict(self.net)
        self.optimizer=torch.optim.Adam(self.net.critic.parameters(),lr=learning_rate)

    def select_action(self,state):
        state=torch.from_numpy(state)
        with torch.no_grad():
            mean,std=self.net.actor(state)
            normal_dist=Normal(mean,std)
            action=normal_dist.sample()
            return action.detach().numpy().squeeze()

    def update(self,states,actions,rewards):

        self.optimizer.zero_grad()
        states=torch.from_numpy(states).to(self.device)
        actions=torch.from_numpy(actions).to(self.device)

        with torch.no_grad():
            values=self.net.critic(states)

        batch_size=rewards.shape[0]
        returns=torch.Tensor(batch_size).to(self.device)
        deltas=torch.Tensor(batch_size).to(self.device)
        advantage=torch.Tensor(batch_size).to(self.device)

        prev_return=0
        prev_value=0
        prev_advantage=0
        for i in reversed(range(batch_size)):
            returns[i]=rewards[i]+self.gamma*prev_return
            deltas[i]=rewards[i]+self.gamma*prev_value-values[i].data.numpy()
            # ref: https://arxiv.org/pdf/1506.02438.pdf(GAE)
            advantage[i]=deltas[i]+self.gamma*self.lambd*prev_advantage

            prev_return=returns[i]
            prev_value=values[i].data.numpy()
            prev_advantage=advantage[i]
        advantage = (advantage - advantage.mean())/(advantage.std()+self.entropy_eps)

        values=self.net.critic(states)

        mean,std=self.net.actor(states)
        normal_dist = Normal(mean, std)
        surr_loss=normal_dist.log_prob(actions).sum(dim=1, keepdim=True)

        surr_grad=torch.autograd.grad(surr_loss,self.net.actor.parameters())
        flat_surr_grad=torch.cat([grad.view(-1) for grad in surr_grad]).data
        nature_grad=self.conjugated_gradient()

    def _fisher_matrix(self):
        pass

    def _get_kl(self):
        pass

    def _conjugated_gradient(self):
        pass



