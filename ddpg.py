import os
from copy import deepcopy

import mujoco_py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from model import OUNoise, Actor, Critic, ReplayBuffer

# Hyper Parameters:

REPLAY_CAPACITY = 100000
MAX_STEP = 2000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

class DDPG(object):
    """DDPG class
    Args:
        gym environment
    """
    def __init__(self, env, path_to_actor=None, path_to_critic=None, tau=0.001, gamma=0.99, batch_size=32, lr=1e-3):
        self.tau = tau
        self.gamma = gamma
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_high = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
        self.buffer_size = REPLAY_CAPACITY
        self.batch_size = batch_size
        self.explore_noise = OUNoise(self.action_dim)
        if path_to_actor:
            self.actorEval = torch.load(path_to_actor)
        else:
            self.actorEval = Actor(self.state_dim, self.action_dim, self.action_high).to(device)
        self.actorTarget = Actor(self.state_dim, self.action_dim, self.action_high).to(device)
        self.actorTarget.load_state_dict(self.actorEval.state_dict())
        if path_to_critic:
            self.criticEval = torch.load(path_to_critic)
        else:
            self.criticEval = Critic(self.state_dim, self.action_dim).to(device)
        self.criticTarget = Critic(self.state_dim, self.action_dim).to(device)
        self.criticTarget.load_state_dict(self.criticEval.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actorEval.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.criticEval.parameters(), lr=lr)

        self.criterion = nn.MSELoss()
        self.exp_counter = 0    # Increment only enqueue buffer
    def choose_action(self, state):
        """ Output deterministic action from Actor-eval
        *Ensure state to be TENSOR
        """
        state = state.squeeze(0)
        return self.actorEval(state).detach()
    def target_network_replace(self):
        """update the Target network by ratio tau
        """
        for eval_param, target_param in zip(self.actorEval.parameters(), self.actorTarget.parameters()):
            target_param = target_param * (1 - self.tau) + eval_param * self.tau
        for eval_param, target_param in zip(self.criticEval.parameters(), self.criticTarget.parameters()):
            target_param = target_param * (1 - self.tau) + eval_param * self.tau
    def enQueue(self, state, action, reward, done, new_state):
        self.replay_buffer.enQueue(state, action, reward, done, new_state)
        self.exp_counter += 1
    def fit(self):
        """ Add exploration noise to action when training(main)
        """
        # fetch a batch
        
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        states = []
        next_states = []
        actions = []
        rewards = []
        terminates = []
        for data in minibatch:
            tmp_state,tmp_action,tmp_reward,tmp_done,tmp_next = data
            states.append(tmp_state)
            actions.append(tmp_action)
            rewards.append(tmp_reward)
            terminates.append(tmp_done)
            next_states.append(tmp_next)
        
        # unsqueeze also
        batch_states = torch.stack(states, dim=0).to(device)
        batch_next_states = torch.stack(next_states, dim=0).to(device)
        actions = torch.stack(actions, dim=0).to(device)
        rewards = torch.stack(rewards,dim=0)
    
        # Actor & update ==> action **
        aeval = self.actorEval(batch_states)
        qtmp = self.criticEval(batch_states, aeval)
        action_loss = - torch.mean(qtmp)   # The loss function is Q itself(minus mean)
        
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()

        # Critic & update
        atarget = self.actorTarget(batch_next_states)
        qtarget = self.criticTarget(batch_next_states, atarget)

        qeval = self.criticEval(batch_states, actions)
        y = self.gamma * qtarget.cpu().detach() + rewards
        y = y.to(device)
        # Computing the loss
        loss = self.criterion(qeval, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        if self.exp_counter % 10000 == 0:
            print("[INFO] Actor Loss:{} | Critic Loss:{}".format(action_loss, loss))
            self.target_network_replace()
        return loss 

    

