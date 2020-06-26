from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        hidden_size = 512
        # Actor ==> deterministic action
        self.a1 = nn.Linear(state_dim, hidden_size)
        self.a1.weight.data.normal_(0.,0.1) # initialization
        self.a2 = nn.Linear(hidden_size, hidden_size//2)
        self.a2.weight.data.normal_(0.,0.1) 
        self.a3 = nn.Linear(hidden_size//2, action_dim)
        self.a3.weight.data.normal_(0.,0.1) 
    def forward(self, state):
        """Return Raw tensor action
        """
        out = F.relu(self.a1(state))
        out = F.relu(self.a2(out))
        # Normalization
        out = F.tanh(self.a3(out)) * self.action_bound
        return out

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_size = 1024

        # Critic ==> Q-value
        self.c_fca = nn.Linear(action_dim, hidden_size//2)
        self.c_fca.weight.data.normal_(0.,0.1) # initialization
        self.c_fcs = nn.Linear(state_dim, hidden_size//2)
        self.c_fcs.weight.data.normal_(0.,0.1) 
        
        self.c_fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.c_fc1.weight.data.normal_(0.,0.1) 
        self.c_fc2 = nn.Linear(hidden_size//2, 1)
        self.c_fc2.weight.data.normal_(0.,0.1) 

    def forward(self, state, action):
        """Return a sampled action given state and policy
        Return: 
            Action (n, action_dim) 
            Continuous action !! 
        Input as numpy array
        """
        # Ensure the Tensor
        out_s = F.relu(self.c_fcs(state))
        out_a = F.relu(self.c_fca(action))
        value = torch.cat((out_s, out_a), dim=-1)
        value = F.relu(self.c_fc1(value))
        value = self.c_fc2(value)
        return value


class ReplayBuffer(object):
    """Replay buffer class
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_exp = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Fetch examples[batch_size] radnomly from the replaybuffer
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def enQueue(self, state, action, reward, done, new_state):
        """raw input state(Tensor, squeezed)
        """
        record = (state, action, reward, done, new_state)
        if self.num_exp < self.buffer_size:
            self.buffer.append(record)
            self.num_exp += 1
        else:
            self.buffer.popleft()
            self.buffer.append(record)

    def length(self):
        # If buffer is full, return buffer size
        # Otherwise, return experience counter
        return self.num_exp

    def clear(self):
        self.buffer = deque()
        self.num_exp = 0


class OUNoise(object):
    """Ornstein-Uhlenbeck Noise applied"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    def decay(self):
        self.sigma *= 0.995


if __name__ == "__main__":
    
    s= np.array([1.,2.,3.,1.,1.], dtype=np.float32)
    s = torch.from_numpy(s)
    print(s.shape)
