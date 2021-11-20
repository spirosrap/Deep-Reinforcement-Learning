import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque
from torch.distributions import Categorical
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

"""
The code in this file implements the REINFORCE algorithm.
The REINFORCE algorithm is an actor-critic algorithm that uses the policy gradient theorem to learn the optimal policy.
The algorithm works by having the agent play a number of episodes and then updating the policy based on the rewards obtained in each episode.
The policy is updated by taking the gradient of the log probability of the action taken in the episode multiplied by the reward.
"""

class Policy(nn.Module):
    # Changing the Full connected layer units to 16,32,64 or more doesn't improve
    # how quick the model converges - if it ever converges. Also, adding an additional layer
    # doesn't improve the effectiveness of the algorithm
    def __init__(self, state_size=4, action_size=2, seed=0, fc1_units=8):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def act(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

policy = Policy().to(device)
policy.load_state_dict(torch.load("reinforce.pth"))

env = gym.make('CartPole-v0')
state = env.reset()
env.render()
rewards = []

for t in range(1000):
    action, log_prob = policy.act(state)
    state, reward, done, _ = env.step(action)
    if done:
        print(reward, "done")
        break    
    rewards.append(reward)
                        #env.render()
    time.sleep(0.05)

print(rewards)
env.close()