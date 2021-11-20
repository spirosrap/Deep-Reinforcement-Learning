import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
import time

gym.logger.set_level(40)
env = gym.make("CartPole-v0")
env.seed(0)
print("observation space",env.observation_space.shape)
print("action space",env.action_space.n)

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
optimizer = optim.Adam(policy.parameters(),lr=1e-2)
def main():
    n_episodes=5000
    max_t=200
    gamma=0.995
    print_every=100
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1,n_episodes+1):
        env.render()
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        discounts = [gamma**i for i in range(len(rewards))]
        R = [a*b for a,b in zip(discounts, rewards)]
        R_sum = sum(R)
        policy_loss = []
        for i,log_prob in enumerate(saved_log_probs):
            policy_loss.append(-log_prob*R_sum)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print("\rEpisode{:d}\tAverage Score {:.2f}".format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print("\rEpisode{:d}\tAverage Score {:.2f}".format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) > 195:
            print("\nSolved in {:d}\tAverage Score {:.2f}".format(i_episode, np.mean(scores_deque)))
            torch.save(policy.state_dict(), "reinforce.pth")
            break

if __name__ == '__main__':
    main()

"""
Explain what the code does.
"""

"""
The code is an implementation of the REINFORCE algorithm.
The REINFORCE algorithm is an actor-critic method.
The actor is the policy network, which takes in the state and outputs the action to take.
The critic is the value network, which takes in the state and outputs the value of the state.
The actor is trained by taking the gradient of the log probability of the action taken times the discounted reward.
The critic is trained by taking the discounted reward.
The algorithm is run for 5000 episodes.
The code runs the environment for a maximum of 200 timesteps.
The code saves the policy network every 100 episodes.
The code prints the average score every 100 episodes.
The code prints the solved message when the average score is above 195.
"""
