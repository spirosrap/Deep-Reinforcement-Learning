import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

import numpy as np
import random
from collections import namedtuple, deque
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

gym.logger.set_level(40)
env = gym.make("CartPole-v0")
env.seed(0)
print("observation space",env.observation_space.shape)
print("action space",env.action_space.n)

class Policy(nn.Module):
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

policy = Policy().to(device) optimizer = optim.Adam(policy.parameters(),lr=1e-02)
def reinforce(n_episodes=1000,max_t=1000,gamma=1.0,print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1,n_episodes+1):
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

    return scores

scores = reinforce()
