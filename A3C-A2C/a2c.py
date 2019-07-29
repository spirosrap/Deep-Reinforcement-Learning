import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import math
import random
from collections import namedtuple, deque
from multiprocessing_env import SubprocVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_env():
    def _thunk():
        env = gym.make("CartPole-v0")
        return env
    return _thunk

envs = [make_env() for i in range(16)]
envs = SubprocVecEnv(envs)
env = gym.make("CartPole-v0")

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, action_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

state_size  = envs.observation_space.shape[0]
action_size = envs.action_space.n
#Hyper params:
fc1_units = 256
LR = 3e-4
rollout_length = 5

model = ActorCritic(state_size, action_size, fc1_units).to(device)
optimizer = optim.Adam(model.parameters(),lr=LR)

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    # discounts = [gamma**i for i in range(len(rewards))]
    # R = [a*b for a,b in zip(discounts, rewards)]
    return returns

def train_a2c():
    max_frames   = 20000
    frame_idx    = 0
    test_rewards = []
    state = envs.reset()
    scores_deque = deque(maxlen=100)
    while frame_idx < max_frames:
        saved_log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        for _ in range(rollout_length):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            saved_log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            frame_idx += 1
            m = np.mean([test_env() for _ in range(10)])
            scores_deque.append(m)
            print("\rFrame {:d}\tAverage total reward: {:.2f}".format(frame_idx, np.mean(scores_deque)), end="")
            if frame_idx % 1000 == 0:
                # print(np.mean([test_env() for _ in range(10)]))
                print("\rFrame {:d}\tAverage total reward: {:.2f}".format(frame_idx, np.mean(scores_deque)))
                test_rewards.append(m)
                # plot(frame_idx, test_rewards)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)

        saved_log_probs = torch.cat(saved_log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values

        actor_loss  = -(saved_log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return test_rewards

rew = train_a2c()
