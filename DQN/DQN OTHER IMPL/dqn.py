import numpy as np
import gym
from gym import wrappers
from atari_wrappers import wrap_deepmind
env_name = "FreewayNoFrameskip-v4"
env = gym.make(env_name)
env = wrap_deepmind(env)
env = wrappers.Monitor(env, env_name + '_results', force=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from memory import RandomMemory

# Raw atari image size: (210, 160, 3)

transform =  T.Compose([
            T.ToTensor()
        ])

def process(img):
    return torch.Tensor(transform(img)).unsqueeze(0)

class Qnet(nn.Module):
    def __init__(self, num_actions):
        super(Qnet, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(4)
        # (84 - 8) / 4  + 1 = 20 (len,width) output size
        self.cnn1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(32)
        # (20 - 4) / 2 + 1 = 9 (len,width) output size
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.bn3 = torch.nn.BatchNorm2d(64)
        # (9 - 3) / 1 + 1 = 7 (len,width )output size
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn4 = torch.nn.BatchNorm2d(64)
        # fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        out = F.relu(self.cnn1(self.bn1(x)))
        out = F.relu(self.cnn2(self.bn2(out)))
        out = F.relu(self.cnn3(self.bn3(out)))
        # Resize from (batch_size, 64, 7, 7) to (batch_size,64*7*7)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)

min_epsilon = 0.1
init_epsilon = 1
eps_decay_steps = 1000000
epsilon = init_epsilon
Q = Qnet(env.action_space.n)
Q_target = Qnet(env.action_space.n)

def e_greedy(state, time):
    epsilon = max(min_epsilon, init_epsilon - (init_epsilon-min_epsilon) * time/eps_decay_steps)
    if np.random.random() < epsilon:
        return np.random.choice(range(env.action_space.n), 1)[0]
    else:
        # since pytorch networks expect batch input, add extra input of zeros
        state = torch.cat((Variable(state,volatile=True),torch.zeros(state.size())),0)
        qvalues = Q(state)
        maxq, actions = torch.max(qvalues, 1)
        return actions[0].data[0]

Q = Qnet(env.action_space.n)
Qtarget = Qnet(env.action_space.n)
batch_size = 32
memory = RandomMemory(1000000, batch_size)
discount = 0.99
target_update_frequency = 10000
learning_rate = 0.00025
#optimizer = torch.optim.Adamax(Q.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(Q.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)
train_frequency = 4
time = 0

for episode in range(10000):
    state = env.reset()
    state = process(state)
    done = False
    while not done: # restart episode if done
        env.render()
        action = e_greedy(state, time)
        newstate, reward, done, _ = env.step(action)
        # need to mark terminal state with done... so q target will just be reward
        memory.add(state, action, reward, process(newstate), done)
        # sample random batch of transitions and train the network if you have enough data
        if memory.current_size >= batch_size and time % train_frequency == 0:
            state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = memory.get_batch()
            # reset stored gradients to 0
            optimizer.zero_grad()
            # get q values of actions
            qvalues = Q(state_batch)[range(batch_size), action_batch]
            # get max q values of next states
            qtargetValues, _ = torch.max(Qtarget(next_state_batch), 1)
            # zero out qtargetVaues for terminal states, since their value is only the reward!
            qtargetValues = not_done_batch * qtargetValues
            # compute target q value: reward + gamma * max q values of next states
            qtarget = reward_batch + discount * qtargetValues
            # prevent backpropagation on target network
            qtarget = qtarget.detach()
            # Use Huber loss instead of manually clamping the gradients
            loss = F.smooth_l1_loss(qvalues, qtarget)
            # backpropagate
            loss.backward()
            # update parameters
            optimizer.step()

        # every C steps reset Qtarget = Q
        if time % target_update_frequency == 0:
            Qtarget.load_state_dict(Q.state_dict())

        time += 1
