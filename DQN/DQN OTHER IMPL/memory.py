# implement different replay methods in this file
from collections import deque
import random
import torch 
from torch.autograd import Variable
import numpy as np

class RandomMemory(object):
    def __init__(self, size, batch_size):
        self.max_size = size
        self.current_size = 0
        self.data = deque(maxlen=size)
        self.batch_size = batch_size

    def add(self, state, action, reward, newstate, done):
        if self.current_size < self.max_size:
            self.data.append((state, int(action), reward, newstate, not done))
            #self.data.append((state.data.numpy(),action,reward,newstate,done))
            self.current_size += 1
        else: # pop the oldest element
            self.data.popleft()
            self.data.append((state, int(action), reward, newstate, not done))
    
    def get_batch(self):
        batch = random.sample(self.data, self.batch_size)
        batch = np.array(batch)
        state = Variable(torch.cat(batch[:,0]))
        action = torch.LongTensor(batch[:,1])
        reward = Variable(torch.FloatTensor(batch[:,2]))
        next_state = Variable(torch.cat(batch[:,3]))
        not_done = Variable(torch.Tensor(batch[:,4]).float())
        return state, action, reward, next_state, not_done
