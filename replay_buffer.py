import collections
import random
import torch

class ReplayBuffer(object):
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, state, action, reward, state_next, done):
        self.buffer.append((state, action, reward, state_next, done))

    def sample(self, n_sample):
        return random.sample(self.buffer, n_sample)

    def clear(self):
        self.buffer.clear()

    @property
    def size(self):
        return len(self.buffer)
    
