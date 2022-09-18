import numpy as np
from collections import deque
import random


# experience-uudiig hadgalah buffer

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=self.size)

    def add(self, state, action, reward, next_state, d):
        exp = (state, action, reward, next_state, d)
        self.buffer.append(exp)

    def sample_exp(self, size):
        batch = []
        size = min(size, len(self.buffer))
        batch = random.sample(self.buffer, size)

        states = np.float32([arr[0] for arr in batch])
        actions = np.float32([arr[1] for arr in batch])
        rewards = np.float32([arr[2] for arr in batch])
        next_states = np.float32([arr[3] for arr in batch])
        d = np.float32([arr[4] for arr in batch])

        return states, actions, rewards, next_states, d
