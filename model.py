import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_dim=3, output_dim=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """存入一筆經驗 (state, action, reward, next_state, done)"""
        self.buffer.append(transition)

    def sample(self, batch_size):
        """隨機取樣一批經驗"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)
