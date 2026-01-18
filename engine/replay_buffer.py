# engine/replay_buffer.py
import random
import torch
import os

class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.data = []  # list of (state_tensor, pi_tensor, value_tensor)

    def add_samples(self, samples):
        self.data.extend(samples)
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]  # keep recent only

    def sample_batch(self, batch_size=32):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        states = torch.stack([b[0] for b in batch])
        pis = torch.stack([b[1] for b in batch])
        vals = torch.stack([b[2] for b in batch])
        return states, pis, vals

    def __len__(self):
        return len(self.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.data, path)

    def load(self, path):
        if os.path.exists(path):
            self.data = torch.load(path, map_location="cpu", weights_only=False)
        else:
            self.data = []
