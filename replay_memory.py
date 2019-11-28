from sample import Sample
import random
import numpy as np


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * capacity  # pre-allocate memory
        self.position = 0
        self.size = 0

    def push(self, *args):
        """Saves a sample."""
        sample = Sample(*args)
        for elem in sample:
            if elem is not None:
                if np.any(np.isnan(elem)):
                    print("Avoided pushing NaN into memory", elem)
                    return

        self.size = min(self.size + 1, self.capacity)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory[0:self.size], batch_size)

    def __len__(self):
        return self.size
