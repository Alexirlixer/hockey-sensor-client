import numpy as np

class fixed_size_buffer:
    def __init__(self, size, dimensions, dtype=float):
        self.size = size
        self.buffer = np.zeros((size, dimensions), dtype=dtype)
        self.index = 0  # Current insertion index
        self.is_full = False

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0 and not self.is_full:
            self.is_full = True

    def get(self):
        if self.is_full:
            return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))
        else:
            return self.buffer[:self.index]

    def __len__(self):
        if self.is_full:
            return self.size
        else:
            return self.index
    def clear(self):
        self.buffer.fill(0)
        self.index = 0
        self.is_full = False

