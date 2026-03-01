import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, trajectory, reward, score_diff):
        self.buffer.append((trajectory, reward, abs(score_diff)))
        if len(self.buffer) > self.capacity:
            self.buffer.sort(key=lambda x: x[2], reverse=True)
            self.buffer = self.buffer[:self.capacity]

    def sample(self, n):
        if len(self.buffer) < n:
            return [(t, r) for t, r, _ in self.buffer]

        weights = [x[2] + 0.1 for x in self.buffer]
        indices = random.choices(range(len(self.buffer)), weights=weights, k=n)
        return [(self.buffer[i][0], self.buffer[i][1]) for i in indices]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
