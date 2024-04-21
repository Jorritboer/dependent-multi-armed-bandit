import numpy as np


class RandomPolicy:
    def __init__(self):
        pass

    def choose_bandit(self, k_array, reward_array, n_bandits):
        return np.random.choice(range(n_bandits), 1)[0]
