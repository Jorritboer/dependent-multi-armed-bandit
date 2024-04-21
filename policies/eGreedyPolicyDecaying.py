import numpy as np


# e-greedy policy with decaying epsilon
class eGreedyPolicyDecaying:
    # epsilon is now a function
    def __init__(self, epsilon_func):
        self.epsilon_func = epsilon_func

    def choose_bandit(self, k_array, reward_array, n_bandits):

        # sucesses and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # try out each arm once, and avoid dividing by 0 below
        for k in range(n_bandits):
            if total_count[k] == 0:
                return k

        success_ratio = success_count / total_count

        epsilon = self.epsilon_func(np.sum(total_count))
        if np.random.random() < epsilon:
            return np.random.choice(
                np.delete(list(range(n_bandits)), np.argmax(success_ratio))
            )
        return np.argmax(success_ratio)
