import numpy as np


# e-greedy policy
class eGreedyPolicy:

    # initializing
    def __init__(self, epsilon):

        # saving epsilon
        self.epsilon = epsilon

    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):

        # sucesses and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # try out each arm once, and avoid dividing by 0 below
        for k in range(n_bandits):
            if total_count[k] == 0:
                return k

        # ratio of sucesses vs total
        success_ratio = success_count / total_count

        # choosing best greedy action or random depending with epsilon probability
        if np.random.random() < self.epsilon:

            # returning random action, excluding best
            return np.random.choice(
                np.delete(list(range(n_bandits)), np.argmax(success_ratio))
            )

        # else return best
        else:

            # returning best greedy action
            return np.argmax(success_ratio)
