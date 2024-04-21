import numpy as np


# Thompson Sampling policy
class TSPolicy:

    # initializing
    def __init__(self):

        # nothing to do here
        pass

    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):

        # list of samples, for each bandit
        samples_list = []

        # sucesses and failures
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count

        # drawing a sample from each bandit distribution
        samples_list = [
            np.random.beta(1 + success_count[bandit_id], 1 + failure_count[bandit_id])
            for bandit_id in range(n_bandits)
        ]

        # returning bandit with best sample
        return np.argmax(samples_list)
