import numpy as np


# upper confidence bound policy
class UCBPolicy:

    # initializing
    def __init__(self):

        # nothing to do here
        pass

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

        # computing square root term
        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        # returning best greedy action
        return np.argmax(success_ratio + sqrt_term)
