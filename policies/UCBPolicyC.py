import numpy as np


# upper confidence bound policy
# version C, where we max out the upper confidence at 1
# but, in case multiple upper bounds are maximal, we pick random from those
class UCBPolicyC:

    # initializing
    def __init__(self):

        # nothing to do here
        pass

    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):
        # sucesses and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # ratio of sucesses vs total
        success_ratio = success_count / total_count

        # try out each arm once, and avoid dividing by 0 below
        for k in range(n_bandits):
            if total_count[k] == 0:
                return k

        # computing square root term
        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        upper_bounds = [
            min(1, success_ratio[i] + sqrt_term[i]) for i in range(n_bandits)
        ]
        # returning best greedy action
        return np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))
