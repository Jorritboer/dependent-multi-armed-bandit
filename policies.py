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

        # ratio of sucesses vs total
        success_ratio = success_count / total_count

        # choosing best greedy action or random depending with epsilon probability
        if np.random.random() < self.epsilon:

            # returning random action, excluding best
            return np.random.choice(
                np.delete(list(range(N_BANDITS)), np.argmax(success_ratio))
            )

        # else return best
        else:

            # returning best greedy action
            return np.argmax(success_ratio)


# e-greedy policy
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

        # ratio of sucesses vs total
        success_ratio = success_count / total_count

        # computing square root term
        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        # returning best greedy action
        return np.argmax(success_ratio + sqrt_term)


# e-greedy policy
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
