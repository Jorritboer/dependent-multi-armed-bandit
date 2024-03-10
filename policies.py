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
                np.delete(list(range(n_bandits)), np.argmax(success_ratio))
            )

        # else return best
        else:

            # returning best greedy action
            return np.argmax(success_ratio)


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

        # ratio of sucesses vs total
        success_ratio = success_count / total_count

        # computing square root term
        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        # returning best greedy action
        return np.argmax(success_ratio + sqrt_term)


# upper confidence bound policy
# version B, where we max out the upper confidence at 1
class UCBPolicyB:

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
        return np.argmax(
            [min(1, success_ratio[i] + sqrt_term[i]) for i in range(n_bandits)]
        )


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

        # computing square root term
        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        upper_bounds = [
            min(1, success_ratio[i] + sqrt_term[i]) for i in range(n_bandits)
        ]
        # returning best greedy action
        return np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))


# UCB2
class UCBPolicy2:

    # if we're currently playing a bandit, this variable will indicate which
    playing_bandit = None
    # indicating how many times left to play
    playing_times = None

    # initializing
    def __init__(self, alpha, n_bandits):
        self.alpha = alpha
        self.r = np.array([0] * n_bandits)
        pass

    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # try out each arm once, and avoid dividing by 0 below
        for k in range(n_bandits):
            if total_count[k] == 0:
                # also reset (this is a bit of a hack to solve a problem for running multiple simulations)
                self.r = np.array([0] * n_bandits)
                self.playing_bandit = None
                self.playing_times = None
                return k

        if self.playing_bandit and self.playing_times > 0:
            bandit = self.playing_bandit
            self.playing_times -= 1
            # if we're done playing, set to None
            if self.playing_times == 0:
                self.playing_bandit = None
            return bandit

        success_ratio = success_count / total_count

        sqrt_term = np.sqrt(
            (1 + self.alpha)
            * np.maximum(
                0,
                np.log(
                    np.e * np.sum(total_count) / (np.ceil((1 + self.alpha) ** self.r))
                ),
            )
            / (2 * np.ceil((1 + self.alpha) ** self.r))
        )

        bandit = np.argmax(success_ratio + sqrt_term)
        self.playing_bandit = bandit
        self.playing_times = np.ceil(
            (1 + self.alpha) ** (self.r[bandit] + 1)
        ) - np.ceil((1 + self.alpha) ** self.r[bandit])
        # print(f"Playing bandit {bandit} {self.playing_times} times")
        # - 1 because we play it once immediately
        self.playing_times -= 1
        self.r[bandit] += 1

        return bandit


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
