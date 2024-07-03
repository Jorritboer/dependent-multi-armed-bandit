import numpy as np


# LinUCB as from Chu et al. but assuming Bernoulli bandits, and constant
class LinUCB:

    # initializing
    def __init__(self, alpha, feature_vectors):
        self.alpha = alpha
        self.feature_vectors = feature_vectors

        # d is the number of features
        d = len(feature_vectors[0])
        self.A = np.identity(d)
        self.b = np.zeros([d, 1])

        # last chosen bandit
        self.previous_action = None

    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):
        total_count = int(k_array.sum(axis=1).sum())
        if total_count > 0:
            # update matrix with last action and observed reward
            x = self.feature_vectors[self.previous_action]
            self.A = self.A + np.dot(x, x.T)

            self.b = self.b + reward_array[self.previous_action][total_count - 1] * x
        else:
            # no actions taken yet, so reset (this is a bit of a hack to solve a problem for running multiple simulations)
            # d is the number of features
            d = len(self.feature_vectors[0])
            self.A = np.identity(d)
            self.b = np.zeros([d, 1])

            # last chosen bandit
            self.previous_action = None

        A_inv = np.linalg.inv(self.A)
        theta_t = np.dot(A_inv, self.b)

        upper_bounds = [
            np.dot(theta_t.T, x) + self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
            for x in self.feature_vectors
        ]
        bandit = np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))
        self.previous_action = bandit

        return bandit
