import numpy as np
import cvxpy as cp


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
