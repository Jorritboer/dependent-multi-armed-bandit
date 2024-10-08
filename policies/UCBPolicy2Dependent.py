import numpy as np
import cvxpy as cp


# UCB2 dependent
# variant that rounds to 3 decimals and picks random on tie
class UCBPolicy2Dependent:

    # if we're currently playing a bandit, this variable will indicate which
    playing_bandit = None
    # indicating how many times left to play
    playing_times = None

    # initializing
    def __init__(self, alpha, variables, bandit_expressions):
        self.alpha = alpha
        self.r = np.array([0] * len(bandit_expressions))

        self.parameters = [
            {"min": cp.Parameter(), "max": cp.Parameter()}
            for _ in range(len(bandit_expressions))
        ]

        constraints = (
            [v >= 0 for v in variables]
            + [v <= 1 for v in variables]
            + [
                bandit_expressions[b] >= self.parameters[b]["min"]
                for b in range(len(bandit_expressions))
            ]
            + [
                bandit_expressions[b] <= self.parameters[b]["max"]
                for b in range(len(bandit_expressions))
            ]
        )

        self.problems = [
            cp.Problem(cp.Maximize(b), constraints) for b in bandit_expressions
        ]

    def dep_max_est(self, estimators, bound):
        for b in range(len(self.parameters)):
            self.parameters[b]["min"].value = estimators[b] - bound[b]
            self.parameters[b]["max"].value = estimators[b] + bound[b]

        return [prob.solve(solver=cp.SCS) for prob in self.problems]

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

        # max(0,.) added to prevent negative roots
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

        dep_bounds = self.dep_max_est(success_ratio, sqrt_term)

        upper_bounds = [
            min(1, (success_ratio + sqrt_term)[i], round(dep_bounds[i], 3))
            for i in range(n_bandits)
        ]

        bandit = np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))

        self.playing_bandit = bandit
        self.playing_times = np.ceil(
            (1 + self.alpha) ** (self.r[bandit] + 1)
        ) - np.ceil((1 + self.alpha) ** self.r[bandit])
        # print(f"Playing bandit {bandit} {self.playing_times} times")
        # - 1 because we play it once immediately
        self.playing_times -= 1
        self.r[bandit] += 1

        return bandit
