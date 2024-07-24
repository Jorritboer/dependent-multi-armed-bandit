import numpy as np
import cvxpy as cp


# dependent UCB
# variant that rounds to 3 decimals and picks random on tie
class UCBDependentB:

    def __init__(self, variables, bandit_expressions, alpha=1):
        self.parameters = [
            {"min": cp.Parameter(), "max": cp.Parameter()}
            for _ in range(len(bandit_expressions))
        ]
        self.alpha = alpha

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

    def choose_bandit(self, k_array, reward_array, n_bandits):
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # try out each arm once, and avoid dividing by 0 below
        for k in range(n_bandits):
            if total_count[k] == 0:
                return k

        success_ratio = success_count / total_count

        sqrt_term = (1 / self.alpha) * np.sqrt(
            2 * np.log(np.sum(total_count)) / total_count
        )

        dep_bounds = self.dep_max_est(success_ratio, sqrt_term)

        # return np.argmax(success_ratio + sqrt_term)
        upper_bounds = [
            min(1, (success_ratio + sqrt_term)[i], round(dep_bounds[i], 3))
            for i in range(n_bandits)
        ]
        return np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))
