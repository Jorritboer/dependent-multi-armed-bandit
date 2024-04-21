import numpy as np
import cvxpy as cp


# dependent UCB1
class UCBDependent:

    def __init__(self, variables, bandit_expressions):
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

    def choose_bandit(self, k_array, reward_array, n_bandits):
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # try out each arm once, and avoid dividing by 0 below
        for k in range(n_bandits):
            if total_count[k] == 0:
                return k

        success_ratio = success_count / total_count

        sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        dep_bounds = self.dep_max_est(success_ratio, sqrt_term)

        # return np.argmax(success_ratio + sqrt_term)
        return np.argmax(
            [
                min((success_ratio + sqrt_term)[i], dep_bounds[i])
                for i in range(n_bandits)
            ]
        )
