import numpy as np
import cvxpy as cp


# upper confidence bound policy, tuned version, dependent
# rest same as version C of UCB: we max out the upper confidence at 1
# variant that rounds to 3 decimals and picks random on tie
class UCBPolicyTunedDependent:

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

        # compute variances
        vars = np.sum(reward_array**2, axis=1) / total_count - (success_ratio**2)
        V = vars + np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

        # computing square root term
        sqrt_term = np.sqrt(
            (np.log(np.sum(total_count)) / total_count) * np.minimum(0.25, V)
        )

        dep_bounds = self.dep_max_est(success_ratio, sqrt_term)

        # return np.argmax(success_ratio + sqrt_term)
        upper_bounds = [
            min(1, (success_ratio + sqrt_term)[i], round(dep_bounds[i], 3))
            for i in range(n_bandits)
        ]
        return np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))
