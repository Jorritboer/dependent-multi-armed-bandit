import numpy as np
import cvxpy as cp


# Explore based on derivatives of the LP solution
# Effectively: eGreedy-decaying + UCB-Tuned + dependent + explore derivatives
class DerivativeExplorePolicy:
    def __init__(self, variables, bandit_expressions, epsilon_func):
        self.epsilon_func = epsilon_func
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

    # compute per bandit the impact that pulling that bandit would have on LP solution
    # this is computed as the sum of the gradients of those parameters on the different problems
    # multiplied by the derivative in n for that bandit
    def compute_derivatives(self, total_count):
        derivatives = np.zeros(len(self.problems))

        for i, prob in enumerate(self.problems):
            # solve again, but now with requires_grad
            prob.solve(solver=cp.SCS, requires_grad=True)
            prob.backward()
            derivatives += [
                abs(param["min"].gradient) + abs(param["max"].gradient)
                for param in self.parameters
            ]
            for p in range(len(self.parameters)):
                param = self.parameters[p]

        n = sum(total_count)
        n_derivatives = -np.sqrt(np.log(n)) / (
            total_count * np.sqrt(2 * total_count)
        ) + 1 / (n * np.sqrt(total_count) * np.sqrt(2 * np.log(n)))

        return derivatives * np.abs(n_derivatives)

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

        if any([abs(dep_bound) == np.inf for dep_bound in dep_bounds]):
            # problem could have been infeasible or unbounded
            # in which case return random
            # to prevent compute_derivatives from crashing
            return np.random.choice(range(n_bandits), 1)[0]

        # with chance epsilon we pick the highest derivative bandit to explore
        epsilon = self.epsilon_func(np.sum(total_count))
        if np.random.random() < epsilon:
            return np.argmax(self.compute_derivatives(total_count))

        # return np.argmax(success_ratio + sqrt_term)
        upper_bounds = [
            min(1, (success_ratio + sqrt_term)[i], round(dep_bounds[i], 3))
            for i in range(n_bandits)
        ]
        return np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))
