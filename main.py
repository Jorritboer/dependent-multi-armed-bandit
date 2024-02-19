import numpy as np
import plot
import policies
import cvxpy as cp
from simulation import simulation


class MAB:

    def __init__(self, bandit_probs):
        self.bandit_probs = bandit_probs

    def draw(self, k):
        return (
            np.random.binomial(1, self.bandit_probs[k]),
            np.max(self.bandit_probs) - self.bandit_probs[k],
        )


v1 = 0.5
v2 = 0.2

bandit_probs = [v1, v1 - v2]
mab = MAB(bandit_probs)


def dep_max_est(estimators, bound):
    v1 = cp.Variable()
    v2 = cp.Variable()

    eq1 = v1
    eq2 = v1 - v2

    constraints = [
        v1 >= 0,
        v1 <= 1,
        v2 >= 0,
        v2 <= 1,
        eq1 >= estimators[0] - bound[0],
        eq1 <= estimators[0] + bound[0],
        eq2 <= estimators[1] + bound[1],
        eq2 >= estimators[1] - bound[1],
        # v1 + v2 <= estimators[2] + bound[2],
        # v1 + v2 >= estimators[2] - bound[2],
    ]

    obj = cp.Maximize(eq1)
    prob = cp.Problem(obj, constraints)
    p1_max_est = prob.solve()
    obj = cp.Maximize(eq2)
    prob = cp.Problem(obj, constraints)
    p2_max_est = prob.solve()
    # obj = cp.Maximize(v1 + v2)
    # prob = cp.Problem(obj, constraints)
    # p3_max_est = prob.solve()
    return p1_max_est, p2_max_est


# dependent UCB
def choose_bandit(k_array, reward_array, n_bandits):
    success_count = reward_array.sum(axis=1)
    total_count = k_array.sum(axis=1)

    # try out each arm once, and avoid dividing by 0 below
    for k in range(n_bandits):
        if total_count[k] == 0:
            return k

    success_ratio = success_count / total_count

    sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

    dep_bounds = dep_max_est(success_ratio, sqrt_term)

    return np.argmax(
        [min((success_ratio + sqrt_term)[i], dep_bounds[i]) for i in range(n_bandits)]
    )


def random_policy(k_array, reward_array, n_bandits):
    return np.random.choice(range(n_bandits), 1)[0]


plot.plot_MAB_experiment(choose_bandit, mab, 100, bandit_probs, "Bounded UCB")


algorithms = {
    "e_greedy": policies.eGreedyPolicy(0.1).choose_bandit,
    "ucb": policies.UCBPolicy().choose_bandit,
    "ts": policies.TSPolicy().choose_bandit,
    "ucb-bounded": policies.TSPolicy().choose_bandit,
}

simulation(mab, algorithms, 100, 100)
