# %%
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


# %%
v1 = 0.5
v2 = 0.2

bandit_probs = [v1, v1 + v2, v2, 0.7 * v1 + v2]
mab = MAB(bandit_probs)

v1 = cp.Variable()
v2 = cp.Variable()
# v3 = cp.Variable()

p1max = cp.Parameter()
p1min = cp.Parameter()
p2max = cp.Parameter()
p2min = cp.Parameter()
p3max = cp.Parameter()
p3min = cp.Parameter()
p4max = cp.Parameter()
p4min = cp.Parameter()

eq1 = v1
eq2 = v1 + v2
eq3 = v2
eq4 = 0.7 * v1 + v2

constraints = [
    v1 >= 0,
    v1 <= 1,
    v2 >= 0,
    v2 <= 1,
    eq1 >= p1min,
    eq1 <= p1max,
    eq2 >= p2min,
    eq2 <= p2max,
    eq3 >= p3min,
    eq3 <= p3max,
    eq4 >= p4min,
    eq4 <= p4max,
]

prob1 = cp.Problem(cp.Maximize(eq1), constraints)
prob2 = cp.Problem(cp.Maximize(eq2), constraints)
prob3 = cp.Problem(cp.Maximize(eq3), constraints)
prob4 = cp.Problem(cp.Maximize(eq4), constraints)


def dep_max_est(estimators, bound):
    # print(estimators, bound)
    p1min.value = estimators[0] - bound[0]
    p1max.value = estimators[0] + bound[0]
    p2min.value = estimators[1] - bound[1]
    p2max.value = estimators[1] + bound[1]
    p3min.value = estimators[2] - bound[2]
    p3max.value = estimators[2] + bound[2]
    p4min.value = estimators[3] - bound[3]
    p4max.value = estimators[3] + bound[3]

    p1_max_est = prob1.solve(solver=cp.SCS)
    p2_max_est = prob2.solve(solver=cp.SCS)
    p3_max_est = prob3.solve(solver=cp.SCS)
    p4_max_est = prob4.solve(solver=cp.SCS)
    return p1_max_est, p2_max_est, p3_max_est, p4_max_est


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

    def print_info():
        print(f"r{int(sum(total_count))}:")
        print(
            f"\t ub [{round((success_ratio - sqrt_term)[0],2)},{round((success_ratio)[0],2)},{round((success_ratio+sqrt_term)[0],5)}] \t dep_ub {round(dep_bounds[0],5)} \t diff? {(dep_bounds[0] < (success_ratio + sqrt_term)[0])}"
        )
        print(
            f"\t ub [{round((success_ratio - sqrt_term)[1],2)},{round((success_ratio)[1],2)},{round((success_ratio+sqrt_term)[1],5)}] \t dep_ub {round(dep_bounds[1],5)} \t diff? {(dep_bounds[1] < (success_ratio + sqrt_term)[1])}"
        )
        print(
            f"\t ub [{round((success_ratio - sqrt_term)[2],2)},{round((success_ratio)[2],2)},{round((success_ratio+sqrt_term)[2],5)}] \t dep_ub {round(dep_bounds[2],5)} \t diff? {(dep_bounds[2] < (success_ratio + sqrt_term)[2])}"
        )
        c1 = (
            np.argmax(
                [
                    min((success_ratio + sqrt_term)[i], dep_bounds[i])
                    for i in range(n_bandits)
                ]
            )
            + 1
        )
        c2 = np.argmax(success_ratio + sqrt_term) + 1
        print(f"Choice: {c1} =? {c2}: {c1==c2} ")
        print("--------------------------------------------------")

    # print_info()

    # return np.argmax(success_ratio + sqrt_term)
    return np.argmax(
        [min((success_ratio + sqrt_term)[i], dep_bounds[i]) for i in range(n_bandits)]
    )


# dependent UCB
# variant that rounds to 3 decimals and picks random on tie
def choose_bandit_B(k_array, reward_array, n_bandits):
    success_count = reward_array.sum(axis=1)
    total_count = k_array.sum(axis=1)

    # try out each arm once, and avoid dividing by 0 below
    for k in range(n_bandits):
        if total_count[k] == 0:
            return k

    success_ratio = success_count / total_count

    sqrt_term = np.sqrt(np.log(2 * np.sum(total_count)) / total_count)

    dep_bounds = dep_max_est(success_ratio, sqrt_term)

    def print_info():
        print(f"r{int(sum(total_count))}:")
        print(
            f"\t ub [{round((success_ratio - sqrt_term)[0],2)},{round((success_ratio)[0],2)},{round((success_ratio+sqrt_term)[0],5)}] \t dep_ub {round(dep_bounds[0],5)} \t diff? {(dep_bounds[0] < (success_ratio + sqrt_term)[0])}"
        )
        print(
            f"\t ub [{round((success_ratio - sqrt_term)[1],2)},{round((success_ratio)[1],2)},{round((success_ratio+sqrt_term)[1],5)}] \t dep_ub {round(dep_bounds[1],5)} \t diff? {(dep_bounds[1] < (success_ratio + sqrt_term)[1])}"
        )
        print(
            f"\t ub [{round((success_ratio - sqrt_term)[2],2)},{round((success_ratio)[2],2)},{round((success_ratio+sqrt_term)[2],5)}] \t dep_ub {round(dep_bounds[2],5)} \t diff? {(dep_bounds[2] < (success_ratio + sqrt_term)[2])}"
        )
        c1 = (
            np.argmax(
                [
                    min((success_ratio + sqrt_term)[i], dep_bounds[i])
                    for i in range(n_bandits)
                ]
            )
            + 1
        )
        c2 = np.argmax(success_ratio + sqrt_term) + 1
        print(f"Choice: {c1} =? {c2}: {c1==c2} ")
        print("--------------------------------------------------")

    # print_info()

    # return np.argmax(success_ratio + sqrt_term)
    upper_bounds = [
        min(1, (success_ratio + sqrt_term)[i], round(dep_bounds[i], 3))
        for i in range(n_bandits)
    ]
    return np.random.choice(np.flatnonzero(upper_bounds == np.max(upper_bounds)))


# UCB2
class UCBPolicy2_dependent:

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

        dep_bounds = dep_max_est(success_ratio, sqrt_term)

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


def random_policy(k_array, reward_array, n_bandits):
    return np.random.choice(range(n_bandits), 1)[0]


# %%
plot.plot_MAB_experiment(
    UCBPolicy2_dependent(0.1, len(bandit_probs)).choose_bandit,
    # choose_bandit_B,
    mab,
    1000,
    bandit_probs,
    "UCB2",
    graph=False,
    video=False,
)


# %%
algorithms = {
    # "random": random_policy,
    # "e_greedy": policies.eGreedyPolicy(0.1).choose_bandit,
    # "ucb": policies.UCBPolicy().choose_bandit,
    "ts": policies.TSPolicy().choose_bandit,
    # "ucb-B": policies.UCBPolicyB().choose_bandit,
    # "ucb-C": policies.UCBPolicyC().choose_bandit,
    # "ucb-dependent": choose_bandit,
    # "ucb-dependent-B": choose_bandit_B,
    "ucb2(0.5)": policies.UCBPolicy2(0.5, len(bandit_probs)).choose_bandit,
    "ucb_dep2(.5)": UCBPolicy2_dependent(0.5, len(bandit_probs)).choose_bandit,
    "ucb2(0.9)": policies.UCBPolicy2(0.9, len(bandit_probs)).choose_bandit,
    "ucb_dep2(.9)": UCBPolicy2_dependent(0.9, len(bandit_probs)).choose_bandit,
}

simulation(mab, algorithms, 2000, 20)

# %%
