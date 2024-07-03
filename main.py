# %%
import numpy as np
import plot
from policies import *
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


def lin_decaying(start, end, steps):
    return lambda n: start - (n / steps) * (start - end) if n < steps else end


# %%
v1 = 0.1
v2 = 0.2
v3 = 0.3

bandit_probs = [v1 + v2, v1 + v3, v2 + v3]
mab = MAB(bandit_probs)

v1 = cp.Variable()
v2 = cp.Variable()
v3 = cp.Variable()
variables = [v1, v2, v3]
bandit_expressions = [v1 + v2, v1 + v3, v2 + v3]
# or list of np arrays?
bandit_expressions_matrix = np.array(
    [
        [[1], [1], [0]],
        [[1], [0], [1]],
        [[0], [1], [1]],
    ]
)


# %%
plot.plot_MAB_experiment(
    LinUCB(1, bandit_expressions_matrix).choose_bandit,
    mab,
    200,
    bandit_probs,
    "LinUCB",
    video=False,
    graph=False,
)


# %%

algorithms = {
    # "random": RandomPolicy().choose_bandit,
    # "e_greedy": eGreedyPolicy(0.1).choose_bandit,
    # "e_greedy_decaying": eGreedyPolicyDecaying(
    #     lin_decaying(0.75, 0.005, 200)
    # ).choose_bandit,
    # "e_greedy_decaying_ucb_tuned_dep": eGreedyPolicyDecayingUCBTunedDependent(
    #     variables, bandit_expressions, lin_decaying(0.75, 0, 200)
    # ).choose_bandit,
    # "DerivativeExplore": DerivativeExplorePolicy(
    #     variables, bandit_expressions, lin_decaying(0.75, 0, 200)
    # ).choose_bandit,
    # "ucb": UCBPolicy().choose_bandit,
    # "ts": TSPolicy().choose_bandit,
    # "ucb-B": UCBPolicyB().choose_bandit,
    # "ucb-C": UCBPolicyC().choose_bandit,
    # "ucb-dependent": UCBDependent(variables, bandit_expressions).choose_bandit,
    # "ucb-dependentB": UCBDependentB(variables, bandit_expressions).choose_bandit,
    # "ucb2(0.5)": UCBPolicy2(0.5, len(bandit_probs)).choose_bandit,
    "ucb2_dep(0.5)": UCBPolicy2Dependent(
        0.5, variables, bandit_expressions
    ).choose_bandit,
    # "ucb-tuned": UCBPolicyTuned().choose_bandit,
    # "ucb-tuned_dep": UCBPolicyTunedDependent(
    #     variables, bandit_expressions
    # ).choose_bandit,
    "LinUCB": LinUCB(1, bandit_expressions_matrix).choose_bandit,
}

simulation(mab, algorithms, 2000, 10, plot_reward=False)

# %%
