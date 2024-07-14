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
bandit_matrix = np.array(
    [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
)
values = [0.1, 0.2, 0.3]

bandit_probs = np.dot(bandit_matrix, values)
mab = MAB(bandit_probs)

# width of matrix determines nr of variables
variables = [cp.Variable() for _ in range(len(bandit_matrix[0]))]
bandit_expressions = np.dot(bandit_matrix, variables)

# create column vectors
column_vecs = [v.reshape(-1, 1) for v in bandit_matrix]


# %%
plot.plot_MAB_experiment(
    LinUCB(1, column_vecs).choose_bandit,
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
    "ts": TSPolicy().choose_bandit,
    # "ucb-B": UCBPolicyB().choose_bandit,
    # "ucb-C": UCBPolicyC().choose_bandit,
    # "ucb-dependent": UCBDependent(variables, bandit_expressions).choose_bandit,
    # "ucb-dependentB": UCBDependentB(variables, bandit_expressions).choose_bandit,
    # "ucb2(0.5)": UCBPolicy2(0.5, len(bandit_probs)).choose_bandit,
    "ucb2_dep(0.5)": UCBPolicy2Dependent(
        0.5, variables, bandit_expressions
    ).choose_bandit,
    # "ucb-tuned": UCBPolicyTuned().choose_bandit,
    "ucb-tuned_dep": UCBPolicyTunedDependent(
        variables, bandit_expressions
    ).choose_bandit,
    # "ucb-tuned_dep2": UCBPolicyTunedDependent2(
    #     variables, bandit_expressions
    # ).choose_bandit,
    "LinUCB": LinUCB(1, column_vecs).choose_bandit,
}

simulation(mab, algorithms, 2000, 5, plot_reward=False)

# %%
