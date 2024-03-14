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
v1 = 0.3
v2 = 0.2
v3 = 0.1

bandit_probs = [
    v1,
    v2,
    v3,
    v1 + v2,
    v1 + v3,
    v2 + v3,
    v1 + v2 + v3,
    -0.2 * v1 + 0.8 * v2 + 0.5 * v3,
]

mab = MAB(bandit_probs)

v1 = cp.Variable()
v2 = cp.Variable()
v2 = cp.Variable()
variables = [v1, v2, v3]
bandit_expressions = [
    v1,
    v2,
    v3,
    v1 + v2,
    v1 + v3,
    v2 + v3,
    v1 + v2 + v3,
    -0.2 * v1 + 0.8 * v2 + 0.5 * v3,
]


# %%
plot.plot_MAB_experiment(
    policies.UCBPolicy2_dependent(0.5, variables, bandit_expressions).choose_bandit,
    mab,
    2000,
    bandit_probs,
    "UCB2 dependent",
    video=False,
)

# %%
algorithms = {
    "random": policies.RandomPolicy().choose_bandit,
    "e_greedy": policies.eGreedyPolicy(0.1).choose_bandit,
    # "ucb": policies.UCBPolicy().choose_bandit,
    # "ts": policies.TSPolicy().choose_bandit,
    # "ucb-B": policies.UCBPolicyB().choose_bandit,
    # "ucb-C": policies.UCBPolicyC().choose_bandit,
    # "ucb-dependent": policies.UCB_Dependent(
    #     variables, bandit_expressions
    # ).choose_bandit,
    # "ucb-dependentB": policies.UCB_DependentB(
    #     variables, bandit_expressions
    # ).choose_bandit,
    "ucb2(0.5)": policies.UCBPolicy2(0.5, len(bandit_probs)).choose_bandit,
    "ucb_dep2(0.5)": policies.UCBPolicy2_dependent(
        0.5, variables, bandit_expressions
    ).choose_bandit,
    # "ucb2(0.9)": policies.UCBPolicy2(0.9, len(bandit_probs)).choose_bandit,
    # "ucb_dep2(0.9)": policies.UCBPolicy2_dependent(
    #     0.9, variables, bandit_expressions
    # ).choose_bandit,
}

simulation(mab, algorithms, 1000, 5, plot_reward=True)

# %%
