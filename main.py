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
bandit_probs = [v1, v1 + v2]

mab = MAB(bandit_probs)

v1 = cp.Variable()
v2 = cp.Variable()
variables = [v1, v2]
bandit_expressions = [v1, v1 + v2]


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
    # "random": random_policy,
    # "e_greedy": policies.eGreedyPolicy(0.1).choose_bandit,
    # "ucb": policies.UCBPolicy().choose_bandit,
    "ts": policies.TSPolicy().choose_bandit,
    # "ucb-B": policies.UCBPolicyB().choose_bandit,
    # "ucb-C": policies.UCBPolicyC().choose_bandit,
    # "ucb-dependent": policies.UCB_Dependent(
    #     variables, bandit_expressions
    # ).choose_bandit,
    # "ucb-dependentB": policies.UCB_DependentB(
    #     variables, bandit_expressions
    # ).choose_bandit,
    "ucb2(0.5)": policies.UCBPolicy2(0.5, len(bandit_probs)).choose_bandit,
    "ucb_dep2(.5)": policies.UCBPolicy2_dependent(
        0.5, variables, bandit_expressions
    ).choose_bandit,
}

simulation(mab, algorithms, 1000, 2)

# %%
