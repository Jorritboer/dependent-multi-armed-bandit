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
v1 = 0.5
v2 = 0.2

bandit_probs = [v1, v1 + v2, v2, 0.7 * v1 + v2]
mab = MAB(bandit_probs)

v1 = cp.Variable()
v2 = cp.Variable()
variables = [v1, v2]
bandit_expressions = [v1, v1 + v2, v2, 0.7 * v1 + v2]


# %%
plot.plot_MAB_experiment(
    DerivativeExplorePolicy(variables, bandit_expressions, lin_decaying(0.75, 0, 50)),
    mab,
    200,
    bandit_probs,
    "Derivative Explore Policy",
    video=False,
    graph=True,
)


# %%

algorithms = {
    # "random": RandomPolicy().choose_bandit,
    # "e_greedy": eGreedyPolicy(0.1).choose_bandit,
    # "e_greedy_decaying": eGreedyPolicyDecaying(
    #     lin_decaying(0.75, 0.005, 200)
    # ).choose_bandit,
    "e_greedy_decaying_ucb_tuned_dep": eGreedyPolicyDecayingUCBTunedDependent(
        variables, bandit_expressions, lin_decaying(0.75, 0, 200)
    ).choose_bandit,
    "DerivativeExplore": DerivativeExplorePolicy(
        variables, bandit_expressions, lin_decaying(0.75, 0, 200)
    ).choose_bandit,
    # "ucb": UCBPolicy().choose_bandit,
    "ts": TSPolicy().choose_bandit,
    # "ucb-B": UCBPolicyB().choose_bandit,
    # "ucb-C": UCBPolicyC().choose_bandit,
    # "ucb-dependent": UCBDependent(variables, bandit_expressions).choose_bandit,
    # "ucb-dependentB": UCBDependentB(variables, bandit_expressions).choose_bandit,
    # "ucb2(0.5)": UCBPolicy2(0.5, len(bandit_probs)).choose_bandit,
    # "ucb2_dep(0.5)": UCBPolicy2Dependent(
    #     0.5, variables, bandit_expressions
    # ).choose_bandit,
    # "ucb-tuned": UCBPolicyTuned().choose_bandit,
    "ucb-tuned_dep": UCBPolicyTunedDependent(
        variables, bandit_expressions
    ).choose_bandit,
}

simulation(mab, algorithms, 3000, 6, plot_reward=False, plot_uncertainty=True)

# %%
