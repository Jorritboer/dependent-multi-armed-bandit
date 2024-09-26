# %%
import numpy as np
from policies import *
import cvxpy as cp
from simulation import simulation
from random_bandits import random_bandits
import pandas as pd
from tqdm import tqdm


class MAB:

    def __init__(self, bandit_probs):
        self.bandit_probs = bandit_probs

    def draw(self, k):
        return (
            np.random.binomial(1, self.bandit_probs[k]),
            np.max(self.bandit_probs) - self.bandit_probs[k],
        )


# %%
theta_length = 3
for n in tqdm([3, 10, 30], desc=f"nr_bandits", leave=False):
    for round in tqdm(range(5), desc="round", leave=False):
        theta, bandit_matrix = random_bandits(theta_length, n)

        bandit_probs = np.dot(bandit_matrix, theta)
        mab = MAB(bandit_probs)

        # width of matrix determines nr of variables
        variables = [cp.Variable() for _ in range(len(bandit_matrix[0]))]
        bandit_expressions = np.dot(bandit_matrix, variables)

        # create column vectors
        column_vecs = [v.reshape(-1, 1) for v in bandit_matrix]

        algorithms = {
            "Thompson Sampling": TSPolicy().choose_bandit,
            "DependentUCB3.5": UCBDependentB(
                variables, bandit_expressions, alpha=3.5
            ).choose_bandit,
            "DependentUCB5": UCBDependentB(
                variables, bandit_expressions, alpha=5
            ).choose_bandit,
            "DependentUCB6.5": UCBDependentB(
                variables, bandit_expressions, alpha=6.5
            ).choose_bandit,
            "LinUCB0.5": LinUCB(1, column_vecs).choose_bandit,
            "LinUCB1": LinUCB(1, column_vecs).choose_bandit,
            "LinUCB2": LinUCB(1, column_vecs).choose_bandit,
        }

        results = simulation(mab, algorithms, 1000, 5)

        data_regret = {}
        data_comptime = {}
        for policy in algorithms.keys():
            data_regret[policy] = np.sum(results[policy]["regret_array"], axis=1)
            data_comptime[policy] = results[policy]["computation_time"]
        df_regret = pd.DataFrame(data_regret)
        df_comptime = pd.DataFrame(data_comptime)
        df_regret.to_csv(f"results/regret/results{theta_length}_{n}_{round}_regret.csv")
        df_comptime.to_csv(
            f"results/comp_time/results{theta_length}_{n}_{round}_comptime.csv"
        )

# %%
