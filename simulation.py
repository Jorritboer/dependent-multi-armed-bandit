import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def simulation(
    mab,
    policies,
    n_rounds=1000,
    n_simulations=1000,
    uncertainty=None,
):
    n_bandits = len(mab.bandit_probs)

    # we save the initial probs, so we can add uncertainty to this
    base_probs = mab.bandit_probs[:]

    # dict storing results for each algorithm and simulation
    results_dict = {
        policy: {
            "k_array": np.zeros((n_bandits, n_rounds)),
            "reward_array": np.zeros((n_bandits, n_rounds)),
            "regret_array": np.zeros((n_simulations, n_rounds)),
        }
        for policy in policies.keys()
    }

    # loop for each simulation
    for simulation in tqdm(range(n_simulations)):

        if uncertainty:
            for i in range(n_bandits):
                e = np.random.uniform(-uncertainty, uncertainty)
                mab.bandit_probs[i] = max(min(base_probs[i] + e, 1), 0)

        # loop for each algorithm
        for key, decision_policy in policies.items():
            # numpy arrays for accumulating draws, bandit choices and rewards, more efficient calculations
            k_array = np.zeros((n_bandits, n_rounds))
            reward_array = np.zeros((n_bandits, n_rounds))
            regret_array = np.zeros((1, n_rounds))[0]

            # loop for each round
            for round_id in tqdm(range(n_rounds), desc=key, leave=False):

                # choosing arm and pulling it
                k = decision_policy(k_array, reward_array, n_bandits)
                reward, regret = mab.draw(k)

                # record information about this draw
                k_array[k, round_id] = 1
                reward_array[k, round_id] = reward
                regret_array[round_id] = regret

            # results for the simulation
            results_dict[key]["k_array"] += k_array
            results_dict[key]["reward_array"] += reward_array
            results_dict[key]["regret_array"][simulation] = regret_array

    return results_dict
