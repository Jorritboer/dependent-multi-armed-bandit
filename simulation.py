import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def simulation(
    mab,
    policies,
    n_rounds=1000,
    n_simulations=1000,
    plot_reward=False,
    plot_uncertainty=False,
):
    n_bandits = len(mab.bandit_probs)
    n_policies = len(policies.keys())

    # dict storing results for each algorithm and simulation
    results_dict = {
        policy: {
            "k_array": np.zeros((n_bandits, n_rounds)),
            "reward_array": np.zeros((n_bandits, n_rounds)),
            "regret_array": np.zeros((n_simulations, n_rounds)),
        }
        for policy in policies.keys()
    }

    # loop for each algorithm
    for key, decision_policy in policies.items():

        # printing progress
        print(key, decision_policy)

        # loop for each simulation
        for simulation in tqdm(range(n_simulations)):

            # numpy arrays for accumulating draws, bandit choices and rewards, more efficient calculations
            k_array = np.zeros((n_bandits, n_rounds))
            reward_array = np.zeros((n_bandits, n_rounds))
            regret_array = np.zeros((1, n_rounds))[0]

            # loop for each round
            for round_id in range(n_rounds):

                # choosing arm nad pulling it
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

    # closing all past figures
    plt.close("all")

    # opening figure to plot regret
    plt.figure(figsize=(10, 3), dpi=150)

    # NUM_COLORS = len(policies.items())
    # cm = plt.get_cmap("inferno")
    # plt.gca().set_prop_cycle(
    #     color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    # )

    # loop for each decision policy
    for policy in policies.keys():

        # plotting data
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        if plot_reward:
            plt.plot(
                np.cumsum(
                    np.sum(results_dict[policy]["reward_array"], axis=0) / n_simulations
                ),
                "--",
                label=f"{policy}-reward",
                linewidth=1.5,
                color=color,
            )
        cumulative_sum = np.cumsum(results_dict[policy]["regret_array"], axis=1)

        std = np.std(cumulative_sum, axis=0)
        cumulative_sum = np.sum(cumulative_sum, axis=0) / n_simulations

        plt.plot(
            cumulative_sum,
            label=policy,
            linewidth=1.5,
            color=color,
        )
        if plot_uncertainty:
            plt.fill_between(
                range(n_rounds),
                cumulative_sum - std,
                cumulative_sum + std,
                color=color,
                alpha=0.2,
            )
            plt.plot(
                cumulative_sum + std,
                ",",
                color=color,
            )
            plt.plot(
                cumulative_sum - std,
                ",",
                color=color,
            )

    # adding title
    plt.title(
        "Comparison of cumulative regret for each method in {} simulations".format(
            n_simulations
        ),
        fontsize=10,
    )

    # adding legend
    plt.legend(fontsize=8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # showing plot
    plt.show()

    # closing all past figures
    plt.close("all")

    # opening figure to plot regret
    plt.figure(figsize=(10, 3), dpi=150)

    # colors for each bandit
    bandit_colors = ["red", "green", "blue", "purple"] * 4

    # loop for each decision policy
    for i, policy in enumerate(policies.keys()):

        # subplots
        plt.subplot(1, n_policies + 1, i + 1)

        # loop for each arm
        for arm in range(n_bandits):

            # plotting data
            plt.plot(
                results_dict[policy]["k_array"][arm] / n_simulations,
                label="Bandit {}".format(arm),
                linewidth=1.5,
                color=bandit_colors[arm],
            )

            # adding title
            plt.title(policy, fontsize=8)

            # adding legend
            plt.legend(fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.ylim([-0.1, 1.1])

    plt.show()
