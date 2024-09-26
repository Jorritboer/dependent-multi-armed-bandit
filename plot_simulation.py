import numpy as np
import matplotlib.pyplot as plt


def plot_simulation(results_dict, n_bandits, plot_reward=False):
    policies = list(results_dict.keys())
    n_simulations = len(results_dict[policies[0]]["regret_array"])
    n_rounds = len(results_dict[policies[0]]["regret_array"][0])
    n_policies = len(policies)

    # closing all past figures
    plt.close("all")

    def plot_regret(plot_uncertainty):
        # opening figure to plot regret
        plt.figure(figsize=(10, 3), dpi=150)

        # NUM_COLORS = len(policies.items())
        # cm = plt.get_cmap("inferno")
        # plt.gca().set_prop_cycle(
        #     color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
        # )

        # loop for each decision policy
        for policy in policies:

            # plotting data
            color = next(plt.gca()._get_lines.prop_cycler)["color"]
            if plot_reward:
                plt.plot(
                    np.cumsum(
                        np.sum(results_dict[policy]["reward_array"], axis=0)
                        / n_simulations
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

    # plot regret without uncertainty
    plot_regret(plot_uncertainty=False)
    plt.close("all")
    # plot regret with uncertainty
    plot_regret(plot_uncertainty=True)
    plt.close("all")

    # opening figure to plot regret
    plt.figure(figsize=(10, 3), dpi=150)

    # colors for each bandit
    bandit_colors = ["red", "green", "blue", "purple"] * 4

    # loop for each decision policy
    for i, policy in enumerate(policies):

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
