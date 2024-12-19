import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from scipy.stats import beta as beta_dist
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.style.use("fivethirtyeight")

bandit_colors = ["red", "green", "blue", "purple"]


# let us create a function that returns the pdf for our beta posteriors
def get_beta_pdf(alpha, beta):
    X = np.linspace(0, 1, 1000)
    return X, beta_dist(1 + alpha, 1 + beta).pdf(X)


def plot_graph(
    graph_ax, k_array, reward_array, k_list, column_vecs, theta_values, alpha
):
    success_count = reward_array.sum(axis=1)
    total_count = k_array.sum(axis=1)

    round = total_count.sum()
    if round < 2:
        return

    # ratio of sucesses vs total
    success_ratio = success_count / total_count

    # computing square root term
    # sqrt_term = np.sqrt(2 * np.log(np.sum(total_count)) / total_count)
    vars = np.sum(reward_array**2, axis=1) / total_count - (success_ratio**2)
    V = vars + np.sqrt(2 * np.log(np.sum(total_count)) / total_count)

    # computing square root term
    sqrt_term = np.sqrt(
        (np.log(np.sum(total_count)) / total_count) * np.minimum(0.25, V)
    )

    D = np.array([column_vecs[k].T[0] for k in k_list])
    A = np.dot(D.T, D) + np.identity(2)
    rewards = np.array(reward_array.sum(axis=0))
    b = np.dot(rewards, D)

    A_inv = np.linalg.inv(A)
    theta_t = np.dot(A_inv, b)
    graph_ax.scatter(theta_t[0], theta_t[1], color="black")

    graph_ax.scatter(theta_values[0], theta_values[1], color="magenta")

    x = np.linspace(-0.5, 1.5, 65)
    y = np.linspace(-0.5, 1.5, 65)
    x, y = np.meshgrid(x, y)
    a = A[0][0]
    b = A[0][1]
    c = A[1][0]
    d = A[1][1]
    graph_ax.contour(
        x,
        y,
        (
            a * (x - theta_t[0]) ** 2
            + d * (y - theta_t[1]) ** 2
            + (b + c) * (x - theta_t[0]) * (y - theta_t[1])
        ),
        [alpha],
        colors="black",
        linewidths=1,
    )

    graph_ax.set_xlim([-0.5, 1.5])
    graph_ax.set_ylim([-0.5, 1.5])
    graph_ax.plot([0, 1], [0, 0], linewidth=1, color="black")
    graph_ax.plot([1, 1], [0, 1], linewidth=1, color="black")
    graph_ax.plot([1, 0], [1, 1], linewidth=1, color="black")
    graph_ax.plot([0, 0], [1, 0], linewidth=1, color="black")

    graph_ax.axhline(-0.5, color="black")
    graph_ax.axvline(-0.5, color="black")
    graph_ax.set_xlabel("$\\theta_1$")
    graph_ax.set_ylabel("$\\theta_2$")

    x = np.linspace(-0.5, 1.5, 10)
    y = np.linspace(-0.5, 1.5, 10)
    x, y = np.meshgrid(x, y)
    for bandit in range(len(column_vecs)):
        if int(total_count[bandit]) == 0:
            continue

        vector = column_vecs[bandit]
        graph_ax.contour(
            x,
            y,
            (vector[0][0] * x + vector[1][0] * y),
            [
                success_ratio[bandit] - sqrt_term[bandit],
                success_ratio[bandit],
                success_ratio[bandit] + sqrt_term[bandit],
            ],
            colors=bandit_colors[bandit],
            linewidths=2,
            linestyles=["-", "--", "-"],
        )


# let us wrap a function that draws the draws and distributions of the bandit experiment
def plot_MAB_experiment(
    decision_policy,
    mab,
    N_DRAWS,
    bandit_probs,
    plot_title,
    graph=True,
    video=False,
    column_vecs=None,
    theta_values=None,
    alpha=None,
):

    # number of bandits
    N_BANDITS = len(bandit_probs)

    # numpy arrays for accumulating draws, bandit choices and rewards, more efficient calculations
    k_array = np.zeros((N_BANDITS, N_DRAWS))
    reward_array = np.zeros((N_BANDITS, N_DRAWS))

    # lists for accumulating draws, bandit choices and rewards
    k_list = []
    reward_list = []

    # animation dict for the posteriors
    posterior_anim_dict = {i: [] for i in range(N_BANDITS)}

    # opening figure
    if graph:
        fig = plt.figure(figsize=(15, 5), dpi=150)
    else:
        fig = plt.figure(figsize=(9, 5), dpi=150)

    # let us position our plots in a grid, the largest being our plays
    ax1 = plt.subplot2grid((5, 6), (0, 0), colspan=4, rowspan=3)
    ax2 = plt.subplot2grid((5, 6), (3, 0), rowspan=2)
    ax3 = plt.subplot2grid((5, 6), (3, 1), rowspan=2)
    axes = [ax2, ax3]
    if N_BANDITS > 2:
        ax4 = plt.subplot2grid((5, 6), (3, 2), rowspan=2)
        axes.append(ax4)
    if N_BANDITS > 3:
        ax5 = plt.subplot2grid((5, 6), (3, 3), rowspan=2)
        axes.append(ax5)
    if graph:
        graph_ax = plt.subplot2grid((5, 6), (0, 4), colspan=4, rowspan=5)

    total_regret = 0
    # loop generating draws
    for draw_number in range(N_DRAWS):

        # record information about this draw
        k = decision_policy(k_array, reward_array, N_BANDITS)
        reward, regret = mab.draw(k)
        total_regret += regret

        # record information about this draw
        k_list.append(k)
        reward_list.append(reward)
        k_array[k, draw_number] = 1
        reward_array[k, draw_number] = reward

        # sucesses and failures for our beta distribution
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count

        # calculating pdfs for each bandit
        for bandit_id in range(N_BANDITS):

            # pdf
            X, curve = get_beta_pdf(success_count[bandit_id], failure_count[bandit_id])

            # appending to posterior animation dict
            posterior_anim_dict[bandit_id].append({"X": X, "curve": curve})

        # getting list of colors that tells us the bandit
        color_list = [bandit_colors[k] for k in k_list]

        # getting list of facecolors that tells us the reward
        facecolor_list = [
            ["none", bandit_colors[k_list[i]]][r] for i, r in enumerate(reward_list)
        ]

    # fixing properties of the plots
    ax1.set(xlim=(-1, N_DRAWS), ylim=(-0.5, N_BANDITS - 0.5))
    ax1.set_title(plot_title, fontsize=10)
    ax1.set_xlabel("Round", fontsize=10)
    ax1.set_ylabel("Bandit", fontsize=10)
    ax1.set_yticks(list(range(N_BANDITS)))
    ax1.set_yticklabels([f"\n($p_{i} = {bandit_probs[i]}$)" for i in range(N_BANDITS)])
    ax1.tick_params(labelsize=10)

    # titles of distribution plots
    for i in range(N_BANDITS):
        axes[i].set_title(f"Estimated $p_{i+1}$", fontsize=10)

    # initializing with first data
    scatter = ax1.scatter(
        y=[k_list[0]],
        x=[list(range(N_DRAWS))[0]],
        color=[color_list[0]],
        linestyle="-",
        marker="o",
        s=30,
        facecolor=[facecolor_list[0]],
    )
    for j in range(N_BANDITS):
        axes[j].fill_between(
            posterior_anim_dict[j][0]["X"],
            0,
            posterior_anim_dict[j][0]["curve"],
            color="red",
            alpha=0.7,
        )

    if graph:
        plot_graph(
            graph_ax, k_array, reward_array, k_list, column_vecs, theta_values, alpha
        )

    # titles
    # plt.title('Random draws from the row of slot machines (MAB)', fontsize=10)
    # plt.xlabel('Round', fontsize=10); plt.ylabel('Bandit', fontsize=10);

    # function for updating
    def animate(i):

        # clearing axes
        ax1.clear()

        for j in range(N_BANDITS):
            axes[j].clear()
        if graph:
            graph_ax.clear()
            plot_graph(
                graph_ax,
                np.array([k_arr[:i] for k_arr in k_array]),
                np.array([reward_arr[:i] for reward_arr in reward_array]),
                k_list[:i],
                column_vecs,
                theta_values,
                alpha,
            )

        # updating game rounds
        scatter = ax1.scatter(
            y=k_list[:i],
            x=list(range(N_DRAWS))[:i],
            color=color_list[:i],
            linestyle="-",
            marker="o",
            s=30,
            facecolor=facecolor_list[:i],
        )

        # fixing properties of the plot
        ax1.set(xlim=(-1, N_DRAWS), ylim=(-0.5, N_BANDITS - 0.5))
        ax1.set_title(plot_title, fontsize=10)
        ax1.set_xlabel("Round", fontsize=10)
        ax1.set_ylabel("Bandit", fontsize=10)
        ax1.set_yticks(list(range(N_BANDITS)))
        ax1.set_yticklabels([f"1\n$p_{j}={bandit_probs[j]}$" for j in range(N_BANDITS)])
        ax1.tick_params(labelsize=10)

        # updating distributions
        for j in range(N_BANDITS):
            axes[j].fill_between(
                posterior_anim_dict[j][i]["X"],
                0,
                posterior_anim_dict[j][i]["curve"],
                color=bandit_colors[j],
                alpha=0.7,
            )
            axes[j].set_title(f"Estimated $p_{j+1}$", fontsize=10)

        # do not need to return
        return ()

    # function for creating animation
    anim = FuncAnimation(fig, animate, frames=N_DRAWS, interval=100, blit=True)

    # fixing the layout
    fig.tight_layout()

    print(f"Total regret: {total_regret}")

    if video:
        return HTML(anim.to_html5_video())
    else:
        animate(N_DRAWS - 1)
