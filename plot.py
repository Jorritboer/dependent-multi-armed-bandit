# plotting inline
# %matplotlib inline

# working directory
# import os; os.chdir('/home/gdmarmerola/ts_demo')

# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from scipy.stats import beta as beta_dist
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# from tqdm import tqdm

# turning off automatic plot showing, and setting style
plt.ioff()
plt.style.use("fivethirtyeight")

bandit_colors = ["red", "green", "blue", "purple"]


# let us create a function that returns the pdf for our beta posteriors
def get_beta_pdf(alpha, beta):
    X = np.linspace(0, 1, 1000)
    return X, beta_dist(1 + alpha, 1 + beta).pdf(X)


# let us wrap a function that draws the draws and distributions of the bandit experiment
def plot_MAB_experiment(
    decision_policy, mab, N_DRAWS, bandit_probs, plot_title, video=False
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
    fig = plt.figure(figsize=(9, 5), dpi=150)

    # let us position our plots in a grid, the largest being our plays
    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=3)
    ax2 = plt.subplot2grid((5, 4), (3, 0), rowspan=2)
    ax3 = plt.subplot2grid((5, 4), (3, 1), rowspan=2)
    ax4 = plt.subplot2grid((5, 4), (3, 2), rowspan=2)
    ax5 = plt.subplot2grid((5, 4), (3, 3), rowspan=2)

    # loop generating draws
    for draw_number in range(N_DRAWS):

        # record information about this draw
        k = decision_policy(k_array, reward_array, N_BANDITS)
        reward, regret = mab.draw(k)

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
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(
        ["{}\n($\\theta = {}$)".format(i, bandit_probs[i]) for i in range(4)]
    )
    ax1.tick_params(labelsize=10)

    # titles of distribution plots
    ax2.set_title("Estimated $\\theta_0$", fontsize=10)
    ax3.set_title("Estimated $\\theta_1$", fontsize=10)
    ax4.set_title("Estimated $\\theta_2$", fontsize=10)
    ax5.set_title("Estimated $\\theta_3$", fontsize=10)

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
    dens1 = ax2.fill_between(
        posterior_anim_dict[0][0]["X"],
        0,
        posterior_anim_dict[0][0]["curve"],
        color="red",
        alpha=0.7,
    )
    dens2 = ax3.fill_between(
        posterior_anim_dict[1][0]["X"],
        0,
        posterior_anim_dict[1][0]["curve"],
        color="green",
        alpha=0.7,
    )
    dens3 = ax4.fill_between(
        posterior_anim_dict[2][0]["X"],
        0,
        posterior_anim_dict[2][0]["curve"],
        color="blue",
        alpha=0.7,
    )
    dens4 = ax5.fill_between(
        posterior_anim_dict[3][0]["X"],
        0,
        posterior_anim_dict[3][0]["curve"],
        color="purple",
        alpha=0.7,
    )

    # titles
    # plt.title('Random draws from the row of slot machines (MAB)', fontsize=10)
    # plt.xlabel('Round', fontsize=10); plt.ylabel('Bandit', fontsize=10);

    # function for updating
    def animate(i):

        # clearing axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()

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
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(
            ["{}\n($\\theta = {}$)".format(i, bandit_probs[i]) for i in range(4)]
        )
        ax1.tick_params(labelsize=10)

        # updating distributions
        dens1 = ax2.fill_between(
            posterior_anim_dict[0][i]["X"],
            0,
            posterior_anim_dict[0][i]["curve"],
            color="red",
            alpha=0.7,
        )
        dens2 = ax3.fill_between(
            posterior_anim_dict[1][i]["X"],
            0,
            posterior_anim_dict[1][i]["curve"],
            color="green",
            alpha=0.7,
        )
        dens3 = ax4.fill_between(
            posterior_anim_dict[2][i]["X"],
            0,
            posterior_anim_dict[2][i]["curve"],
            color="blue",
            alpha=0.7,
        )
        dens4 = ax5.fill_between(
            posterior_anim_dict[3][i]["X"],
            0,
            posterior_anim_dict[3][i]["curve"],
            color="purple",
            alpha=0.7,
        )

        # titles of distribution plots
        ax2.set_title("Estimated $\\theta_0$", fontsize=10)
        ax3.set_title("Estimated $\\theta_1$", fontsize=10)
        ax4.set_title("Estimated $\\theta_2$", fontsize=10)
        ax5.set_title("Estimated $\\theta_3$", fontsize=10)

        # do not need to return
        return ()

    # function for creating animation
    anim = FuncAnimation(fig, animate, frames=N_DRAWS, interval=100, blit=True)

    # fixing the layout
    fig.tight_layout()

    if video:
        return HTML(anim.to_html5_video())
    else:
        animate(N_DRAWS - 1)
