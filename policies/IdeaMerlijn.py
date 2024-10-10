import numpy as np


#
class IdeaMerlijn:

    # initializing
    def __init__(self, bandit_matrix, alpha=1):
        self.bandit_matrix = bandit_matrix
        self.alpha = alpha
        self.weight_per_var = [np.sum(var) for var in bandit_matrix.T]

    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):
        # sucesses and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)

        # ratio of sucesses vs total
        success_ratio = success_count / total_count

        print(success_ratio)

        # estimating theta
        theta = np.dot(success_ratio, self.bandit_matrix)

        # replace nan with 0
        theta = np.nan_to_num(theta)

        # divide by weight per var
        theta = theta / self.weight_per_var

        contribution_weight = [0] * len(theta)
        for j in range(len(theta)):
            w = 0
            for i in range(n_bandits):
                tot_weight = 0
                for jj in range(len(theta)):
                    tot_weight += self.bandit_matrix[i][jj] * theta[jj]
                w += ((self.bandit_matrix[i][j] * theta[j]) / tot_weight) * total_count[
                    i
                ]
            contribution_weight[j] = w / sum(total_count)

        # compute bounds
        bounds = np.nan_to_num(
            (1 / self.alpha)
            * np.sqrt(
                2
                * np.log(np.sum(total_count))
                / contribution_weight
                * np.sum(total_count)
            )
        )

        print(
            f"theta {theta}, contribution_weights {contribution_weight}, estimations {bounds}, t+b {theta+bounds}"
        )

        theta = theta + bounds

        est = np.dot(self.bandit_matrix, theta)

        # returning best greedy action
        return np.random.choice(np.flatnonzero(est == np.max(est)))
