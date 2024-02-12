import numpy as np
import plot
import policies


class MAB:

    def __init__(self, bandit_probs):
        self.bandit_probs = bandit_probs

    def draw(self, k):
        return (
            np.random.binomial(1, self.bandit_probs[k]),
            np.max(self.bandit_probs) - self.bandit_probs[k],
        )


bandit_probs = [0.35, 0.40, 0.80, 0.25]
mab = MAB(bandit_probs)


def random_policy(k_array, reward_array, n_bandits):
    return np.random.choice(range(n_bandits), 1)[0]


plot.plot_MAB_experiment(
    policies.UCBPolicy().choose_bandit, mab, 100, bandit_probs, "hi", True
)
