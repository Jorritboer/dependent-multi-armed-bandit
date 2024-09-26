import numpy as np


def random_bandits(nr_parameters, nr_bandits, max_prob=0.65):
    theta = [np.random.uniform(0, 1) for _ in range(nr_parameters)]

    bandits = []
    for _ in range(nr_bandits):
        x = [np.random.uniform(-1, 1) for _ in range(nr_parameters)]

        p = np.dot(x, theta)
        # if not (0<= p <= 1) try again
        while p > max_prob or p < 0:
            # # with a small proability allow p>0.65 anyway
            # if p > 0.65 and p < 1 and np.random.uniform(0, 1) < 0.05:
            #     break
            x = [np.random.uniform(-1, 1) for _ in range(nr_parameters)]
            p = np.dot(x, theta)
        bandits.append(x)

    return theta, bandits
