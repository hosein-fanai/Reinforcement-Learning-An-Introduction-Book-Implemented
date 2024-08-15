import numpy as np


class KArmedBanditProblem:

    def __init__(self, k: int, nonstationarity: bool = False):
        self.k = k
        self.nonstationarity = nonstationarity

    def reset(self, mean_qs: float = 0., equal_all_qs: bool = False):
        if equal_all_qs:
            q_star = np.random.normal(loc=mean_qs, scale=1.)
            self.q_stars = [q_star for _ in range(self.k)]
        else:
            self.q_stars = [np.random.normal(loc=mean_qs, scale=1.) for _ in range(self.k)]

    def step(self, action: int):
        assert 0 <= action < self.k

        q_star_action = self.q_stars[action]

        if self.nonstationarity:
            for i in range(self.k):
                random_walk = np.random.normal(loc=0, scale=0.01)
                self.q_stars[i] += random_walk

        return np.random.normal(loc=q_star_action, scale=1.)