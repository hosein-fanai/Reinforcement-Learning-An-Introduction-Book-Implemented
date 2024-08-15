import numpy as np

from k_armed_bandit_problem import KArmedBanditProblem


class KArmedBanditLearner:
    EPS = np.array(1e-7, dtype=np.float64)

    def __init__(self):
        pass

    def _softmax(self, arr: np.ndarray):
        exp_arr = np.exp(arr + KArmedBanditLearner.EPS)
        softmax = exp_arr / np.sum(exp_arr)

        return softmax.astype(np.float64)

    def _randomized_argmax(self, arr: np.ndarray):
        max_val_indices = np.where(arr==np.max(arr))[0]
        max_val_index = np.random.choice(max_val_indices)

        return max_val_index

    def reset(self, bandit_problem: KArmedBanditProblem, 
            alpha: float | None = None, optimistic_init: float | None = None, 
            gradient_bandit_algo: bool = False, reward_baseline: bool = False):
        self.bandit_problem = bandit_problem
        self.alpha = (lambda action: np.array(alpha, dtype=np.float64)) if alpha is not None else self.n_inversed

        k = bandit_problem.k

        self.H = np.zeros((k,), dtype=np.float64) if gradient_bandit_algo else None
        self.reward_baseline = reward_baseline
        if reward_baseline:
            self.rewards_avg = 0.
            self.t = 0

        init_val = optimistic_init if optimistic_init is not None else 0
        self.Q_actions = np.zeros((k,), dtype=np.float64) + init_val
        self.num_actions = np.zeros((k,), dtype=np.float64)

    def n_inversed(self, action: int):
        return 1 / self.num_actions[action]

    def update_Q(self, action: int, reward: float):
        self.num_actions[action] += 1

        self.Q_actions[action] += self.alpha(action) * (reward - self.Q_actions[action])

    def update_rewards_avg(self, reward: float):
        self.t += 1
        self.rewards_avg += 1 / self.t * (reward - self.rewards_avg)

        return self.rewards_avg

    def act_greedy(self):
        greedy_action = np.argmax(self.Q_actions)
        # greedy_action = self._randomized_argmax(self.Q_actions)
        reward = self.bandit_problem.step(greedy_action)

        self.update_Q(greedy_action, reward)

        return greedy_action, reward

    def act_random(self):
        random_action = np.random.randint(self.bandit_problem.k)
        reward = self.bandit_problem.step(random_action)

        self.update_Q(random_action, reward)

        return random_action, reward

    def act_epsilon_greedy(self, epsilon: float):
        if epsilon <= np.random.rand():
            return self.act_greedy()
        else:
            return self.act_random()
    
    def act_ucb(self, c: float):
        t = sum(self.num_actions) + 1
        ucb_values = self.Q_actions + c * np.sqrt(np.log(t) / (self.num_actions + KArmedBanditLearner.EPS))

        ucb_action = np.argmax(ucb_values)
        reward = self.bandit_problem.step(ucb_action)

        self.update_Q(ucb_action, reward)

        return ucb_action, reward
    
    def act_gradient_bandit(self):
        assert self.H is not None

        pi_action = self._softmax(self.H)
        grad_bandit_action = np.argmax(np.random.multinomial(1, pvals=pi_action))
        reward = self.bandit_problem.step(grad_bandit_action)

        not_grad_bandit_actions = np.ones(self.H.shape, dtype=np.bool8)
        not_grad_bandit_actions[grad_bandit_action] = False

        rew_baseline = self.update_rewards_avg(reward) if self.reward_baseline else 0

        self.H[grad_bandit_action] += self.alpha(grad_bandit_action) * (reward - rew_baseline) * (1 - pi_action[grad_bandit_action])
        self.H[not_grad_bandit_actions] -= self.alpha(not_grad_bandit_actions) * (reward - rew_baseline) * pi_action[not_grad_bandit_actions]

        return grad_bandit_action, reward