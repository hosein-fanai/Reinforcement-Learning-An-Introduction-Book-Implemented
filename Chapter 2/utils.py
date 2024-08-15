import numpy as np

from matplotlib import pyplot as plt

from k_armed_bandit_problem import KArmedBanditProblem
from k_armed_bandit_learner import KArmedBanditLearner


def plot_problem(problem, bandit_sample_step_num: int):
        action_rewards_list = [[problem.step(action) for _ in range(bandit_sample_step_num)] 
                                for action in range(problem.k)]

        _, ax = plt.subplots()

        ax.violinplot(action_rewards_list, showmeans=True, showextrema=False)
        ax.plot([0, 11], [0, 0], linestyle="--")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(1, 10)

        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        ax.set_xticks(list(range(1, problem.k+1)))
        ax.set_xlim([0.5, 10.5])

        ax.set_xlabel("Action")
        ax.set_ylabel("Reward\ndistribution", rotation=0, labelpad=30)

        means = [sum(rewards)/len(rewards) for rewards in action_rewards_list]
        for i, mean_val in enumerate(means):
            ax.text(i+1.25, mean_val, f"q*({i+1})", verticalalignment="center", fontsize=7)

        plt.show()


def create_problems(bandit_problem_num: int = 2_000, k: int = 10, mean_qs: float = 0., 
                    nonstationarity: bool = False, equal_all_qs: bool = False):
    problems = []

    for _ in range(bandit_problem_num):
        problem = KArmedBanditProblem(k=k, nonstationarity=nonstationarity)
        problem.reset(mean_qs=mean_qs, equal_all_qs=equal_all_qs)

        problems.append(problem)

    return problems


def learn_problems(problems: list[KArmedBanditProblem], bandit_learning_step_num: int = 1_000, 
            epsilon=None, c=None, alpha=None, optimistic_init=None, 
            grad_bandit_algo=False, reward_baseline=False):
    assert epsilon is not None or c is not None or grad_bandit_algo

    agent = KArmedBanditLearner()

    rewards_list = np.zeros((len(problems), bandit_learning_step_num), dtype=np.float16)
    optimal_action_rates_list = np.zeros((len(problems), bandit_learning_step_num), dtype=np.bool8)
    for i, problem in enumerate(problems):
        print(f"\r{i+1}/{len(problems)}", end='')

        optimal_action = np.argmax(problem.q_stars)

        agent.reset(problem, alpha=alpha, 
                optimistic_init=optimistic_init, 
                gradient_bandit_algo=grad_bandit_algo,
                reward_baseline=reward_baseline)

        rewards = np.zeros((bandit_learning_step_num,), dtype=np.float16)
        optimal_action_rates = np.zeros((bandit_learning_step_num,), dtype=np.bool8)
        for step in range(bandit_learning_step_num):
            if epsilon is not None:
                action, reward = agent.act_epsilon_greedy(epsilon=epsilon)
            elif c is not None:
                action, reward = agent.act_ucb(c=c)
            elif grad_bandit_algo:
                action, reward = agent.act_gradient_bandit()

            rewards[step] = reward
            optimal_action_rates[step] = 1 if action==optimal_action else 0

        rewards_list[i, :] = rewards
        optimal_action_rates_list[i, :] = optimal_action_rates

    print()

    average_rewards = np.mean(rewards_list, axis=0)
    average_optimal_action_rates = np.mean(optimal_action_rates_list, axis=0)

    return average_rewards, average_optimal_action_rates