import numpy as np

from collections import deque

from copy import deepcopy

import random


class QAgent: # This is not the method mentioned in this chapter.

    def __init__(self, table_shape=(3**9, 9)):
        self.Q_table = np.zeros(table_shape)

    def save(self, path):
        np.save(path, self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path)

    def train(self, env, n_episodes, max_steps=100, 
            gamma=0.95, lr=0.1, exploration_portion=0.6, 
            oppo_policy=None):
        oppo_policy = self.act_random if oppo_policy is None else oppo_policy

        for episode in range(1, n_episodes+1):
            epsilon = max(0.01, 1 - episode/(exploration_portion*n_episodes))
            env.reset()

            rewards = 0.
            for _ in range(max_steps):
                state = env._calc_state()
                action = self.act_epsilon_greedy(state, epsilon)

                state_new, reward, done, _ = env.step(action, self_play=True, opponent_policy=oppo_policy)
                # if episode % 250 == 0:
                #     env.render()

                self.Q_table[state][action] += lr * (reward + gamma * self.Q_table[state_new].max() - self.Q_table[state][action])

                rewards += reward

                if done:
                    break
            
            print(f"\r---Episode: {episode}, Epsilon: {epsilon:.4f}, Rewards: {rewards:.4f}", end="")

        print()

    def train_with_self_play_algo(self, env, n_episodes, 
                                max_steps=100,  gamma=0.95, 
                                lr=0.1, exploration_portion=0.6, 
                                n_epochs=50, bank_len=8):
        policy_bank = deque(maxlen=bank_len)
        policy_bank.append(self.act_random)

        for epoch in range(n_epochs):
            print(f"Self-play epoch: {epoch+1}/{n_epochs}")

            oppo_policy_index = random.randint(0, len(policy_bank)-1)
            oppo_policy = policy_bank[oppo_policy_index]

            self.train(env, n_episodes, max_steps, gamma, lr, exploration_portion, oppo_policy=oppo_policy)
            policy_bank.append(deepcopy(self).act_greedy)

    def act_greedy(self, state):
        values = self.Q_table[state]
        max_values = values.max()
        action = random.choice(np.where(values == max_values)[0])

        return action

    def act_random(self, state):
        return np.random.randint(9)

    def act_epsilon_greedy(self, state, epsilon=0.01):
        if np.random.rand() > epsilon:
            action = self.act_greedy(state)
        else:
            action = self.act_random(state)

        return action

    def act_greedy_sampled(self, state, temperature=0.5):
        preds = self.Q_table[state]
        exp_preds = np.exp(np.log(preds + np.abs(preds.min()) + 1e-5) / temperature)
        exp_preds /= np.sum(exp_preds) + 1e-7

        action = np.random.multinomial(1, exp_preds, 1)[0].argmax()

        return action

    def play_against_opponent(self, env, opponent_policy_fn=None): # TODO: fix the bug for opponent's moves when it is faul
        opponent_policy_fn = (lambda state: int(input("O's turn: "))) if opponent_policy_fn is None else self.act_greedy

        env.reset()
        done = False

        rewards = 0.
        while not done:
            state = env._calc_state()
            action = self.act_greedy(state)

            _, reward, done, _  = env.step(action, self_play=True, render=True,
                                    opponent_policy=opponent_policy_fn)

            rewards += reward


if __name__ == "__main__":
    from environment import Environment


    agent = QAgent()
    agent.load("./policies/policy (against random opponent).npy")

    env = Environment()
    agent.play_against_human(opponent_policy_fn=agent.act_greedy_sampled)