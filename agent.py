import numpy as np


class Agent:

    def __init__(self, table_shape=(3**9, 9)):
        self.Q_table = np.zeros(table_shape)
    
    def save(self, path):
        np.save(path, self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path)

    def train(self, env, n_epochs, max_steps=100, gamma=0.95, lr=0.1, self_play_type="q_policy"):
        opo_policy_fn = None if self_play_type == "random" else self.act_epsilon_greedy

        for epoch in range(1, n_epochs+1):
            epsilon = max(0.01, 1 - epoch/(0.6*n_epochs))

            state = env.reset()
            rewards = 0.

            for step in range(max_steps):
                action = self.act_epsilon_greedy(state, epsilon)

                state_new, reward, done, info = env.step(action, self_play_type=self_play_type, opponent_policy_fn=opo_policy_fn)

                self.Q_table[state][action] += lr * (reward + gamma * self.Q_table[state_new].max() - self.Q_table[state][action])

                state = state_new
                rewards += reward

                if done:
                    break
            
            print(f"\rEpoch: {epoch}, Epsilon: {epsilon:.4f}, Rewards: {rewards:.4f}", end="")

    def act_greedy(self, state):
        action = self.Q_table[state].argmax()

        return action

    def act_epsilon_greedy(self, state, epsilon=0.01):
        if np.random.rand() > epsilon:
            action = self.act_greedy(state)
        else:
            action = np.random.randint(9)

        return action
    
    def act_greedy_sampled(self, state, temperature=0.5):
        preds = self.Q_table[state]
        exp_preds = np.exp(np.log(preds + np.abs(preds.min()) + 1e-5) / temperature)
        exp_preds /= np.sum(exp_preds) + 1e-7
        
        action = np.random.multinomial(1, exp_preds, 1).argmax()

        return action