import numpy as np


class Agent:

    def __init__(self):    
        pass

    def learn_pi_value_function(self, env, policy, iters_num=100):
        self.V_pi = np.zeros((env.grid_size, env.grid_size))

        for _ in range(iters_num):
            V_pi = np.zeros((env.grid_size, env.grid_size))
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    s = (i, j)
                    for act_index, action in enumerate(env.ACTIONS):
                        env.reset(start_pos=s)
                        s_prime, reward = env.step(action)

                        V_pi[*s] += policy(s)[act_index] * 1. * (reward + env.discount * self.V_pi[*s_prime])

            self.V_pi = V_pi

    def learn_optimal_value_function(self, env, iters_num=100):
        self.V_pi = np.zeros((env.grid_size, env.grid_size))

        for _ in range(iters_num):
            V_pi = np.zeros((env.grid_size, env.grid_size))
            for i in range(env.grid_size):
                for j in range(env.grid_size):
                    s = (i, j)

                    current_state_values = []
                    for action in env.ACTIONS:
                        env.reset(start_pos=s)
                        s_prime, reward = env.step(action)

                        current_state_values.append(1. * (reward + env.discount * self.V_pi[*s_prime]))

                    V_pi[*s] = max(current_state_values)
            
            self.V_pi = V_pi

    def compute_policy(self, env):
        policy = {}
        for i in range(self.V_pi.shape[0]):
            for j in range(self.V_pi.shape[1]):
                s = (i, j)

                action_values = []
                for action in env.ACTIONS:
                    env.reset(start_pos=s)
                    s_prime, _ = env.step(action)
                    action_values.append(self.V_pi[*s_prime])
                
                policy[str(s)] = np.where(action_values==max(action_values))[0].tolist()
        
        return policy

    def run_policy(self, env, policy=None, max_steps=10, render=False):
        if policy is None:
            policy = self.policy
        
        start_pos = np.random.randint(low=0, high=env.grid_size, size=2)
        state = env.reset(start_pos=start_pos)

        if render:
            env.render()

        rewards = 0.
        for step in range(max_steps):
            state = str(state).replace('[', '(').replace(']', ')')
            action = env.ACTIONS[np.random.choice(policy[state])]

            state, reward = env.step(action)

            rewards += reward
            print(f"Step: {step+1}, Current Reward: {reward}, Total Rewards: {rewards}")

            if render:
                env.render()

    def show_value_function(self):
        grid_size = self.V_pi.shape[0]

        print('_'*80)

        for i in range(grid_size):
            print('|', end='\t')
            for j in range(grid_size):
                print(f"{self.V_pi[i, j]:.2f}", end="\t|\t")

            print()
            print(('|'+'_'*15)*grid_size, end="|\n")

    @staticmethod
    def show_policy(policy, env):
        grid_size = int(np.sqrt(len(policy)))

        print('_'*80)

        for i in range(grid_size):
            print('|', end='\t')
            for j in range(grid_size):
                cell = ' '.join([env.ACTIONS[action_index][0] for action_index in policy[f"({i}, {j})"]])
                print(cell, end="\t|\t")

            print()
            print(('|'+'_'*15)*grid_size, end="|\n")