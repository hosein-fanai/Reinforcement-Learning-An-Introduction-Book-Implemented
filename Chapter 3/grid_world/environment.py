class Environment:
    ACTIONS = ["West", "North", "East", "South"]

    def __init__(self, grid_size=5, discount=0.9, 
                A_pos=(0, 1), A_prime_pos=(4, 1), A_reward=10., 
                B_pos=(0, 3), B_prime_pos=(2, 3), B_reward=5.):
        self.grid_size = grid_size
        self.discount = discount
        self.A_pos = list(A_pos)
        self.A_prime_pos = list(A_prime_pos)
        self.A_reward  = A_reward
        self.B_pos = list(B_pos)
        self.B_prime_pos = list(B_prime_pos)
        self.B_reward = B_reward
    
    def reset(self, start_pos=(0, 0)):
        self.agent_pos = list(start_pos)

        return self.agent_pos
    
    def step(self, action):
        assert action in Environment.ACTIONS, "Wrong action for the step function!"

        reward = 0.

        if self.agent_pos[0] == self.A_pos[0] and \
            self.agent_pos[1] == self.A_pos[1]:
            reward += self.A_reward
            self.agent_pos = self.A_prime_pos.copy()

            return self.agent_pos, reward
        
        if self.agent_pos[0] == self.B_pos[0] and \
            self.agent_pos[1] == self.B_pos[1]:
            reward += self.B_reward
            self.agent_pos = self.B_prime_pos.copy()

            return self.agent_pos, reward

        match action:
            case "West":
                self.agent_pos[1] += -1
            case "North":
                self.agent_pos[0] += -1
            case "East":
                self.agent_pos[1] += 1
            case "South":
                self.agent_pos[0] += 1
            case _:
                raise Exception("Wrong action for the step function!")
    
        if self.agent_pos[0] < 0 or \
            self.agent_pos[1] < 0 or \
            self.agent_pos[0] >= self.grid_size or \
            self.agent_pos[1] >= self.grid_size:
            reward += -1.
        
        self.agent_pos[0] = max(0, self.agent_pos[0])
        self.agent_pos[1] = max(0, self.agent_pos[1])
        self.agent_pos[0] = min(self.grid_size-1, self.agent_pos[0])
        self.agent_pos[1] = min(self.grid_size-1, self.agent_pos[1])

        return self.agent_pos, reward
    
    def render(self):
        print('_'*80)

        for i in range(self.grid_size):
            print('|', end='\t')

            for j in range(self.grid_size):
                cell = ''
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    cell += ' X'
                if i == self.A_pos[0] and j == self.A_pos[1]:
                    cell += ' A'
                if i == self.A_prime_pos[0] and j == self.A_prime_pos[1]:
                    cell += ' A`'
                if i == self.B_pos[0] and j == self.B_pos[1]:
                    cell += ' B'
                if i == self.B_prime_pos[0] and j == self.B_prime_pos[1]:
                    cell += ' B`'

                cell += ' '

                print(cell, end="\t|\t")

            print()
            print(('|'+'_'*15)*self.grid_size, end="|\n")


if __name__ == "__main__":
    env = Environment()
    env.reset()
    env.render()

    while True:
        action_index = int(input("Action Index: "))
        _, reward = env.step(env.ACTIONS[action_index])
        print("Reward:", reward)

        env.render()