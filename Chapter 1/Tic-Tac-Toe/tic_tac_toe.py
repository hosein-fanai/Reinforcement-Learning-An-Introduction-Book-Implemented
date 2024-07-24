import numpy as np


class TicTacToe:

    def __init__(self):
        self.reset()

    def _check_board_is_full(self):
        return (self.board == 0).sum() == 0

    def _calc_reward_and_done(self):
        reward = -0.1
        done = False

        for i in range(3):
            row = self.board[i].tolist()
            if row == [1, 1, 1]:
                reward += 10.1
                done = True
            elif row == [2, 2, 2]:
                reward -= 10.
                done = True

        for j in range(3):
            col = self.board[:, j].tolist()
            if col == [1, 1, 1]:
                reward += 10.1
                done = True
            elif col == [2, 2, 2]:
                reward -= 10.
                done = True

        diag_main = [self.board[k, k] for k in range(3)]
        diag_aux = [self.board[0, 2], self.board[1, 1], self.board[2, 0]]
        if diag_main == [1, 1, 1] or diag_aux == [1, 1, 1]:
            reward += 10.1
            done = True
        elif diag_main == [2, 2, 2] or diag_aux == [2, 2, 2]:
            reward -= 10.
            done = True

        filled = self._check_board_is_full()
        if filled and not done:
            done = filled
            reward -= 10.

        return reward, done

    def _calc_state(self):
        return np.sum([digit*pow(3, i) for i, digit in enumerate(self.board.flatten().tolist())])

    def _take_action(self, action, act_as_O):
        self.board = self.board.flatten()
        self.board[action] = 1 if not act_as_O else 2
        self.board = self.board.reshape((3, 3))

        return self._calc_state()

    def _play_as_opponent(self, state, self_play_type, act_as_O, opponent_policy_fn=None):
        while (self.board == 0).sum() != 0:
            if self_play_type == "q_policy":
                opponent_action = opponent_policy_fn(state)
            elif self_play_type == "random":
                opponent_action = np.random.randint(9)

            if self.board.flatten()[opponent_action] == 0:
                self._take_action(opponent_action, act_as_O=act_as_O)
                break
            else:
                self_play_type = "random"

    def step(self, action, self_play=True, self_play_type="random", act_as_O=False, opponent_policy_fn=None): # TODO: Make self.turn to step turn-wise
        if self.board.flatten()[action] != 0: # 1 is X (as in the player) and 2 is O (as in the opponent)
            state = self._calc_state()
            reward = -1.1
            done = False
            info = {"wrong input": True}

            return state, reward, done, info

        state = self._take_action(action, act_as_O)
        reward, done = self._calc_reward_and_done()
        info = {"wrong input": False}

        if self_play and not done:
            self._play_as_opponent(state, self_play_type, not act_as_O, opponent_policy_fn)

        return state, reward, done, info

    def render(self):
        print('_'*25)
        for i in range(3):
            print('|'+' '*3, end="")
    
            for j in range(3):
                if self.board[i, j] == 0:
                    print(' ', end="")
                elif self.board[i, j] == 1:
                    print('X', end="")
                elif self.board[i, j] == 2:
                    print('O', end="")

                if j != 2:
                    print(' '*3+'|'+' '*3, end="")
            
            print(' '*3+'|')
            if i != 2:
                print('|'+'_'*7+'|'+'_'*7+'|'+'_'*7+'|')
        
        print('|'+'_'*7+'|'+'_'*7+'|'+'_'*7+'|')
        print()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.uint8)
        state = self._calc_state()

        return state

    def play_against_opponent(self, opponent_policy_fn): # TODO: fix the bug for opponent's moves when it is faul
        state = self.reset()
        info = {"wrong input": False}

        while True:
            if not info["wrong input"]:
                action = opponent_policy_fn(state)
                _, _, done, _  = self.step(action, self_play=False)
                self.render()
                if done:
                    break
            else:
                print("Wrong choice! Choose again.")

            action_opponent = int(input("O's turn: "))
            state, _, done, info = self.step(action_opponent, self_play=False, act_as_O=True)
            self.render()
            if done:
                break

if __name__ == "__main__":
    from agent import Agent


    agent = Agent()
    agent.load("./policies/policy (against random opponent).npy")

    env = TicTacToe()
    env.play_against_opponent(opponent_policy_fn=agent.act_greedy_sampled)