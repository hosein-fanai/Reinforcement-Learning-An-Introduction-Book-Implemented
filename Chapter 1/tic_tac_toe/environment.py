import numpy as np


class Environment:

    def __init__(self):
        self.reset()

    def _check_board_is_full(self):
        return (self.board == 0).sum() == 0

    def _calc_reward_and_done(self):
        friendly_no = 2 if self.act_as_O else 1
        enemy_no = 1 if self.act_as_O else 2

        reward = -0.1 # 0.
        done = False

        for i in range(3):
            row = self.board[i].tolist()
            if row == [friendly_no]*3:
                reward += 10.1 # 1.
                done = True
            elif row == [enemy_no]*3:
                reward -= 10. # 1.
                done = True

        for j in range(3):
            col = self.board[:, j].tolist()
            if col == [friendly_no]*3:
                reward += 10.1
                done = True
            elif col == [enemy_no]*3:
                reward -= 10.
                done = True

        diag_main = [self.board[k, k] for k in range(3)]
        diag_aux = [self.board[0, 2], self.board[1, 1], self.board[2, 0]]
        if diag_main == [friendly_no]*3 or diag_aux == [friendly_no]*3:
            reward += 10.1
            done = True
        elif diag_main == [enemy_no]*3 or diag_aux == [enemy_no]*3:
            reward -= 10.
            done = True

        filled = self._check_board_is_full()
        if filled and not done:
            done = filled
            reward -= 10.

        return reward, done

    def _calc_state(self, board=None):
        board = self.board if board is None else board

        return np.sum([digit*pow(3, i) for i, digit in enumerate(board.flatten().tolist())])

    def _take_action(self, action, render=False):
        self.board = self.board.flatten()

        if self.board[action] != 0:
            state = self._calc_state()
            reward = -1.1 # 0
            done = False
            info = {"wrong input": True}

            self.board = self.board.reshape((3, 3))
        else:
            self.board[action] = 2 if self.act_as_O else 1
            self.board = self.board.reshape((3, 3))

            state = self._calc_state()
            reward, done = self._calc_reward_and_done()
            info = {"wrong input": False}

        self.act_as_O = not self.act_as_O

        if render:
            self.render()

        return state, reward, done, info

    def reset(self):
        self.act_as_O = False

        self.board = np.zeros((3, 3), dtype=np.uint8) # 1 is X (as in the player) and 2 is O (as in the opponent)
        state = self._calc_state()

        return state

    def step(self, action, self_play=False, opponent_policy=None, render=False):
        state, reward, done, info = self._take_action(action, render)

        if self_play and not done and not info["wrong input"]:
            oppo_board = self.board.copy()
            ones_indices = np.where(oppo_board == 1)[0]
            twos_indices = np.where(oppo_board == 2)[0]
            oppo_board[ones_indices] = 2
            oppo_board[twos_indices] = 1

            oppo_state = self._calc_state(oppo_board)
            oppo_action = opponent_policy(oppo_state)

            _, _, done, _ = self._take_action(oppo_action, render)

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