import cv2
import inspect
import numpy as np
import yaml
from abc import ABCMeta
from gym import spaces
from gym_mupen64plus.envs.mupen64plus_env import (ControllerState,
                                                  Mupen64PlusEnv)
from gym_mupen64plus.envs.MarioTennis.score_tracker import ScoreParser
from os.path import dirname, join


class MarioTennisEnv(Mupen64PlusEnv):
    __metaclass__ = ABCMeta

    def __init__(self, player='mario', opponent='luigi', court='open'):
        self._set_players(player, opponent)
        self.counter_im = 0
        self.score_parser = ScoreParser(player, opponent)
        self._episode_over = False

        super(MarioTennisEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([[-128, 127],
                                                  [-128, 127],
                                                  [   0,   1],
                                                  [   0,   1],
                                                  [   0,   0],
                                                  [   0,   1],
                                                  [   0,   1],
                                                  [   0,   1]])

    def _step(self, action):
        # Append unneeded inputs
        num_missing = len(ControllerState.A_BUTTON) - len(action)
        full_action = action + [0] * num_missing
        return super(MarioTennisEnv, self)._step(full_action)

    def _reset(self):
        if self.reset_count > 0:
            with self.controller_server.frame_skip_disabled():
                if self._episode_over:
                    self._navigate_postgame()
                    self._navigate_player_select()
                    self._navigate_game()
                    self._select_opponent()
                    self._select_court()
                    self.score_parser.queue.clear()
                    self.score_parser._serving = True
                    self._episode_over = False
                else:
                    self._navigate_endgame()
                    self._navigate_player_select()
                    self._navigate_game()
                    self._select_opponent()
                    self._select_court()
                    self.score_parser.queue.clear()
                    self.score_parser._serving = True
        return super(MarioTennisEnv, self)._reset()

    def _get_reward(self):
        reward = self.score_parser.reward(self.pixel_array)
        if reward != 0.0:
            print(reward)
        #print(reward)
        #return reward
        self.counter_im += 1
        img = cv2.cvtColor(self.pixel_array[165:275, 145:495, :], cv2.COLOR_BGR2HSV)
        lower = np.array([85, 200, 220])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(img, lower, upper)
        img = cv2.bitwise_and(img, img, mask=mask)
        lower = np.array([5, 5, 5])
        upper = np.array([255, 255, 255])
        img[mask != 0] = [255, 255, 255]
        best = 0.0
        best_key = None

        for key, image in self.score_parser._templates.items():
            intersection = np.sum(np.logical_and(img, image))
            union = np.sum(np.logical_or(img, image))
            miou = intersection / float(union)
            if miou > best and miou > 0.35:
                best = miou
                best_key = key
        #if best == 0.0:
        #    return 1.0
        cv2.imwrite('%s.jpg' % self.counter_im, img)
        return reward

    def _evaluate_end_state(self):
        self._episode_over = True
        return self.score_parser.match_complete

    def _load_config(self):
        config = yaml.safe_load(open(join(dirname(inspect.stack()[0][1]),
                                          'mario_tennis_config.yml')))
        self.config.update(config)

    def _validate_config(self):
        print('validate config')
        gfx_plugin = self.config['GFX_PLUGIN']

    def _navigate_menu(self):
        self._navigate_start_menu()
        self._navigate_player_select()
        self._navigate_game()
        self._select_opponent()
        self._select_court()
        # Skip intro
        self._wait(count=200, wait_for='Pregame intro')
        self._act(ControllerState.A_BUTTON, count=2)
        self._wait(count=50, wait_for='Versus Screen')
        self._act(ControllerState.A_BUTTON, count=2)

    def _navigate_start_menu(self):
        self._wait(count=140, wait_for='Nintendo Screen')
        self._act(ControllerState.A_BUTTON)

        self._wait(count=230, wait_for='Intro Video')
        self._act(ControllerState.A_BUTTON)

        self._wait(count=190, wait_for='Splash Screen')
        self._act(ControllerState.START_BUTTON)

        # Select single player
        self._wait(count=70, wait_for='Main Menu')
        self._act(ControllerState.A_BUTTON, count=2)

    def _navigate_player_select(self):
        self._wait(count=60, wait_for='Player Select')
        self._player_select(self._player_pos)

    def _player_select(self, player_pos):
        row, col = player_pos
        # Navigate to the requested player according to their pre-defined row
        # and column values.
        self._press_button(ControllerState.JOYSTICK_DOWN, times=row)
        self._press_button(ControllerState.JOYSTICK_RIGHT, times=col)
        # Select the requested player
        self._act(ControllerState.A_BUTTON, count=2)

    def _navigate_game(self):
        # Select Exhibition mode
        self._wait(count=100, wait_for='Game Selection')
        self._act(ControllerState.A_BUTTON, count=2)

        # Select singles
        self._wait(count=50, wait_for='Number of Players')
        self._act(ControllerState.A_BUTTON, count=2)

        # Select 1 set with 2 games
        self._wait(count=50, wait_for='Match duration')
        self._act(ControllerState.A_BUTTON, count=2)

    def _select_opponent(self):
        # Select computer opponent
        self._wait(count=50, wait_for='Opponent select')
        #self._player_select(self._opponent_pos)
        self._act(ControllerState.A_BUTTON, count=2)

        # Select difficulty
        self._wait(count=10, wait_for='Difficulty select')
        self._act(ControllerState.JOYSTICK_LEFT)
        self._act(ControllerState.A_BUTTON, count=2)

    def _select_court(self):
        self._wait(count=150, wait_for='Court selection')
        self._act(ControllerState.A_BUTTON, count=2)

    def _navigate_postgame(self):
        # Wait for final replay screen
        self._wait(count=200, wait_for='Replay Start')
        self._act(ControllerState.A_BUTTON, count=2)
        # Exit final replay
        self._wait(count=400, wait_for='Post Game Stats')
        self._act(ControllerState.A_BUTTON, count=2)
        # Exit post game stats back to main menu
        self._wait(count=200, wait_for='Post Game')
        self._act(ControllerState.A_BUTTON, count=2)
        # Select single player
        self._wait(count=170, wait_for='Main Menu')
        self._act(ControllerState.A_BUTTON, count=2)

    def _navigate_endgame(self):
        # Can't pause the match until fully loaded
        if (self.step_count * self.controller_server.frame_skip) < 30:
            steps_to_wait = 30 - (self.step_count * self.controller_server.frame_skip)
            self._wait(count=steps_to_wait, wait_for='Match to load')
        # Load the in-game menu
        self._act(ControllerState.START_BUTTON, count=2)
        self._wait(count=20, wait_for='Pause screen loaded')
        # Navigate to the save progress option
        for _ in range(3):
            self._act(ControllerState.JOYSTICK_RIGHT, count=2)
            self._wait(count=5, wait_for='Next selection')
        # Select the save progress sub-menu
        self._act(ControllerState.A_BUTTON, count=2)
        self._wait(count=40, wait_for='Save progress screen')
        # Navigate to the end game button
        self._act(ControllerState.JOYSTICK_UP, count=2)
        self._wait(count=5, wait_for='Save progress to end game')
        # Select the end game button
        self._act(ControllerState.A_BUTTON, count=2)
        self._wait(count=10, wait_for='End game menu')
        # Navigate to the main menu button
        self._act(ControllerState.JOYSTICK_LEFT, count=2)
        self._wait(count=5, wait_for='Navigate to menu button')
        # Select the main menu button
        self._act(ControllerState.A_BUTTON, count=2)
        self._wait(count=170, wait_for='Main menu to load')
        # Select single player
        self._act(ControllerState.A_BUTTON, count=2)

    def _set_players(self, player, opponent):
        players = {
            'mario': (0, 0),
            'luigi': (0, 1),
            'peach': (0, 2),
            'babymario': (0, 3),
            'yoshi': (0, 4),
            'donkeykong': (0, 5),
            'paratroopa': (0, 6),
            'donkeykongjr': (0, 7),
            'wario': (1, 0),
            'waluigi': (1, 1),
            'daisy': (1, 2),
            'toad': (1, 3),
            'birdo': (1, 4),
            'bowser': (1, 5),
            'boo': (1, 6),
            'shyguy': (1,7)
        }

        self._player_pos = players[player]
        self._opponent_pos = players[opponent]
