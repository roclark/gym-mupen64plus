import inspect
import yaml
from abc import ABCMeta
from gym import spaces
from gym_mupen64plus.envs.mupen64plus_env import (ControllerState,
                                                  Mupen64PlusEnv)
from os.path import dirname, join


class MarioTennisEnv(Mupen64PlusEnv):
    __metaclass__ = ABCMeta

    def __init__(self, player='mario', opponent='luigi', court='open'):
        self._set_players(player, opponent)

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
                pass
        return super(MarioTennisEnv, self)._reset()

    def _get_reward(self):
        return 1.0

    def _evaluate_end_state(self):
        return False

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
