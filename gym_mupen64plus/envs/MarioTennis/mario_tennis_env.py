import inspect
import yaml
from gym import spaces
from gym_mupen64plus.envs.mupen64plus_env import (ControllerState,
                                                  Mupen64PlusEnv)
from os.path import dirname, join


class MarioTennisEnv(Mupen64PlusEnv):
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

    def _reset(self):
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
        return True

    def _navigate_menu(self):
        self._navigate_start_menu()
        self._navigate_player_select()

    def _navigate_start_menu(self):
        self._wait(count=100, wait_for='Splash Screen')
        self._press_button(ControllerState.START_BUTTON)
        self._wait(count=30, wait_for='Main Menu')
        # Select single player
        self._press_button(ControllerState.A_BUTTON)

    def _navigate_player_select(self):
        self._wait(count=30, wait_for='Player Select')
        self._player_select()

    def _player_select(self):
        row, col = self._player_pos
        # Navigate to the requested player according to their pre-defined row
        # and column values.
        self._press_button(ControllerState.JOYSTICK_DOWN, times=row)
        self._press_button(ControllerState.JOYSTICK_RIGHT, times=col)
        # Select the requested player
        self._press_button(ControllerState.A_BUTTON)

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
