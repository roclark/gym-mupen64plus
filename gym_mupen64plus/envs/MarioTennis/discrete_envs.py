from gym_mupen64plus.envs.MarioTennis.mario_tennis_env import MarioTennisEnv
from gym import spaces


class DiscreteActions:
    ACTION_MAP = [
        ("NO_OP", [0, 0, 0, 0, 0])
    ]

    @staticmethod
    def get_action_space():
        return spaces.Discrete(len(DiscreteActions.ACTION_MAP))

    @staticmethod
    def get_controls_from_action(action):
        return DiscreteActions.ACTION_MAP[action][1]


class MarioTennisDiscreteEnv(MarioTennisEnv):
    def __init__(self, player='mario', opponent='luigi', court='open'):
        super(MarioTennisDiscreteEnv, self).__init__(player=player,
                                                     opponent=opponent,
                                                     court=court)

        # This needs to happen after the parent class init to effectively
        # override the action space.
        self.action_space = DiscreteActions.get_action_space()

    def _step(self, action):
        # Interpret the action choice and get the actual controller state for
        # this step.
        controls = DiscreteActions.get_controls_from_action(action)

        return super(MarioTennisDiscreteEnv, self)._step(controls)
