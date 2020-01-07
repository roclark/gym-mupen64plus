from gym.envs.registration import register

from gym_mupen64plus.envs.MarioTennis.mario_tennis_env import MarioTennisEnv
from gym_mupen64plus.envs.MarioTennis.discrete_envs import MarioTennisDiscrete

characters = ['mario', 'luigi', 'peach', 'daisy', 'toad', 'waluigi', 'shyguy',
              'wario', 'bowser', 'donkeykong', 'donkeykongjr', 'yoshi',
              'birdo', 'babymario', 'paratroopa', 'boo']

for character in characters:
    # Continuous Action Space:
    register(
        id='Mario-Tennis-%s-v0' % character,
        entry_point='gym_mupen64plus.envs.MarioTennis:MarioTennisEnv',
        kwargs={'my_character': character},
        tags={
            'mupen': True,
            'wrapper_config.TimeLimit.max_episode_steps': 2147483647
        },
        nondeterministic=True
    )

    # Discrete Action Space:
    register(
        id='Mario-Tennis-Discrete-%s-v0' % character,
        entry_point='gym_mupen64plus.envs.MarioTennis:MarioTennisDiscreteEnv',
        kwargs={'my_character': character},
        tags={
            'mupen': True,
            'wrapper_config.TimeLimit.max_episode_steps': 2147483647
        },
        nondeterministic=True
    )
