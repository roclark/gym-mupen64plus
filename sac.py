import gym
import gym_mupen64plus
import ray
from argparse import ArgumentParser
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.env.atari_wrappers import (MonitorEnv,
                                          NoopResetEnv,
                                          WarpFrame,
                                          FrameStack)
from tabulate import tabulate


def parse_args():
    parser = ArgumentParser(description='Train an agent to beat Super Mario '
                            'Bros. levels.')
    parser.add_argument('--checkpoint', help='Specify an existing checkpoint '
                        'which can be used to restore progress from a previous'
                        ' training run.')
    parser.add_argument('--dimension', help='The image dimensions to resize to'
                        ' while preprocessing the game states.', type=int,
                        default=84)
    parser.add_argument('--environment', help='The Super Mario Bros level to '
                        'train on.', type=str,
                        default='Mario-Kart-Discrete-Luigi-Raceway-v0')
    parser.add_argument('--framestack', help='The number of frames to stack '
                        'together to feed into the network.', type=int,
                        default=4)
    parser.add_argument('--gpus', help='Number of GPUs to include in the '
                        'cluster.', type=float, default=0)
    parser.add_argument('--iterations', help='Number of iterations to train '
                        'for.', type=int, default=1000000)
    parser.add_argument('--workers', help='Number of workers to launch on the '
                        'cluster. Hint: Must be less than the number of CPU '
                        'cores available.', type=int, default=4)
    return parser.parse_args()


def env_creator(env_name, config, dim, framestack):
    import gym_mupen64plus
    env = gym.make(env_name)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = WarpFrame(env, dim)
    if framestack:
        env = FrameStack(env, framestack)
    return env


def print_results(result, iteration):
    table = [['SAC',
              iteration,
              result['timesteps_total'],
              round(result['episode_reward_max'], 3),
              round(result['episode_reward_min'], 3),
              round(result['episode_reward_mean'], 3)]]
    print(tabulate(table,
                   headers=['Agent',
                            'Iteration',
                            'Steps',
                            'Max Reward',
                            'Min Reward',
                            'Mean Reward'],
                   tablefmt='psql',
                   showindex="never"))
    print()


def main():
    def env_creator_lambda(env_config):
        return env_creator(args.environment,
                           config,
                           args.dimension,
                           args.framestack)

    args = parse_args()
    config = {
        'env': 'mario_kart',
        'framework': 'torch',
        'gamma': 0.99,
        'use_state_preprocessor': True,
        'Q_model': {
            'fcnet_activation': 'relu',
            'fcnet_hiddens': [512]
        },
        'policy_model': {
            'fcnet_activation': 'relu',
            'fcnet_hiddens': [512]
        },
        'tau': 1.0,
        'target_network_update_freq': 8000,
        'target_entropy': 'auto',
        'no_done_at_end': False,
        'n_step': 1,
        'rollout_fragment_length': 1,
        'buffer_size': int(1e6),
        'lr': 0.01,
        'prioritized_replay': True,
        'train_batch_size': 64,
        'timesteps_per_iteration': 4,
        'learning_starts': 100000,
        'num_workers': args.workers,
        'num_envs_per_worker': 1,
        'num_gpus': args.gpus
    }
    ray.init(address='head:6379', _redis_password='5241590000000000')

    register_env('mario_kart', env_creator_lambda)
    trainer = SACTrainer(config=config)

    if args.checkpoint:
        trainer.restore(args.checkpoint)

    for iteration in range(args.iterations):
        result = trainer.train()
        print_results(result, iteration)

        if iteration % 50 == 0:
            checkpoint = trainer.save()
            print('Checkpoint saved at', checkpoint)


if __name__ == "__main__":
    main()
