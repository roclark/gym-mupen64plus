import gym
import gym_mupen64plus
import ray
from argparse import ArgumentParser
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
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
    table = [['PPO',
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
        'lr': 0.0003,
        'lambda': 0.95,
        'gamma': 0.99,
        'sgd_minibatch_size': tune.grid_search([256, 512, 1024]),
        'train_batch_size': tune.grid_search([4000, 10000, 40000]),
        'vf_clip_param': tune.grid_search([0.1, 10, 100, 1000]),
        'clip_param': 0.2,
        'num_sgd_iter': 10,
        'rollout_fragment_length': 200,
        'num_workers': args.workers,
        'num_envs_per_worker': 1,
        'num_gpus': args.gpus,
        'model': {
            'fcnet_hiddens': [512, 512]
        }
    }
    ray.init(address='head:6379', _redis_password='5241590000000000')

    register_env('mario_kart', env_creator_lambda)
    import time
    time.sleep(5)
    tune.run('PPO', stop={'timesteps_total': 500000}, config=config)
    #trainer = PPOTrainer(config=config)

    #if args.checkpoint:
    #    trainer.restore(args.checkpoint)

    #for iteration in range(args.iterations):
    #    result = trainer.train()
    #    print_results(result, iteration)

    #    if iteration % 50 == 0:
    #        checkpoint = trainer.save()
    #        print('Checkpoint saved at', checkpoint)


if __name__ == "__main__":
    main()
