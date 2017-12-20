import argparse
import time
import datetime
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
from sandbox.vime.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.vime.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from sandbox.vime.envs.half_cheetah_env_x import HalfCheetahEnvX
from sandbox.vime.envs.swimmer_env_x import SwimmerEnvX
from sandbox.vime.envs.double_pendulum_env_x import DoublePendulumEnvX
from rllab.envs.normalized_env import normalize

import gym
import tensorflow as tf
from mpi4py import MPI
import pdb

# ===========================#
# SPARSE REWARD EXPERIMENTS #
# ===========================#

mc_experiment = {
    'env_name': 'MountainCarEnvX',
    'task_type': 'classic',
    'env_call': MountainCarEnvX,
    'normalize_env': False
}

cps_experiment = {
    'env_name': 'CartpoleSwingupEnvX',
    'task_type': 'classic',
    'env_call': CartpoleSwingupEnvX,
    'normalize_env': False
}

hc_experiment = {
    'env_name': 'HalfCheetahEnvX',
    'task_type': 'locomotion',
    'env_call': HalfCheetahEnvX,
    'normalize_env': True
}

swim_experiment = {
    'env_name': 'SwimmerEnvX',
    'task_type': 'locomotion',
    'env_call': SwimmerEnvX,
    'normalize_env': False
}

pendulum_experiment = {
    'env_name': 'DoublePendulumEnvX',
    'task_type': 'classic',
    'env_call': DoublePendulumEnvX,
    'normalize_env': False
}

experiments = {
    'MountainCarX': mc_experiment,
    'CartpoleSwingupX': cps_experiment,
    'HalfCheetahX': hc_experiment,
    'SwimmerX': swim_experiment,
    'DoublePendulumX': pendulum_experiment,
}


def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    # env = CartpoleSwingupEnvX()
    env = experiments[env_id]['env_call']()
    if experiments[env_id]['normalize_env']:
        env = normalize(env)
    env = bench.Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    # env = gym.wrappers.Monitor(env, log_dir, video_callable=False,
    # force=True)
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank == 0:
        eval_env = experiments[env_id]['env_call']()
        if experiments[env_id]['normalize_env']:
            eval_env = normalize(eval_env)
        eval_env = bench.Monitor(
            eval_env,
            os.path.join(logger.get_dir(),
                         'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(
                initial_stddev=float(stddev),
                desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError(
                'unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(
        limit=int(1e6),
        action_shape=env.action_space.shape,
     observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info(
        'rank {}: seed={}, logdir={}'.format(rank,
                                             seed,
                                             logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    # env.seed(seed)
    # if eval_env is not None:
        # eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
                   action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=100)
                        # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument(
        '--nb-train-steps',
        type=int,
     default=50)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--nb-eval-steps',
        type=int,
     default=100)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--nb-rollout-steps',
        type=int,
     default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')
                        # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(
            args.num_timesteps == args.nb_epochs *
            args.nb_epoch_cycles *
            args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    env_name = args['env_id']
    root_dir = '/usr/local/tianbing/baidu/idl/baselines/baselines/ddpg/logs/'
    for nb_epoch_cycles in [100]:
        args['nb_epoch_cycles'] = nb_epoch_cycles
        args['nb_rollout_steps'] = int(1e4 / nb_epoch_cycles)
        log_dir = os.path.join(root_dir,
            "env={:}_{:}_nc={:}_nr={:}_{:}".format(env_name,
             args['noise_type'], args['nb_epoch_cycles'],
             args['nb_rollout_steps'],
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.configure(dir=log_dir)
        # Run actual script.
        run(**args)
