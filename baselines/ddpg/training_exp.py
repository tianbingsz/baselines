import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg_exp import DDPGExp
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import pdb


def train(
    env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise,
    actor, critic, explorer,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_train_onpolicy_steps, 
    nb_rollout_steps, nb_eval_steps, batch_size, memory,
          on_policy_mem, tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()
            # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info(
        'scaling actions by {} before executing in env'.format(max_action))
    agent = DDPGExp(actor, critic, explorer, memory, on_policy_mem,
                    env.observation_space.shape, env.action_space.shape,
                    gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
                    batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                    actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                    reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        eval_obs = obs
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        eval_episode_rewards = []
        eval_qs = []
        for epoch in range(nb_epochs):
            # 1.inner loop cycle, learn to explore
            # clear exploration buffer D_0, on-policy
            agent.clear_on_policy_mem()
            for cycle in range(nb_epoch_cycles):
                # a) generate rollouts {D_0} with exploration policy $\pi_0$,
                for t_rollout in range(nb_rollout_steps):
                    # predict next action, first a = a + N
                    action, q = agent.pi_noisy_exp(
                        obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape
                    # exec next action
                    assert max_action.shape == action.shape
                    # scale for execution in env (as far as DDPG is concerned,
                    # every action is in [-1, 1])
                    new_obs, r, done, info = env.step(max_action * action)
                    t += 1
                    episode_reward += r
                    episode_step += 1
                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    # save to on policy mem
                    agent.store_on_policy_transition(
                        obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # b) update exploitation policy $\pi_1$ and Q with D_0, on-policy
                # todo, save old policy
                # todo, log actor, critic losses
                # todo, off policy train, but seems on policy is pretty good?
                for t_op_train in range(nb_train_onpolicy_steps):
                    cl, al = agent.train_on_policy()
                    agent.update_target_net()
                # c) generate rollouts {D_1} with $\pi_1$, evaluate the performace $R_t = R_{new_1}(D_1) - R_{old_1}(D_1)$
                # Evaluation for only one trajectory, 
                # todo, more trajectories, but maybe we could use evaluation
                # Q instead of monte carlo R?
                # evaluation for old policy
                eval_episode_reward = 0.
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q = agent.pi(
                        eval_obs, apply_noise=False, compute_Q=True)
                    # scale for execution in env
                    new_eval_obs, eval_r, eval_done, eval_info = env.step(
                        max_action * eval_action)
                    eval_episode_reward += eval_r
                    eval_qs.append(eval_q)
                    agent.store_transition(
                        eval_obs,
                        eval_action,
                        eval_r,
                        new_eval_obs,
                        eval_done)
                    eval_obs = new_eval_obs
                    if eval_done:
                        eval_obs = env.reset()
                        # R_t = eval_reward
                        eval_episode_rewards.append(eval_episode_reward)
                        eval_episode_rewards_history.append(
                            eval_episode_reward)
                        eval_episode_reward = 0.
                    # todo, dR_t = R_t - old_R
                    # maxR = max(maxR, R_t), save max policy

            # 2.update exploration policy $\pi_0$ with {D_0} and R_t
            # \sum_t \partial{\log \pi_0(D_0t)}{\theta} R_t
            # first use the same exploration policy here, a = a + N

            # 3.update exploitation policy $\pi_1$ and Q (or \pi_1 = \argmax_t {R_t})
            # Policy Upate, todo, use max policy
            epoch_actor_losses = []
            epoch_critic_losses = []
            for t_train in range(nb_train_steps):
                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(
                np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(
                epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

            # Train statistics.
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)

            # Evaluation statistics.
            combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
            combined_stats['eval/return_history'] = mpi_mean(
                np.mean(eval_episode_rewards_history))
            combined_stats['eval/Q'] = mpi_mean(eval_qs)
            combined_stats['eval/episodes'] = mpi_mean(
                len(eval_episode_rewards))

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(
                float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)
