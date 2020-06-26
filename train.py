import gym
import mujoco_py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ddpg import *


def train(env_name, path_to_actor=None, path_to_critic=None, batch_size=32, num_episodes=300, max_steps=2000, render=False):
    """Train the agent to learn, and finally save checkpoints
    """
    env = gym.make(env_name)
    if path_to_actor and path_to_critic:
        ddpg = DDPG(env, path_to_actor, path_to_critic)
    else:
        ddpg = DDPG(env)
    episode_rewards = []
    episode_losses = []
    total_steps = 0
    action_noise = OUNoise(env.action_space.shape[0])
    action_noise.reset()
    print('[INFO] Deep DPG built: [action_dim]: {}, [obs_dim]: {}'.format(env.action_space.shape[0], env.observation_space.shape[0]))
    
    for episode in range(num_episodes):
        steps = 0
        obs = env.reset()
        obs = torch.from_numpy(obs)   # single obs
        # Concat the 4 * frame to be small action process
        terminate = False
        episode_reward = 0
        episode_loss = 0
        for episode_step in range(max_steps):
            env.render()
            # Noisy action
            action = ddpg.choose_action(obs)
            action = action + action_noise.noise()
            next_obs, reward, terminate, _ = env.step(action.numpy())
            next_obs = torch.from_numpy(next_obs)
            reward = torch.Tensor([reward])
            ddpg.enQueue(obs, action, reward, terminate, next_obs)

            if total_steps > 100 and (total_steps + 1) % 25 == 0:
                loss = ddpg.fit()
                episode_loss += loss
            episode_reward += reward
            obs = next_obs
            total_steps += 1
            steps += 1
            if terminate:
                break
        print("[INFO] Episode {} terminate with [Reward={}] and [Steps={}]"
                .format(episode + 1, episode_reward, steps))
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss)
        # reduce exploration noise variance
        action_noise.decay()
    
    plt.plot(episode_rewards)
    plt.ylabel('Moving average episode reward')
    plt.xlabel('Episodes')
    plt.savefig(env_name + '-rewards.png')
    plt.close()

    plt.plot(episode_losses)
    plt.ylabel('Moving average episode loss')
    plt.xlabel('Episodes')
    plt.savefig(env_name + '-loss.png')
    plt.close()
    
    torch.save(ddpg.actorEval, 'ddpg-Actor-{}-{}.ckpt'.format(env_name,num_episodes))
    torch.save(ddpg.criticEval, 'ddpg-Critic-{}-{}.ckpt'.format(env_name,num_episodes))


def test(env_name, path_to_actor, path_to_critic):
    """Test the performance of given model 
    """
    env = gym.make(env_name)
    ddpg = DDPG(env, path_to_actor, path_to_critic)
    episode_rewards = []
    num_episodes = 10
    total_steps = 0
    # action_noise = OUNoise(env.action_space.shape[0])
    # action_noise.reset()
    print('[INFO] Deep DPG built: [action_dim]: {}, [obs_dim]: {}'.format(env.action_space.shape[0], env.observation_space.shape[0]))
    
    for episode in range(num_episodes):
        steps = 0
        obs = env.reset()
        obs = torch.from_numpy(obs)   # single obs
        terminate = False
        episode_reward = 0
        for _ in range(2000):
            env.render()
            # Noisy action
            action = ddpg.choose_action(obs)
            # action = action + action_noise.noise()
            next_obs, reward, terminate, _ = env.step(action.numpy())
            next_obs = torch.from_numpy(next_obs)
            reward = torch.Tensor([reward])
            episode_reward += reward
            obs = next_obs
            total_steps += 1
            steps += 1
            if terminate:
                break
        print("[INFO] Episode {} terminate with [Reward={}] and [Steps={}]"
                .format(episode + 1, episode_reward, steps))
        episode_rewards.append(episode_reward)
        # action_noise.decay() 

if __name__ == "__main__":
    train('Ant-v2')
    #train('Ant-v2', 'ddpg-Actor-Ant-v2-300.ckpt', 'ddpg-Critic-Ant-v2-300.ckpt')
    #test('Ant-v2', 'ddpg-Actor-Ant-v2-500.ckpt', 'ddpg-Critic-Ant-v2-500.ckpt')
    #test('HalfCheetah-v2', 'ddpg-Actor-HalfCheetah-v2-800.ckpt', 'ddpg-Critic-HalfCheetah-v2-800.ckpt')
    
