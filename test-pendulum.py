import gym
import matplotlib.pyplot as plt
import numpy as np
import sys
from ddpg import DDPGagent
from utils import *

# Initialize environment, agent and noise
env = gym.make('Pendulum-v1')
agent = DDPGagent(env)
noise = OUNoise(env.action_space)

# Training parameters
n_episodes = 200
max_t = 500
batch_size = 64

# Storage for rewards for plotting
rewards = []
avg_rewards = []

for episode in range(n_episodes):
    print(f"Starting episode {episode}")
    state = env.reset()[0]
    noise.reset()
    episode_reward = 0

    for t in range(max_t):
        action = agent.get_action(state)
        action = noise.get_action(action, t)
        new_state, reward, done, _, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(
                episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
