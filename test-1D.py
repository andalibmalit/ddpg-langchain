import gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

class Simple1DEnvironment:
    def __init__(self):
        self.current_state = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def step(self, action):
        self.current_state += action
        reward = -abs(self.current_state)
        if abs(self.current_state) > 10:
            done = True
            reward -= 100  # Large penalty if out of bounds
        else:
            done = False
        return np.array([self.current_state]), reward, done

    def reset(self):
        self.current_state = np.random.uniform(low=-1, high=1)
        return np.array([self.current_state])


env = Simple1DEnvironment()
agent = DDPGagent(env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000)
noise = OUNoise(env.action_space)

rewards = []
avg_rewards = []

# Parameters
batch_size = 64
num_episodes = 50

for episode in range(num_episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done = env.step(action)

        # Push transition to memory
        agent.memory.push(state, action, reward, new_state, done)

        # Update agent if enough transitions are stored in memory
        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            print("Episode: {}, Reward: {}, Average Reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:] if len(rewards) > 10 else rewards)))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:] if len(rewards) > 10 else rewards))

plt.figure(figsize=(10,5))
plt.plot(rewards, label='Reward')
plt.plot(avg_rewards, label='Average Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()