import os
from getpass import getpass
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from ecosystem import *
from DDPGagent import *
import matplotlib.pyplot as plt

# HUGGINGFACEHUB_API_TOKEN = getpass()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# repo_id = "google/flan-t5-xl"
# llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "Who won the FIFA World Cup in the year 1994? "

# print(llm_chain.run(question))

# Create the environment
env = Ecosystem()

# Initialize DDPG agent
agent = DDPGagent()

# Set parameters
num_episodes = 200  # Total episodes of interaction with the environment
episode_length = 300  # Maximum length of an episode
batch_size = 64  # Size of the batch to be sampled and used for training
reward_threshold = 1000  # The reward threshold for considering the task solved

# Initialize variables
episode_rewards = []  # Store the total reward for each episode
avg_rewards = []  # Store the average reward over the last 100 episodes

# Main loop
for episode in range(num_episodes):
    state = np.array([env.rabbit_pop, env.fox_pop])
    episode_reward = 0

    for step in range(episode_length):
        # Agent takes action
        action = agent.get_action(state)

        # Apply the action to the environment
        next_state, reward, done = env.step(action)

        # Save the experience in the replay memory
        agent.memory.push(state, action, reward, next_state, done)

        # Update the total reward and state
        episode_reward += reward
        state = next_state

        # If the memory is sufficiently populated, perform an update
        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        # If the episode is done, break
        if done:
            break

    episode_rewards.append(episode_reward)

    # Calculate average reward over the last 100 episodes
    if episode >= 100:
        avg_reward = np.mean(episode_rewards[episode-100:episode+1])
        avg_rewards.append(avg_reward)

    # Log progress
    print(f"Episode {episode}, Total Reward: {episode_reward}")

    # Check if task is solved
    if np.mean(avg_rewards[-100:]) > reward_threshold:
        print(f"Task solved in {episode} episodes!")
        break

# Plot the training progress
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(avg_rewards, label='Average Reward (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

# Finally, we can save our trained model
torch.save(agent.actor.state_dict(), 'actor_model.pth')
torch.save(agent.critic.state_dict(), 'critic_model.pth')