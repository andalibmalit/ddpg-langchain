'''
    At the start of a simulation, the ecosystem parameters are initialized randomly within some interval.
    The goal is for the LLM to train a DDPG agent that can bring the system to equilibrium within a certain
        number of steps for arbitrary initial conditions.
    The user can also interface with the LLM at any point during the simulated ecosystem run and upset the
        system by commanding the LLM to set the parameters to some value. The DDPG agent will then try to balance
        the ecosystem under the new conditions.

    Here are all possible way to manipulate the ecosystem:
    * The LLM can directly access all parameters.
    * The user can change parameter values through the LLM.
    * The user can only change rabbit and fox population values.
    * The DDPG agent can only Lotka-Volterra parameters values (they are a continuous action space).
'''

import numpy as np

class Ecosystem:
    def __init__(self, alpha=0.1, beta=0.02, gamma=0.3, delta=0.01, rabbit_pop = 1.0, fox_pop=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rabbit_pop = rabbit_pop
        self.fox_pop = fox_pop

    def reward(self):
        # Define the ideal equilibrium populations
        ideal_rabbit_pop = self.gamma / (self.delta + 1e-6)
        ideal_fox_pop = self.alpha / (self.beta + 1e-6)

        # Calculate the difference between the current and ideal populations
        rabbit_diff = abs(self.rabbit_pop - ideal_rabbit_pop)
        fox_diff = abs(self.fox_pop - ideal_fox_pop)

        # Calculate the reward based on the difference
        reward = -(rabbit_diff + fox_diff)

        return reward

    def step(self, action):
        """
        action is a list of 4 continuous values corresponding to modifications to alpha, beta, gamma, delta
        """
        action = action[0] # Initially it's a list of a list
        # Apply modifications to parameters. Use tanh to ensure the new parameter is in the range [-1, 1]
        self.alpha = max(1e-5, self.alpha + action[0])
        self.beta = max(1e-5, self.beta + action[1])
        self.gamma = max(1e-5, self.gamma + action[2])
        self.delta = max(1e-5, self.delta + action[3])

        # Update populations
        new_rabbit_pop = self.rabbit_pop + (self.alpha * self.rabbit_pop - self.beta * self.rabbit_pop * self.fox_pop)
        new_fox_pop = self.fox_pop + (self.delta * self.rabbit_pop * self.fox_pop - self.gamma * self.fox_pop)
        
        self.rabbit_pop = new_rabbit_pop
        self.fox_pop = new_fox_pop

        # Observations for the DDPG agent are rabbit and fox populations
        state = np.array([self.rabbit_pop, self.fox_pop])

        # Calculate reward
        reward = self.reward()
        print("REWARD:", reward)

        # Send done signal if either population dies out
        done = False
        if self.rabbit_pop == 0.0 or self.fox_pop == 0.0:
            done = True

        return state, reward, done

    def set_rabbits(self, r):
        self.rabbit_pop = r
    
    def set_foxes(self, f):
        self.fox_pop = f