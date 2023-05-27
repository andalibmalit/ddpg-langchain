# Maintaining ecosystem equilibrium with LangChain and a DDPG agent
This is a toy project to get warmed up on working with LangChain + DDPG.

## Progress
Currently, the reward function ends up returning nan after several steps of an epoch. This may be due to instability because the reward function is based on the four parameters we are optimizing for... but hoping this is not the case. Current approach to solve this is to squeeze the ouput of `get_action()` in DDPGagent.py into some interval, say [-1, 1]...

## Credits
Thanks to Chris Yoon on [Medium](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) for guidance on DDPG implementation!
