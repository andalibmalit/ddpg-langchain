# Maintaining ecosystem equilibrium with LangChain and a DDPG agent
This is a toy project to get warmed up on working with LangChain + DDPG.

## What sort of things do we want our language model to be able to do?
* In our actual project (microgrid), here are some good ideas:
  * A language-based dashboard/console - give the language model access to all the solar panel and energy grid data, so it can answer any question users can think of.
  * Control the DDPG agent policy - if we want to change the way we optimize for power, e.g. pursue selling more aggressively, prioritize storing power for emergencies etc.; users can give commands via the language model
    * Actually, this may not be a good idea if users end up causing blackouts for themselves for ex.... but, maybe it's good functionality and we can implement safeguards (and we could make those safeguards optional for power users)
  * Interface with a [classic] visual dashboard - a visual dashboard with numbers, graphs and buttons seems like a more natural interface with your home power grid than a chat UI. LLM direct querying can be a specific mode, but having the default be more of a classic dashboard seems appealing. If we make the commands (i.e. buttons) in the dashboard call the language model, we want to be sure they always work as intended (as in pure logic) and are not negatively affected by stochasticity of the language model. (Maybe we build a hybrid dashboard, where different "classic" logic UIs can be activated based on what the user tells the language model to do.)


## Progress
Currently, the reward function ends up returning nan after several steps of an epoch. This may be due to instability because the reward function is based on the four parameters we are optimizing for... but hoping this is not the case. Current approach to solve this is to squeeze the ouput of `get_action()` in DDPGagent.py into some interval, say [-1, 1]...

## Credits
Thanks to Chris Yoon on [Medium](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) for guidance on DDPG implementation!