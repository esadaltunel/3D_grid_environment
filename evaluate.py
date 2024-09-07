import torch
from dqn import DQNAgent
from env_obs_free import Env

done = False
env = Env()
state = env._get_obs()

agent = DQNAgent(len(state), len(env._action_to_direction), 42)
state_dict = torch.load("2024-09-07_model.pth")
agent.qnetwork_local.load_state_dict(state_dict)

while not done:
    action = agent.act(state, 0.0)
    next_state, reward, done, info = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state

env.render()